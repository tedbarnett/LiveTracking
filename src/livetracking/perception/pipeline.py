"""Perception pipeline orchestrator.

For one (color, depth) frame:

  1. Stage 1 (`geometry.detect_blobs`) finds depth-foreground blobs inside
     the projector footprint.
  2. Run Grounding DINO once on the whole color frame.
  3. Filter DINO detections: keep only those whose bbox center lies inside
     the projector footprint. This is the standing rule (skill #1).
  4. For each surviving DINO detection, point-prompt SAM 2 at its bbox
     center, take the best mask, AND with the footprint, and turn that
     into a FreshDetection.
  5. Hand the FreshDetections to the ObjectTracker for stable ids.
  6. Warp each cam_mask into projector coords using H.

Why DINO-driven prompts rather than Stage-1-driven? Because Stage 1 reliably
gives us ONE giant "stuff in front of the wall" blob (sofa + everything on
it). DINO gives a per-object guess for each thing on the sofa. Combining
"DINO says where the objects are" with "Stage 1 + footprint restricts where
we'll accept them" is the best of both: open-vocab labels AND no off-axis
false positives.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .footprint import footprint_mask_in_camera
from .geometry import GeometryParams, detect_blobs
from .recognize import DEFAULT_DINO_PROMPT, Recognizer
from .tracker import FreshDetection, ObjectTracker
from .types import DetectedObject


@dataclass
class PipelineConfig:
    intrinsics: Tuple[float, float, float, float] = (615.0, 615.0, 424.0, 240.0)
    proj_w: int = 3840
    proj_h: int = 2160
    geometry: GeometryParams = field(default_factory=GeometryParams)
    dino_prompt: str = DEFAULT_DINO_PROMPT
    dino_box_thresh: float = 0.30
    dino_text_thresh: float = 0.25
    min_dino_score: float = 0.35
    min_obj_area_px: int = 800
    require_geometric_overlap: bool = True   # require some Stage-1 overlap
    geometric_overlap_min: float = 0.15
    iou_match_threshold: float = 0.25
    stale_after_s: float = 5.0


class Pipeline:
    def __init__(
        self,
        H: np.ndarray,
        cam_w: int,
        cam_h: int,
        config: Optional[PipelineConfig] = None,
        recognizer: Optional[Recognizer] = None,
    ):
        self.H = H
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.cfg = config or PipelineConfig()
        self.footprint = footprint_mask_in_camera(
            H, self.cfg.proj_w, self.cfg.proj_h, cam_w, cam_h
        )
        self.recognizer = recognizer or Recognizer()
        self.tracker = ObjectTracker(
            iou_match_threshold=self.cfg.iou_match_threshold,
            stale_after_s=self.cfg.stale_after_s,
        )
        # Cache the last DINO pass and re-run every N frames for performance.
        self._dino_every_n = 1
        self._frame_idx = 0
        self._last_dino: List[dict] = []
        self.last_stage1_debug: dict = {}

    def _dedupe_dino(self, dets: List[dict]) -> List[dict]:
        """Drop DINO bboxes that are mostly redundant with a higher-scoring one."""
        if not dets:
            return []
        dets = sorted(dets, key=lambda d: -d["score"])
        keep: List[dict] = []
        for d in dets:
            x, y, w, h = d["bbox"]
            ok = True
            for k in keep:
                kx, ky, kw, kh = k["bbox"]
                ix0 = max(x, kx); iy0 = max(y, ky)
                ix1 = min(x + w, kx + kw); iy1 = min(y + h, ky + kh)
                iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
                inter = iw * ih
                small = min(w * h, kw * kh)
                if small > 0 and inter / small > 0.6:
                    ok = False
                    break
            if ok:
                keep.append(d)
        return keep

    def _inside_footprint(self, bbox: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = bbox
        cx, cy = int(x + w / 2), int(y + h / 2)
        if not (0 <= cx < self.cam_w and 0 <= cy < self.cam_h):
            return False
        return bool(self.footprint[cy, cx] > 0)

    def step(self, color: np.ndarray, depth_m: np.ndarray) -> List[DetectedObject]:
        t0 = time.perf_counter()
        # ---- Stage 1: depth-foreground (gives us proof there's stuff inside the footprint) ----
        blobs, dbg = detect_blobs(
            color, depth_m, self.footprint, self.cfg.intrinsics, self.cfg.geometry
        )
        self.last_stage1_debug = dbg
        # Union of all Stage-1 blob masks — used to require any new object to
        # overlap the depth-foreground region (kills DINO hallucinations on the
        # bare wall).
        geom_union = np.zeros((self.cam_h, self.cam_w), np.uint8)
        for b in blobs:
            geom_union = cv2.bitwise_or(geom_union, b.cam_mask)
        t_stage1 = time.perf_counter()

        # ---- DINO (every frame for now; can be throttled later) ----
        dino_all = self.recognizer.label_image(
            color, prompt=self.cfg.dino_prompt,
            box_threshold=self.cfg.dino_box_thresh,
            text_threshold=self.cfg.dino_text_thresh,
        )
        dino_filt = [d for d in dino_all
                     if d["score"] >= self.cfg.min_dino_score
                     and self._inside_footprint(d["bbox"])]
        dino_filt = self._dedupe_dino(dino_filt)
        t_dino = time.perf_counter()

        # ---- SAM2 prompted by each surviving DINO detection's bbox center ----
        if dino_filt:
            prompts = [(d["bbox"][0] + d["bbox"][2] / 2,
                        d["bbox"][1] + d["bbox"][3] / 2) for d in dino_filt]
            sam_results = self.recognizer.segment_with_points(color, prompts)
        else:
            sam_results = []
        t_sam = time.perf_counter()

        # ---- Build FreshDetections (mask ∩ footprint, then geometric-overlap gate) ----
        fresh: List[FreshDetection] = []
        for d, (sam_mask, sam_score) in zip(dino_filt, sam_results):
            cam_mask = cv2.bitwise_and(sam_mask, self.footprint * 255)
            area = int((cam_mask > 0).sum())
            if area < self.cfg.min_obj_area_px:
                continue
            if self.cfg.require_geometric_overlap:
                overlap = int(np.logical_and(cam_mask > 0, geom_union > 0).sum())
                if overlap / max(1, area) < self.cfg.geometric_overlap_min:
                    continue
            # Median depth on the cam_mask
            z = depth_m[(cam_mask > 0) & (depth_m > 0)]
            med_z = float(np.median(z)) if z.size else 0.0
            # Warp into projector space
            proj_mask = cv2.warpPerspective(
                cam_mask, self.H, (self.cfg.proj_w, self.cfg.proj_h),
                flags=cv2.INTER_NEAREST,
            )
            proj_cy, proj_cx = np.where(proj_mask > 0)
            if proj_cy.size:
                proj_centroid = (float(proj_cx.mean()), float(proj_cy.mean()))
            else:
                proj_centroid = None
            fresh.append(FreshDetection(
                cam_mask=cam_mask,
                label=d["label"],
                label_score=float(d["score"] * sam_score),
                median_depth_m=med_z,
                proj_mask=proj_mask if proj_mask.any() else None,
                proj_centroid=proj_centroid,
            ))

        objects = self.tracker.update(fresh)
        t_end = time.perf_counter()
        self.last_timings_ms = {
            "stage1_ms": (t_stage1 - t0) * 1000,
            "dino_ms": (t_dino - t_stage1) * 1000,
            "sam_ms": (t_sam - t_dino) * 1000,
            "merge_ms": (t_end - t_sam) * 1000,
            "total_ms": (t_end - t0) * 1000,
            "n_dino_raw": len(dino_all),
            "n_dino_kept": len(dino_filt),
            "n_objects": len(objects),
        }
        self._frame_idx += 1
        return objects

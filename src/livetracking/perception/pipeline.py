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
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .footprint import footprint_mask_in_camera
from .geometry import GeometryParams, detect_blobs
from .recognize import DEFAULT_DINO_PROMPT, Recognizer
from .tracker import FreshDetection, ObjectTracker
from .types import DetectedObject


def _bbox_contains(bbox, pt) -> bool:
    """True iff (x,y) lies within (x,y,w,h)."""
    x, y, w, h = bbox
    px, py = pt
    return x <= px <= x + w and y <= py <= y + h


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
    # 60s of DINO silence before we retire a track. With async DINO running
    # every ~2s, this gives a track ~30 chances to be re-seen. Combined with
    # fast_step keeping `last_seen_t` fresh (it calls refresh_track every fast
    # frame for matched tracks), retirement now only fires when an object
    # genuinely leaves the scene for a full minute.
    stale_after_s: float = 60.0
    # Heavy DINO+SAM runs every N frames. The N-1 frames between use Stage-1
    # blobs to refresh each tracked object's cam_mask, parallax and proj_mask
    # in-place. 1 = full pass every frame (old behaviour, ~1-2 fps).
    # 6 ≈ 5-7 fps end-to-end on the 5090 with 30 Hz Stage-1.
    recognize_every_n: int = 6
    # On fast frames, a Stage-1 blob is accepted as the same tracked object if
    # IoU(prior cam_mask, blob mask) >= this. Below it, we fall back to nearest
    # centroid (within fast_centroid_max_px), otherwise the object is left at
    # its last known state (likely momentarily occluded).
    fast_iou_min: float = 0.10
    fast_centroid_max_px: int = 80
    # Stage-1-driven provisional promotion (Option B fix for "objects keep
    # appearing/disappearing"). Any unmatched Stage-1 blob inside the footprint
    # that persists across this many consecutive fast_step frames gets promoted
    # to a real track with a placeholder name. The async recognizer attaches a
    # real label later. At ~10 Hz fast frames, 25 → ~2.5 s of "really there"
    # evidence before we add an object. Set higher to be more conservative.
    provisional_promote_frames: int = 25
    provisional_match_dist_px: float = 60.0
    provisional_min_blob_area_px: int = 1500
    # Max consecutive fast frames a provisional can be missing before its
    # hit-count is reset (kills flickering noise from accruing hits across
    # long gaps). 2 frames = ~200 ms tolerance.
    provisional_max_gap_frames: int = 2
    # Async mode: when True, DINO+SAM run in a background thread and the main
    # loop always runs the FAST path. The heavy thread refreshes the tracker
    # whenever it finishes. Set False to use sync FAST/FULL alternation.
    async_recognize: bool = True


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
        # Async recognize machinery — only used when cfg.async_recognize=True.
        # The main loop submits the most recent (color, depth_m) into a 1-slot
        # box; the background thread pops it, runs DINO+SAM, calls
        # tracker.update(...) under tracker_lock, and the result becomes
        # visible to the next fast_step.
        self._async_thread = None
        self._async_box = None              # (color, depth_m) — newest only
        self._async_lock = threading.Lock()
        self._async_cv = threading.Condition(self._async_lock)
        self._async_running = False
        self.tracker_lock = threading.Lock()
        self.last_async_timings_ms: dict = {}
        # Provisional blob history for Stage-1-driven track promotion.
        # Each entry: {"centroid": (x,y), "last_seen_frame": int, "hits": int,
        #              "last_cam_mask": ndarray, "last_depth": float}
        self._provisional_blobs: List[dict] = []

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

    def _warp_with_parallax(
        self,
        cam_mask: np.ndarray,
        depth_m: np.ndarray,
        plane,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]], float]:
        """Compute median depth + parallax-shifted proj_mask + proj_centroid.

        Returns (proj_mask_or_None, proj_centroid_or_None, median_depth_m).
        Used by both step() (full pass) and fast_step() (Stage-1-driven update).
        """
        fx, fy, cx, cy = self.cfg.intrinsics
        a, b, c, d_plane = plane

        z_pix = depth_m[(cam_mask > 0) & (depth_m > 0)]
        med_z = float(np.median(z_pix)) if z_pix.size else 0.0

        ys_m, xs_m = np.where(cam_mask > 0)
        cam_mask_for_warp = cam_mask
        if med_z > 0.1 and ys_m.size:
            cx_m = float(xs_m.mean()); cy_m = float(ys_m.mean())
            u = (cx_m - cx) / fx
            v = (cy_m - cy) / fy
            denom = a * u + b * v + c
            z_wall = (-d_plane / denom) if abs(denom) > 1e-6 else 0.0
            if z_wall > 0.1:
                scale = (z_wall - med_z) / z_wall
                shift_x = (cx_m - cx) * scale
                shift_y = (cy_m - cy) * scale
                if abs(shift_x) > 0.5 or abs(shift_y) > 0.5:
                    M = np.array([[1.0, 0.0, shift_x],
                                  [0.0, 1.0, shift_y]], dtype=np.float32)
                    cam_mask_for_warp = cv2.warpAffine(
                        cam_mask, M, (self.cam_w, self.cam_h),
                        flags=cv2.INTER_NEAREST, borderValue=0,
                    )

        proj_mask = cv2.warpPerspective(
            cam_mask_for_warp, self.H, (self.cfg.proj_w, self.cfg.proj_h),
            flags=cv2.INTER_NEAREST,
        )
        if not proj_mask.any():
            return None, None, med_z
        py, px = np.where(proj_mask > 0)
        return proj_mask, (float(px.mean()), float(py.mean())), med_z

    def step_auto(self, color: np.ndarray, depth_m: np.ndarray) -> List[DetectedObject]:
        """Pick the right path based on config.

        async_recognize=True (default for the live daemon): always run fast_step;
            the background recognize thread refreshes tracks/labels in parallel.
            Submit the freshest frame to the recognize thread on the way through.
        async_recognize=False: sync FAST/FULL alternation by recognize_every_n.
        """
        if self.cfg.async_recognize:
            if not self._async_running:
                self.start_async()
            # Hand the freshest frame to the recognize thread (replaces any
            # pending one — we always work on the latest).
            with self._async_cv:
                self._async_box = (color, depth_m)
                self._async_cv.notify()
            # If tracker has nothing yet, run a sync full pass so we boot.
            if not self.tracker.active():
                with self.tracker_lock:
                    return self.step(color, depth_m)
            return self.fast_step(color, depth_m)

        if (self.cfg.recognize_every_n <= 1
                or self._frame_idx % self.cfg.recognize_every_n == 0
                or not self.tracker.active()):
            return self.step(color, depth_m)
        return self.fast_step(color, depth_m)

    # ---- async recognize thread ----
    def start_async(self) -> None:
        if self._async_running:
            return
        self._async_running = True
        self._async_thread = threading.Thread(
            target=self._async_loop, name="recognize", daemon=True
        )
        self._async_thread.start()

    def stop_async(self) -> None:
        self._async_running = False
        with self._async_cv:
            self._async_cv.notify_all()

    def _async_loop(self) -> None:
        while self._async_running:
            with self._async_cv:
                while self._async_running and self._async_box is None:
                    self._async_cv.wait(timeout=0.5)
                if not self._async_running:
                    return
                color, depth_m = self._async_box
                self._async_box = None
            try:
                fresh, timings = self._compute_fresh(color, depth_m)
                with self.tracker_lock:
                    objects = self.tracker.update(fresh)
                timings["n_objects"] = len(objects)
                timings["total_ms"] = timings["stage1_ms"] + timings["dino_ms"] + timings["sam_ms"]
                self.last_async_timings_ms = timings
            except Exception as e:
                print(f"[pipeline.async] error: {e!r}")

    def fast_step(self, color: np.ndarray, depth_m: np.ndarray) -> List[DetectedObject]:
        """Cheap update: re-run Stage 1 only and bind blobs to existing tracks.

        For each currently-tracked object, find the Stage-1 blob with greatest
        IoU against its prior cam_mask. If none qualifies, fall back to the
        nearest blob within fast_centroid_max_px. Refresh the track's cam_mask,
        parallax-corrected proj_mask, centroids and median depth in-place.
        """
        t0 = time.perf_counter()
        blobs, dbg = detect_blobs(
            color, depth_m, self.footprint, self.cfg.intrinsics, self.cfg.geometry
        )
        self.last_stage1_debug = dbg
        plane = dbg.get("plane", [0, 0, -1, 3.0])
        t_stage1 = time.perf_counter()

        with self.tracker_lock:
            tracked = list(self.tracker.active())
        used_blob_idx: set = set()
        refresh_calls = []  # apply under lock at the end
        for obj in tracked:
            prior = obj.cam_mask
            best_i, best_iou = -1, 0.0
            for i, b in enumerate(blobs):
                if i in used_blob_idx:
                    continue
                inter = int(np.logical_and(prior > 0, b.cam_mask > 0).sum())
                union = int(np.logical_or(prior > 0, b.cam_mask > 0).sum())
                if union == 0:
                    continue
                iou = inter / union
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i < 0 or best_iou < self.cfg.fast_iou_min:
                # Centroid fallback
                pcx, pcy = obj.centroid_cam
                best_c, best_dist = -1, float("inf")
                for i, b in enumerate(blobs):
                    if i in used_blob_idx:
                        continue
                    ys, xs = np.where(b.cam_mask > 0)
                    if not ys.size:
                        continue
                    bcx, bcy = float(xs.mean()), float(ys.mean())
                    d = ((bcx - pcx) ** 2 + (bcy - pcy) ** 2) ** 0.5
                    if d < best_dist:
                        best_dist, best_c = d, i
                if best_c >= 0 and best_dist < self.cfg.fast_centroid_max_px:
                    best_i = best_c
            if best_i < 0:
                continue  # leave object untouched — momentarily lost
            used_blob_idx.add(best_i)
            new_cam_mask = cv2.bitwise_and(blobs[best_i].cam_mask,
                                           self.footprint * 255)
            if int((new_cam_mask > 0).sum()) < self.cfg.min_obj_area_px:
                continue
            proj_mask, proj_centroid, med_z = self._warp_with_parallax(
                new_cam_mask, depth_m, plane
            )
            refresh_calls.append((obj.object_id, new_cam_mask, proj_mask, proj_centroid, med_z))

        # ---- Provisional promotion (Stage-1-driven, no DINO required) -----
        # For each blob NOT bound to an existing track, see if it's also
        # outside every track's bbox. If so, treat it as a candidate provisional.
        # Match across fast frames by centroid distance; promote after N hits.
        new_promotions: List[dict] = []
        with self.tracker_lock:
            active_objs = list(self.tracker.active())
        for i, b in enumerate(blobs):
            if i in used_blob_idx:
                continue
            ys, xs = np.where(b.cam_mask > 0)
            if not ys.size:
                continue
            area = int((b.cam_mask > 0).sum())
            if area < self.cfg.provisional_min_blob_area_px:
                continue
            bcx, bcy = float(xs.mean()), float(ys.mean())
            # Skip if this blob overlaps any active track meaningfully
            # (avoids creating a duplicate provisional for a partially-occluded
            # tracked object that fast_step couldn't match this frame).
            already_tracked = False
            for o in active_objs:
                if _bbox_contains(o.bbox_cam, (bcx, bcy)):
                    already_tracked = True
                    break
                ti = int(np.logical_and(b.cam_mask > 0, o.cam_mask > 0).sum())
                if ti / max(1, area) > 0.05:
                    already_tracked = True
                    break
            if already_tracked:
                continue
            # Match against existing provisional history by centroid distance
            best_p, best_dist = -1, float("inf")
            for pi, p in enumerate(self._provisional_blobs):
                pcx, pcy = p["centroid"]
                d = ((bcx - pcx) ** 2 + (bcy - pcy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist, best_p = d, pi
            if best_p >= 0 and best_dist < self.cfg.provisional_match_dist_px:
                p = self._provisional_blobs[best_p]
                gap = self._frame_idx - p["last_seen_frame"]
                p["centroid"] = (bcx, bcy)
                p["last_seen_frame"] = self._frame_idx
                p["last_cam_mask"] = b.cam_mask
                # If we lost sight of this blob for > max_gap frames, treat
                # the previous run as unreliable and restart the count. Stops
                # one-off Stage-1 noise from accumulating hits across seconds
                # of absence.
                if gap > self.cfg.provisional_max_gap_frames:
                    p["hits"] = 1
                else:
                    p["hits"] += 1
                if p["hits"] >= self.cfg.provisional_promote_frames:
                    new_promotions.append(p)
                    self._provisional_blobs.pop(best_p)
            else:
                self._provisional_blobs.append({
                    "centroid": (bcx, bcy),
                    "last_seen_frame": self._frame_idx,
                    "hits": 1,
                    "last_cam_mask": b.cam_mask,
                })
        # Reap provisionals not seen in the last 3 fast frames.
        self._provisional_blobs = [
            p for p in self._provisional_blobs
            if self._frame_idx - p["last_seen_frame"] <= 3
        ]

        with self.tracker_lock:
            for oid, cm, pm, pc, mz in refresh_calls:
                self.tracker.refresh_track(
                    oid, cam_mask=cm, proj_mask=pm,
                    proj_centroid=pc, median_depth_m=mz,
                )
            for p in new_promotions:
                cam_mask = cv2.bitwise_and(p["last_cam_mask"], self.footprint * 255)
                proj_mask, proj_centroid, med_z = self._warp_with_parallax(
                    cam_mask, depth_m, plane
                )
                self.tracker.promote_provisional(
                    cam_mask=cam_mask,
                    proj_mask=proj_mask,
                    proj_centroid=proj_centroid,
                    median_depth_m=med_z,
                )
            objects = self.tracker.active()
        t_end = time.perf_counter()
        self.last_timings_ms = {
            "stage1_ms": (t_stage1 - t0) * 1000,
            "dino_ms": 0.0,
            "sam_ms": 0.0,
            "merge_ms": (t_end - t_stage1) * 1000,
            "total_ms": (t_end - t0) * 1000,
            "n_dino_raw": 0,
            "n_dino_kept": 0,
            "n_objects": len(objects),
            "fast": True,
        }
        self._frame_idx += 1
        return objects

    def step(self, color: np.ndarray, depth_m: np.ndarray) -> List[DetectedObject]:
        """Synchronous full pass: Stage 1 + DINO + SAM + tracker.update."""
        t0 = time.perf_counter()
        fresh, timings = self._compute_fresh(color, depth_m)
        objects = self.tracker.update(fresh)
        t_end = time.perf_counter()
        timings["merge_ms"] = (t_end - t0) * 1000 - timings["stage1_ms"] - timings["dino_ms"] - timings["sam_ms"]
        timings["total_ms"] = (t_end - t0) * 1000
        timings["n_objects"] = len(objects)
        self.last_timings_ms = timings
        self._frame_idx += 1
        return objects

    def _compute_fresh(
        self, color: np.ndarray, depth_m: np.ndarray
    ) -> Tuple[List[FreshDetection], dict]:
        """Heavy DINO + SAM + parallax → list of FreshDetection. No tracker.

        Used by step() (sync) and the async recognize thread.
        """
        t0 = time.perf_counter()
        blobs, dbg = detect_blobs(
            color, depth_m, self.footprint, self.cfg.intrinsics, self.cfg.geometry
        )
        self.last_stage1_debug = dbg
        geom_union = np.zeros((self.cam_h, self.cam_w), np.uint8)
        for b in blobs:
            geom_union = cv2.bitwise_or(geom_union, b.cam_mask)
        t_stage1 = time.perf_counter()

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

        if dino_filt:
            prompts = [(d["bbox"][0] + d["bbox"][2] / 2,
                        d["bbox"][1] + d["bbox"][3] / 2) for d in dino_filt]
            sam_results = self.recognizer.segment_with_points(color, prompts)
        else:
            sam_results = []
        t_sam = time.perf_counter()

        plane = dbg.get("plane", [0, 0, -1, 3.0])
        fresh: List[FreshDetection] = []
        for det, (sam_mask, sam_score) in zip(dino_filt, sam_results):
            cam_mask = cv2.bitwise_and(sam_mask, self.footprint * 255)
            area = int((cam_mask > 0).sum())
            if area < self.cfg.min_obj_area_px:
                continue
            if self.cfg.require_geometric_overlap:
                overlap = int(np.logical_and(cam_mask > 0, geom_union > 0).sum())
                if overlap / max(1, area) < self.cfg.geometric_overlap_min:
                    continue
            proj_mask, proj_centroid, med_z = self._warp_with_parallax(
                cam_mask, depth_m, plane
            )
            fresh.append(FreshDetection(
                cam_mask=cam_mask,
                label=det["label"],
                label_score=float(det["score"] * sam_score),
                median_depth_m=med_z,
                proj_mask=proj_mask,
                proj_centroid=proj_centroid,
            ))
        timings = {
            "stage1_ms": (t_stage1 - t0) * 1000,
            "dino_ms": (t_dino - t_stage1) * 1000,
            "sam_ms": (t_sam - t_dino) * 1000,
            "n_dino_raw": len(dino_all),
            "n_dino_kept": len(dino_filt),
        }
        return fresh, timings

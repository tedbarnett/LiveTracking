"""Stage 2: open-vocabulary recognition + clean masks via SAM 2 + Grounding DINO.

Inputs from Stage 1: a list of Blob objects (with cam_mask, centroid, bbox).
Outputs: a list of DetectedObject — clean SAM-refined masks and DINO labels.

Models (all auto-downloaded from HuggingFace on first use, cached in ~/.cache/huggingface):
  - SAM 2.1 Hiera-Large  via `transformers` Sam2Model
      checkpoint: facebook/sam2.1-hiera-large
  - Grounding DINO Tiny  via `transformers` AutoModelForZeroShotObjectDetection
      checkpoint: IDEA-Research/grounding-dino-tiny

Heavy imports (torch, transformers) are lazy so importing this module doesn't
slow down lightweight CLI work that only needs Stage 1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .types import Blob


DEFAULT_DINO_PROMPT = (
    "guitar. acoustic guitar. electric guitar. drum. drum head. snare drum. "
    "tambourine. cymbal. piano. keyboard. microphone. cushion. throw pillow. "
    "sofa. couch. armchair. chair. ottoman. picture frame. painting. map. "
    "poster. mirror. lamp. plant. book. mug. cup. bottle. can. plate. bowl. "
    "remote. phone. laptop. notebook. hand. arm. person. face. box. bag. hat. "
    "ball. toy. blanket. towel. shoe. object."
)


@dataclass
class RecognitionResult:
    sam_mask_cam: np.ndarray          # uint8 {0,255}, camera-space, refined mask
    label: str                        # best DINO label
    label_score: float                # 0..1
    sam_score: float                  # SAM's own quality score
    source_blob_id: int


class Recognizer:
    """Loads SAM2 + Grounding DINO once, then segments+labels on demand."""

    def __init__(
        self,
        sam_checkpoint: str = "facebook/sam2.1-hiera-large",
        dino_checkpoint: str = "IDEA-Research/grounding-dino-tiny",
        device: Optional[str] = None,
        dtype: str = "float16",
    ):
        import torch                                    # heavy import deferred
        from transformers import (                       # noqa: F401
            AutoProcessor,
            AutoModelForZeroShotObjectDetection,
            Sam2Model,
        )

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype)

        print(f"[recognize] loading SAM2 ({sam_checkpoint}) on {self.device}...")
        t0 = time.perf_counter()
        self.sam_processor = AutoProcessor.from_pretrained(sam_checkpoint)
        self.sam_model = Sam2Model.from_pretrained(sam_checkpoint).to(self.device)
        self.sam_model.eval()
        print(f"[recognize] SAM2 loaded in {time.perf_counter()-t0:.1f}s")

        print(f"[recognize] loading Grounding DINO ({dino_checkpoint})...")
        t0 = time.perf_counter()
        self.dino_processor = AutoProcessor.from_pretrained(dino_checkpoint)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_checkpoint
        ).to(self.device)
        self.dino_model.eval()
        print(f"[recognize] DINO loaded in {time.perf_counter()-t0:.1f}s")

    @property
    def torch(self):
        return self._torch

    # ----- SAM 2 ---------------------------------------------------------------
    def segment_with_points(
        self, color_bgr: np.ndarray, points_cam: List[Tuple[float, float]]
    ) -> List[Tuple[np.ndarray, float]]:
        """For each (x, y) point in camera pixels, return (mask uint8, score).

        Internally batched: all points are sent to SAM2 in a single forward
        pass (one object per point), so cost scales much better than the old
        per-point Python loop. ~600 ms for 16 points on a 5090, vs ~7 s if
        looped.
        """
        torch = self._torch
        if not points_cam:
            return []
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        # Shape (B=1, N_obj=len(points), N_pts=1, 2)
        input_points = [[[[float(px), float(py)]] for (px, py) in points_cam]]
        input_labels = [[[1] for _ in points_cam]]  # 1 = foreground
        inputs = self.sam_processor(
            images=rgb, input_points=input_points, input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.sam_model(**inputs, multimask_output=True)
        masks = self.sam_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
        )[0]  # (N_obj, N_masks, H, W) bool
        # iou_scores: (B, N_obj, N_masks)
        scores_all = outputs.iou_scores[0].cpu().numpy()  # (N_obj, N_masks)
        results: List[Tuple[np.ndarray, float]] = []
        for i in range(len(points_cam)):
            scores = scores_all[i]
            best = int(np.argmax(scores))
            m = masks[i, best].numpy().astype(np.uint8) * 255
            results.append((m, float(scores[best])))
        return results

    # ----- SAM 2 automatic mask generator (for splitting mega-blobs) ----------
    def auto_segment_in_bbox(
        self,
        color_bgr: np.ndarray,
        bbox_xywh: Tuple[int, int, int, int],
        parent_mask: Optional[np.ndarray] = None,
        grid_n: int = 4,
        score_thresh: float = 0.55,
        dedupe_iou: float = 0.55,
        max_area_frac: float = 0.85,
        min_area_px: int = 800,
    ) -> List[Tuple[np.ndarray, float]]:
        """Split a single Stage-1 mega-blob into per-object sub-masks.

        Generates a grid_n x grid_n grid of point prompts inside ``bbox_xywh``,
        filters points to those lying inside ``parent_mask`` (so we don't
        waste SAM forwards on background pixels), runs SAM2 in one batched
        forward, then dedupes the resulting masks by IoU > ``dedupe_iou``.

        Heuristics applied to suppress SAM's typical failure modes:
          * Drop masks whose area is > ``max_area_frac`` of the parent — that's
            SAM "expanding to the whole couch" when no clear sub-object is at
            that point.
          * Drop masks with area < ``min_area_px`` (noise).
          * Drop masks below ``score_thresh`` (low SAM confidence).
          * Greedy NMS: keep highest-score mask first, drop later masks whose
            IoU vs any kept mask exceeds ``dedupe_iou``.

        Returns [(uint8 mask, sam_score), ...] in score-descending order.
        """
        x, y, w, h = bbox_xywh
        if w <= 0 or h <= 0:
            return []
        H_full, W_full = color_bgr.shape[:2]
        parent_area = int((parent_mask > 0).sum()) if parent_mask is not None else (w * h)
        if parent_area <= 0:
            return []
        # Grid points across the bbox, with inset so we don't sample right on
        # the boundary where SAM tends to grab the background.
        inset_x = w / (grid_n + 1)
        inset_y = h / (grid_n + 1)
        points: List[Tuple[float, float]] = []
        for gy in range(grid_n):
            for gx in range(grid_n):
                px = x + inset_x * (gx + 1)
                py = y + inset_y * (gy + 1)
                if not (0 <= px < W_full and 0 <= py < H_full):
                    continue
                if parent_mask is not None and parent_mask[int(py), int(px)] == 0:
                    continue
                points.append((px, py))
        if not points:
            return []
        raw = self.segment_with_points(color_bgr, points)
        # Filter by score / area / parent overlap
        candidates: List[Tuple[np.ndarray, float]] = []
        for mask, score in raw:
            if score < score_thresh:
                continue
            area = int((mask > 0).sum())
            if area < min_area_px:
                continue
            if area > parent_area * max_area_frac:
                continue
            if parent_mask is not None:
                inside = int(np.logical_and(mask > 0, parent_mask > 0).sum())
                if inside / max(1, area) < 0.5:
                    continue
            candidates.append((mask, score))
        candidates.sort(key=lambda t: -t[1])
        # Greedy NMS by IoU
        kept: List[Tuple[np.ndarray, float]] = []
        for mask, score in candidates:
            ok = True
            for kmask, _ in kept:
                inter = int(np.logical_and(mask > 0, kmask > 0).sum())
                union = int(np.logical_or(mask > 0, kmask > 0).sum())
                if union > 0 and inter / union > dedupe_iou:
                    ok = False
                    break
            if ok:
                kept.append((mask, score))
        return kept

    # ----- Grounding DINO ------------------------------------------------------
    def label_image(
        self,
        color_bgr: np.ndarray,
        prompt: str = DEFAULT_DINO_PROMPT,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> List[dict]:
        """Return [{'label': str, 'score': float, 'bbox': (x,y,w,h)}, ...] over the whole image."""
        torch = self._torch
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.dino_processor(
            images=rgb, text=prompt, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.dino_model(**inputs)
        H, W = rgb.shape[:2]
        target_sizes = torch.tensor([[H, W]], device=self.device)
        post = self.dino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]
        out = []
        # transformers 5.x returns 'text_labels'; older returns 'labels'.
        label_list = post.get("text_labels", post.get("labels", []))
        for score, label, box in zip(
            post["scores"].cpu().numpy(),
            label_list,
            post["boxes"].cpu().numpy(),
        ):
            x0, y0, x1, y1 = box
            out.append({
                "label": str(label),
                "score": float(score),
                "bbox": (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
            })
        return out

    # ----- Combined ------------------------------------------------------------
    def recognize(
        self,
        color_bgr: np.ndarray,
        blobs: List[Blob],
        prompt: str = DEFAULT_DINO_PROMPT,
    ) -> List[RecognitionResult]:
        """Per Stage-1 blob: SAM-segment around its centroid, label via DINO IoU."""
        results: List[RecognitionResult] = []
        if not blobs:
            return results
        sam_out = self.segment_with_points(
            color_bgr, [b.centroid_cam for b in blobs]
        )
        dino_dets = self.label_image(color_bgr, prompt=prompt)

        for blob, (sam_mask, sam_score) in zip(blobs, sam_out):
            # Assign label by best IoU between sam_mask bbox and any DINO bbox
            bx, by, bw, bh = _mask_bbox(sam_mask)
            best_label = "object"
            best_score = 0.0
            for det in dino_dets:
                iou = _bbox_iou((bx, by, bw, bh), det["bbox"])
                weighted = iou * det["score"]
                if weighted > best_score:
                    best_score = weighted
                    best_label = det["label"] or "object"
            results.append(RecognitionResult(
                sam_mask_cam=sam_mask,
                label=best_label,
                label_score=best_score,
                sam_score=sam_score,
                source_blob_id=blob.blob_id,
            ))
        return results


# ---- helpers --------------------------------------------------------------
def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def _bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix0 = max(ax, bx); iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw); iy1 = min(ay + ah, by + bh)
    iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

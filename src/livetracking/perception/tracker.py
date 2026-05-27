"""Object tracker — stable ids and palette colors across frames.

Stage 1 + Stage 2 produce a fresh list of (mask, label) per frame. The tracker
matches each fresh detection to an existing tracked object by mask IoU, so
``DetectedObject.object_id`` stays stable as the scene jitters or the user
moves.

Within a session:
- ids are assigned 1, 2, 3, ... and never reused.
- each id gets a stable palette color (cycled by ``id % len(PALETTE)``).
- names default to the DINO label and persist to ``runtime/object_names.json``
  so user-edited names survive perception_daemon restarts.
- an object whose mask drops out for > ``stale_after_s`` is retired.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from livetracking.paths import OBJECT_NAMES_FILE
from .types import DetectedObject


# 12-color palette — distinct under projector light, all sufficiently saturated.
PALETTE: List[Tuple[int, int, int]] = [
    (255, 80, 80),    # red
    (80, 255, 80),    # green
    (80, 80, 255),    # blue
    (255, 200, 0),    # amber
    (255, 80, 255),   # magenta
    (80, 255, 255),   # cyan
    (255, 130, 0),    # orange
    (160, 80, 255),   # purple
    (0, 255, 160),    # teal
    (255, 255, 80),   # yellow
    (255, 80, 160),   # pink
    (80, 200, 255),   # sky
]


@dataclass
class _Track:
    obj: DetectedObject
    age_frames: int = 0
    misses: int = 0


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    ab = (a > 0)
    bb = (b > 0)
    inter = int(np.logical_and(ab, bb).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(ab, bb).sum())
    return inter / union if union > 0 else 0.0


def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


@dataclass
class FreshDetection:
    """What perception delivers per frame, pre-tracking."""
    cam_mask: np.ndarray
    label: str
    label_score: float
    median_depth_m: float
    proj_mask: Optional[np.ndarray] = None
    proj_centroid: Optional[Tuple[float, float]] = None


class ObjectTracker:
    def __init__(
        self,
        iou_match_threshold: float = 0.3,
        stale_after_s: float = 2.0,
        names_path: str = OBJECT_NAMES_FILE,
    ):
        self.iou_match_threshold = iou_match_threshold
        self.stale_after_s = stale_after_s
        self.names_path = names_path
        self._tracks: Dict[int, _Track] = {}
        self._next_id: int = 1
        self._names: Dict[str, str] = self._load_names()

    # ---- name persistence ----
    def _load_names(self) -> Dict[str, str]:
        if not os.path.exists(self.names_path):
            return {}
        try:
            with open(self.names_path) as f:
                return dict(json.load(f))
        except Exception:
            return {}

    def _save_names(self) -> None:
        os.makedirs(os.path.dirname(self.names_path), exist_ok=True)
        with open(self.names_path, "w") as f:
            json.dump(self._names, f, indent=2)

    def rename(self, object_id: int, new_name: str) -> bool:
        track = self._tracks.get(object_id)
        if track is None:
            return False
        track.obj.name = new_name
        self._names[str(object_id)] = new_name
        self._save_names()
        return True

    # ---- per-frame update ----
    def update(self, fresh: List[FreshDetection]) -> List[DetectedObject]:
        now = time.time()
        # Greedy matching: prefer mask IoU, fall back to centroid proximity.
        unmatched_track_ids = set(self._tracks.keys())
        matched_fresh = set()
        pairs: List[Tuple[int, int, float]] = []  # (fresh_idx, track_id, score)
        for fi, fd in enumerate(fresh):
            f_cent = _mask_centroid(fd.cam_mask)
            for tid, tr in self._tracks.items():
                iou = _mask_iou(fd.cam_mask, tr.obj.cam_mask)
                # Secondary score: centroid distance, normalized to image diag.
                t_cent = tr.obj.centroid_cam
                dx, dy = f_cent[0] - t_cent[0], f_cent[1] - t_cent[1]
                dist = (dx * dx + dy * dy) ** 0.5
                # Combine: IoU dominates above threshold; otherwise allow a
                # close (< 40 px) re-match for transient mask wobble.
                if iou >= self.iou_match_threshold:
                    score = 1.0 + iou
                elif dist < 40.0:
                    score = 0.5 - dist / 100.0
                else:
                    continue
                pairs.append((fi, tid, score))
        pairs.sort(key=lambda x: -x[2])
        used_tracks = set()
        for fi, tid, iou in pairs:
            if fi in matched_fresh or tid in used_tracks:
                continue
            matched_fresh.add(fi)
            used_tracks.add(tid)
            unmatched_track_ids.discard(tid)
            # update existing track in place
            tr = self._tracks[tid]
            fd = fresh[fi]
            tr.obj.cam_mask = fd.cam_mask
            tr.obj.proj_mask = fd.proj_mask
            tr.obj.centroid_cam = _mask_centroid(fd.cam_mask)
            tr.obj.centroid_proj = fd.proj_centroid
            tr.obj.bbox_cam = _mask_bbox(fd.cam_mask)
            tr.obj.median_depth_m = fd.median_depth_m
            tr.obj.label_score = max(tr.obj.label_score, fd.label_score)
            # Only overwrite label/name with DINO's pick if user hasn't renamed.
            user_named = str(tid) in self._names
            if not user_named and fd.label and fd.label_score > tr.obj.label_score - 0.05:
                tr.obj.name = fd.label
            tr.obj.last_seen_t = now
            tr.age_frames += 1
            tr.misses = 0

        # Create new tracks for unmatched fresh detections.
        for fi, fd in enumerate(fresh):
            if fi in matched_fresh:
                continue
            new_id = self._next_id
            self._next_id += 1
            color = PALETTE[(new_id - 1) % len(PALETTE)]
            saved_name = self._names.get(str(new_id))
            obj = DetectedObject(
                object_id=new_id,
                name=saved_name or fd.label or f"object {new_id}",
                color_rgb=color,
                cam_mask=fd.cam_mask,
                proj_mask=fd.proj_mask,
                centroid_cam=_mask_centroid(fd.cam_mask),
                centroid_proj=fd.proj_centroid,
                bbox_cam=_mask_bbox(fd.cam_mask),
                median_depth_m=fd.median_depth_m,
                last_seen_t=now,
                label_score=fd.label_score,
            )
            self._tracks[new_id] = _Track(obj=obj, age_frames=1, misses=0)

        # Retire stale tracks.
        for tid in list(self._tracks.keys()):
            tr = self._tracks[tid]
            if tid in unmatched_track_ids:
                tr.misses += 1
            if now - tr.obj.last_seen_t > self.stale_after_s:
                del self._tracks[tid]

        return [tr.obj for tr in self._tracks.values()]

    def active(self) -> List[DetectedObject]:
        return [tr.obj for tr in self._tracks.values()]

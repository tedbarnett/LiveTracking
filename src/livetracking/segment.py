"""Depth-cluster foreground segmentation.

Takes a depth map (float32 meters; 0 = no data) and returns connected
components of pixels in the foreground band (typically 0.4-3.0 m).
Components are filtered by area, sorted by depth (nearest first), and
limited to a small count for keyboard selection (1-9).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


NEAR_M = 0.4
FAR_M = 3.0
MIN_AREA_PX = 1500       # at 848x480 — tune per scene
MAX_OBJECTS = 9          # number keys 1..9
MORPH_KERNEL = 5


@dataclass
class Segment:
    index: int                   # 1..N (0 reserved for background)
    mask: np.ndarray             # (H, W) bool
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    centroid_xy: Tuple[float, float]
    area_px: int
    median_depth_m: float


def _foreground_mask(depth_m: np.ndarray, near: float, far: float) -> np.ndarray:
    valid = depth_m > 0.001
    band = (depth_m >= near) & (depth_m <= far)
    return (valid & band).astype(np.uint8)


def segment(depth_m: np.ndarray,
            near: float = NEAR_M,
            far: float = FAR_M,
            min_area_px: int = MIN_AREA_PX,
            max_objects: int = MAX_OBJECTS) -> List[Segment]:
    """Return up to `max_objects` foreground segments, nearest first."""
    if depth_m.ndim != 2:
        raise ValueError(f"expected 2D depth, got shape {depth_m.shape}")

    fg = _foreground_mask(depth_m, near, far)
    if fg.sum() == 0:
        return []

    # Morphological clean-up: close small holes, then open to drop speckle.
    k = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)

    segs: List[Segment] = []
    for label in range(1, n_labels):  # 0 is background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        mask = labels == label
        depths = depth_m[mask]
        depths = depths[depths > 0]
        med = float(np.median(depths)) if depths.size else float("inf")
        cx, cy = centroids[label]
        segs.append(Segment(
            index=0,  # filled in after sorting
            mask=mask,
            bbox=(x, y, w, h),
            centroid_xy=(float(cx), float(cy)),
            area_px=area,
            median_depth_m=med,
        ))

    # Nearest first → most likely the thing the user wants.
    segs.sort(key=lambda s: s.median_depth_m)
    segs = segs[:max_objects]
    for i, s in enumerate(segs, start=1):
        s.index = i
    return segs

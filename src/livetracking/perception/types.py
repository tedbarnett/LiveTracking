"""Dataclasses passed between perception stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class Blob:
    """Stage-1 candidate region — depth foreground inside the projector footprint."""
    blob_id: int                       # transient, only valid for one frame
    cam_mask: np.ndarray               # (H, W) uint8 {0,255} in camera space
    centroid_cam: Tuple[float, float]  # (x, y) in camera pixels
    bbox_cam: Tuple[int, int, int, int]  # (x, y, w, h) in camera pixels
    area_px: int
    median_depth_m: float


@dataclass
class DetectedObject:
    """A tracked object — survives across frames, has a stable id and name."""
    object_id: int                     # 1-indexed, stable for the session
    name: str                          # editable in the UI
    color_rgb: Tuple[int, int, int]    # assigned palette color for highlight
    cam_mask: np.ndarray               # camera-space mask
    proj_mask: Optional[np.ndarray]    # projector-space mask (warpPerspective result)
    centroid_cam: Tuple[float, float]
    centroid_proj: Optional[Tuple[float, float]]
    bbox_cam: Tuple[int, int, int, int]
    median_depth_m: float
    last_seen_t: float                 # epoch seconds
    label_score: float = 0.0           # confidence from Grounding DINO (0..1)
    dino_label: str = ""               # last DINO class for this track (survives rename)
    hidden: bool = False               # user-hidden — filtered out of UI/projection
    effect: str = "color"              # render mode: "color" (flat) | "flame" | "cloud"
    aux: dict = field(default_factory=dict)

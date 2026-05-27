"""Projector footprint utilities.

The camera↔projector homography ``H`` maps camera pixels to projector pixels.
Its inverse maps the 4 projector-frame corners back to camera space, which
gives us the "footprint mask" — the polygon inside the camera frame that the
projector can actually light. ALL detection must be restricted to this mask
(see `computer-vision/projection-mapping` skill, rule #1).
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import cv2
import numpy as np

from livetracking.paths import HOMOGRAPHY_FILE, CALIB_META_FILE


def save_homography(
    H: np.ndarray,
    proj_size: Tuple[int, int],
    cam_size: Tuple[int, int],
    n_correspondences: int,
    extra: Optional[dict] = None,
    path: str = HOMOGRAPHY_FILE,
    meta_path: str = CALIB_META_FILE,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, H)
    meta = {
        "homography_file": os.path.basename(path),
        "proj_w": int(proj_size[0]),
        "proj_h": int(proj_size[1]),
        "cam_w": int(cam_size[0]),
        "cam_h": int(cam_size[1]),
        "n_correspondences": int(n_correspondences),
    }
    if extra:
        meta.update(extra)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_homography(
    path: str = HOMOGRAPHY_FILE,
    meta_path: str = CALIB_META_FILE,
) -> Tuple[np.ndarray, dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Homography not found at {path}. "
            "Run `python scripts/calibrate_homography.py` first."
        )
    H = np.load(path)
    meta: dict = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return H, meta


def footprint_mask_in_camera(
    H: np.ndarray, proj_w: int, proj_h: int, cam_w: int, cam_h: int
) -> np.ndarray:
    """Return a uint8 {0,1} mask of the projector footprint in camera space."""
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError as e:
        raise ValueError("Singular homography; recalibrate.") from e
    corners = np.array(
        [[[0, 0]], [[proj_w - 1, 0]], [[proj_w - 1, proj_h - 1]], [[0, proj_h - 1]]],
        dtype=np.float32,
    )
    in_cam = cv2.perspectiveTransform(corners, H_inv).reshape(-1, 2)
    mask = np.zeros((cam_h, cam_w), dtype=np.uint8)
    poly = np.round(in_cam).astype(np.int32)
    cv2.fillConvexPoly(mask, poly, 1)
    return mask


def footprint_corners_in_camera(
    H: np.ndarray, proj_w: int, proj_h: int
) -> np.ndarray:
    """Return the 4 projector-frame corners mapped into camera pixel coords."""
    H_inv = np.linalg.inv(H)
    corners = np.array(
        [[[0, 0]], [[proj_w - 1, 0]], [[proj_w - 1, proj_h - 1]], [[0, proj_h - 1]]],
        dtype=np.float32,
    )
    return cv2.perspectiveTransform(corners, H_inv).reshape(-1, 2)

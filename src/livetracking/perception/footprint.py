"""Projector footprint utilities.

The projector footprint is the set of camera pixels where the projector
actually lands light. We MEASURE it (full-white vs full-black diff during
calibration) rather than computing it via `H⁻¹·corners`, because the
homography is single-plane and the projector typically lights multiple
planes (wall + sofa fronts). Extrapolating H to scene corners produces a
badly skewed quadrilateral; the measured mask is the truth.

If no measured mask exists (older calibration), we fall back to the H-based
quad as a best-effort.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import cv2
import numpy as np

from livetracking.paths import (
    CALIB_DIR,
    CALIB_META_FILE,
    HOMOGRAPHY_FILE,
    MEASURED_FOOTPRINT_FILE,
)


def _dot_based_footprint_quad(
    cam_w: int, cam_h: int, calib_dir: str = CALIB_DIR
) -> Optional[np.ndarray]:
    """Return a 4-corner quad (in camera pixels) bounding the projector
    footprint, derived from the actually-observed dot positions.

    The 9-dot calibration grid is sampled at projector fractions
    (0.2, 0.5, 0.8). So the dot hull covers ~60% of the projector frame.
    We linearly extrapolate to (0.0, 1.0) along the projector grid axes to
    recover the FULL footprint corners. This is much more accurate than
    H⁻¹·corners because (a) it uses only the dots we actually saw, and
    (b) the extrapolation is tiny (40% one-shot), well within the linear
    regime even when the scene is multi-plane.
    """
    cam_pts_path = os.path.join(calib_dir, "dot_cam_pts.npy")
    proj_pts_path = os.path.join(calib_dir, "dot_proj_pts.npy")
    if not (os.path.exists(cam_pts_path) and os.path.exists(proj_pts_path)):
        return None
    cam_pts = np.load(cam_pts_path)
    proj_pts = np.load(proj_pts_path)
    if len(cam_pts) < 4:
        return None
    # Solve H_dot: projector px -> camera px, using ONLY the measured dots.
    # This is well-conditioned over the dot region. Then map the 4 projector
    # corners back through this H_dot — which is exactly the bounding quad
    # of where the dot rig predicts the projector frame edges land.
    try:
        H_dot, _ = cv2.findHomography(proj_pts, cam_pts, cv2.RANSAC, 5.0)
        if H_dot is None:
            return None
        PW = float(proj_pts[:, 0].max() / 0.8)   # recover PW from grid 0.8 frac
        PH = float(proj_pts[:, 1].max() / 0.8)
        corners_proj = np.array(
            [[[0, 0]], [[PW - 1, 0]], [[PW - 1, PH - 1]], [[0, PH - 1]]],
            dtype=np.float32,
        )
        corners_cam = cv2.perspectiveTransform(corners_proj, H_dot).reshape(-1, 2)
        # Clip to camera frame
        corners_cam[:, 0] = np.clip(corners_cam[:, 0], 0, cam_w - 1)
        corners_cam[:, 1] = np.clip(corners_cam[:, 1], 0, cam_h - 1)
        return corners_cam.astype(np.float32)
    except Exception:
        return None


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
    H: np.ndarray, proj_w: int, proj_h: int, cam_w: int, cam_h: int,
    measured_path: str = MEASURED_FOOTPRINT_FILE,
) -> np.ndarray:
    """Return a uint8 {0,1} mask of the projector footprint in camera space.

    Preference order:
      1. Quadrilateral derived from the actually-measured dot positions.
         This is the most reliable estimate when the scene is multi-plane
         (e.g. wall + sofa cushion fronts).
      2. H-extrapolated quadrilateral (kept as a last-resort fallback).
    """
    quad = _dot_based_footprint_quad(cam_w, cam_h)
    if quad is None:
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError as e:
            raise ValueError("Singular homography; recalibrate.") from e
        corners = np.array(
            [[[0, 0]], [[proj_w - 1, 0]], [[proj_w - 1, proj_h - 1]], [[0, proj_h - 1]]],
            dtype=np.float32,
        )
        quad = cv2.perspectiveTransform(corners, H_inv).reshape(-1, 2)
    mask = np.zeros((cam_h, cam_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 1)
    return mask


def footprint_outline_in_camera(
    H: np.ndarray, proj_w: int, proj_h: int, cam_w: int, cam_h: int,
    measured_path: str = MEASURED_FOOTPRINT_FILE,
) -> np.ndarray:
    """Return an (N, 2) int32 polyline tracing the projector footprint."""
    quad = _dot_based_footprint_quad(cam_w, cam_h)
    if quad is not None:
        return quad.astype(np.int32)
    return footprint_corners_in_camera(H, proj_w, proj_h).astype(np.int32)


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

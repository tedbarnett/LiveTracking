"""Stage 1: depth-plane foreground detection inside the projector footprint.

Given:
  - color  : (H, W, 3) uint8 BGR  (unused here; passed through for Stage 2)
  - depth_m: (H, W) float32 meters  (0 = invalid)
  - footprint_mask: (H, W) uint8 {0, 1}  from `footprint.footprint_mask_in_camera`

We:
  1. Fit a dominant plane to the depth data inside the *upper band* of the
     footprint (the wall — avoids fitting the sofa as "the plane").
  2. Mark pixels foreground when they are at least ``foreground_offset_m``
     CLOSER to the camera than the fitted plane.
  3. Restrict the foreground mask to the footprint.
  4. Morphologically clean it and run connected-components.
  5. Return one ``Blob`` per surviving component, with centroid + bbox +
     median depth.

This is the load-bearing "where might an object be" pass. Deterministic, no
ML, ~1 ms on CPU. See `computer-vision/projection-mapping` skill, rule #1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .types import Blob


@dataclass
class GeometryParams:
    wall_band_top: float = 0.0           # fraction of footprint height — top of wall band
    wall_band_bottom: float = 0.45       # fraction of footprint height — bottom of wall band
    ransac_iters: int = 200
    ransac_inlier_thresh_m: float = 0.025  # 2.5 cm
    foreground_offset_m: float = 0.05    # pixel must be 5 cm closer than wall
    min_blob_area_px: int = 1500
    morph_open: int = 5
    # Morph close kernel. KEEP SMALL (≤ ~9 px). The previous 25 px kernel
    # welded the bodhran + guitar + pillow + couch back into a single mega-
    # blob that DINO could only label once ("Pillow"). With 9 px we still
    # smooth over per-pixel noise + thin depth gaps but distinct objects on
    # the same plane stay separable.
    morph_close: int = 9
    footprint_inside_frac: float = 0.7   # blob must be ≥ this fraction inside footprint


def fit_plane_ransac(
    points_xyz: np.ndarray,
    iters: int,
    thresh: float,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """RANSAC plane fit. Returns (plane [a,b,c,d] with a*x+b*y+c*z+d=0,
    inlier_fraction, inlier_mask).

    points_xyz: (N, 3) float32. Must have N >= 3.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = points_xyz.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points to fit a plane.")

    best_inliers = np.zeros(n, dtype=bool)
    best_plane = np.array([0, 0, 1, 0], dtype=np.float64)
    best_count = 0

    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points_xyz[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        nrm = np.linalg.norm(normal)
        if nrm < 1e-8:
            continue
        normal = normal / nrm
        d = -float(np.dot(normal, p0))
        # signed distance
        dist = np.abs(points_xyz @ normal + d)
        inliers = dist < thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)

    # Refit with least squares on inliers for accuracy
    if best_count >= 3:
        pts = points_xyz[best_inliers]
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        # smallest-singular-value vector = plane normal
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[-1]
        normal = normal / np.linalg.norm(normal)
        d = -float(np.dot(normal, centroid))
        best_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)
        # recompute inliers
        dist = np.abs(points_xyz @ normal + d)
        best_inliers = dist < thresh
        best_count = int(best_inliers.sum())

    return best_plane, best_count / max(1, n), best_inliers


def _depth_to_xyz(
    depth_m: np.ndarray, mask: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project masked depth pixels to camera-frame 3D points.

    Returns (xyz (N,3) float32, pixel_indices (N,2) int — column, row).
    """
    ys, xs = np.where(mask & (depth_m > 0))
    if ys.size == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.int32)
    z = depth_m[ys, xs].astype(np.float32)
    X = (xs.astype(np.float32) - cx) * z / fx
    Y = (ys.astype(np.float32) - cy) * z / fy
    xyz = np.stack([X, Y, z], axis=1)
    px = np.stack([xs, ys], axis=1).astype(np.int32)
    return xyz, px


def detect_blobs(
    color: np.ndarray,
    depth_m: np.ndarray,
    footprint_mask: np.ndarray,
    intrinsics: Tuple[float, float, float, float],
    params: GeometryParams | None = None,
) -> Tuple[List[Blob], dict]:
    """Stage 1 detection.

    intrinsics: (fx, fy, cx, cy) of the color stream.
    Returns (blobs, debug) where debug carries arrays for diagnostic dumps:
        - 'plane': [a,b,c,d]
        - 'inlier_frac': float
        - 'wall_band_mask', 'foreground_mask_raw', 'foreground_mask_clean',
          'labels' (connected-components output), 'wall_depth_at_center'
    """
    p = params or GeometryParams()
    H, W = depth_m.shape
    fx, fy, cx, cy = intrinsics

    # Define wall band: top fraction of the footprint bbox
    ys_fp, xs_fp = np.where(footprint_mask > 0)
    if ys_fp.size == 0:
        raise ValueError("Empty footprint mask — bad homography or calibration.")
    y_top, y_bot = int(ys_fp.min()), int(ys_fp.max())
    band_top = int(y_top + (y_bot - y_top) * p.wall_band_top)
    band_bot = int(y_top + (y_bot - y_top) * p.wall_band_bottom)
    wall_band_mask = np.zeros_like(footprint_mask)
    wall_band_mask[band_top:band_bot + 1, :] = footprint_mask[band_top:band_bot + 1, :]

    # Sample plane points from the wall band (subsample for speed)
    xyz_wall, _ = _depth_to_xyz(
        depth_m,
        wall_band_mask.astype(bool),
        fx, fy, cx, cy,
    )
    if xyz_wall.shape[0] < 500:
        raise ValueError(
            f"Too few wall-band depth points ({xyz_wall.shape[0]}); "
            "increase wall_band_bottom or check exposure/depth quality."
        )
    # subsample to 5000 points for RANSAC speed
    if xyz_wall.shape[0] > 5000:
        idx = np.random.default_rng(0).choice(xyz_wall.shape[0], 5000, replace=False)
        xyz_wall_s = xyz_wall[idx]
    else:
        xyz_wall_s = xyz_wall

    plane, inlier_frac, _ = fit_plane_ransac(
        xyz_wall_s, iters=p.ransac_iters, thresh=p.ransac_inlier_thresh_m
    )
    a, b, c, d = plane

    # For every pixel inside the footprint with valid depth, compute signed
    # distance to the plane along the camera's view ray. Foreground = the
    # measured point is closer to the camera than the plane along that ray.
    fp_bool = footprint_mask.astype(bool) & (depth_m > 0)
    ys, xs = np.where(fp_bool)
    z = depth_m[ys, xs].astype(np.float32)
    X = (xs.astype(np.float32) - cx) * z / fx
    Y = (ys.astype(np.float32) - cy) * z / fy
    # Signed plane value: > 0 on the camera side of the plane if normal faces camera,
    # but we don't know normal sign. So: compute the depth at which the camera ray
    # through this pixel would hit the plane, and compare to measured z.
    # Ray: (X, Y, Z) = (u*Z, v*Z, Z) where u = (xs-cx)/fx, v = (ys-cy)/fy.
    u = (xs.astype(np.float32) - cx) / fx
    v = (ys.astype(np.float32) - cy) / fy
    denom = a * u + b * v + c
    # Avoid division by ~0
    safe = np.abs(denom) > 1e-6
    z_plane = np.full_like(z, np.nan)
    z_plane[safe] = -d / denom[safe]
    fg_pix = safe & (z_plane > 0.1) & (z < z_plane - p.foreground_offset_m)

    foreground_raw = np.zeros((H, W), dtype=np.uint8)
    foreground_raw[ys[fg_pix], xs[fg_pix]] = 255

    # Morphological clean
    k_open = np.ones((p.morph_open, p.morph_open), np.uint8)
    k_close = np.ones((p.morph_close, p.morph_close), np.uint8)
    fg_clean = cv2.morphologyEx(foreground_raw, cv2.MORPH_OPEN, k_open)
    fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_CLOSE, k_close)
    fg_clean = cv2.bitwise_and(fg_clean, footprint_mask * 255)

    # Connected components
    n_lab, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_clean, 8)
    blobs: List[Blob] = []
    next_id = 1
    for lab in range(1, n_lab):
        x, y, w, h, area = stats[lab]
        if area < p.min_blob_area_px:
            continue
        blob_mask = (labels == lab).astype(np.uint8) * 255
        # require ≥ footprint_inside_frac of blob to be inside footprint
        inside = int(np.logical_and(blob_mask > 0, footprint_mask > 0).sum())
        if inside / float(area) < p.footprint_inside_frac:
            continue
        # Median depth on the blob (ignoring zeros)
        z_blob = depth_m[(blob_mask > 0) & (depth_m > 0)]
        med_z = float(np.median(z_blob)) if z_blob.size else 0.0
        cx_blob, cy_blob = float(centroids[lab][0]), float(centroids[lab][1])
        blobs.append(Blob(
            blob_id=next_id,
            cam_mask=blob_mask,
            centroid_cam=(cx_blob, cy_blob),
            bbox_cam=(int(x), int(y), int(w), int(h)),
            area_px=int(area),
            median_depth_m=med_z,
        ))
        next_id += 1

    # Sort blobs left-to-right, top-to-bottom so numbering is stable on a static scene
    blobs.sort(key=lambda b: (round(b.centroid_cam[1] / 100), b.centroid_cam[0]))
    for i, b in enumerate(blobs, start=1):
        b.blob_id = i

    debug = {
        "plane": plane.tolist(),
        "inlier_frac": float(inlier_frac),
        "wall_band_mask": wall_band_mask,
        "foreground_mask_raw": foreground_raw,
        "foreground_mask_clean": fg_clean,
        "labels": labels,
        "n_components_raw": int(n_lab - 1),
    }
    return blobs, debug

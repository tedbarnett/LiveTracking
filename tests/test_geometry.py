"""Unit tests for the depth-plane Stage 1 detector.

These run without a camera by synthesizing a (depth, color, footprint) input:
  - A flat "wall" plane at z=3.0 m.
  - Two "objects" closer (a box at z=1.5 m and a cylinder at z=2.0 m) inside
    the footprint.
  - One off-axis "distractor" object OUTSIDE the footprint that the detector
    must reject (this is the LiveTracking standing rule).
"""
from __future__ import annotations

import numpy as np
import pytest

from livetracking.perception.geometry import (
    GeometryParams,
    detect_blobs,
    fit_plane_ransac,
)


W, H = 848, 480
FX, FY = 615.0, 615.0
CX, CY = 424.0, 240.0
INTR = (FX, FY, CX, CY)


def _synth_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (color, depth_m, footprint_mask)."""
    depth = np.full((H, W), 3.0, dtype=np.float32)  # flat wall at 3 m
    color = np.full((H, W, 3), 80, dtype=np.uint8)

    # Footprint: centered rectangle 400 wide x 240 tall
    fp = np.zeros((H, W), dtype=np.uint8)
    fp[120:360, 224:624] = 1

    # Object A: 80x80 box at depth 1.5 m, fully inside footprint
    depth[180:260, 280:360] = 1.5
    color[180:260, 280:360] = (240, 220, 200)

    # Object B: 60x60 box at depth 2.0 m, fully inside footprint
    depth[200:260, 460:520] = 2.0
    color[200:260, 460:520] = (180, 230, 240)

    # Distractor: 100x100 box at depth 1.2 m, OUTSIDE footprint (top-left)
    depth[20:120, 30:130] = 1.2
    color[20:120, 30:130] = (255, 100, 100)

    # Add a tiny bit of depth noise so RANSAC has something to chew on
    rng = np.random.default_rng(42)
    depth += rng.normal(0.0, 0.003, size=depth.shape).astype(np.float32)
    return color, depth, fp


def test_fit_plane_ransac_recovers_z_const_plane():
    rng = np.random.default_rng(0)
    # Plane z = 3.0 — points sampled on it with small noise
    xs = rng.uniform(-1, 1, size=2000)
    ys = rng.uniform(-1, 1, size=2000)
    zs = 3.0 + rng.normal(0.0, 0.002, size=2000)
    pts = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    plane, inlier_frac, _ = fit_plane_ransac(pts, iters=100, thresh=0.01)
    a, b, c, d = plane
    # Normal should be ~ (0, 0, ±1)
    assert abs(abs(c) - 1.0) < 1e-2, f"plane normal not z-aligned: {plane}"
    # d should be ~ -3.0 * c sign
    assert abs(d + 3.0 * c) < 1e-2, f"plane d off: {plane}"
    assert inlier_frac > 0.95


def test_detect_blobs_finds_two_objects_inside_footprint_and_rejects_distractor():
    color, depth, fp = _synth_scene()
    params = GeometryParams(
        wall_band_top=0.0,
        wall_band_bottom=0.35,    # top 35% of footprint is wall-only
        ransac_iters=80,
        ransac_inlier_thresh_m=0.015,
        foreground_offset_m=0.10,
        min_blob_area_px=500,
        morph_open=3,
        morph_close=9,
    )
    blobs, dbg = detect_blobs(color, depth, fp, INTR, params)
    assert len(blobs) == 2, (
        f"expected 2 blobs (A + B), got {len(blobs)}: "
        f"{[(b.blob_id, b.area_px, b.centroid_cam) for b in blobs]}"
    )
    # No blob centroid should land outside the footprint
    for b in blobs:
        cx, cy = b.centroid_cam
        assert fp[int(cy), int(cx)] == 1, (
            f"blob #{b.blob_id} centroid ({cx:.0f},{cy:.0f}) is OUTSIDE the footprint"
        )
    # Depths should be roughly correct
    depths = sorted(b.median_depth_m for b in blobs)
    assert depths[0] == pytest.approx(1.5, abs=0.05)
    assert depths[1] == pytest.approx(2.0, abs=0.05)
    # And RANSAC plane should be near z = 3
    a, b, c, d = dbg["plane"]
    z0_at_center = -d / c if abs(c) > 1e-6 else 0.0
    assert z0_at_center == pytest.approx(3.0, abs=0.05)

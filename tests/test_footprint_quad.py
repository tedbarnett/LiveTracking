"""Tests for projector_quad_footprint — the wall-plane projector rectangle
mapped into camera space. This footprint must be STABLE (same shape every
calibration) and must match the projector's actual throw, not shrink to the
contrast-gate's bright center.
"""
import cv2
import numpy as np

from livetracking.perception.footprint import projector_quad_footprint


PW, PH = 3840, 2160          # projector native
CW, CH = 848, 480            # D455 capture


def _h_cam_to_proj(quad_cam):
    """Build H (cam->proj) such that the given 4 camera-pixel corners map to
    the projector frame corners — i.e. H^-1 maps the projector rectangle onto
    quad_cam. Order: TL, TR, BR, BL."""
    proj_corners = np.array(
        [[0, 0], [PW, 0], [PW, PH], [0, PH]], dtype=np.float32
    )
    return cv2.getPerspectiveTransform(
        np.asarray(quad_cam, dtype=np.float32), proj_corners
    )


def test_quad_recovers_known_rectangle():
    # Projector lands as an axis-aligned rect occupying most of the frame.
    quad = [[100, 60], [740, 60], [740, 420], [100, 420]]
    H = _h_cam_to_proj(quad)
    mask = projector_quad_footprint(H, PW, PH, CW, CH)
    assert mask is not None
    ys, xs = np.where(mask > 0)
    # Filled region matches the rectangle within a couple px of rounding.
    assert abs(xs.min() - 100) <= 2
    assert abs(xs.max() - 740) <= 2
    assert abs(ys.min() - 60) <= 2
    assert abs(ys.max() - 420) <= 2


def test_quad_fills_keystone_trapezoid():
    # Keystoned throw (top edge narrower than bottom) — area should be large,
    # not collapsed to a tiny center blob.
    quad = [[250, 80], [600, 80], [760, 400], [90, 400]]
    H = _h_cam_to_proj(quad)
    mask = projector_quad_footprint(H, PW, PH, CW, CH)
    assert mask is not None
    frac = (mask > 0).sum() / mask.size
    assert frac > 0.25  # covers a real chunk of frame, not a sliver


def test_quad_is_stable_across_noise():
    # The whole point: footprint shape must not jitter. Two H's that differ
    # only by sub-pixel correspondence noise must yield near-identical masks
    # (unlike the contrast-gate hull, which reshuffles dim edges each run).
    quad = [[120, 70], [720, 70], [720, 410], [120, 410]]
    H1 = _h_cam_to_proj(quad)
    rng = np.random.default_rng(0)
    quad2 = (np.array(quad, dtype=np.float64)
             + rng.normal(0, 0.4, size=(4, 2))).tolist()
    H2 = _h_cam_to_proj(quad2)
    m1 = projector_quad_footprint(H1, PW, PH, CW, CH)
    m2 = projector_quad_footprint(H2, PW, PH, CW, CH)
    assert m1 is not None and m2 is not None
    inter = np.logical_and(m1 > 0, m2 > 0).sum()
    union = np.logical_or(m1 > 0, m2 > 0).sum()
    assert inter / union > 0.99  # essentially identical


def test_singular_h_returns_none():
    H = np.zeros((3, 3), dtype=np.float64)  # singular
    assert projector_quad_footprint(H, PW, PH, CW, CH) is None


def test_exploded_quad_rejected():
    # H that maps the projector frame far outside the camera image => bad fit.
    # Should be rejected (None) so the caller falls back to the cone hull.
    quad = [[1e5, 1e5], [2e5, 1e5], [2e5, 2e5], [1e5, 2e5]]
    H = _h_cam_to_proj(quad)
    assert projector_quad_footprint(H, PW, PH, CW, CH) is None


def test_clips_to_frame_bounds():
    # Projector overshoots frame on all sides; mask must stay within bounds
    # and still be large (clipped rectangle, not rejected).
    quad = [[-50, -40], [900, -40], [900, 520], [-50, 520]]
    H = _h_cam_to_proj(quad)
    mask = projector_quad_footprint(H, PW, PH, CW, CH)
    assert mask is not None
    assert mask.shape == (CH, CW)
    # Edges are saturated (clipped to frame), so most of the frame is filled.
    assert (mask > 0).sum() / mask.size > 0.9

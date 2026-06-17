"""Tests for the select-all highlight hysteresis gate.

The select-all path re-warps every object from its fresh SAM mask on every
DINO+SAM pass. SAM boundaries wobble a few px run-to-run, so a STATIC object
visibly jerks every ~2 s. _highlight_mask_stable() decides when to HOLD the
on-screen mask (stable) vs re-push (real motion/reshape).
"""
import numpy as np

from livetracking.daemon.perception import (
    _cam_mask_iou,
    _highlight_mask_stable,
)


def _disk(cx, cy, r, w=848, h=480):
    """Filled disk mask — stand-in for a SAM blob."""
    yy, xx = np.ogrid[:h, :w]
    m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    return (m.astype(np.uint8) * 255)


MOVE_PX = 6.0
IOU = 0.90


def test_iou_identical_is_one():
    m = _disk(400, 240, 80)
    assert _cam_mask_iou(m, m) == 1.0


def test_iou_empty_vs_empty_is_one():
    z = np.zeros((480, 848), np.uint8)
    assert _cam_mask_iou(z, z) == 1.0


def test_iou_disjoint_is_zero():
    a = _disk(100, 100, 40)
    b = _disk(700, 400, 40)
    assert _cam_mask_iou(a, b) == 0.0


def test_static_object_with_sam_wobble_is_stable():
    # Same object, radius wobbles by 1 px (SAM boundary noise), centroid
    # unmoved -> must be judged STABLE so the wash holds (no jerk).
    prev = _disk(400, 240, 80)
    new = _disk(400, 240, 81)
    assert _highlight_mask_stable(
        (400, 240), (400, 240), prev, new, MOVE_PX, IOU) is True


def test_translation_beyond_threshold_updates():
    # Object slid 20 px -> NOT stable (must re-push so the wash follows).
    prev = _disk(400, 240, 80)
    new = _disk(420, 240, 80)
    assert _highlight_mask_stable(
        (400, 240), (420, 240), prev, new, MOVE_PX, IOU) is False


def test_small_subpixel_drift_is_stable():
    # 3 px centroid drift, near-identical shape -> hold.
    prev = _disk(400, 240, 80)
    new = _disk(402, 242, 80)
    assert _highlight_mask_stable(
        (400, 240), (402, 242), prev, new, MOVE_PX, IOU) is True


def test_reshape_same_centroid_updates():
    # Centroid barely moves but the shape changes a lot (object rotated /
    # occluded / merged) -> IoU drops below threshold -> update.
    prev = _disk(400, 240, 80)
    new = _disk(400, 240, 130)  # much bigger blob, same center
    assert _highlight_mask_stable(
        (400, 240), (400, 240), prev, new, MOVE_PX, IOU) is False


def test_missing_prior_state_updates():
    # First time an object is shown (no prior) -> not stable -> paint now.
    new = _disk(400, 240, 80)
    assert _highlight_mask_stable(
        None, (400, 240), None, new, MOVE_PX, IOU) is False


def test_missing_centroid_updates():
    prev = _disk(400, 240, 80)
    new = _disk(400, 240, 80)
    assert _highlight_mask_stable(
        (400, 240), None, prev, new, MOVE_PX, IOU) is False

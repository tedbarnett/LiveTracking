"""Tests for the inter-pass FastTracker fusion (Step 2 of fast-following).

Synthetic frames only (no camera/torch/pygame): a disc at a known depth is
moved around a depth image and we assert the fused estimate follows it, the
fusion branches pick the right source, and the tracker freezes (rather than
chasing garbage) when both estimators lose the object.
"""
from __future__ import annotations

import numpy as np
import pytest

from livetracking.perception.fasttrack import (
    FastTracker, _fuse_estimates, _make_csrt,
)


H, W = 480, 848


def _disc_mask(cx, cy, r=40):
    y, x = np.ogrid[:H, :W]
    return (((y - cy) ** 2 + (x - cx) ** 2 <= r * r).astype(np.uint8)) * 255


def _depth_frame(cx, cy, obj_depth=1.5, wall_depth=2.8, r=40):
    """A wall at wall_depth with a disc at obj_depth — mimics an object
    sitting in front of the wall, which is what depth-blob keys on."""
    depth = np.full((H, W), wall_depth, dtype=np.float32)
    m = _disc_mask(cx, cy, r) > 0
    depth[m] = obj_depth
    return depth


def _bbox_of(cx, cy, r=40):
    return (cx - r, cy - r, 2 * r, 2 * r)


# ---- pure fusion logic --------------------------------------------------

class TestFuseEstimates:
    def test_both_unavailable_freezes(self):
        cx, cy, conf, src = _fuse_estimates(
            None, 0.0, None, 0.0, last_xy=(100.0, 200.0), agree_px=40)
        assert (cx, cy) == (100.0, 200.0)
        assert conf == 0.0
        assert src == "frozen"

    def test_depth_only(self):
        cx, cy, conf, src = _fuse_estimates(
            (300.0, 250.0), 0.9, None, 0.0, last_xy=(0, 0), agree_px=40)
        assert (cx, cy) == (300.0, 250.0)
        assert src == "depth"
        assert conf == 0.9

    def test_csrt_only_is_bridge(self):
        cx, cy, conf, src = _fuse_estimates(
            None, 0.0, (310.0, 240.0), 0.7, last_xy=(0, 0), agree_px=40)
        assert (cx, cy) == (310.0, 240.0)
        assert src == "csrt"
        assert conf == pytest.approx(0.56)  # 0.7 * 0.8 bridge discount

    def test_agree_uses_depth_and_boosts_confidence(self):
        # Depth and CSRT within agree_px => trust depth, conf boosted.
        cx, cy, conf, src = _fuse_estimates(
            (300.0, 250.0), 0.6, (320.0, 250.0), 0.6, last_xy=(0, 0),
            agree_px=40)
        assert (cx, cy) == (300.0, 250.0)  # depth wins (drift-free)
        assert src == "fused"
        assert conf > 0.6  # boosted because they agree

    def test_disagree_prefers_confident_depth(self):
        cx, cy, conf, src = _fuse_estimates(
            (300.0, 250.0), 0.9, (500.0, 250.0), 0.5, last_xy=(0, 0),
            agree_px=40)
        assert (cx, cy) == (300.0, 250.0)
        assert src == "depth"
        assert conf < 0.9  # penalized for disagreement

    def test_disagree_prefers_csrt_when_depth_weak(self):
        cx, cy, conf, src = _fuse_estimates(
            (300.0, 250.0), 0.2, (500.0, 250.0), 0.7, last_xy=(0, 0),
            agree_px=40)
        assert (cx, cy) == (500.0, 250.0)
        assert src == "csrt"


# ---- depth-blob tracking through the public API -------------------------

class TestFastTrackerDepth:
    def _seed(self, ft, cx, cy, obj_depth=1.5):
        mask = _disc_mask(cx, cy)
        # color=None => depth-blob-only mode (isolates the depth estimator).
        ft.reseed(1, mask, _bbox_of(cx, cy), obj_depth, color=None)

    def test_follows_moving_object_by_depth(self):
        ft = FastTracker()
        self._seed(ft, 200, 240)
        # Move the disc right in small steps; the depth-blob estimate should
        # track it without any appearance tracker.
        last_cx = 200.0
        for cx in range(220, 421, 20):
            depth = _depth_frame(cx, 240)
            est = ft.update(1, None, depth)
            assert est is not None
            assert est.source in ("depth", "fused")
            assert est.cx > last_cx - 5  # monotonic-ish follow
            last_cx = est.cx
        # Ended near the final position (cx=420), not stuck at the seed.
        assert last_cx > 380

    def test_freezes_when_object_disappears(self):
        ft = FastTracker(freeze_after_misses=3)
        self._seed(ft, 300, 240)
        # Object vanishes: a flat wall with no near-depth blob anywhere.
        flat = np.full((H, W), 2.8, dtype=np.float32)
        last = None
        for _ in range(5):
            last = ft.update(1, None, flat)
        assert last is not None
        assert last.source == "frozen"
        assert last.confidence == 0.0
        # Frozen at the last good position (the seed centroid).
        assert last.cx == pytest.approx(300.0, abs=2.0)
        assert last.cy == pytest.approx(240.0, abs=2.0)

    def test_reseed_reanchors_after_jump(self):
        ft = FastTracker()
        self._seed(ft, 200, 240)
        ft.update(1, None, _depth_frame(220, 240))
        # A big SAM-pass jump (object teleported in the scene). Depth-blob in
        # the OLD window would lose it; reseed re-anchors to the new spot.
        self._seed(ft, 600, 300)
        est = ft.update(1, None, _depth_frame(610, 300))
        assert est is not None
        assert est.cx > 560  # tracking near the new anchor, not the old one

    def test_moved_px_reports_displacement_from_anchor(self):
        ft = FastTracker()
        self._seed(ft, 200, 240)
        # Continuous motion (small per-frame steps, as at 30-60 fps) so the
        # object stays inside the search window each frame.
        est = None
        for cx in range(220, 301, 20):
            est = ft.update(1, None, _depth_frame(cx, 240))
        assert est is not None
        # Ended ~100 px from the anchor at x=200.
        assert est.moved_px > 80


# ---- lifecycle ----------------------------------------------------------

class TestLifecycle:
    def test_retain_only_drops_unselected(self):
        ft = FastTracker()
        ft.reseed(1, _disc_mask(200, 240), _bbox_of(200, 240), 1.5)
        ft.reseed(2, _disc_mask(400, 240), _bbox_of(400, 240), 1.6)
        ft.reseed(3, _disc_mask(600, 240), _bbox_of(600, 240), 1.7)
        ft.retain_only([2])
        assert set(ft.active_ids()) == {2}

    def test_update_unknown_id_returns_none(self):
        ft = FastTracker()
        assert ft.update(99, None, _depth_frame(200, 240)) is None


# ---- CSRT availability (informational) ----------------------------------
def test_csrt_available_in_this_build():
    # Not a hard requirement (depth-blob works alone), but on the rig's cv2
    # build CSRT should be present — assert so we notice if a cv2 upgrade
    # silently drops it.
    assert _make_csrt() is not None


class TestFastTrackerFusionLive:
    def test_csrt_plus_depth_follow_textured_object(self):
        """End-to-end with a real CSRT tracker on a textured color frame +
        matching depth. Exercises the fused path, not just depth-only."""
        if _make_csrt() is None:
            pytest.skip("no CSRT in this cv2 build")
        ft = FastTracker()

        def color_frame(cx, cy, r=40):
            img = np.zeros((H, W, 3), dtype=np.uint8)
            # textured square so CSRT has features to lock onto
            rng = np.random.default_rng(0)
            patch = rng.integers(0, 255, size=(2 * r, 2 * r, 3),
                                  dtype=np.uint8)
            y0, x0 = cy - r, cx - r
            img[y0:y0 + 2 * r, x0:x0 + 2 * r] = patch
            return img

        cx0, cy0 = 200, 240
        ft.reseed(1, _disc_mask(cx0, cy0), _bbox_of(cx0, cy0), 1.5,
                  color=color_frame(cx0, cy0))
        last_cx = float(cx0)
        for cx in range(220, 361, 20):
            est = ft.update(1, color_frame(cx, cy0), _depth_frame(cx, cy0))
            assert est is not None
            assert est.source in ("depth", "csrt", "fused")
            last_cx = est.cx
        assert last_cx > 320  # followed the object across the frame

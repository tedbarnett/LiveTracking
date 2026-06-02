"""Unit tests for livetracking.perception.depth_probe.

These cover the failure mode we hit in the field on 2026-06-02: a NEAR
target (bodhran) physically present but smaller than the centerbox in
front of a back wall, where median_depth_in_centerbox returned the wall
depth instead of the foreground. nearest_depth_in_centerbox(pct=10) is
the fix; these tests pin both behaviors so we don't regress either.

Pure-numpy module — no pyrealsense2, no pygame, no torch needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from livetracking.perception.depth_probe import (
    median_depth_in_centerbox,
    nearest_depth_in_centerbox,
    _MIN_VALID_PIXELS,
)


# ---- shared fixtures -----------------------------------------------------

def _uniform_depth(h: int = 480, w: int = 848, value: float = 3.7) -> np.ndarray:
    """HxW float32 depth map, all pixels at `value` meters."""
    return np.full((h, w), value, dtype=np.float32)


def _small_target_on_wall(target_depth: float, wall_depth: float,
                          target_radius: int) -> np.ndarray:
    """Wall plane with a small circular target at frame center.

    target_radius is in pixels. Used to reproduce the bodhran-on-couch
    failure mode.
    """
    depth = _uniform_depth(value=wall_depth)
    h, w = depth.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    mask = (y - cy) ** 2 + (x - cx) ** 2 <= target_radius ** 2
    depth[mask] = target_depth
    return depth


# ---- median probe --------------------------------------------------------

class TestMedianDepth:
    def test_uniform_field_returns_value(self):
        depth = _uniform_depth(value=2.5)
        assert median_depth_in_centerbox(depth) == pytest.approx(2.5)

    def test_invalid_pixels_ignored(self):
        depth = _uniform_depth(value=2.5)
        # Punch a hole through half the centerbox.
        depth[200:280, 400:500] = 0.0
        assert median_depth_in_centerbox(depth) == pytest.approx(2.5)

    def test_all_invalid_returns_zero(self):
        depth = np.zeros((480, 848), dtype=np.float32)
        assert median_depth_in_centerbox(depth) == 0.0

    def test_too_few_valid_pixels_returns_zero(self):
        depth = np.zeros((480, 848), dtype=np.float32)
        # Put fewer than _MIN_VALID_PIXELS valid pixels in the centerbox.
        depth[240, 424:424 + (_MIN_VALID_PIXELS - 1)] = 1.5
        assert median_depth_in_centerbox(depth) == 0.0

    def test_small_target_gets_swamped_by_wall(self):
        """REGRESSION GUARD: this is the 2026-06-02 failure. With a small
        foreground target (e.g. bodhran @ 1.2 m, radius 30 px) in front of
        a 3.7 m wall, the median MUST return wall depth. If a future change
        makes the median magically find the foreground, the comment above
        nearest_depth_in_centerbox becomes a lie and the calibration UI
        will lose its purpose-built nearest-decile probe.
        """
        depth = _small_target_on_wall(target_depth=1.2, wall_depth=3.7,
                                       target_radius=30)
        # Default half=60 -> 120x120 box -> 14400 px. Target covers
        # pi*30^2 ~= 2827 px = 20% of the box. Median = wall.
        assert median_depth_in_centerbox(depth) == pytest.approx(3.7)

    def test_half_argument_respected(self):
        depth = _uniform_depth(value=2.0)
        # Outside the (half=10)^2 box, put a different depth. Probe should
        # still see 2.0 because we restrict the window.
        depth[:200, :] = 99.0
        depth[280:, :] = 99.0
        assert median_depth_in_centerbox(depth, half=10) == pytest.approx(2.0)


# ---- nearest-decile probe ------------------------------------------------

class TestNearestDepth:
    def test_uniform_field_returns_value(self):
        depth = _uniform_depth(value=2.5)
        assert nearest_depth_in_centerbox(depth) == pytest.approx(2.5)

    def test_small_target_recovered(self):
        """THE FIX. Same scene as test_small_target_gets_swamped_by_wall,
        but nearest_depth_in_centerbox(pct=10) MUST recover the foreground
        target (within 0.1 m) — this is the behavior that makes the bodhran
        calibration actually work next time."""
        depth = _small_target_on_wall(target_depth=1.2, wall_depth=3.7,
                                       target_radius=30)
        # Target is 20% of box -> the 10th percentile lands inside it.
        z = nearest_depth_in_centerbox(depth, pct=10.0)
        assert z == pytest.approx(1.2, abs=0.01)

    def test_too_small_target_falls_back_to_wall(self):
        """Sanity: if the target is < 10% of the box, the 10th-percentile
        probe will land in the wall pixels. Surface this so the operator
        sees red on the traffic light and aims the camera better."""
        depth = _small_target_on_wall(target_depth=1.2, wall_depth=3.7,
                                       target_radius=10)
        # pi*10^2 = 314 px out of 14400 = 2.2% -> 10th percentile = wall.
        z = nearest_depth_in_centerbox(depth, pct=10.0)
        assert z == pytest.approx(3.7, abs=0.01)

    def test_pct_zero_returns_minimum(self):
        depth = _uniform_depth(value=3.0)
        depth[240, 424] = 0.5
        # pct=0 -> true min. With 14399 wall pixels + 1 spike pixel,
        # percentile(0) returns the spike.
        z = nearest_depth_in_centerbox(depth, pct=0.0)
        assert z == pytest.approx(0.5)

    def test_pct_fifty_equals_median(self):
        """pct=50 of the nearest probe must equal the bare median — both
        compute the same statistic on the same window."""
        depth = _small_target_on_wall(target_depth=1.0, wall_depth=4.0,
                                       target_radius=40)
        z_near = nearest_depth_in_centerbox(depth, pct=50.0)
        z_med = median_depth_in_centerbox(depth)
        assert z_near == pytest.approx(z_med)

    def test_invalid_pixels_ignored(self):
        depth = _small_target_on_wall(target_depth=1.2, wall_depth=3.7,
                                       target_radius=30)
        # Knock out the target — should now return wall.
        h, w = depth.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        mask = (y - cy) ** 2 + (x - cx) ** 2 <= 30 ** 2
        depth[mask] = 0.0
        z = nearest_depth_in_centerbox(depth, pct=10.0)
        assert z == pytest.approx(3.7)

    def test_too_few_valid_pixels_returns_zero(self):
        depth = np.zeros((480, 848), dtype=np.float32)
        depth[240, 424:424 + (_MIN_VALID_PIXELS - 1)] = 1.5
        assert nearest_depth_in_centerbox(depth) == 0.0

    def test_all_invalid_returns_zero(self):
        depth = np.zeros((480, 848), dtype=np.float32)
        assert nearest_depth_in_centerbox(depth) == 0.0

    def test_d455_noise_spike_not_amplified(self):
        """D455 occasionally returns 0.05 m noise spikes. pct=10 must NOT
        amplify a handful of these into the reported z_near; pct=0 (true
        min) would. This is why the default is 10.0 not 0.0."""
        depth = _uniform_depth(value=3.7)
        # 5 noise pixels at 0.05 m. _MIN_VALID_PIXELS is 50 so the centerbox
        # is mostly wall; nearest decile (>1400 pixels) easily skips past
        # the 5 spikes.
        depth[240, 420:425] = 0.05
        z = nearest_depth_in_centerbox(depth, pct=10.0)
        assert z == pytest.approx(3.7)
        # But pct=0 catches the spike (documenting the trap).
        z_min = nearest_depth_in_centerbox(depth, pct=0.0)
        assert z_min == pytest.approx(0.05)


# ---- integration: replay the 2026-06-02 bodhran scene -------------------

def test_bodhran_on_couch_scene():
    """End-to-end reproduction of the field failure: bodhran on couch in
    front of back wall. nearest probe recovers the foreground, median does
    not. This test exists so a future refactor of the centerbox probe
    breaks loudly when it regresses the live behavior."""
    # 18-inch bodhran ~30 px radius at 1.2 m on a D455 @ 848x480.
    depth = _small_target_on_wall(target_depth=1.2, wall_depth=3.7,
                                   target_radius=30)
    z_median = median_depth_in_centerbox(depth)
    z_near = nearest_depth_in_centerbox(depth, pct=10.0)
    # The whole point: the two probes MUST disagree on this scene.
    assert z_median > 3.0  # swamped by wall
    assert z_near < 2.0   # recovered foreground
    # And the alpha = (1/z_near - 1/z_wall) / ... denominator must be
    # large enough to actually produce parallax correction.
    inv_denom = abs(1.0 / z_near - 1.0 / 3.7)
    assert inv_denom > 0.3  # ~0.564 in this scene

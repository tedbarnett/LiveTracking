"""Unit tests for livetracking.perception.depth_probe.

These cover the failure mode we hit in the field on 2026-06-02: a NEAR
target (bodhran) physically present but smaller than the centerbox in
front of a back wall, where median_depth_in_centerbox returned the wall
depth instead of the foreground. nearest_depth_in_centerbox(pct=10) is
the fix; these tests pin both behaviors so we don't regress either.

The click-probe family (depth_at_point + projector->camera inverse
homography math) is also exercised here so the live calibration UI
isn't the first place we discover sign/transpose errors.

Pure-numpy module — no pyrealsense2, no pygame, no torch needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from livetracking.perception.depth_probe import (
    depth_at_point,
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
        depth[240, 424:424 + (_MIN_VALID_PIXELS - 1)] = 1.5
        assert median_depth_in_centerbox(depth) == 0.0

    def test_small_target_gets_swamped_by_wall(self):
        """REGRESSION GUARD: the 2026-06-02 failure. Small bodhran target
        @ 1.2 m in front of 3.7 m wall — median MUST return wall depth.
        If a future change makes the median magically find the foreground,
        the docstring on nearest_depth_in_centerbox becomes a lie."""
        depth = _small_target_on_wall(target_depth=1.2, wall_depth=3.7,
                                       target_radius=30)
        assert median_depth_in_centerbox(depth) == pytest.approx(3.7)

    def test_half_argument_respected(self):
        depth = _uniform_depth(value=2.0)
        depth[:200, :] = 99.0
        depth[280:, :] = 99.0
        assert median_depth_in_centerbox(depth, half=10) == pytest.approx(2.0)


# ---- nearest-decile probe ------------------------------------------------

class TestNearestDepth:
    def test_uniform_field_returns_value(self):
        depth = _uniform_depth(value=2.5)
        assert nearest_depth_in_centerbox(depth) == pytest.approx(2.5)

    def test_small_target_recovered(self):
        """THE FIX: nearest_depth_in_centerbox(pct=10) recovers the
        foreground target (within 0.01 m) — this is the behavior that
        makes the bodhran calibration actually work next time."""
        depth = _small_target_on_wall(target_depth=1.2, wall_depth=3.7,
                                       target_radius=30)
        z = nearest_depth_in_centerbox(depth, pct=10.0)
        assert z == pytest.approx(1.2, abs=0.01)

    def test_too_small_target_falls_back_to_wall(self):
        """If target < 10% of the box, the 10th-percentile lands in wall.
        Surfaces this so the operator sees red on the traffic light."""
        depth = _small_target_on_wall(target_depth=1.2, wall_depth=3.7,
                                       target_radius=10)
        z = nearest_depth_in_centerbox(depth, pct=10.0)
        assert z == pytest.approx(3.7, abs=0.01)

    def test_pct_zero_returns_minimum(self):
        depth = _uniform_depth(value=3.0)
        depth[240, 424] = 0.5
        z = nearest_depth_in_centerbox(depth, pct=0.0)
        assert z == pytest.approx(0.5)

    def test_pct_fifty_equals_median(self):
        depth = _small_target_on_wall(target_depth=1.0, wall_depth=4.0,
                                       target_radius=40)
        z_near = nearest_depth_in_centerbox(depth, pct=50.0)
        z_med = median_depth_in_centerbox(depth)
        assert z_near == pytest.approx(z_med)

    def test_invalid_pixels_ignored(self):
        depth = _small_target_on_wall(target_depth=1.2, wall_depth=3.7,
                                       target_radius=30)
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
        """pct=10 ignores a handful of D455 0.05 m noise spikes; pct=0 does
        not. This is why the default is 10.0, not 0.0."""
        depth = _uniform_depth(value=3.7)
        depth[240, 420:425] = 0.05
        z = nearest_depth_in_centerbox(depth, pct=10.0)
        assert z == pytest.approx(3.7)
        z_min = nearest_depth_in_centerbox(depth, pct=0.0)
        assert z_min == pytest.approx(0.05)


# ---- click-probe depth sampler -------------------------------------------

class TestDepthAtPoint:
    def test_uniform_field(self):
        depth = _uniform_depth(value=1.5)
        assert depth_at_point(depth, 400, 240) == pytest.approx(1.5)

    def test_off_center_target_recovered(self):
        """Centerbox is blind to off-center targets; click probe is not."""
        depth = _uniform_depth(value=3.7)
        h, w = depth.shape
        y, x = np.ogrid[:h, :w]
        mask = (y - 100) ** 2 + (x - 700) ** 2 <= 30 ** 2
        depth[mask] = 1.2
        assert median_depth_in_centerbox(depth) == pytest.approx(3.7)
        assert depth_at_point(depth, 700, 100) == pytest.approx(1.2)

    def test_edge_clamping(self):
        depth = _uniform_depth(value=2.0)
        assert depth_at_point(depth, 5, 5, half=20) == pytest.approx(2.0)
        assert depth_at_point(depth, 843, 475, half=20) == pytest.approx(2.0)

    def test_too_few_valid_returns_zero(self):
        depth = np.zeros((480, 848), dtype=np.float32)
        depth[240, 424] = 1.5
        assert depth_at_point(depth, 424, 240) == 0.0


# ---- projector -> camera inverse homography ------------------------------

class TestProjectorToCameraInverse:
    """The click-probe inverse-maps projector coords through composed H to
    camera coords. Validate the math here so the live UI isn't the first
    place we discover sign/transpose errors."""

    def test_identity_homography(self):
        H = np.eye(3)
        H_inv = np.linalg.inv(H)
        for px, py in [(100, 50), (400, 240), (847, 479)]:
            v = H_inv @ np.array([px, py, 1.0])
            assert (int(v[0] / v[2]), int(v[1] / v[2])) == (px, py)

    def test_scaling_homography(self):
        """2x scale projector -> inverse halves projector coords."""
        H = np.diag([2.0, 2.0, 1.0])
        H_inv = np.linalg.inv(H)
        v = H_inv @ np.array([1000.0, 480.0, 1.0])
        assert (int(v[0] / v[2]), int(v[1] / v[2])) == (500, 240)

    def test_translation_homography(self):
        """Projector shifted +100 px right of camera -> click (300, 200) ->
        camera (200, 200)."""
        H = np.array([[1.0, 0.0, 100.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        H_inv = np.linalg.inv(H)
        v = H_inv @ np.array([300.0, 200.0, 1.0])
        assert (int(v[0] / v[2]), int(v[1] / v[2])) == (200, 200)


# ---- integration: replay the 2026-06-02 bodhran scene -------------------

def test_bodhran_on_couch_scene():
    """End-to-end repro of the field failure: bodhran on couch in front
    of back wall. nearest probe recovers the foreground, median does
    not. This test exists so a future refactor of the centerbox probe
    breaks loudly when it regresses the live behavior."""
    depth = _small_target_on_wall(target_depth=1.2, wall_depth=3.7,
                                   target_radius=30)
    z_median = median_depth_in_centerbox(depth)
    z_near = nearest_depth_in_centerbox(depth, pct=10.0)
    assert z_median > 3.0   # swamped by wall
    assert z_near < 2.0    # recovered foreground
    inv_denom = abs(1.0 / z_near - 1.0 / 3.7)
    assert inv_denom > 0.3  # ~0.564 in this scene

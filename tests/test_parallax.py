"""Unit tests for the dual-depth parallax math.

These cover both calibration paths used by the perception pipeline:
 - Two-plane homography lerp (`alpha_from_depth`, `interp_homography`,
   `homography_for_depth`) — the post-calibration path.
 - Constant-K x-shift fallback (`constant_k_shift_px`, `shift_matrix`)
   — used when no two-plane calibration has been captured yet.

Pure numpy / Python — no camera, no projector, no torch.
"""
from __future__ import annotations

import numpy as np
import pytest

from livetracking.perception.parallax import (
    alpha_from_depth,
    constant_k_shift_px,
    homography_for_depth,
    interp_homography,
    shift_matrix,
)


# ---------------------------------------------------------------------------
# alpha_from_depth
# ---------------------------------------------------------------------------

class TestAlphaFromDepth:
    def test_alpha_zero_at_wall(self):
        # Object sitting on the wall plane -> alpha 0 -> use H_wall.
        a = alpha_from_depth(z_obj=3.4, z_wall=3.4, z_near=2.0)
        assert a == pytest.approx(0.0, abs=1e-9)

    def test_alpha_one_at_near(self):
        # Object at the near-plane depth -> alpha 1 -> use H_near.
        a = alpha_from_depth(z_obj=2.0, z_wall=3.4, z_near=2.0)
        assert a == pytest.approx(1.0, abs=1e-9)

    def test_alpha_halfway_in_inverse_depth(self):
        # alpha is linear in 1/z, NOT in z. With z_wall=4, z_near=2 the
        # 1/z midpoint is 1/z = (1/4 + 1/2)/2 = 0.375 -> z = 2.6666...
        z_wall, z_near = 4.0, 2.0
        z_mid = 1.0 / ((1.0 / z_wall + 1.0 / z_near) / 2.0)
        a = alpha_from_depth(z_obj=z_mid, z_wall=z_wall, z_near=z_near)
        assert a == pytest.approx(0.5, abs=1e-9)

    def test_alpha_clamped_at_max_for_very_close_objects(self):
        # Object much closer than NEAR -> alpha would overshoot far past 1.
        # Default cap is 1.5 so hands in front of the camera don't blow up.
        a = alpha_from_depth(z_obj=0.3, z_wall=3.4, z_near=2.0)
        assert a == pytest.approx(1.5, abs=1e-9)

    def test_alpha_clamped_at_zero_for_far_objects(self):
        # Object farther than the wall -> alpha would go negative; clamp 0.
        a = alpha_from_depth(z_obj=10.0, z_wall=3.4, z_near=2.0)
        assert a == pytest.approx(0.0, abs=1e-9)

    def test_alpha_zero_for_zero_depth(self):
        # No depth reading -> safe fallback (treat as wall).
        a = alpha_from_depth(z_obj=0.0, z_wall=3.4, z_near=2.0)
        assert a == 0.0

    def test_alpha_zero_for_negative_depth(self):
        a = alpha_from_depth(z_obj=-1.0, z_wall=3.4, z_near=2.0)
        assert a == 0.0

    def test_alpha_zero_for_degenerate_calibration(self):
        # z_wall == z_near -> no parallax basis.
        a = alpha_from_depth(z_obj=2.5, z_wall=3.0, z_near=3.0)
        assert a == 0.0

    def test_alpha_zero_for_bad_wall_depth(self):
        a = alpha_from_depth(z_obj=2.0, z_wall=0.0, z_near=2.0)
        assert a == 0.0

    def test_alpha_vectorized_over_array(self):
        zs = np.array([3.4, 2.0, 10.0, 0.0])
        result = alpha_from_depth(zs, z_wall=3.4, z_near=2.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(0.0)  # far -> clamped
        assert result[3] == pytest.approx(0.0)  # bad depth

    def test_alpha_max_is_configurable(self):
        a = alpha_from_depth(
            z_obj=0.3, z_wall=3.4, z_near=2.0, alpha_max=2.0
        )
        assert a == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# interp_homography
# ---------------------------------------------------------------------------

class TestInterpHomography:
    def setup_method(self):
        self.H_wall = np.eye(3, dtype=np.float64)
        self.H_near = np.array(
            [[1.0, 0.0, 100.0],
             [0.0, 1.0, 50.0],
             [0.0, 0.0, 1.0]], dtype=np.float64,
        )

    def test_alpha_zero_returns_wall(self):
        M = interp_homography(self.H_wall, self.H_near, alpha=0.0)
        np.testing.assert_allclose(M, self.H_wall)

    def test_alpha_one_returns_near(self):
        M = interp_homography(self.H_wall, self.H_near, alpha=1.0)
        np.testing.assert_allclose(M, self.H_near)

    def test_alpha_half_is_midpoint(self):
        M = interp_homography(self.H_wall, self.H_near, alpha=0.5)
        expected = 0.5 * (self.H_wall + self.H_near)
        np.testing.assert_allclose(M, expected)

    def test_overshoot_extrapolates(self):
        # alpha=1.5 -> 50% past the NEAR plane in the WALL->NEAR direction.
        M = interp_homography(self.H_wall, self.H_near, alpha=1.5)
        # Translation x should be 150 (=1.5 * 100), y should be 75.
        assert M[0, 2] == pytest.approx(150.0)
        assert M[1, 2] == pytest.approx(75.0)

    def test_returns_fresh_array_not_view(self):
        M = interp_homography(self.H_wall, self.H_near, alpha=0.0)
        M[0, 0] = 999.0
        assert self.H_wall[0, 0] == 1.0, "interp returned a view into H_wall"

    def test_rejects_non_3x3(self):
        bad = np.eye(4)
        with pytest.raises(ValueError):
            interp_homography(bad, self.H_near, alpha=0.5)
        with pytest.raises(ValueError):
            interp_homography(self.H_wall, bad, alpha=0.5)


# ---------------------------------------------------------------------------
# homography_for_depth (end-to-end convenience)
# ---------------------------------------------------------------------------

class TestHomographyForDepth:
    def setup_method(self):
        self.H_wall = np.eye(3, dtype=np.float64)
        self.H_near = np.array(
            [[1.0, 0.0, 200.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]], dtype=np.float64,
        )

    def test_object_at_wall_uses_wall_homography(self):
        M = homography_for_depth(
            self.H_wall, self.H_near, z_wall=3.4, z_near=2.0, z_obj=3.4,
        )
        np.testing.assert_allclose(M, self.H_wall)

    def test_object_at_near_uses_near_homography(self):
        M = homography_for_depth(
            self.H_wall, self.H_near, z_wall=3.4, z_near=2.0, z_obj=2.0,
        )
        np.testing.assert_allclose(M, self.H_near)

    def test_degenerate_calibration_falls_back_to_wall(self):
        M = homography_for_depth(
            self.H_wall, self.H_near, z_wall=3.0, z_near=3.0, z_obj=2.0,
        )
        np.testing.assert_allclose(M, self.H_wall)

    def test_missing_depth_falls_back_to_wall(self):
        M = homography_for_depth(
            self.H_wall, self.H_near, z_wall=3.4, z_near=2.0, z_obj=0.0,
        )
        np.testing.assert_allclose(M, self.H_wall)


# ---------------------------------------------------------------------------
# constant_k_shift_px (legacy fallback)
# ---------------------------------------------------------------------------

class TestConstantKShift:
    def test_zero_when_object_at_wall(self):
        s = constant_k_shift_px(z_obj=3.4, z_wall=3.4, k_px_m=1200.0)
        assert s == 0.0

    def test_zero_when_object_just_in_front_of_wall(self):
        # Inside the min_gap_m=0.05 dead band.
        s = constant_k_shift_px(z_obj=3.38, z_wall=3.4, k_px_m=1200.0)
        assert s == 0.0

    def test_zero_for_bad_depth(self):
        assert constant_k_shift_px(z_obj=0.0, z_wall=3.4, k_px_m=1200.0) == 0.0
        assert constant_k_shift_px(z_obj=2.0, z_wall=0.0, k_px_m=1200.0) == 0.0

    def test_zero_when_object_behind_wall(self):
        # Object behind the wall plane (e.g. through-doorway artifact).
        s = constant_k_shift_px(z_obj=5.0, z_wall=3.4, k_px_m=1200.0)
        assert s == 0.0

    def test_positive_sign_default(self):
        # Default sign=+1: object closer than wall -> positive shift.
        s = constant_k_shift_px(z_obj=2.0, z_wall=3.4, k_px_m=1200.0)
        # disparity = 1/2 - 1/3.4 = 0.20588..., * 1200 ~ 247 px
        assert s > 0
        assert s == pytest.approx(1200.0 * (1.0 / 2.0 - 1.0 / 3.4), abs=1e-6)

    def test_sign_negative_flips(self):
        s_pos = constant_k_shift_px(z_obj=2.0, z_wall=3.4, k_px_m=1200.0, sign=+1.0)
        s_neg = constant_k_shift_px(z_obj=2.0, z_wall=3.4, k_px_m=1200.0, sign=-1.0)
        assert s_neg == pytest.approx(-s_pos)

    def test_scale_multiplies_shift(self):
        s1 = constant_k_shift_px(z_obj=2.0, z_wall=3.4, k_px_m=1200.0, scale=1.0)
        s2 = constant_k_shift_px(z_obj=2.0, z_wall=3.4, k_px_m=1200.0, scale=2.5)
        assert s2 == pytest.approx(2.5 * s1)

    def test_shift_grows_as_object_approaches_camera(self):
        s_far = constant_k_shift_px(z_obj=3.0, z_wall=3.4, k_px_m=1200.0)
        s_mid = constant_k_shift_px(z_obj=2.0, z_wall=3.4, k_px_m=1200.0)
        s_near = constant_k_shift_px(z_obj=1.0, z_wall=3.4, k_px_m=1200.0)
        assert s_far < s_mid < s_near


# ---------------------------------------------------------------------------
# shift_matrix
# ---------------------------------------------------------------------------

class TestShiftMatrix:
    def test_identity_when_no_shift(self):
        np.testing.assert_allclose(shift_matrix(0.0, 0.0), np.eye(3))

    def test_translation_applies_correctly(self):
        T = shift_matrix(100.0, -50.0)
        pt = np.array([10.0, 20.0, 1.0])
        out = T @ pt
        assert out[0] == pytest.approx(110.0)
        assert out[1] == pytest.approx(-30.0)
        assert out[2] == pytest.approx(1.0)

    def test_returns_3x3_float64(self):
        T = shift_matrix(1.0)
        assert T.shape == (3, 3)
        assert T.dtype == np.float64

"""Pure-numpy parallax math, extracted so it's unit-testable without
loading the full perception pipeline (which pulls in torch, SAM2, DINO,
RealSense, etc.).

Two depth-aware parallax strategies live here:

1. Two-plane homography interpolation (`alpha_from_depth`,
   `interp_homography`). Given calibration matrices `H_wall` and
   `H_near` and their reference depths, the homography for an object
   at depth `z_obj` is a straight-line lerp in inverse-depth space:

       alpha = (1/z_obj - 1/z_wall) / (1/z_near - 1/z_wall)
       H(z)  = (1 - alpha) * H_wall + alpha * H_near

   Inverse-depth is the right axis because parallax disparity is
   linear in 1/z (camera-projector baseline geometry, small-angle).

2. Constant-K x-shift (`constant_k_shift_px`). The legacy fallback
   used before manual two-plane calibration: a single tunable K
   (units: pixels * meters) scales the disparity `(1/z_obj -
   1/z_wall)` directly into a projector-x shift.

Conventions:
- `alpha` is clamped to [0, alpha_max] (default 1.5). Slight overshoot
  is allowed for objects closer than the NEAR plane (extrapolation),
  capped so a hand right in front of the camera doesn't blow up.
- All depths are meters; non-positive depths short-circuit to alpha=0
  (no parallax correction = behave like the wall plane).
- Functions are vectorized: `z_obj` may be a scalar or array.
"""
from __future__ import annotations

from typing import Union

import numpy as np

ArrayOrFloat = Union[float, np.ndarray]


def alpha_from_depth(
    z_obj: ArrayOrFloat,
    z_wall: float,
    z_near: float,
    alpha_max: float = 1.5,
) -> ArrayOrFloat:
    """Compute interpolation weight for two-plane parallax.

    alpha = 0 -> use H_wall (object at wall depth).
    alpha = 1 -> use H_near (object at near-plane depth).
    alpha clamped to [0, alpha_max]; values outside that range are
    extrapolation regions.

    Degenerate inputs return alpha=0:
      * z_obj <= 0 (no depth reading)
      * z_wall <= 0 or z_near <= 0 (bad calibration)
      * z_wall == z_near (degenerate calibration; no parallax basis)
    """
    z_obj_arr = np.asarray(z_obj, dtype=np.float64)
    if z_wall <= 0.0 or z_near <= 0.0 or abs(z_wall - z_near) < 1e-9:
        out = np.zeros_like(z_obj_arr)
        return float(out) if out.ndim == 0 else out

    inv_zw = 1.0 / z_wall
    inv_zn = 1.0 / z_near
    denom = inv_zn - inv_zw  # nonzero by the guard above

    # For non-positive z_obj, fall back to alpha=0 (treat as wall).
    safe_z = np.where(z_obj_arr > 0.0, z_obj_arr, z_wall)
    inv_zo = 1.0 / safe_z
    raw = (inv_zo - inv_zw) / denom
    raw = np.where(z_obj_arr > 0.0, raw, 0.0)
    clipped = np.clip(raw, 0.0, alpha_max)
    return float(clipped) if clipped.ndim == 0 else clipped


def interp_homography(
    H_wall: np.ndarray,
    H_near: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Element-wise lerp between two 3x3 homographies.

    Geometrically a straight-line homotopy in homography space — valid
    here because both H_wall and H_near map the SAME camera plane to
    the SAME projector frame; only the depth of the alignment target
    differs.

    Returns a fresh float64 3x3.
    """
    H_wall = np.asarray(H_wall, dtype=np.float64)
    H_near = np.asarray(H_near, dtype=np.float64)
    if H_wall.shape != (3, 3) or H_near.shape != (3, 3):
        raise ValueError(
            f"homographies must be 3x3; got {H_wall.shape} and {H_near.shape}"
        )
    return (1.0 - alpha) * H_wall + alpha * H_near


def homography_for_depth(
    H_wall: np.ndarray,
    H_near: np.ndarray,
    z_wall: float,
    z_near: float,
    z_obj: float,
    alpha_max: float = 1.5,
) -> np.ndarray:
    """Convenience: compute alpha for `z_obj` then lerp H_wall->H_near.

    When calibration is degenerate (see `alpha_from_depth`) this
    returns `H_wall` unchanged.
    """
    a = alpha_from_depth(z_obj, z_wall, z_near, alpha_max=alpha_max)
    return interp_homography(H_wall, H_near, float(a))


def constant_k_shift_px(
    z_obj: float,
    z_wall: float,
    k_px_m: float,
    sign: float = 1.0,
    scale: float = 1.0,
    min_gap_m: float = 0.05,
) -> float:
    """Legacy constant-K x-shift in projector pixels.

    Returns 0.0 unless the object sits clearly in front of the wall
    (`z_obj < z_wall - min_gap_m`). Formula:

        shift = sign * scale * k_px_m * (1/z_obj - 1/z_wall)

    Positive shift = move warped projector mask right; negative = left.
    The caller composes this into a translation matrix and folds it
    into the warp.
    """
    if z_obj <= 0.1 or z_wall <= 0.1:
        return 0.0
    if z_obj >= z_wall - min_gap_m:
        return 0.0
    disparity = (1.0 / z_obj) - (1.0 / z_wall)
    return float(sign * scale * k_px_m * disparity)


def shift_matrix(shift_x: float, shift_y: float = 0.0) -> np.ndarray:
    """Build a 3x3 translation matrix usable as `T @ H`."""
    return np.array(
        [
            [1.0, 0.0, float(shift_x)],
            [0.0, 1.0, float(shift_y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

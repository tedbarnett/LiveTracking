"""Predict a target's projector position from already-converged peer targets.

The closed-loop convergence's quality depends heavily on the *initial*
projector guess. The default init uses a 2D planar homography of the
projection rectangle's 4 corners, which assumes the projector image lands
on a flat-rectangular region of the camera's view. For a real Kodak Pocket
Projector pointing at a wall at an angle, that assumption breaks for
target positions outside the calibration quad - search wanders, diverges
to a frame edge.

Better: if 2+ other targets have already converged successfully, we have
a small set of known (cam_xy, proj_xy) correspondences. Fit a local model
on those pairs and predict the new target's proj_xy from its cam_xy.

This module has NO opencv / pygame / RealSense deps so it's unit-testable.

Functions
---------
predict_proj_from_peers(target_cam, peers) -> [px, py] | None
    target_cam: (cx, cy)
    peers: list of dicts with keys "target_cam" and "converged_proj",
           each (x, y). Use winners list from the daemon.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


PointLike = Sequence[float]


def _to_array(points: Iterable[PointLike]) -> np.ndarray:
    return np.asarray([[float(p[0]), float(p[1])] for p in points],
                       dtype=np.float64)


def predict_proj_from_peers(
    target_cam: PointLike,
    peers: List[dict],
) -> Optional[List[float]]:
    """Predict projector (x, y) for target_cam from converged peers.

    Returns a Python list [px, py] (not numpy) so downstream pygame code
    doesn't choke on array operations. Returns None if fewer than 2 peers
    or the peers degenerate (collinear, identical cam positions).
    """
    if len(peers) < 2:
        return None

    cam_pts = _to_array(p["target_cam"] for p in peers)
    proj_pts = _to_array(p["converged_proj"] for p in peers)
    target = np.asarray([float(target_cam[0]), float(target_cam[1])],
                          dtype=np.float64)

    if len(peers) == 2:
        # Two-point local similarity: translate + rotate + uniform scale.
        cam_a, cam_b = cam_pts[0], cam_pts[1]
        proj_a, proj_b = proj_pts[0], proj_pts[1]
        cam_span = cam_b - cam_a
        proj_span = proj_b - proj_a
        cam_len = float(np.linalg.norm(cam_span))
        proj_len = float(np.linalg.norm(proj_span))
        if cam_len < 1e-3 or proj_len < 1e-3:
            return None
        # Rotation: angle between cam_span and proj_span
        cos_t = float(np.dot(cam_span, proj_span) / (cam_len * proj_len))
        # Cross product (in 2D, sign of z component)
        sin_t = float(cam_span[0] * proj_span[1]
                       - cam_span[1] * proj_span[0]) / (cam_len * proj_len)
        s = proj_len / cam_len
        # Apply: pred - proj_mid = s * R * (target - cam_mid)
        cam_mid = (cam_a + cam_b) / 2.0
        proj_mid = (proj_a + proj_b) / 2.0
        d = target - cam_mid
        rotated = np.array([cos_t * d[0] - sin_t * d[1],
                              sin_t * d[0] + cos_t * d[1]])
        pred = proj_mid + s * rotated
        if not np.isfinite(pred).all():
            return None
        return [float(pred[0]), float(pred[1])]

    # 3+ peers: full affine via least squares.
    # Build [cx, cy, 1] x M = [px, py] for each peer.
    A = np.hstack([cam_pts, np.ones((len(cam_pts), 1))])
    # Solve A @ M = proj_pts for M (3x2). lstsq is robust to near-singular.
    try:
        M, *_ = np.linalg.lstsq(A, proj_pts, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cam_h = np.array([target[0], target[1], 1.0])
    pred = cam_h @ M
    if not np.isfinite(pred).all():
        return None
    return [float(pred[0]), float(pred[1])]


# ----- self-test (run as `python -m livetracking.calib_v1.peer_init`) -----

# Self-tests for this module live in tests/ (pytest-runnable).

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

if __name__ == "__main__":
    # Test 1: two peers, predict midpoint scenario.
    peers_2 = [
        {"target_cam": (100, 100), "converged_proj": (200, 200)},
        {"target_cam": (300, 100), "converged_proj": (600, 200)},
    ]
    # Target at (200, 100) - same y, midway in x. Expect proj (400, 200).
    pred = predict_proj_from_peers((200, 100), peers_2)
    print(f"Test 1 (2 peers midpoint): pred={pred} expected=[400, 200]")
    assert pred is not None
    assert abs(pred[0] - 400) < 0.01 and abs(pred[1] - 200) < 0.01

    # Test 2: two peers, predict an off-axis target.
    # Linear scale x2 along x axis. Target (200, 200) -> proj (400, 400)?
    # Actually with similarity (uniform scale + rotation), y also scales.
    # cam_span = (200, 0) proj_span = (400, 0). Same direction, scale=2.
    # Target (200, 200) - cam_mid (200, 100) = (0, 100).
    # Rotated by 0 (cos=1 sin=0), times scale 2 = (0, 200).
    # proj_mid (400, 200) + (0, 200) = (400, 400). Yes.
    pred = predict_proj_from_peers((200, 200), peers_2)
    print(f"Test 2 (2 peers off-axis): pred={pred} expected=[400, 400]")
    assert abs(pred[0] - 400) < 0.01 and abs(pred[1] - 400) < 0.01

    # Test 3: three peers, full affine.
    peers_3 = [
        {"target_cam": (0, 0), "converged_proj": (0, 0)},
        {"target_cam": (100, 0), "converged_proj": (200, 0)},
        {"target_cam": (0, 100), "converged_proj": (0, 300)},
    ]
    # Linear: px = 2*cx, py = 3*cy. Target (50, 50) -> (100, 150).
    pred = predict_proj_from_peers((50, 50), peers_3)
    print(f"Test 3 (3 peers affine): pred={pred} expected=[100, 150]")
    assert abs(pred[0] - 100) < 0.01 and abs(pred[1] - 150) < 0.01

    # Test 4: realistic - matches the actual T2/T3 -> T1 scenario.
    # T2 cam (523, 230) proj (340, 282)
    # T3 cam (577, 206) proj (584, 178)
    # T1 cam (483, 266) - predict?
    peers_real = [
        {"target_cam": (523, 230), "converged_proj": (340, 282)},
        {"target_cam": (577, 206), "converged_proj": (584, 178)},
    ]
    pred = predict_proj_from_peers((483, 266), peers_real)
    print(f"Test 4 (T1 from T2+T3): pred={pred}")
    assert pred is not None
    # T1 should be somewhere left-and-down of T2 in projector space.
    # Sanity: pred should be in the projector frame (1280 x 720),
    # not at clamped edges.
    assert 0 < pred[0] < 1280, f"T1 pred x {pred[0]} outside [0, 1280]"
    assert 0 < pred[1] < 720, f"T1 pred y {pred[1]} outside [0, 720]"
    print(f"   (verified T1 prediction is inside the projector frame)")

    # Test 5: degenerate - identical peer positions
    peers_bad = [
        {"target_cam": (100, 100), "converged_proj": (200, 200)},
        {"target_cam": (100, 100), "converged_proj": (300, 300)},
    ]
    pred = predict_proj_from_peers((50, 50), peers_bad)
    print(f"Test 5 (degenerate peers): pred={pred} expected=None")
    assert pred is None

    # Test 6: too few peers
    pred = predict_proj_from_peers((50, 50),
                                       [{"target_cam": (0, 0),
                                          "converged_proj": (0, 0)}])
    print(f"Test 6 (1 peer): pred={pred} expected=None")
    assert pred is None

    # Test 7: returns list not ndarray
    pred = predict_proj_from_peers((200, 100), peers_2)
    print(f"Test 7 (return type): type={type(pred).__name__} contents_type={type(pred[0]).__name__}")
    assert isinstance(pred, list)
    assert isinstance(pred[0], float)
    assert isinstance(pred[1], float)

    print("\nAll 7 self-tests passed.")

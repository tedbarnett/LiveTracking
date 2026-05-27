"""Pytest port of the in-module self-tests previously at the bottom of
``src/livetracking/calib_v1/peer_init.py``.

Test cases are unchanged - only restructured so they run under ``pytest``
(CI-friendly, isolated, prints captured) instead of being executed at module
import time.
"""
from __future__ import annotations

import pytest

from livetracking.calib_v1.peer_init import predict_proj_from_peers


def test_two_peers_midpoint():
    peers = [
        {"target_cam": (100, 100), "converged_proj": (200, 200)},
        {"target_cam": (300, 100), "converged_proj": (600, 200)},
    ]
    pred = predict_proj_from_peers((200, 100), peers)
    assert pred is not None
    assert pred[0] == pytest.approx(400, abs=0.01)
    assert pred[1] == pytest.approx(200, abs=0.01)


def test_two_peers_off_axis():
    peers = [
        {"target_cam": (100, 100), "converged_proj": (200, 200)},
        {"target_cam": (300, 100), "converged_proj": (600, 200)},
    ]
    pred = predict_proj_from_peers((200, 200), peers)
    assert pred is not None
    assert pred[0] == pytest.approx(400, abs=0.01)
    assert pred[1] == pytest.approx(400, abs=0.01)


def test_three_peers_affine():
    peers = [
        {"target_cam": (0, 0), "converged_proj": (0, 0)},
        {"target_cam": (100, 0), "converged_proj": (200, 0)},
        {"target_cam": (0, 100), "converged_proj": (0, 300)},
    ]
    # Linear: px = 2*cx, py = 3*cy. Target (50, 50) -> (100, 150).
    pred = predict_proj_from_peers((50, 50), peers)
    assert pred is not None
    assert pred[0] == pytest.approx(100, abs=0.01)
    assert pred[1] == pytest.approx(150, abs=0.01)


def test_realistic_t1_from_t2_t3():
    # Matches the actual scenario from the daemon's run.
    peers = [
        {"target_cam": (523, 230), "converged_proj": (340, 282)},
        {"target_cam": (577, 206), "converged_proj": (584, 178)},
    ]
    pred = predict_proj_from_peers((483, 266), peers)
    assert pred is not None
    # Should land inside the projector frame, not at a clamped edge.
    assert 0 < pred[0] < 1280, f"x {pred[0]} outside [0, 1280]"
    assert 0 < pred[1] < 720, f"y {pred[1]} outside [0, 720]"


def test_degenerate_identical_peers():
    peers = [
        {"target_cam": (100, 100), "converged_proj": (200, 200)},
        {"target_cam": (100, 100), "converged_proj": (300, 300)},
    ]
    assert predict_proj_from_peers((50, 50), peers) is None


def test_too_few_peers():
    pred = predict_proj_from_peers(
        (50, 50),
        [{"target_cam": (0, 0), "converged_proj": (0, 0)}],
    )
    assert pred is None


def test_return_type_is_list_of_float():
    peers = [
        {"target_cam": (100, 100), "converged_proj": (200, 200)},
        {"target_cam": (300, 100), "converged_proj": (600, 200)},
    ]
    pred = predict_proj_from_peers((200, 100), peers)
    assert isinstance(pred, list)
    assert all(isinstance(v, float) for v in pred)

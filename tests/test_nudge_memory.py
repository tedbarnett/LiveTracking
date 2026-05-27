"""Pytest port of the in-module self-tests previously at the bottom of
``src/livetracking/calib_v1/nudge_memory.py``.

Behaviour-identical to the original ``__main__`` block - each former test gets
its own function with proper isolation via tmp_path.
"""
from __future__ import annotations

import os
import time

import pytest

from livetracking.calib_v1.nudge_memory import NudgeMemory


@pytest.fixture
def memfile(tmp_path):
    return str(tmp_path / "test.jsonl")


def test_empty_lookup(memfile):
    m = NudgeMemory(memfile)
    assert m.lookup((100, 100)) is None


def test_append_and_exact_lookup(memfile):
    m = NudgeMemory(memfile)
    m.append((100, 100), (5, -3))
    assert m.lookup((100, 100)) == [5.0, -3.0]


def test_lookup_within_radius(memfile):
    m = NudgeMemory(memfile)
    m.append((100, 100), (5, -3))
    # ~11 px away, within default 50 px radius.
    assert m.lookup((110, 105)) == [5.0, -3.0]


def test_lookup_outside_radius(memfile):
    m = NudgeMemory(memfile)
    m.append((100, 100), (5, -3))
    assert m.lookup((200, 200)) is None


def test_persistence_reload(memfile):
    m = NudgeMemory(memfile)
    m.append((100, 100), (5, -3))
    m2 = NudgeMemory(memfile)
    assert len(m2.entries) == 1
    assert m2.lookup((100, 100)) == [5.0, -3.0]


def test_newest_wins_in_region(memfile):
    m = NudgeMemory(memfile)
    m.append((100, 100), (5, -3))
    time.sleep(0.01)
    m.append((105, 105), (10, -10))  # 7 px from first, newer
    assert m.lookup((100, 100)) == [10.0, -10.0]


def test_far_lookup_none(memfile):
    m = NudgeMemory(memfile)
    m.append((100, 100), (5, -3))
    assert m.lookup((500, 500)) is None


def test_return_type_is_list_of_float(memfile):
    m = NudgeMemory(memfile)
    m.append((100, 100), (5, -3))
    result = m.lookup((100, 100))
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


def test_clear(memfile):
    m = NudgeMemory(memfile)
    m.append((100, 100), (5, -3))
    m.clear()
    assert m.lookup((100, 100)) is None
    assert not os.path.exists(memfile)


def test_real_world_t1_two_positions(memfile):
    """T1 had a -5,0 nudge at its original cam position. After moving, it got
    a much larger (135, -280) nudge. Both should be recoverable independently
    via cam-position lookup."""
    m = NudgeMemory(memfile)
    m.append((488, 200), (-5, 0))
    time.sleep(0.01)
    m.append((483, 266), (135, -280))
    assert m.lookup((488, 200)) == [-5.0, 0.0]
    assert m.lookup((483, 266)) == [135.0, -280.0]
    assert m.lookup((600, 400)) is None

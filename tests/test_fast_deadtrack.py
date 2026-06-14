"""Regression test: a dead highlighted track must clear its projector offset.

The bug (observed live, wash flew onto a door): when the *only* highlighted
object's track dies between SAM passes — its id churns away / disappears — the
old ``_fast_track_step`` computed an empty ``offsets`` dict, hit the
``if offsets:`` guard, and sent nothing. But ``set_offsets`` fully *replaces*
the projector's offset table, so skipping the send left the projector applying
that object's last (large) offset to its still-cached mask forever. The wash
ran away across the room instead of falling back to its anchored position.

The fix: detect highlighted ids whose track is no longer live, prune them from
the fast tracker + telemetry, and ALWAYS emit ``set_offsets`` when a track died
(an empty dict for that object is the clear).

This drives the REAL ``PerceptionDaemon._fast_track_step`` against lightweight
fakes (a real FastTracker, a fake pipeline/tracker, a recording push socket) —
no RealSense, torch, or pygame. It asserts the wire behavior the projector
relies on, not an internal detail.
"""
from __future__ import annotations

import threading
import types

import numpy as np
import pytest

pytest.importorskip("cv2")

from livetracking.daemon.perception import PerceptionDaemon
from livetracking.perception.fasttrack import FastTracker


# ---- fakes --------------------------------------------------------------

class _FakeObj:
    """Stand-in for a DetectedObject: only the attributes _fast_track_step reads."""

    def __init__(self, object_id, cx, cy, r=40, depth=1.5):
        self.object_id = object_id
        self.median_depth_m = depth
        h, w = 480, 848
        y, x = np.ogrid[:h, :w]
        self.cam_mask = (((y - cy) ** 2 + (x - cx) ** 2 <= r * r)
                         .astype(np.uint8)) * 255
        self.bbox_cam = (cx - r, cy - r, 2 * r, 2 * r)
        self.centroid_cam = (float(cx), float(cy))
        self.hidden = False


class _FakeTracker:
    def __init__(self):
        self._visible = []

    def visible(self):
        return list(self._visible)


class _FakePipeline:
    def __init__(self):
        self.tracker = _FakeTracker()
        self.tracker_lock = threading.Lock()
        self.recognize_seq = 0

    def cam_to_proj_point(self, xy, med_z):
        # Linear 1:1 mapping is enough — the test asserts presence/absence of a
        # clear, not the exact offset magnitude.
        return (float(xy[0]), float(xy[1]))


class _RecordingPush:
    def __init__(self):
        self.sent = []

    def send_json(self, msg):
        self.sent.append(msg)


def _make_daemon():
    """A bare object carrying just the state _fast_track_step touches, with the
    real methods bound to it."""
    d = types.SimpleNamespace()
    d.pipeline = _FakePipeline()
    d.proj_push = _RecordingPush()
    d._fast = FastTracker()
    d._fast_seed_seq = -1
    d._fast_stats = {}
    d._last_highlight = None
    # Bind the real methods.
    d._fast_track_step = types.MethodType(
        PerceptionDaemon._fast_track_step, d)
    d._highlighted_ids = types.MethodType(
        PerceptionDaemon._highlighted_ids, d)
    return d


def _last_set_offsets(push):
    for msg in reversed(push.sent):
        if msg.get("type") == "set_offsets":
            return msg
    return None


# ---- tests --------------------------------------------------------------

class TestDeadTrackClearsOffset:
    def test_dead_highlighted_track_emits_clear(self):
        d = _make_daemon()
        color = np.zeros((480, 848, 3), dtype=np.uint8)
        depth = np.full((480, 848), 1.5, dtype=np.float32)

        # Frame 1: guitar #1 is live and highlighted -> offset emitted for it.
        guitar = _FakeObj(1, cx=400, cy=240)
        d.pipeline.tracker._visible = [guitar]
        d._last_highlight = {"kind": "single", "id": 1}
        d.pipeline.recognize_seq = 1  # fresh pass -> reseed
        d._fast_track_step(color, depth)

        msg1 = _last_set_offsets(d.proj_push)
        assert msg1 is not None, "should emit offsets while object is live"
        assert "1" in msg1["offsets"], "live object must have an offset"
        assert 1 in d._fast.active_ids(), "fast tracker should hold the live id"

        # Frame 2: the track DIED (id churned away). Still highlighted as #1,
        # but #1 is no longer in the visible list.
        d.pipeline.tracker._visible = []
        d.pipeline.recognize_seq = 1  # no new pass; mid-pass churn
        n_before = len(d.proj_push.sent)
        d._fast_track_step(color, depth)

        assert len(d.proj_push.sent) > n_before, (
            "a dead highlighted track MUST emit a set_offsets clear "
            "(the old bug was sending nothing, leaving a stale offset)"
        )
        msg2 = _last_set_offsets(d.proj_push)
        assert msg2 is not None
        assert "1" not in msg2["offsets"], (
            "dead object's offset must be gone so the projector drops it"
        )
        assert 1 not in d._fast.active_ids(), (
            "dead id must be pruned from the fast tracker"
        )
        assert 1 not in d._fast_stats, "dead id must be pruned from telemetry"

    def test_live_track_still_follows_normally(self):
        """The fix must not break the happy path: a live highlighted object
        keeps getting a non-empty offset every frame."""
        d = _make_daemon()
        color = np.zeros((480, 848, 3), dtype=np.uint8)
        depth = np.full((480, 848), 1.5, dtype=np.float32)

        guitar = _FakeObj(1, cx=400, cy=240)
        d.pipeline.tracker._visible = [guitar]
        d._last_highlight = {"kind": "single", "id": 1}
        d.pipeline.recognize_seq = 1
        d._fast_track_step(color, depth)

        # Next frame, same object still present (no new pass).
        d.pipeline.recognize_seq = 1
        d._fast_track_step(color, depth)
        msg = _last_set_offsets(d.proj_push)
        assert msg is not None and "1" in msg["offsets"]
        assert 1 in d._fast.active_ids()

    def test_one_of_two_dies_keeps_the_survivor(self):
        """With two highlighted objects, if one track dies the survivor must
        keep its offset and the dead one must be dropped — in a single emit."""
        d = _make_daemon()
        color = np.zeros((480, 848, 3), dtype=np.uint8)
        depth = np.full((480, 848), 1.5, dtype=np.float32)

        g1 = _FakeObj(1, cx=300, cy=240)
        g2 = _FakeObj(2, cx=560, cy=240)
        d.pipeline.tracker._visible = [g1, g2]
        d._last_highlight = {"kind": "set", "ids": [1, 2]}
        d.pipeline.recognize_seq = 1
        d._fast_track_step(color, depth)
        msg1 = _last_set_offsets(d.proj_push)
        assert set(msg1["offsets"].keys()) == {"1", "2"}

        # #2 dies, #1 survives.
        d.pipeline.tracker._visible = [g1]
        d.pipeline.recognize_seq = 1
        d._fast_track_step(color, depth)
        msg2 = _last_set_offsets(d.proj_push)
        assert "1" in msg2["offsets"], "survivor keeps its offset"
        assert "2" not in msg2["offsets"], "dead object dropped"
        assert 2 not in d._fast.active_ids()
        assert 1 in d._fast.active_ids()

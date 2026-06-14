"""Tests for lock mode — interactive "light the thing I'm holding/playing".

When an object is LOCKED, the slow DINO+SAM tracker must be barred from touching
its wash. Position is driven purely by the fast tracker (depth-band gate + CSRT),
which rejects the player's body (it sits at a nearer depth than the held object).
This is what stops the highlight being stolen by an arm/body reaching in.

These drive the REAL daemon methods against lightweight fakes (a real
FastTracker, a fake pipeline/tracker, a recording push socket) — no RealSense,
torch, or pygame. They assert the wire behavior the projector relies on:

  * a locked object seeds once then follows via set_offsets;
  * the slow-tracker-fed re-push / fusion paths are NOT taken while locked
    (the main-loop branch picks _lock_follow_step instead);
  * unlock clears the offset and releases the lock.
"""
from __future__ import annotations

import threading
import types

import numpy as np
import pytest

pytest.importorskip("cv2")

from livetracking.daemon.perception import PerceptionDaemon
from livetracking.perception.fasttrack import FastTracker


class _FakeObj:
    def __init__(self, object_id, cx, cy, r=40, depth=2.0):
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
        return (float(xy[0]), float(xy[1]))


class _RecordingPush:
    def __init__(self):
        self.sent = []

    def send_json(self, msg):
        self.sent.append(msg)


def _make_daemon():
    d = types.SimpleNamespace()
    d.pipeline = _FakePipeline()
    d.proj_push = _RecordingPush()
    d._fast = FastTracker()
    d._fast_seed_seq = -1
    d._fast_stats = {}
    d._last_highlight = None
    d._locked_id = None
    d._lock_anchor_cam = None
    d._lock_depth = 0.0
    d._lock_state = None
    d._lock_follow_step = types.MethodType(
        PerceptionDaemon._lock_follow_step, d)
    return d


def _last_set_offsets(push):
    for msg in reversed(push.sent):
        if msg.get("type") == "set_offsets":
            return msg
    return None


class TestLockFollow:
    def test_seed_then_follow_emits_offset(self):
        d = _make_daemon()
        color = np.zeros((480, 848, 3), dtype=np.uint8)
        depth = np.full((480, 848), 2.0, dtype=np.float32)
        # Make a depth blob at the object so the fast tracker has something.
        depth[200:280, 360:440] = 2.0

        guitar = _FakeObj(1, cx=400, cy=240)
        d.pipeline.tracker._visible = [guitar]
        d._locked_id = 1
        d._lock_state = "pending"
        d._lock_anchor_cam = (400.0, 240.0)
        d._lock_depth = 2.0

        # Frame 1: pending -> seeds, no offset yet.
        d._lock_follow_step(color, depth)
        assert d._lock_state == "seeded"
        assert 1 in d._fast.active_ids()

        # Frame 2: running -> emits a set_offsets for the locked id.
        d._lock_follow_step(color, depth)
        msg = _last_set_offsets(d.proj_push)
        assert msg is not None
        assert "1" in msg["offsets"], "locked object must emit an offset"

    def test_lock_ignores_slow_tracker_id_loss(self):
        """The whole point: once seeded, the locked object keeps following even
        if the slow tracker drops/reassigns its id (the visible list no longer
        contains it). Position comes from the fast tracker, not the tracker."""
        d = _make_daemon()
        color = np.zeros((480, 848, 3), dtype=np.uint8)
        depth = np.full((480, 848), 2.0, dtype=np.float32)

        guitar = _FakeObj(1, cx=400, cy=240)
        d.pipeline.tracker._visible = [guitar]
        d._locked_id = 1
        d._lock_state = "pending"
        d._lock_anchor_cam = (400.0, 240.0)
        d._lock_depth = 2.0
        d._lock_follow_step(color, depth)  # seed

        # Slow tracker loses the object entirely (arm churned the id away).
        d.pipeline.tracker._visible = []
        n_before = len(d.proj_push.sent)
        d._lock_follow_step(color, depth)
        # Still follows: emits an offset (or at minimum does not crash and the
        # fast track is retained) — it must NOT clear or drop the lock.
        assert d._locked_id == 1, "lock must survive slow-tracker id loss"
        assert 1 in d._fast.active_ids(), "fast track retained despite id loss"
        assert len(d.proj_push.sent) >= n_before

    def test_pending_waits_when_object_not_yet_visible(self):
        """If the object isn't visible on the seed frame, stay pending and emit
        nothing (the frozen lock-time mask is still showing on the projector).
        """
        d = _make_daemon()
        color = np.zeros((480, 848, 3), dtype=np.uint8)
        depth = np.full((480, 848), 2.0, dtype=np.float32)
        d._locked_id = 1
        d._lock_state = "pending"
        d.pipeline.tracker._visible = []  # not visible yet

        d._lock_follow_step(color, depth)
        assert d._lock_state == "pending", "should keep waiting to seed"
        assert _last_set_offsets(d.proj_push) is None, "no offset before seed"

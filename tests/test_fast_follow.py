"""Tests for Step-1 fast object-following (auto re-push on SAM pass).

When a highlighted object moves, the projector wash only re-lands on it when
perception re-pushes the highlight. Step 1 makes the perception main loop
re-push the active highlight whenever a heavy DINO+SAM pass produces fresh
positions, so the wash follows a moving object within one SAM pass instead of
staying frozen until an unrelated UI event.

Two layers are tested, both fully offline (no RealSense, no torch, no pygame):

1. ``_repush_decision`` — the pure gating logic that decides whether to
   re-push and what the new seq marker is. This is where all the edge cases
   live (flag off, seq not advanced, no highlight, test-hold window).

2. ``Pipeline.recognize_seq`` — the trigger the daemon watches. We drive the
   real Pipeline in sync mode with a fake recognizer that emits a mask at a
   moving centroid, and assert the seq advances once per pass AND the tracked
   object's centroid actually moves (so the re-push, when it fires, carries a
   new position — the whole point).
"""
from __future__ import annotations

import numpy as np
import pytest

from livetracking.daemon.perception import _repush_decision


# ---- layer 1: pure decision logic ---------------------------------------

class TestRepushDecision:
    def test_flag_off_never_pushes_and_keeps_seq(self):
        # Even with a fresh pass and an active highlight, flag off => no push,
        # and the seq marker is left untouched (so flipping the flag on later
        # re-evaluates the latest pass rather than treating it as consumed).
        should, new_seq = _repush_decision(
            fast_track=False, cur_seq=9, last_repush_seq=4,
            has_active_highlight=True, now=0.0, test_hold_until=0.0,
        )
        assert should is False
        assert new_seq == 4

    def test_seq_not_advanced_no_push(self):
        # No new SAM pass since last re-push => no redundant push. This is the
        # guard that protects the projector's 4K-mask decode cache from being
        # invalidated every frame between passes.
        should, new_seq = _repush_decision(
            fast_track=True, cur_seq=7, last_repush_seq=7,
            has_active_highlight=True, now=0.0, test_hold_until=0.0,
        )
        assert should is False
        assert new_seq == 7

    def test_new_pass_with_active_highlight_pushes(self):
        should, new_seq = _repush_decision(
            fast_track=True, cur_seq=8, last_repush_seq=7,
            has_active_highlight=True, now=100.0, test_hold_until=0.0,
        )
        assert should is True
        assert new_seq == 8

    def test_new_pass_no_highlight_consumes_seq_but_no_push(self):
        # Nothing lit on the wall: don't push, but DO advance the marker so we
        # don't re-check this same pass next frame.
        should, new_seq = _repush_decision(
            fast_track=True, cur_seq=8, last_repush_seq=7,
            has_active_highlight=False, now=100.0, test_hold_until=0.0,
        )
        assert should is False
        assert new_seq == 8

    def test_test_hold_window_suppresses_push_but_consumes_seq(self):
        # A /test_light square is being held on screen; perception must not
        # stomp it with its own highlight. Marker still advances so we resume
        # cleanly once the hold expires.
        should, new_seq = _repush_decision(
            fast_track=True, cur_seq=8, last_repush_seq=7,
            has_active_highlight=True, now=100.0, test_hold_until=105.0,
        )
        assert should is False
        assert new_seq == 8

    def test_hold_just_expired_pushes(self):
        should, new_seq = _repush_decision(
            fast_track=True, cur_seq=8, last_repush_seq=7,
            has_active_highlight=True, now=105.0, test_hold_until=105.0,
        )
        assert should is True
        assert new_seq == 8

    def test_seq_jump_multiple_passes_still_single_push(self):
        # If the loop fell behind several passes, one push catches us up to
        # the newest seq (we always paint the freshest positions, not stale).
        should, new_seq = _repush_decision(
            fast_track=True, cur_seq=20, last_repush_seq=7,
            has_active_highlight=True, now=100.0, test_hold_until=0.0,
        )
        assert should is True
        assert new_seq == 20


# ---- layer 2: pipeline.recognize_seq trigger ----------------------------

def _fresh_disc(cx, cy, r=40, depth=1.5, label="bodhran", score=0.9):
    """A FreshDetection with a circular cam_mask at (cx, cy)."""
    from livetracking.perception.tracker import FreshDetection
    h, w = 480, 848
    y, x = np.ogrid[:h, :w]
    mask = (((y - cy) ** 2 + (x - cx) ** 2 <= r * r).astype(np.uint8)) * 255
    return FreshDetection(
        cam_mask=mask, label=label, label_score=score, median_depth_m=depth,
    )


def _make_pipeline_stubbed(monkeypatch, mask_state):
    """Build a sync-mode Pipeline whose heavy pass (_recognize_one) is stubbed
    to return one FreshDetection at the centroid in ``mask_state['cx']``. This
    bypasses DINO/SAM/depth-gates/warp — all env- and calibration-dependent —
    so we test only the real contract the daemon relies on: recognize_seq
    bumps once per pass, and tracker.update reflects the fresh position.
    """
    pytest.importorskip("cv2")
    from livetracking.perception.pipeline import Pipeline, PipelineConfig

    cfg = PipelineConfig(proj_w=1920, proj_h=1080)
    cfg.async_recognize = False
    H = np.eye(3, dtype=np.float64)
    pipe = Pipeline(H, 848, 480, cfg, recognizer=object())

    def _fake_recognize_one(color, depth_m):
        fresh = [_fresh_disc(mask_state["cx"], mask_state["cy"])]
        timings = {"total_ms": 1.0, "dino_ms": 0.0, "sam_ms": 0.0,
                   "stage1_ms": 0.0, "fast": False,
                   "n_dino_raw": 1, "n_dino_kept": 1, "n_objects": 1,
                   "dino_n": 1, "kept_n": 1, "sam_n": 1}
        return fresh, timings

    monkeypatch.setattr(pipe, "_recognize_one", _fake_recognize_one)
    return pipe


class TestRecognizeSeqTrigger:
    def test_seq_advances_once_per_sync_pass(self, monkeypatch):
        state = {"cx": 200, "cy": 200}
        try:
            pipe = _make_pipeline_stubbed(monkeypatch, state)
        except Exception as e:  # pragma: no cover - env-dependent
            pytest.skip(f"pipeline unavailable in this env: {e!r}")

        color = np.zeros((480, 848, 3), dtype=np.uint8)
        depth = np.full((480, 848), 1.5, dtype=np.float32)

        start = pipe.recognize_seq
        pipe.step_auto(color, depth)
        pipe.step_auto(color, depth)
        pipe.step_auto(color, depth)
        assert pipe.recognize_seq == start + 3

    def test_moving_object_changes_tracked_centroid(self, monkeypatch):
        """The seq bump is only useful if positions actually change between
        passes. Move the stubbed object and assert the tracker's centroid for
        the same id follows — this is the data the re-push carries to the
        projector.
        """
        state = {"cx": 200, "cy": 200}
        try:
            pipe = _make_pipeline_stubbed(monkeypatch, state)
        except Exception as e:  # pragma: no cover
            pytest.skip(f"pipeline unavailable in this env: {e!r}")

        color = np.zeros((480, 848, 3), dtype=np.uint8)
        depth = np.full((480, 848), 1.5, dtype=np.float32)

        # Promote to a stable track (promote_after_frames defaults to 3).
        for _ in range(5):
            pipe.step_auto(color, depth)
        with pipe.tracker_lock:
            vis = pipe.tracker.visible()
        assert vis, "object should have promoted to a visible track"
        obj_id = vis[0].object_id
        start_x = vis[0].centroid_cam[0]
        seq_before = pipe.recognize_seq

        # Slide the object right in small steps so the tracker keeps the same
        # id (a single 200 px jump would break the track — IoU 0, beyond the
        # centroid-match radius — which is correct behavior, not a follow).
        # Real objects move continuously, which is what we simulate here.
        for step in range(1, 9):
            state["cx"] = 200 + step * 30
            pipe.step_auto(color, depth)
        with pipe.tracker_lock:
            vis2 = pipe.tracker.visible()
        same = next((o for o in vis2 if o.object_id == obj_id), None)
        assert same is not None, "track id should survive a continuous move"
        assert same.centroid_cam[0] > start_x + 100, (
            "tracked centroid should follow the moved object"
        )
        # And the seq advanced so the daemon would have re-pushed.
        assert pipe.recognize_seq > seq_before

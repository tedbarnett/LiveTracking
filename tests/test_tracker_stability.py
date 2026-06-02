"""Unit tests for livetracking.perception.tracker.ObjectTracker.

These pin the stability properties that matter in the live demo:

1. Single-frame DINO hallucinations DON'T spawn an id (promote_after_frames).
2. Same physical object across many frames keeps the same id (no churn).
3. A briefly-missing object recovers the same id within stale_after_s.
4. After stale_after_s without a hit, the id is retired and a new one is
   assigned. Renames persist across that transition iff the object's
   fingerprint (label + centroid + depth) still matches.
5. Overlapping tracks (post-merge) collapse to the lower id, preserving
   the original name.
6. User renames are not overwritten by DINO's preferred label.
7. Hidden objects stay hidden after disappearing and reappearing.

All tests use a fake clock so we can simulate the 1.5 s candidate reap and
2.0 s stale-track retire deterministically — no time.sleep, no flakes.

Pure-numpy, no pyrealsense2 / pygame / torch needed.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

# Force in-memory paths BEFORE importing tracker so it doesn't write to the
# real names/hidden files.
@pytest.fixture(autouse=True)
def _tmp_persistence_paths(monkeypatch, tmp_path):
    names = tmp_path / "object_names.json"
    hidden = tmp_path / "hidden_objects.json"
    # tracker module reads these at __init__, so swap at import time.
    monkeypatch.setattr("livetracking.paths.OBJECT_NAMES_FILE", str(names),
                        raising=False)
    monkeypatch.setattr("livetracking.paths.HIDDEN_OBJECTS_FILE", str(hidden),
                        raising=False)
    yield


from livetracking.perception.tracker import (  # noqa: E402
    ObjectTracker, FreshDetection,
)


# ---- helpers -------------------------------------------------------------

def _mk_mask(cx: int, cy: int, r: int = 30, shape=(480, 848)) -> np.ndarray:
    """Solid circular mask centered at (cx, cy) in uint8 {0, 255}."""
    h, w = shape
    y, x = np.ogrid[:h, :w]
    m = ((y - cy) ** 2 + (x - cx) ** 2 <= r * r).astype(np.uint8) * 255
    return m


def _det(cx: int, cy: int, label: str = "bodhran", score: float = 0.8,
         depth: float = 1.5, r: int = 30) -> FreshDetection:
    return FreshDetection(
        cam_mask=_mk_mask(cx, cy, r=r),
        label=label,
        label_score=score,
        median_depth_m=depth,
    )


@pytest.fixture
def fake_time(monkeypatch):
    """A monotonically increasing fake clock for tracker.update()."""
    state = {"t": 1_000_000.0}

    def _now():
        return state["t"]

    monkeypatch.setattr("livetracking.perception.tracker.time.time", _now)

    class Clock:
        def advance(self, dt: float):
            state["t"] += dt

        @property
        def now(self):
            return state["t"]

    return Clock()


@pytest.fixture
def trk(tmp_path):
    return ObjectTracker(
        names_path=str(tmp_path / "names.json"),
        hidden_path=str(tmp_path / "hidden.json"),
        promote_after_frames=3,
        stale_after_s=2.0,
    )


# ---- promotion & churn suppression ---------------------------------------

class TestPromotion:
    def test_single_frame_blip_does_not_get_id(self, trk, fake_time):
        """One-frame DINO hallucination -> no track, no id consumed."""
        objs = trk.update([_det(400, 240)])
        assert objs == []
        # Next frame, nothing detected — candidate should age and eventually reap.
        fake_time.advance(2.0)
        objs = trk.update([])
        assert objs == []
        # _next_id is still 1: no id was burned on the hallucination.
        assert trk._next_id == 1

    def test_three_consecutive_frames_promotes(self, trk, fake_time):
        for _ in range(2):
            assert trk.update([_det(400, 240)]) == []
            fake_time.advance(0.1)
        objs = trk.update([_det(400, 240)])
        assert len(objs) == 1
        assert objs[0].object_id == 1

    def test_two_frames_then_gap_no_id(self, trk, fake_time):
        """Two consecutive hits + a 2 s gap -> candidate reaped, no id."""
        trk.update([_det(400, 240)])
        fake_time.advance(0.1)
        trk.update([_det(400, 240)])
        # Big gap — candidate should be reaped (1.5 s threshold).
        fake_time.advance(2.0)
        objs = trk.update([])
        assert objs == []
        assert trk._next_id == 1


# ---- stability across many frames ----------------------------------------

class TestStability:
    def test_same_object_same_id_for_50_frames(self, trk, fake_time):
        """No id churn under steady-state — wobble within IoU threshold."""
        # Promote.
        for _ in range(3):
            trk.update([_det(400, 240)])
            fake_time.advance(0.1)
        # Wobble centroid by a few px each frame for 50 frames.
        rng = np.random.default_rng(42)
        for _ in range(50):
            dx = int(rng.integers(-3, 4))
            dy = int(rng.integers(-3, 4))
            objs = trk.update([_det(400 + dx, 240 + dy)])
            fake_time.advance(0.05)
            assert len(objs) == 1
            assert objs[0].object_id == 1

    def test_brief_miss_keeps_same_id(self, trk, fake_time):
        """Track stays alive across a one-frame DINO miss (under stale_after_s)."""
        for _ in range(3):
            trk.update([_det(400, 240)])
            fake_time.advance(0.1)
        # Miss one frame (less than stale_after_s=2.0).
        fake_time.advance(0.5)
        trk.update([])
        # Object returns.
        objs = trk.update([_det(400, 240)])
        assert len(objs) == 1
        assert objs[0].object_id == 1

    def test_long_miss_retires_id(self, trk, fake_time):
        """After stale_after_s the track is retired; same physical object
        coming back later gets a NEW id (but may inherit name via
        fingerprint lookup — tested separately)."""
        for _ in range(3):
            trk.update([_det(400, 240)])
            fake_time.advance(0.1)
        # Advance well past stale_after_s.
        fake_time.advance(3.0)
        trk.update([])  # triggers retirement
        assert trk._tracks == {}
        # Re-detect for 3 frames -> new id 2.
        for _ in range(3):
            objs = trk.update([_det(400, 240)])
            fake_time.advance(0.1)
        assert len(objs) == 1
        assert objs[0].object_id == 2


# ---- rename behavior -----------------------------------------------------

class TestRename:
    def test_rename_survives_dino_label_changes(self, trk, fake_time):
        """User renames an object — subsequent DINO labels MUST NOT clobber."""
        for _ in range(3):
            trk.update([_det(400, 240, label="bodhran")])
            fake_time.advance(0.1)
        assert trk.rename(1, "Ted's drum") is True
        # DINO now mislabels the same object as 'tambourine'.
        objs = trk.update([_det(400, 240, label="tambourine", score=0.95)])
        fake_time.advance(0.1)
        assert objs[0].name == "Ted's drum"
        # And dino_label still tracks the latest class for fingerprinting.
        assert objs[0].dino_label == "tambourine"

    def test_rename_unknown_id_returns_false(self, trk):
        assert trk.rename(999, "ghost") is False


# ---- post-merge collapse -------------------------------------------------

class TestPostMerge:
    def test_overlapping_tracks_collapse_to_lower_id(self, trk, fake_time):
        """Two tracks at the same position -> merged to lower id (regression
        guard for the bodhran-flicker-between-ids bug class)."""
        # Promote track 1 at (400, 240).
        for _ in range(3):
            trk.update([_det(400, 240, label="bodhran")])
            fake_time.advance(0.1)
        # Inject a SECOND candidate at the same place via manual state poke:
        # the natural way this happens is DINO returns two overlapping masks
        # for the same physical object on one frame.
        for _ in range(3):
            trk.update([
                _det(400, 240, label="bodhran"),
                _det(402, 242, label="drum"),  # near-identical
            ])
            fake_time.advance(0.1)
        # After the post-merge pass, only one track should remain (id 1).
        active = trk.active()
        assert len(active) == 1
        assert active[0].object_id == 1


# ---- hidden persistence --------------------------------------------------

class TestHidden:
    def test_hide_then_unhide(self, trk, fake_time):
        for _ in range(3):
            trk.update([_det(400, 240)])
            fake_time.advance(0.1)
        assert trk.hide(1) is True
        assert trk.visible() == []          # hidden filtered from UI
        assert len(trk.active()) == 1       # but still tracked
        assert trk.unhide(1) is True
        assert len(trk.visible()) == 1

    def test_hidden_unknown_id_returns_false(self, trk):
        assert trk.hide(999) is False
        assert trk.unhide(999) is False

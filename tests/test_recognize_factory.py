"""Unit tests for detector-selection plumbing in `recognize`.

Only the lightweight, hardware-free bits are covered here:
 - `VALID_DETECTORS` vs `create_recognizer` branches (regression guard
   in case a new backend is added and someone forgets either side).
 - `read_active_detector` / `write_active_detector` round-trip.
 - Graceful fallback when the active_detector.json file is missing,
   malformed JSON, or names an unknown backend.

The actual recognizer classes pull in torch + transformers + ultralytics +
mediapipe + CUDA weights, so we patch `_active_detector_path` to a temp
file and never construct a real recognizer here. The full instantiation
path is exercised on-rig (marked `hardware` elsewhere).
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from livetracking.perception import recognize


# ---------------------------------------------------------------------------
# Factory <-> VALID_DETECTORS consistency
# ---------------------------------------------------------------------------

def _factory_branch_names() -> set[str]:
    """Pull the literal detector names out of create_recognizer's source.

    This is deliberately textual: it catches the case where someone adds
    a new branch but forgets to extend VALID_DETECTORS (or vice versa).
    """
    src = inspect.getsource(recognize.create_recognizer)
    names: set[str] = set()
    for token in ('"dino"', '"yolo"', '"yoloworld"', '"mediapipe"'):
        if token in src:
            names.add(token.strip('"'))
    return names


class TestFactoryDetectorParity:
    def test_default_detector_is_in_valid_set(self):
        assert recognize.DEFAULT_DETECTOR in recognize.VALID_DETECTORS

    def test_every_valid_detector_has_a_factory_branch(self):
        missing = set(recognize.VALID_DETECTORS) - _factory_branch_names()
        assert not missing, (
            f"VALID_DETECTORS names with no create_recognizer branch: {missing}"
        )

    def test_every_factory_branch_is_in_valid_detectors(self):
        extra = _factory_branch_names() - set(recognize.VALID_DETECTORS)
        assert not extra, (
            f"create_recognizer branches missing from VALID_DETECTORS: {extra}"
        )

    def test_unknown_name_raises_valueerror(self):
        # We can't call create_recognizer() with a valid name without
        # loading torch/SAM2, but the unknown-name path returns BEFORE
        # importing anything heavy.
        with pytest.raises(ValueError, match="unknown detector"):
            recognize.create_recognizer("not_a_real_detector")


# ---------------------------------------------------------------------------
# active_detector.json round-trip
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_active_detector_path(tmp_path, monkeypatch):
    """Redirect _active_detector_path() to a temp file for the test."""
    target = tmp_path / "active_detector.json"
    monkeypatch.setattr(
        recognize, "_active_detector_path", lambda: str(target)
    )
    return target


class TestActiveDetectorIO:
    def test_round_trip_each_valid_detector(self, patched_active_detector_path):
        for name in recognize.VALID_DETECTORS:
            recognize.write_active_detector(name)
            assert recognize.read_active_detector() == name

    def test_write_normalizes_case(self, patched_active_detector_path):
        recognize.write_active_detector("DINO")
        assert recognize.read_active_detector() == "dino"

    def test_write_rejects_unknown(self, patched_active_detector_path):
        with pytest.raises(ValueError, match="unknown detector"):
            recognize.write_active_detector("nonsense")

    def test_read_missing_file_returns_default(self, patched_active_detector_path):
        # No file written yet.
        assert not patched_active_detector_path.exists()
        assert recognize.read_active_detector() == recognize.DEFAULT_DETECTOR

    def test_read_malformed_json_returns_default(self, patched_active_detector_path):
        patched_active_detector_path.write_text("{not valid json")
        assert recognize.read_active_detector() == recognize.DEFAULT_DETECTOR

    def test_read_unknown_name_returns_default(self, patched_active_detector_path):
        patched_active_detector_path.write_text(
            json.dumps({"detector": "made_up_backend"})
        )
        assert recognize.read_active_detector() == recognize.DEFAULT_DETECTOR

    def test_read_missing_key_returns_default(self, patched_active_detector_path):
        patched_active_detector_path.write_text(json.dumps({"other": "x"}))
        assert recognize.read_active_detector() == recognize.DEFAULT_DETECTOR

    def test_write_creates_parent_dir(self, tmp_path, monkeypatch):
        # Point at a path whose parent does NOT exist yet.
        nested = tmp_path / "deep" / "nested" / "active_detector.json"
        monkeypatch.setattr(
            recognize, "_active_detector_path", lambda: str(nested)
        )
        recognize.write_active_detector("dino")
        assert nested.exists()
        assert json.loads(nested.read_text())["detector"] == "dino"


# ---------------------------------------------------------------------------
# DEFAULT_DINO_PROMPT -> class-list helper
# ---------------------------------------------------------------------------

class TestPromptToClasses:
    def test_splits_and_lowercases(self):
        out = recognize._prompt_to_classes("Cat. Dog. Bird.")
        assert out == ["cat", "dog", "bird"]

    def test_dedupes_preserving_order(self):
        out = recognize._prompt_to_classes("cat. dog. cat. bird. dog.")
        assert out == ["cat", "dog", "bird"]

    def test_strips_whitespace(self):
        out = recognize._prompt_to_classes("  cat .  dog  ")
        assert out == ["cat", "dog"]

    def test_handles_commas_as_separators(self):
        out = recognize._prompt_to_classes("cat, dog, bird")
        assert out == ["cat", "dog", "bird"]

    def test_default_prompt_includes_expected_classes(self):
        # Anchors: the demo specifically wants these classes.
        out = recognize._prompt_to_classes(recognize.DEFAULT_DINO_PROMPT)
        for expected in ("couch", "guitar", "bodhran", "picture frame"):
            assert expected in out, f"missing {expected!r} in DEFAULT_DINO_PROMPT"

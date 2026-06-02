"""Unit tests for livetracking.config.env.

These pin the warn-and-default policy that keeps the perception daemon
booting through garbage env-var values. Critical because the daemon runs
unattended via Task Scheduler — a ValueError at startup means the service
silently dies and the demo goes dark.

Scenarios pinned:
    - unset / empty -> default
    - garbage numeric -> warn + default (NO raise)
    - out-of-range -> warn + clamp
    - unknown enum choice -> warn + default
    - bool truthiness (yes/no/on/off/0/1/empty)
    - integer accepts float-literal (1e3, 1200.5)
    - logger is callable, captures messages
"""
from __future__ import annotations

import pytest

from livetracking.config.env import (
    parse_bool, parse_float, parse_int, parse_str,
)


@pytest.fixture
def caplog_messages():
    """Drop-in logger that captures warnings as a list[str]."""
    msgs: list[str] = []
    return msgs, msgs.append


# ---- parse_bool ----------------------------------------------------------

class TestParseBool:
    def test_unset_returns_default_false(self, monkeypatch):
        monkeypatch.delenv("FOO", raising=False)
        assert parse_bool("FOO", default=False) is False

    def test_unset_returns_default_true(self, monkeypatch):
        monkeypatch.delenv("FOO", raising=False)
        assert parse_bool("FOO", default=True) is True

    def test_empty_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("FOO", "")
        assert parse_bool("FOO", default=True) is True

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "Y", "TRUE", "anything-else"])
    def test_truthy(self, monkeypatch, val):
        monkeypatch.setenv("FOO", val)
        assert parse_bool("FOO", default=False) is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "n", "F", "FALSE"])
    def test_falsy(self, monkeypatch, val):
        monkeypatch.setenv("FOO", val)
        assert parse_bool("FOO", default=True) is False


# ---- parse_float ---------------------------------------------------------

class TestParseFloat:
    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("FOO", raising=False)
        assert parse_float("FOO", default=1.5) == 1.5

    def test_valid_number(self, monkeypatch):
        monkeypatch.setenv("FOO", "3.14")
        assert parse_float("FOO", default=0.0) == pytest.approx(3.14)

    def test_garbage_warns_and_defaults(self, monkeypatch, caplog_messages):
        msgs, log = caplog_messages
        monkeypatch.setenv("FOO", "1200x")
        result = parse_float("FOO", default=1200.0, logger=log)
        assert result == 1200.0
        assert len(msgs) == 1
        assert "FOO" in msgs[0] and "1200x" in msgs[0]

    def test_below_min_clamps(self, monkeypatch, caplog_messages):
        msgs, log = caplog_messages
        monkeypatch.setenv("FOO", "-5")
        result = parse_float("FOO", default=0.0, min_value=0.0, logger=log)
        assert result == 0.0
        assert "below min" in msgs[0]

    def test_above_max_clamps(self, monkeypatch, caplog_messages):
        msgs, log = caplog_messages
        monkeypatch.setenv("FOO", "999999")
        result = parse_float("FOO", default=0.0, max_value=10.0, logger=log)
        assert result == 10.0
        assert "above max" in msgs[0]

    def test_in_range_passes(self, monkeypatch):
        monkeypatch.setenv("FOO", "5")
        assert parse_float("FOO", default=0.0, min_value=0.0,
                           max_value=10.0) == 5.0

    def test_scientific_notation(self, monkeypatch):
        monkeypatch.setenv("FOO", "1.2e3")
        assert parse_float("FOO", default=0.0) == 1200.0

    def test_realistic_parallax_k_typo(self, monkeypatch, caplog_messages):
        """REGRESSION GUARD: LIVETRACKING_PARALLAX_K=1200x must not crash
        the perception daemon — this is the exact scenario this module
        exists to handle."""
        msgs, log = caplog_messages
        monkeypatch.setenv("LIVETRACKING_PARALLAX_K", "1200x")
        result = parse_float("LIVETRACKING_PARALLAX_K", default=1200.0,
                              min_value=0.0, max_value=10000.0, logger=log)
        assert result == 1200.0
        assert len(msgs) == 1


# ---- parse_int -----------------------------------------------------------

class TestParseInt:
    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("FOO", raising=False)
        assert parse_int("FOO", default=42) == 42

    def test_valid_int(self, monkeypatch):
        monkeypatch.setenv("FOO", "17")
        assert parse_int("FOO", default=0) == 17

    def test_float_literal_truncated(self, monkeypatch):
        monkeypatch.setenv("FOO", "1200.7")
        assert parse_int("FOO", default=0) == 1200

    def test_scientific_notation_truncated(self, monkeypatch):
        monkeypatch.setenv("FOO", "1e3")
        assert parse_int("FOO", default=0) == 1000

    def test_garbage_warns_and_defaults(self, monkeypatch, caplog_messages):
        msgs, log = caplog_messages
        monkeypatch.setenv("FOO", "abc")
        result = parse_int("FOO", default=42, logger=log)
        assert result == 42
        assert "FOO" in msgs[0]

    def test_clamping(self, monkeypatch, caplog_messages):
        msgs, log = caplog_messages
        monkeypatch.setenv("FOO", "5000")
        result = parse_int("FOO", default=0, min_value=0, max_value=100,
                            logger=log)
        assert result == 100
        assert "above max" in msgs[0]


# ---- parse_str -----------------------------------------------------------

class TestParseStr:
    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("FOO", raising=False)
        assert parse_str("FOO", default="bar") == "bar"

    def test_passthrough(self, monkeypatch):
        monkeypatch.setenv("FOO", "hello")
        assert parse_str("FOO", default="bar") == "hello"

    def test_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("FOO", "  hello  ")
        assert parse_str("FOO", default="bar") == "hello"

    def test_choices_valid(self, monkeypatch):
        monkeypatch.setenv("FOO", "yolo")
        assert parse_str("FOO", default="dino",
                         choices=["dino", "yolo", "yoloworld"]) == "yolo"

    def test_choices_invalid_warns_and_defaults(self, monkeypatch,
                                                caplog_messages):
        msgs, log = caplog_messages
        monkeypatch.setenv("FOO", "tensorflow")
        result = parse_str("FOO", default="dino",
                            choices=["dino", "yolo"], logger=log)
        assert result == "dino"
        assert "tensorflow" in msgs[0]
        assert "not in" in msgs[0]


# ---- integration: realistic perception-daemon boot scenario --------------

class TestDaemonBootScenarios:
    """The exact sequences perception.py runs at startup. If any of these
    raise, the daemon crashes silently in Task Scheduler."""

    def test_all_unset_uses_defaults(self, monkeypatch):
        for k in ["LIVETRACKING_PARALLAX_COMPENSATE",
                  "LIVETRACKING_PARALLAX_SIGN",
                  "LIVETRACKING_PARALLAX_SCALE",
                  "LIVETRACKING_PARALLAX_K",
                  "LIVETRACKING_DETECTOR"]:
            monkeypatch.delenv(k, raising=False)
        compensate = parse_bool("LIVETRACKING_PARALLAX_COMPENSATE", True)
        sign = parse_float("LIVETRACKING_PARALLAX_SIGN", 1.0,
                            min_value=-1.0, max_value=1.0)
        scale = parse_float("LIVETRACKING_PARALLAX_SCALE", 1.0,
                             min_value=0.0, max_value=10.0)
        k = parse_float("LIVETRACKING_PARALLAX_K", 1200.0,
                         min_value=0.0, max_value=10000.0)
        det = parse_str("LIVETRACKING_DETECTOR", "",
                        choices=["dino", "yolo", "yoloworld", "mediapipe"])
        assert (compensate, sign, scale, k, det) == (True, 1.0, 1.0, 1200.0, "")

    def test_garbage_env_does_not_crash(self, monkeypatch, caplog_messages):
        msgs, log = caplog_messages
        monkeypatch.setenv("LIVETRACKING_PARALLAX_COMPENSATE", "")
        monkeypatch.setenv("LIVETRACKING_PARALLAX_SIGN", "uppercase")
        monkeypatch.setenv("LIVETRACKING_PARALLAX_SCALE", "1.0kg")
        monkeypatch.setenv("LIVETRACKING_PARALLAX_K", "1200x")
        monkeypatch.setenv("LIVETRACKING_DETECTOR", "tensorflow")
        # None of these should raise.
        compensate = parse_bool("LIVETRACKING_PARALLAX_COMPENSATE", True)
        sign = parse_float("LIVETRACKING_PARALLAX_SIGN", 1.0,
                            min_value=-1.0, max_value=1.0, logger=log)
        scale = parse_float("LIVETRACKING_PARALLAX_SCALE", 1.0,
                             min_value=0.0, max_value=10.0, logger=log)
        k = parse_float("LIVETRACKING_PARALLAX_K", 1200.0,
                         min_value=0.0, max_value=10000.0, logger=log)
        det = parse_str("LIVETRACKING_DETECTOR", "",
                        choices=["dino", "yolo", "yoloworld", "mediapipe"],
                        logger=log)
        assert (compensate, sign, scale, k, det) == (True, 1.0, 1.0, 1200.0, "")
        # 4 warnings (compensate=empty is silent, the other 4 garbage).
        assert len(msgs) == 4

    def test_out_of_range_clamps(self, monkeypatch, caplog_messages):
        msgs, log = caplog_messages
        monkeypatch.setenv("LIVETRACKING_PARALLAX_K", "99999")
        k = parse_float("LIVETRACKING_PARALLAX_K", 1200.0,
                         min_value=0.0, max_value=10000.0, logger=log)
        assert k == 10000.0

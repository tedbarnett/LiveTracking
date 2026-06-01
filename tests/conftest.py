"""Shared pytest config for the LiveTracking suite.

Defines a `hardware` marker so tests that need a RealSense camera, a
projector, CUDA, or heavy ML weights can be skipped on plain dev boxes:

    pytest -m "not hardware"   # default for CI / desktop
    pytest -m hardware         # only the on-rig integration tests
"""
from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "hardware: requires RealSense D455, projector, CUDA, or ML weights.",
    )

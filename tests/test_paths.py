"""Smoke tests for the central path/config module.

These verify that the env-var overrides actually take effect when the module
is reimported in a clean state, which is the contract the daemon and web UI
rely on for portability across hosts.
"""
from __future__ import annotations

import importlib
import os

import pytest


def _reimport_paths():
    import livetracking.paths as P
    return importlib.reload(P)


def test_repo_root_resolves_to_pyproject_dir():
    P = _reimport_paths()
    assert os.path.isfile(os.path.join(P.REPO_ROOT, "pyproject.toml"))


def test_runtime_and_tmp_dirs_exist_after_import():
    P = _reimport_paths()
    assert os.path.isdir(P.RUNTIME_DIR)
    assert os.path.isdir(P.TMP_DIR)


def test_env_override_runtime_dir(tmp_path, monkeypatch):
    target = tmp_path / "alt-runtime"
    monkeypatch.setenv("LIVETRACKING_RUNTIME_DIR", str(target))
    P = _reimport_paths()
    try:
        assert P.RUNTIME_DIR == str(target)
        assert os.path.isdir(P.RUNTIME_DIR)
        assert P.STATE_FILE == os.path.join(str(target), "state.json")
    finally:
        # Restore module state so other tests don't see the override.
        monkeypatch.delenv("LIVETRACKING_RUNTIME_DIR", raising=False)
        _reimport_paths()


def test_env_override_display(monkeypatch):
    monkeypatch.setenv("LIVETRACKING_DISPLAY_INDEX", "2")
    monkeypatch.setenv("LIVETRACKING_DISPLAY_W", "3840")
    monkeypatch.setenv("LIVETRACKING_DISPLAY_H", "2160")
    P = _reimport_paths()
    try:
        assert P.DISPLAY_INDEX == 2
        assert P.DISPLAY_W == 3840
        assert P.DISPLAY_H == 2160
    finally:
        for k in (
            "LIVETRACKING_DISPLAY_INDEX",
            "LIVETRACKING_DISPLAY_W",
            "LIVETRACKING_DISPLAY_H",
        ):
            monkeypatch.delenv(k, raising=False)
        _reimport_paths()


def test_display_index_none_by_default(monkeypatch):
    monkeypatch.delenv("LIVETRACKING_DISPLAY_INDEX", raising=False)
    P = _reimport_paths()
    assert P.DISPLAY_INDEX is None

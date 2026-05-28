"""Central path + display config for LiveTracking.

Resolves the repo root from $LIVETRACKING_ROOT or by walking up to find
``pyproject.toml``. All runtime/scratch directories are env-overridable so
the same code runs on the laptop, the PC-5090 desktop, and CI without
host-specific edits.
"""
from __future__ import annotations

import os


def _find_repo_root() -> str:
    env = os.environ.get("LIVETRACKING_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    for _ in range(6):
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # Fallback: <root>/src/livetracking/paths.py -> two up.
    return os.path.normpath(os.path.join(here, "..", ".."))


REPO_ROOT: str = _find_repo_root()
RUNTIME_DIR: str = os.environ.get(
    "LIVETRACKING_RUNTIME_DIR", os.path.join(REPO_ROOT, "runtime")
)
TMP_DIR: str = os.environ.get(
    "LIVETRACKING_TMP_DIR", os.path.join(REPO_ROOT, "tmp")
)
CALIB_DIR: str = os.path.join(RUNTIME_DIR, "calibration")
MASK_DIR: str = os.path.join(RUNTIME_DIR, "masks")
SCRIPT_OUT_DIR: str = os.path.join(REPO_ROOT, "scripts", "out")

for _d in (RUNTIME_DIR, TMP_DIR, CALIB_DIR, MASK_DIR, SCRIPT_OUT_DIR):
    os.makedirs(_d, exist_ok=True)

# Calibration artifacts
HOMOGRAPHY_FILE: str = os.path.join(CALIB_DIR, "H.npy")
CALIB_META_FILE: str = os.path.join(CALIB_DIR, "calib.json")
MEASURED_FOOTPRINT_FILE: str = os.path.join(CALIB_DIR, "footprint_measured.png")

# Object naming persistence
OBJECT_NAMES_FILE: str = os.path.join(RUNTIME_DIR, "object_names.json")
HIDDEN_OBJECTS_FILE: str = os.path.join(RUNTIME_DIR, "object_hidden.json")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_int_or_none(name: str) -> int | None:
    v = os.environ.get(name)
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


# Projector / display
DISPLAY_INDEX: int | None = _env_int_or_none("LIVETRACKING_DISPLAY_INDEX")

# RealSense exposure (manual lock; see projection-mapping skill)
RS_EXPOSURE: int = _env_int("LIVETRACKING_RS_EXPOSURE", 700)
RS_WHITE_BALANCE: int = _env_int("LIVETRACKING_RS_WB", 4600)
RS_WIDTH: int = _env_int("LIVETRACKING_RS_WIDTH", 848)
RS_HEIGHT: int = _env_int("LIVETRACKING_RS_HEIGHT", 480)
RS_FPS: int = _env_int("LIVETRACKING_RS_FPS", 30)

# Web UI port (Cloudflare tunnel `livetracking-laptop` targets this)
WEB_UI_PORT: int = _env_int("LIVETRACKING_WEB_PORT", 5070)

# ZMQ endpoints (localhost IPC between the three daemons)
ZMQ_OBJECTS_PUB: str = os.environ.get(
    "LIVETRACKING_ZMQ_OBJECTS", "tcp://127.0.0.1:5571"
)
ZMQ_PROJECTOR_PULL: str = os.environ.get(
    "LIVETRACKING_ZMQ_PROJECTOR", "tcp://127.0.0.1:5572"
)


def describe() -> str:
    return (
        f"REPO_ROOT={REPO_ROOT} RUNTIME_DIR={RUNTIME_DIR} "
        f"DISPLAY_INDEX={DISPLAY_INDEX} "
        f"RS={RS_WIDTH}x{RS_HEIGHT}@{RS_FPS} exp={RS_EXPOSURE} "
        f"WEB={WEB_UI_PORT}"
    )

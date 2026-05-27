"""Central path + display config for LiveTracking.

Everything that used to be hardcoded as ``D:\\Github-D\\LiveTracking\\...`` or
``DISPLAY_X = 5120`` lives here, so the same code runs on the PC-5090 desktop
and the laptop (and any other machine) without per-host edits.

Repo root resolution
--------------------
1. ``$LIVETRACKING_ROOT`` env var if set (most explicit override).
2. Otherwise: walk up from this file to find a directory containing
   ``pyproject.toml``. Falls back to three-levels-up if that fails (matches
   the historical layout: ``<root>/src/livetracking/paths.py``).

Directories
-----------
- ``RUNTIME_DIR``  - daemon <-> web UI state IPC. Auto-created on import.
- ``TMP_DIR``      - debug/scratch images (``diag_*.png``, ``iv*_final.png``).
                     Auto-created on import.

Files derived from those dirs are exposed as module-level constants so callers
don't reassemble paths themselves.

Display / projector window
--------------------------
JMGO / Dangbei / Kodak / etc. all enumerate differently depending on Windows
display arrangement and DPI scaling. Pick whichever knob is convenient:

- ``$LIVETRACKING_DISPLAY_INDEX`` - pygame display index (preferred; see
  README laptop notes). When set, ``projection_daemon`` uses
  ``pygame.display.set_mode(..., display=int(idx))`` and ignores SDL window
  position.
- ``$LIVETRACKING_DISPLAY_X`` / ``_Y`` - SDL window top-left in the virtual
  desktop. Legacy approach; kept because it still works on the PC-5090
  desktop where ``DISPLAY_X=5120`` was the historical value.
- ``$LIVETRACKING_DISPLAY_W`` / ``_H`` - projector resolution (defaults
  1280x720).
"""
from __future__ import annotations

import os


def _find_repo_root() -> str:
    env = os.environ.get("LIVETRACKING_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    # Walk up looking for pyproject.toml.
    cur = here
    for _ in range(6):
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # Fallback: <root>/src/livetracking/paths.py -> three up.
    return os.path.normpath(os.path.join(here, "..", ".."))


REPO_ROOT: str = _find_repo_root()
RUNTIME_DIR: str = os.environ.get(
    "LIVETRACKING_RUNTIME_DIR",
    os.path.join(REPO_ROOT, "runtime"),
)
TMP_DIR: str = os.environ.get(
    "LIVETRACKING_TMP_DIR",
    os.path.join(REPO_ROOT, "tmp"),
)

os.makedirs(RUNTIME_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# Runtime IPC files (filesystem-based so daemon + web UI restart independently)
STATE_FILE: str = os.path.join(RUNTIME_DIR, "state.json")
FRAME_FILE: str = os.path.join(RUNTIME_DIR, "latest_frame.jpg")
COMMAND_FILE: str = os.path.join(RUNTIME_DIR, "command.txt")
NUDGES_FILE: str = os.path.join(RUNTIME_DIR, "nudges.json")
LEARNED_OFFSETS_FILE: str = os.path.join(RUNTIME_DIR, "learned_offsets.jsonl")

# Debug screenshot paths (replaces the old D:\Github-D\LiveTracking\tmp\... refs)
DIAG_BLACK_FILE: str = os.path.join(TMP_DIR, "diag_v10_black.png")
DIAG_WHITE_FILE: str = os.path.join(TMP_DIR, "diag_v10_white.png")
FINAL_SCREENSHOT_FILE: str = os.path.join(TMP_DIR, "iv10_final.png")


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_int_or_none(name: str) -> int | None:
    val = os.environ.get(name)
    if val is None or val == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


# Projector display config. DISPLAY_INDEX None => fall back to SDL window pos.
DISPLAY_INDEX: int | None = _env_int_or_none("LIVETRACKING_DISPLAY_INDEX")
DISPLAY_X: int = _env_int("LIVETRACKING_DISPLAY_X", 0)
DISPLAY_Y: int = _env_int("LIVETRACKING_DISPLAY_Y", 0)
DISPLAY_W: int = _env_int("LIVETRACKING_DISPLAY_W", 1280)
DISPLAY_H: int = _env_int("LIVETRACKING_DISPLAY_H", 720)

# Web UI port (Cloudflare tunnel still targets 5070; keep that as default).
WEB_UI_PORT: int = _env_int("LIVETRACKING_WEB_PORT", 5070)


def describe() -> str:
    """One-line summary of resolved paths/config (for daemon startup log)."""
    return (
        f"REPO_ROOT={REPO_ROOT} RUNTIME_DIR={RUNTIME_DIR} "
        f"DISPLAY={DISPLAY_W}x{DISPLAY_H} "
        f"@ {'index='+str(DISPLAY_INDEX) if DISPLAY_INDEX is not None else f'({DISPLAY_X},{DISPLAY_Y})'}"
    )

# `src/livetracking` — Phase 0 prototype

Python package that runs the LiveTracking pipeline against the RealSense
D455 (or webcam fallback) and a normal window as the "virtual projector"
until the physical projector arrives.

## Install

From the repo root (PowerShell):

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

If the `pyrealsense2` wheel fails to install (it only ships for select
Python/Windows combos), comment it out in `requirements.txt`; the app
detects the missing import and falls back to a webcam + synthetic depth.

## Run

```pwsh
python -m livetracking              # full interactive app
python -m livetracking --help
python -m livetracking --test       # 30-frame boot test, prints FPS + errors
python -m livetracking --no-camera  # skip RealSense/webcam, synthetic only
python -m livetracking --fullscreen # send to the projector once it arrives
```

## Hotkeys

| Key   | Action                                           |
|-------|--------------------------------------------------|
| `C`   | Run 3-second calibration cycle                   |
| `S`   | Enter object-selection mode (segment + halos)    |
| `1-9` | Select object N → switch to effect mode          |
| `E`   | Cycle effect: fire → glow → colorshift           |
| `R`   | Re-segment current frame                         |
| `Esc` | Back to idle                                     |
| `Q`   | Quit                                             |

## Modules

| File         | Role                                                     |
|--------------|----------------------------------------------------------|
| `capture.py` | RealSense / webcam / synthetic frame source              |
| `pattern.py` | Cobblestone generator (8×6 grid, idle + calibration)     |
| `calibrate.py` | Structured-light driver (Phase 0 stub homography)      |
| `segment.py` | Depth-cluster foreground segmentation                    |
| `effects.py` | ModernGL fragment-shader compositor (fire/glow/shift)    |
| `ui.py`      | Pygame window, mode state machine, keyboard handler      |
| `__main__.py`| CLI entry: arg parsing, lifecycle                        |

## Capture fallback chain

```
pyrealsense2 → cv2.VideoCapture(0) → fully synthetic moving blobs
```

Synthetic mode is deterministic and always available, so the rest of
the pipeline stays debuggable when no hardware is connected.

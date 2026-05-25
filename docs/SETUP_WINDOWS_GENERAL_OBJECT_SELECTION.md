# Windows Setup — General Object Selection Branch

This branch adds a fast depth-first object-selection path. It is separate from the slower post-it closed-loop calibration experiment.

## Pull the branch

```powershell
cd C:\Users\Ted\Documents\GitHub\LiveTracking
git fetch origin
git switch feature/general-object-selection
```

If the repo is not cloned yet:

```powershell
cd C:\Users\Ted\Documents\GitHub
git clone https://github.com/tedbarnett/LiveTracking.git
cd LiveTracking
git switch feature/general-object-selection
```

## Create Python environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run hardware diagnostics

Plug in the Intel RealSense D455, then run:

```powershell
python scripts/diagnose_hardware.py
```

Expected output:

- OpenCV version prints
- RealSense device count is at least 1
- D455 color/depth frames capture at roughly 60 FPS
- diagnostic files are written to `diagnostics/latest/`

## Run general object-selection demo

```powershell
python -m livetracking.app.live_select_demo
```

Controls:

| Key | Action |
|---|---|
| `B` | Capture empty background depth model |
| `Tab` | Cycle selected object |
| `Esc` | Quit |

## Test procedure

1. Aim the RealSense at the projection wall or test area.
2. Clear the projector rectangle / camera view.
3. Launch the demo.
4. Press `B` to capture the empty background.
5. Put objects into the view: guitar, hand, box, bottle, book, prop.
6. The app should draw boxes and IDs around foreground objects.
7. Press `Tab` to cycle selected object.

## Important limitation

This first pass is optimized for objects with depth separation from the wall/background. It will not reliably detect flat post-it notes stuck directly on the wall. Flat wall objects need the slower RGB/edge fallback path, which should remain a separate mode.

## Why this should be faster

The live loop detects all depth-separated objects in one pass using connected components. It does not project a marker, wait for a camera diff, correct, and repeat per target.

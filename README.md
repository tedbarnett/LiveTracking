# LiveTracking

Real-time projection-mapping prototype. A RealSense D455 watches a scene
(currently a couch / living-room corner); a JMGO N3 Ultimate projector lights
objects back. Hover an object name in the web UI and the projector
illuminates that physical object. Public UI at
[livetracking.barnettlabs.tech](https://livetracking.barnettlabs.tech) when
the laptop is running.

> Fresh start — 2026-05-27. Prior work preserved on branch
> [`backup-05-27-26`](../../tree/backup-05-27-26).

---

## Architecture

Three long-running processes plus an on-demand calibration job. They talk
over local ZMQ:

| Process | Purpose | Lifecycle on Windows |
| --- | --- | --- |
| `livetracking.daemon.perception` | Holds the D455. Stage 1 depth-plane foreground → Stage 2 async DINO + SAM 2 (RTX 5090). Publishes detected objects + MJPEG over ZMQ. | Scheduled task `LiveTrackingPerception` (at-logon, Interactive — needs user desktop session for USB + GPU). |
| `livetracking.daemon.projector` | Holds pygame fullscreen on the JMGO (display 1). Subscribes (PULL) for `highlight` / `clear` / `set_intensity` / `set_white_light` commands. | Scheduled task `LiveTrackingProjector` (at-logon, Interactive — needs the user desktop session for display 1). |
| `livetracking.daemon.flame_web` | Flask UI on `:5070`. Serves the hover-to-illuminate page, MJPEG stream, control endpoints. Trampolines control messages into perception over ZMQ. | NSSM service `LiveTrackingFlameWeb` (LocalSystem, auto-start). Cloudflare tunnel `livetracking-laptop` exposes it as `livetracking.barnettlabs.tech`. |
| `scripts/run_calibration.py` | Stops perception+projector, runs camera→projector homography calibration, restarts them. | Scheduled task `LiveTrackingCalibrate` (on-demand). Triggered by the **Re-calibrate** button in the web UI. |

Perception + projector run in the user session because they need the
desktop (pygame display 1, USB camera passthrough). The Flask UI runs as
a Session-0 service so it survives logoff. The Re-calibrate orchestrator
also lives in the user session so it can hold display 1 + the camera.

## Web UI

`https://livetracking.barnettlabs.tech` (or `http://localhost:5070`):

- **Live MJPEG** of the perception camera.
- **Object list** (numbered, color-swatched). Hover a row → projector
  illuminates that physical object. Click a row → pin. Click the swatch →
  cycle color. ✕ → hide (won't reappear until "Unhide all").
- **⏸ Pause / ▶ Run** — freeze the pipeline.
- **Highlight all** — wash all detected objects at once.
- **Clear** — projector to black.
- **☀ White Light** — projector throws full white at the scene (toggle).
  Useful for verifying cone position, lighting the room, or smoke-testing
  the projector pipeline.
- **⟳ Re-calibrate** — triggers `LiveTrackingCalibrate` and live-updates
  through `stopping_perception → stopping_projector → calibrating →
  restarting_projector → restarting_perception → done`. The whole cycle is
  ~30 s.
- **Projector intensity** slider — alpha on the highlight wash (0–100 %).

## Calibration

Camera↔projector homography is solved via **time-multiplexed ArUco
markers**:

1. Stop perception + projector (release D455 + display 1).
2. Project a 4×4 grid of large ArUco markers (DICT_4X4_50), **one marker at
   a time**, each ~22 % of the screen, surrounded by a white quiet-zone tile
   on an otherwise-black projector frame.
3. Capture each frame, detect with `cv2.aruco.ArucoDetector` (sub-pixel
   corner refinement). Each detected marker contributes **4 corner
   correspondences**.
4. `cv2.findHomography(..., cv2.RANSAC, 15.0)` over all detected corners.
5. Persist `runtime/calibration/H.npy`, `calib.json`, `dot_cam_pts.npy`,
   `dot_proj_pts.npy`, plus a synthesized `footprint_measured.png` and
   diagnostic PNGs to `scripts/out/`.

Why time-multiplexed ArUco (not white dots or a single grid frame):

- ArUco detection is shape+code based — robust against ambient daylight
  that drowned the old white-dot diff thresholds in normal-lit rooms.
- Per marker: large (~22 % of screen) so each ArUco square survives the
  848×480 camera's spatial averaging at 4 K projector resolution.
- Per frame: only one marker on screen, full-frame DLP contrast around it.
  No "competing white background" washing out wall paint.
- Each marker contributes 4 corner points instead of 1 dot center → ~30+
  correspondences typical → comfortable RANSAC inlier counts even when
  half the markers are blocked by objects.
- Filenames preserved from the old dot calibrator so
  `livetracking.perception.footprint` keeps working unchanged.

Diagnostic outputs after every run (look in `scripts/out/`):

- `calib_aruco_overview.png` — baseline camera frame with all detected
  markers outlined and labeled.
- `calib_aruco_miss_NN.png` — full camera frame for any marker that
  failed to detect (useful for tuning).

## Setting up a fresh laptop

### Prereqs

- Windows 10/11. NSSM (`choco install nssm`).
- Python 3.11 (the repo's `.venv` is built against 3.11.9).
- CUDA-capable GPU for the async DINO+SAM pass (current target: RTX 5090
  Laptop GPU, 24 GB).
- Intel RealSense D455 on USB-C.
- 4 K projector reachable as display 1 (auto-detected by biggest-area
  heuristic, or set `LIVETRACKING_DISPLAY_INDEX`).

### Install

```powershell
git clone https://github.com/tedbarnett/LiveTracking.git
cd LiveTracking
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
# Then, from an elevated PowerShell:
scripts\install_services.ps1
```

The installer registers `LiveTrackingFlameWeb` (NSSM service),
`LiveTrackingPerception`, `LiveTrackingProjector`, and
`LiveTrackingCalibrate` (Task Scheduler tasks). It's idempotent.

To stop everything: `scripts\install_services.ps1 -Uninstall`.

### First-time calibration

1. Make sure the JMGO is showing the HDMI feed (dismiss its setup menu).
2. Open `livetracking.barnettlabs.tech` (or `http://localhost:5070`).
3. Click **⟳ Re-calibrate**. Watch the projector cycle through 16 ArUco
   frames over ~10 s; the button reports progress and final status.

If detection rate is low, dim the room. The bigger the ambient/projector
contrast ratio, the more markers ArUco recovers.

## Layout

```
src/livetracking/
  daemon/
    perception.py       # D455 owner; Stage 1 + async Stage 2; ZMQ PUB/REP
    projector.py        # pygame on JMGO; ZMQ PULL; highlight/clear/white-light
    flame_web.py        # Flask UI + control trampoline
    templates/index.html
  perception/
    capture.py          # RealSense aligned color+depth, exposure/WB lock
    footprint.py        # H persistence + dot-quad footprint reconstruction
    pipeline.py         # fast_step (geometry) + async recognize
    tracker.py          # IoU/bbox-contains matcher, provisional promotion, hide
    geometry.py         # depth-plane RANSAC, per-object parallax warp
  paths.py              # env-driven paths, display index, ZMQ endpoints
scripts/
  calibrate_homography.py   # time-multiplexed ArUco calibration
  run_calibration.py        # orchestrator: stop/calibrate/restart + status JSON
  install_services.ps1      # idempotent service/task installer
  out/                      # diagnostic dumps from calibration + tests
runtime/                    # H.npy, calib.json, hidden_objects.json, logs (gitignored)
```

## Logs

- `runtime/service-logs/flame_web.{stdout,stderr}.log` — Flask UI.
- `runtime/service-logs/calibrate.log` — calibration orchestrator.
- Perception / Projector — Task Scheduler → History tab.
- `runtime/calibration/calib.json` — last calibration's metadata
  (markers detected, RANSAC inliers, det, marker size).

## Environment overrides

| Variable | Effect | Default |
| --- | --- | --- |
| `LIVETRACKING_DISPLAY_INDEX` | Pin the projector to a specific pygame display index | auto (biggest desktop) |
| `LIVETRACKING_RUNTIME_DIR` | Where to write H, masks, logs | `<repo>/runtime` |
| `LIVETRACKING_RS_EXPOSURE` | RealSense color exposure (locked) | `700` |
| `LIVETRACKING_WEB_UI_PORT` | Flask port | `5070` |

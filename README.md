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
| `livetracking.daemon.perception` | Holds the D455. Image-first pipeline: Grounding DINO (open-vocab detection) → SAM 2 (per-bbox mask) → wall-plane depth gate → parallax-compensated warp through H. Publishes detected objects + MJPEG over ZMQ. | Scheduled task `LiveTrackingPerception` (at-logon, Interactive — needs user desktop session for USB + GPU). |
| `livetracking.daemon.projector` | Holds pygame fullscreen on the JMGO (display 1). Subscribes (PULL) for `highlight` / `clear` / `set_intensity` / `set_white_light` commands. | Scheduled task `LiveTrackingProjector` (at-logon, Interactive — needs the user desktop session for display 1). |
| `livetracking.daemon.flame_web` | Flask UI on `:5070`. Serves the hover-to-illuminate page, MJPEG stream, control endpoints. Trampolines control messages into perception over ZMQ. | NSSM service `LiveTrackingFlameWeb` (LocalSystem, auto-start). Cloudflare tunnel `livetracking-laptop` exposes it as `livetracking.barnettlabs.tech`. |
| `scripts/run_calibration.py` | Stops perception+projector, runs camera→projector homography calibration, restarts them. | Scheduled task `LiveTrackingCalibrate` (on-demand). Triggered by the **Re-calibrate** button in the web UI. |

Perception + projector run in the user session because they need the
desktop (pygame display 1, USB camera passthrough). The Flask UI runs as
a Session-0 service so it survives logoff. The Re-calibrate orchestrator
also lives in the user session so it can hold display 1 + the camera.

## Perception pipeline

Image-first. For each (color, depth) frame:

1. **Grounding DINO** (`IDEA-Research/grounding-dino-tiny`) runs on the color
   frame with an open-vocab prompt (`"guitar. bodhran. picture frame. sofa
   couch. ..."`) → list of bboxes with labels and scores.
2. Drop low-score detections (`min_dino_score`, default 0.30) and any whose
   bbox center is outside the projector footprint (the homography's preimage
   in camera pixels — the area the JMGO can physically reach).
3. **SAM 2** (`facebook/sam2.1-hiera-large`) is point-prompted at each
   surviving bbox center → pixel-accurate `cam_mask`. Batched across all
   detections in one forward pass.
4. AND each mask with the footprint; drop tiny masks
   (`min_obj_area_px`, default 800).
5. Sample median depth at the mask. If the **calibrated wall plane** is
   loaded (see *Calibration* below) and the mask's median depth is
   `wall_gate_m` (default 0.40 m) BEHIND the wall, drop it — catches
   detections seen through a doorway, in a mirror, etc.
6. **Parallax warp.** The homography is solved on the back wall, so naïve
   `warpPerspective(H)` puts every object's projector wash on the wall
   behind it. We shift each mask in camera-pixel space by `(centroid −
   image_center) × (z_wall − z_obj) / z_wall` before warping; for an object
   sitting 1 m in front of a 3.5 m wall this is ~30 % of the radial offset.
   Sign tuned for projector-right-of-camera; flip `parallax_sign` for
   the opposite mount.
7. Hand the resulting `FreshDetection`s to `ObjectTracker` for stable IDs
   across frames (IoU-matched, sticky for `stale_after_s` = 60 s).

Async mode: the heavy DINO+SAM pass (~600 ms on a 5090) runs in a background
thread on the freshest submitted frame; the main loop streams annotated
JPEGs at full camera rate (~30 Hz) using the most recent tracker state.

Depth is **advisory** — used for parallax, gating, and the UI's "this object
is 2.4 m away" readout — not for "is this a thing." Earlier iterations
used depth-foreground blobs as the primary segmentation driver; that
fused touching same-depth objects (bodhran on couch, guitar on cushion)
into one giant blob labeled "sofa couch."

## Web UI

`https://livetracking.barnettlabs.tech` (or `http://localhost:5070`):

**Auth:** every route except `/healthz` requires a shared-secret token
(the tunnel is public internet). First browser visit:
`https://livetracking.barnettlabs.tech/?token=<token>` — sets a 180-day
cookie and redirects to a clean URL. The token lives in
`runtime/auth_token.txt` on the rig (auto-generated on first run;
override with `LIVETRACKING_AUTH_TOKEN`). For curl / remote-ops, send
header `X-LiveTracking-Token: <token>`. Set
`LIVETRACKING_AUTH_DISABLED=1` to turn the gate off (LAN-only dev).

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
- **Edge softness** slider — Gaussian kernel half-width applied to SAM
  masks before warping to the projector. 0 = sharp/pixelated, 3 = soft
  (default), 7+ = airy glow. Compensates for the ~5× upscale from
  848×480 camera to 3840×2160 projector. Live-tunable, no restart;
  changes the active highlight immediately (the perception daemon
  re-warps and re-pushes the current object on slider release, so you
  don't have to re-hover).

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
5. **Wall-plane fit.** Sample depth at each detected marker's center; keep
   the farthest half (markers that actually landed on the back wall, not
   on couches or objects in front of it); back-project to 3D using the
   D455 intrinsics; SVD-fit a plane. This is the surface the H was solved
   on, so it's the correct reference for parallax compensation.
6. Persist `runtime/calibration/H.npy`, `calib.json`, `dot_cam_pts.npy`,
   `dot_proj_pts.npy`, `wall_plane.npy` (a 4-vector `(a, b, c, d)` so
   `aX+bY+cZ+d=0` for points on the wall in camera 3D), plus a synthesized
   `footprint_measured.png` and diagnostic PNGs to `scripts/out/`.

### Two-plane parallax calibration — must be re-run after any rig move

`scripts/calibrate_parallax.py` (UI: **Parallax calibrate**) writes
`H_wall.npy`, `H_near.npy`, and `parallax_depths.json`. When all three
exist, the pipeline **prefers** the two-plane interpolation over the
constant-K fallback — even if they were captured against an old camera/
projector pose.

**Pitfall (hit 2026-06-10):** after the rig moved, re-running the ArUco
homography calibration fixed `H.npy` + `wall_plane.npy` but left stale
June-6 `H_wall`/`H_near` in place — and the pipeline kept lerping toward
the old geometry, washing every object off to one side. Re-calibrating H
alone is NOT enough.

Rules of thumb:

- Moved the camera or projector? Re-run **both** calibrations, or delete
  `runtime/calibration/H_wall.npy` / `H_near.npy` /
  `parallax_depths.json` to drop back to the constant-K fallback (which
  follows the fresh wall plane automatically).
- Check the perception startup log: `two-plane parallax calib loaded`
  means the lerp is active; only `loaded calibrated wall plane` means
  constant-K. If washes are uniformly offset after a recalibration,
  suspect stale two-plane files first.
- Stale sets from past poses are parked in
  `runtime/calibration/stale-<date>/` rather than deleted, in case a
  pose is restored.

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
    perception.py       # D455 owner; image-first pipeline; ZMQ PUB/REP
    projector.py        # pygame on JMGO; ZMQ PULL; highlight/clear/white-light
    flame_web.py        # Flask UI + control trampoline
    templates/index.html
  perception/
    capture.py          # RealSense aligned color+depth, exposure/WB lock
    footprint.py        # H persistence + projector-footprint reconstruction
    pipeline.py         # DINO -> SAM -> depth gate -> parallax warp -> tracker
    recognize.py        # Grounding DINO + SAM 2 model wrappers, DEFAULT_DINO_PROMPT
    tracker.py          # IoU matcher, stable IDs, hide/rename/color cycle
    geometry.py         # depth-plane utilities (debug-only; unused by daemon)
  paths.py              # env-driven paths, display index, ZMQ endpoints
scripts/
  calibrate_homography.py   # time-multiplexed ArUco + wall-plane fit
  run_calibration.py        # orchestrator: stop/calibrate/restart + status JSON
  install_services.ps1      # idempotent service/task installer
  probe_parallax.py         # QA: drive test_point on each tracked object
  out/                      # diagnostic dumps from calibration + tests
runtime/                    # H.npy, wall_plane.npy, calib.json, masks, logs (gitignored)
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
| `LIVETRACKING_WEB_UI_PORT` | Flask port (`LIVETRACKING_WEB_PORT` also accepted) | `5070` |
| `LIVETRACKING_AUTH_TOKEN` | Web-UI auth token override (default: auto-generated `runtime/auth_token.txt`) | — |
| `LIVETRACKING_AUTH_DISABLED` | `1` disables web auth entirely (LAN-only dev) | — |
| `LIVETRACKING_ZMQ_CTRL` | Perception control REP endpoint | `tcp://127.0.0.1:5573` |
| `LIVETRACKING_PARALLAX_COMPENSATE` | Enable per-object parallax shift before warp (`0`/`false` to disable) | `1` |
| `LIVETRACKING_PARALLAX_SIGN` | Baseline direction. `+1` = projector RIGHT of camera (default, shifts mask RIGHT in projector pixels for near objects); `-1` = projector LEFT of camera. | `+1.0` |
| `LIVETRACKING_PARALLAX_SCALE` | Final multiplier on the parallax shift (live tuning). | `1.0` |
| `LIVETRACKING_PARALLAX_K` | Effective `f_proj · B` in pixels·meters. Shift in projector pixels = `sign · scale · K · (1/z_obj − 1/z_wall)`. Tuned live; current default sized to ~4" of correction at 1 m off the wall on the JMGO. | `1200.0` |

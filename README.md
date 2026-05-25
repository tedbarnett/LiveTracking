# LiveTracking

Real-time projection mapping. Track a hand-held object with an Intel RealSense depth camera, project effects (fire, glow, particles) back onto it using TouchDesigner.

**Status:** Active build. **Working in two regimes:**
- 2026-05-24 night (dark room, locked exposure): closed-loop iterative search lands 3/3 `+` signs on post-its, errors 10-16 px.
- 2026-05-25 morning (daylight): the night detector broke. Replaced with `diff_detect.py` (project black + white, diff = projection rectangle, ambient-independent). 3/3 sub-pixel convergence under morning sun.

**Working modules** at [`src/livetracking/calib_v1/`](src/livetracking/calib_v1/):
- `diff_detect.py` — **canonical detector**. Ambient-independent projector↔scene differencing. Use this.
- `closed_loop_search.py` — per-target iterative search to land `+` signs / fills on detected targets.
- `edge_detect.py` — original Otsu detector. Kept for dark-room reference; breaks at sunrise.

**Standing rule (Ted, 2026-05-25):** analysis happens ONLY inside the projection rectangle. Everything outside (walls, chairs, desks) is masked out before detection runs. Detection looking outside the rectangle is the source of every false-positive we ever hit.

**Tried and shelved 2026-05-25:**
- `iter_v11_grid_homography.py` — 9×7 checkerboard for one-shot projector↔camera homography. Kodak's contrast is too low; `findChessboardCorners` fails. Targets occluding the board (post-its on the wall) also break the all-corners-required detector.
- `iter_v12_aruco_homography.py` — 4 corner ArUco markers. Kodak's contrast is too low to decode the markers' internal 4×4 binary patterns; 8 candidate quads detected, 0 decoded. **Retry on JMGO Tuesday.**

The right v2 architecture (Ted's idea, 2026-05-25): project the camera frame back onto the wall, gradient-descend the projector↔camera homography to minimize image-registration error. Continuous calibration, no per-target detection. Build on JMGO.

**Full product spec:** see [`docs/SPEC.md`](docs/SPEC.md).

**Calibration math:** see [`docs/CALIBRATION-MATH.md`](docs/CALIBRATION-MATH.md) for how we handle camera/projector parallax with full 3D calibration.

**Product vision (codename Cobblestone):** see [`docs/PRODUCT-VISION.md`](docs/PRODUCT-VISION.md). Self-calibrating projection mapping for events, ten-minute setup, no measuring tapes. LiveTracking is the prototype platform; Cobblestone is what we ship.

**Calibration pattern art:** see [`assets/calibration-patterns/`](assets/calibration-patterns). Naturalistic running-bond cobblestone matching real deep-joint reference. Muted palette (Ted's call — saturated patterns read "cartoony"). The v5 variant embeds 18% calibration tint per stone so calibration can run continuously without ever visibly flashing.

**Design language:** the calibration pattern is an animated **cobblestone field** — doubles as branding (Cobblestone Labs, TimeWalk 1664 Manhattan) and as a strong structured-light feature pattern. Stones encode position via color; layout follows a hidden 8×6 grid for fast decoding.

## Web UI

The projection daemon publishes its state to a Flask web app for monitoring
and control:

- **Public:** `https://livetracking.barnettlabs.tech` (Cloudflare tunnel)
- **Local:** `http://192.168.1.197:5070` (PC-5090 on the home LAN)

Features:
- Live RealSense RGB camera frame (refreshes ~1×/sec)
- Tracked-targets table: cam coords, projector coords, sub-pixel error, rotation angle, current nudge
- Per-target color coding (target 1 = red, 2 = green, 3 = blue) matching the projected + signs
- Mode toggle: render `+` signs (default) or animated colored fills per target
- **Per-target manual nudge controls** (▲ ▼ ◀ ▶ arrow grid + reset per row). Compensates for residual parallax / per-target wiggle that no algorithm fully solves. Each click = 5 projector pixels. Persisted to `runtime/nudges.json` so nudges survive daemon restarts.
- Restart / Recalibrate button (daemon exits 42, supervisor relaunches, fresh detection)
- Screenshot button (forces a fresh camera capture)
- Status: uptime, heal-cycle count, current mode, last-status
- Cobblestone favicon + apple-touch-icon + webmanifest for proper iOS/Android "Add to Home Screen"

Calibration improvements landed 2026-05-25 morning:
- Full 2x2 Jacobian for projector↔camera mapping (replaces diagonal sx/sy approximation that broke on rotated targets)
- minAreaRect bbox center for post-it detection (replaces brightness centroid that biased toward whichever side of the post-it reflected more light)
- Per-target manual nudge for residual offset correction

Architecture:
- [`projection_daemon.py`](src/livetracking/daemon/projection_daemon.py) —
  long-running pygame + RealSense process. Owns the projector window and the
  camera pipeline. Publishes `runtime/state.json` + `runtime/latest_frame.jpg`
  every ~500ms. Reads `runtime/command.txt` to handle web UI commands.
  Exit code 42 = restart-requested.
- [`supervisor.py`](src/livetracking/daemon/supervisor.py) — relaunches
  daemon on exit 42, waits 5s + relaunches on crash, exits cleanly on 0.
- [`web_ui.py`](src/livetracking/daemon/web_ui.py) — Flask app on port 5070.
  All daemon↔UI communication via filesystem so each can restart
  independently.

Run:
```powershell
cd D:\Github-D\LiveTracking
python src\livetracking\daemon\supervisor.py   # in one terminal
python src\livetracking\daemon\web_ui.py      # in another
```

*Last updated by Helm — 2026-05-25, after the web UI ship + Cloudflare route.*

## Hardware

- **PC:** Windows, RTX 5090 32GB (PC-5090)
- **Depth camera:** Intel RealSense D455 (USB-C, 87°×58° FOV, 0.4-6m depth range)
- **Projector (current):** Dangbei MP1 MAX 4K Triple Laser
  - 3100 ISO lumens, 4K UHD native, HDMI 2.1
  - Game Mode input lag: ~12-35ms (auto ALLM + VRR)
  - Sitting on the office shelf, aimed at the wall opposite, white guitar as test subject
  - Good enough for crude MVP tests; physical setup constrained by couch and existing office furniture for now
- **Projector (incoming Tuesday 2026-05-26):** JMGO N3 Ultimate 4K Triple Laser
  - 5800 ISO lumens (brighter, better for larger / well-lit rooms)
  - 3-in-1 Lens Shift, Optical Zoom
  - AI Gimbal (no remounting when repositioning)
  - 20000:1 contrast
  - 1ms low latency mode + VRR + ALLM (huge — wipes out the projector latency floor)
  - Dolby Vision
  - Use case: larger demos (Cobblestone Labs room, etc.) and the daily driver once it arrives

## Software

- **TouchDesigner** (full license, installed on PC-5090)
- **Intel RealSense SDK 2.0** (installed at `C:\Program Files (x86)\Intel RealSense SDK 2.0`)
- **TouchDesigner Realsense TOP** for direct sensor access (built into TD 2023.11+)

## Latency Budget

End-to-end target: under 80ms (the "feels attached" threshold).

**With Dangbei MP1 MAX (current setup):**

- Camera capture (D455 @ 60fps): ~16ms
- Depth → mask processing in TD: ~10-15ms
- Effect compositing: ~10-15ms
- HDMI out → Dangbei in (Game Mode): ~12-35ms
- **Total: ~50-80ms** — right at the threshold, should feel attached

**With JMGO N3 Ultimate (incoming, will be the primary projector):**

- Camera capture: ~16ms
- TD processing: ~25-30ms
- HDMI out → JMGO in (1ms low-latency mode + ALLM): ~1ms
- **Total: ~42-47ms** — well under threshold, comfortable margin

## Repository Layout

- `td/` — TouchDesigner project files (`.toe`)
- `docs/` — Build plan, calibration notes, network sketches
- `scripts/` — Helper scripts (calibration, capture, diagnostics)
- `media/` — Reference photos, demo recordings (large files via Git LFS — see below)
- `assets/` — Particle textures, effect resources

## Build Phases

See `docs/PLAN.md` for the full build plan with phases:

1. **Setup verification** — confirm hardware and SDK (mostly done)
2. **MVP: silhouette + fire** — guitar held in projection field, fire effect tracks silhouette
3. **Calibration** — projector/camera alignment
4. **Effect library** — multiple effects, switchable live
5. **Hands-free interaction** — gesture triggers, motion-reactive effects

**Currently paused** at Phase 1 / pre-Phase-2 pending the new projector.

## Cloning on Other Machines

```powershell
gh repo clone tedbarnett/LiveTracking
# OR
git clone https://github.com/tedbarnett/LiveTracking.git
```

For the laptop:

- Install [TouchDesigner](https://derivative.ca/download) (free non-commercial is enough)
- Install [Intel RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense/releases)
- Plug in the D455
- Open `td/LiveTracking.toe`

## Notes on Git LFS

Large binary files (`.toe` project files, video recordings, particle textures over a few MB) should go through Git LFS. See `.gitattributes` for tracked patterns.

---

*Repo created 2026-05-23 by Helm + Ted. Updated 2026-05-24 to reflect on-hold status pending new projector. Project doc: `docs/PLAN.md`.*

## How-to (reusable recipe for projector-camera calibration)

**Goal:** Drop a projector + camera in any room, point them at a wall with targets, and have the system auto-detect targets and paint content on them. Self-heals when things move.

### Steps

1. **Set up the rig**: projector on Windows extended display (note the offset, e.g. (5120, 0)); USB camera pointed at the same wall area; room lights stable.

2. **Lock the camera exposure** (
s.option.enable_auto_exposure=0, rs.option.exposure=150). Auto-exposure compensates for projector content shifts and breaks the differencing logic.

3. **White-flood capture** the scene -> Otsu threshold finds the projection rectangle in camera coords -> Canny edges + contour filtering finds your targets inside it. For post-it-style rectangles: area 800-8000 px, 4-6 vertices, aspect < 1.6, solidity > 0.85.

4. **For each detected target** (camera_xy + rotation from cv2.minAreaRect):
   - Use the projection-quad-corners planar homography as the initial projector position estimate
   - Closed-loop search: project a + sign, diff camera frame against baseline (	hreshold=30 on the diff!), find blob centroid restricted to 200 px around target, compute error, apply proportional correction with damping (alternate Y sign every 2 iterations to handle inverted projector mounting), iterate until error < 6 px
   - Once converged, probe local scale by projecting at converged + (30, 0) and converged + (0, 30), measuring camera-pixel response. Gives scale_x and scale_y for that target.

5. **Render content** at each target by computing the projector-space rectangle: `half_w_proj = (cam_w * 0.85 / 2) / scale_x`. Apply rotation from `minAreaRect.angle`. Use `cv2.warpPerspective` to map animated content onto the rotated destination corners.

### Empirical tuning constants

(Default values from the 2026-05-24 session; adjust per scene:)
- `FILL_SHRINK = 0.85` (fills are 85% of detected target size)
- `VERTICAL_OFFSET_FRAC = 0.10` (shift fills down by 10% of target height)
- `THRESHOLD_DIFF = 30` (above noise floor of ~7 mean / 10 std)
- `CONVERGENCE_THRESHOLD_PX = 6`, `MAX_ITERATIONS = 20`, `SEARCH_RADIUS = 200`

### Self-healing

A 30-60 sec watchdog re-runs detection in the background; if any target shifts > 20 px (camera bumped, post-its repositioned), it re-converges just that target.

### The two non-obvious fixes that took hours to find

1. **Threshold the diff at 30, not 15.** Lower threshold connects scene-wide noise into one giant fake blob via morphology, swamping the real marker.
2. **Lock camera auto-exposure.** Auto-exposure breaks differencing by compensating for the projector's content changes.

Without these, the algorithm oscillates and converges to false bright spots (lamps, reflections, the brightest blob in the camera image happens to be the projector itself rather than the small marker it's showing).

### Reusable skill

The Helm workspace has this codified as a reusable skill at `skills/projector-calibration/` with the full SKILL.md, working recipes (`closed_loop_postits.py`, `animated_fills.py`), and lessons learned.

*Last updated by Helm — 2026-05-24 midnight, after 3/3 + animated fills working.*

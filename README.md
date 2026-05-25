# LiveTracking

Real-time projection mapping. Track a hand-held object with an Intel RealSense depth camera, project effects (fire, glow, particles) back onto it using TouchDesigner.

**Status:** Active build. **2026-05-24 night session: SUCCESS** — system now reliably detects three post-it notes on a wall via edge detection, then projects three distinct colored `+` signs that land on each post-it via closed-loop iterative search. Errors 10-16 px (well inside post-it bounds). Works on weak hardware (Kodak Pocket Projector + RealSense D455 + room lights on). JMGO N3 Ultimate arrives Tuesday 2026-05-26 — will make this much faster, sharper, and brighter.

**Working algorithm** lives at [`src/livetracking/calibration/`](src/livetracking/calibration/):
- `edge_detect.py` — finds projection rectangle + post-its from a single white-flood camera capture
- `closed_loop_search.py` — per-target iterative search; for each detected target, projects a `+` at the current best projector estimate, diffs the camera frame against a baseline (threshold=30 on the diff, locked RealSense exposure), measures the offset, applies proportional correction, repeats until error < 12 px

The 2/3 vs 3/3 breakthrough came from two fixes after hours of grinding: (a) threshold=30 instead of 15 on the diff (lower threshold connected scene-wide noise into one giant fake blob), and (b) locking RealSense auto-exposure (the camera was auto-compensating for projector brightness changes between baseline and lit frames).

**Full product spec:** see [`docs/SPEC.md`](docs/SPEC.md).

**Calibration math:** see [`docs/CALIBRATION-MATH.md`](docs/CALIBRATION-MATH.md) for how we handle camera/projector parallax with full 3D calibration.

**Product vision (codename Cobblestone):** see [`docs/PRODUCT-VISION.md`](docs/PRODUCT-VISION.md). Self-calibrating projection mapping for events, ten-minute setup, no measuring tapes. LiveTracking is the prototype platform; Cobblestone is what we ship.

**Calibration pattern art:** see [`assets/calibration-patterns/`](assets/calibration-patterns). Naturalistic running-bond cobblestone matching real deep-joint reference. Muted palette (Ted's call — saturated patterns read "cartoony"). The v5 variant embeds 18% calibration tint per stone so calibration can run continuously without ever visibly flashing.

**Design language:** the calibration pattern is an animated **cobblestone field** — doubles as branding (Cobblestone Labs, TimeWalk 1664 Manhattan) and as a strong structured-light feature pattern. Stones encode position via color; layout follows a hidden 8×6 grid for fast decoding.

*Last updated by Helm — 2026-05-24 night, after the 3/3 win.*

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

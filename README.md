# LiveTracking

Real-time projection mapping. Track a hand-held object with an Intel RealSense depth camera, project effects (fire, glow, particles) back onto it using TouchDesigner.

**Status:** Setup phase. Projector arrives Tuesday 2026-05-26. MVP target: that weekend.

## Hardware

- **PC:** Windows, RTX 5090 32GB (PC-5090)
- **Depth camera:** Intel RealSense D455 (USB-C, 87°×58° FOV, 0.4-6m depth range)
- **Projector:** JMGO N3 Ultimate 4K Triple Laser
  - 5800 ISO lumens (bright enough for ambient light)
  - **1ms low latency mode + ALLM** — the key spec, removes projector from latency budget
  - VRR support
  - AI Gimbal (360° pan / 150° tilt) for repositioning without remounting

## Software

- **TouchDesigner** (full license, installed on PC-5090)
- **Intel RealSense SDK 2.0** (installed at `C:\Program Files (x86)\Intel RealSense SDK 2.0`)
- **TouchDesigner Realsense TOP** for direct sensor access (built into TD 2023.11+)

## Latency Budget

End-to-end target: under 80ms (the "feels attached" threshold). Expected:

- Camera capture (D455 @ 60fps): ~16ms
- Depth → mask processing in TD: ~10-15ms
- Effect compositing: ~10-15ms
- HDMI out → projector in (ALLM mode): ~1ms
- **Total: ~40ms** — well under threshold

## Repository Layout

- `td/` — TouchDesigner project files (`.toe`)
- `docs/` — Build plan, calibration notes, network sketches
- `scripts/` — Helper scripts (calibration, capture, diagnostics)
- `media/` — Reference photos, demo recordings (large files via Git LFS — see below)
- `assets/` — Particle textures, effect resources

## Build Phases

See `docs/PLAN.md` for the full build plan with phases:

1. **Setup verification** (pre-Tuesday) — confirm hardware and SDK
2. **MVP: silhouette + fire** (Tuesday-Sunday) — guitar held in projection field, fire effect tracks silhouette
3. **Calibration** — projector/camera alignment
4. **Effect library** — multiple effects, switchable live
5. **Hands-free interaction** — gesture triggers, motion-reactive effects

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

*Repo created 2026-05-23 by Helm + Ted. Project doc: `docs/PLAN.md`.*

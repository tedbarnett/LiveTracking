# LiveTracking

Real-time projection mapping. Track a hand-held object with an Intel RealSense depth camera, project effects (fire, glow, particles) back onto it using TouchDesigner.

**Status:** On hold pending new projector. Ted is waiting on the new projector before doing more work on this. Current Dangbei MP1 MAX setup is still usable for crude tests at the office, but active build work is paused. Resume when projector arrives.

*Last updated by Helm — 2026-05-24*

## Hardware

- **PC:** Windows, RTX 5090 32GB (PC-5090)
- **Depth camera:** Intel RealSense D455 (USB-C, 87°×58° FOV, 0.4-6m depth range)
- **Projector (current):** Dangbei MP1 MAX 4K Triple Laser
  - 3100 ISO lumens, 4K UHD native, HDMI 2.1
  - Game Mode input lag: ~12-35ms (auto ALLM + VRR)
  - Sitting on the office shelf, aimed at the wall opposite, white guitar as test subject
  - Good enough for crude MVP tests; physical setup constrained by couch and existing office furniture for now
- **Projector (incoming, model TBD):** Ted is sourcing a new projector. Specs and arrival date pending confirmation. Update this section when the order is placed.

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

The incoming projector's latency profile will be added once the model is confirmed.

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

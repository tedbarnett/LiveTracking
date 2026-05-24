# LiveTracking — Product Spec v0.1

*Last updated 2026-05-24 by Helm.*

## One-line pitch

Point a projector at any scene, throw a beautiful cobblestone pattern at it, pick an object from a live highlight UI, and the projector starts painting effects (fire, glow, light) onto that object in real time.

## Why cobblestones

The calibration pattern doubles as brand: cobblestones are the visual language of 1664 Manhattan (TimeWalk) and the namesake of Cobblestone Labs. The same animated pattern that aligns the projector to the world also tells a story about what the world used to look like.

Technically, cobblestones work because:

- Sharp edges between stones = strong feature points for camera decoding
- Variable sizes = naturally multi-scale, robust to projection distance
- Color variation between stones = positional encoding hidden in plain sight
- Mortar lines = implicit grid the decoder rectifies against

## Product flow

### 1. Power-on / launch (no calibration yet)

Projector is on, RealSense is on, software is running. Projector shows a soft idle animation (gentle moving cobblestones in a low-contrast palette) — not blank black, not glaring white. Camera is watching but doing nothing.

### 2. Coarse calibration (~3 seconds)

User presses `C` (or taps a button on a tablet UI). The cobblestone pattern lights up at full brightness, animated for ~3 seconds. During those 3 seconds:

- Each "stone" in the pattern has a unique color signature drawn from a known palette
- Stones are laid out in a hidden 8×6 grid for predictable decoding
- The camera captures ~90 frames of the projected pattern landing on whatever's in the scene
- A structured-light decoder solves for the projector-to-camera homography per depth slice

Output: we now know which projector pixel corresponds to which camera pixel at each depth.

### 3. Object selection

The projector dims the cobblestones to a faint background and lights up candidate objects in the scene with soft outlines (glow halos in different colors per object). Candidates come from depth-cluster segmentation: connected regions of foreground depth (~0.4-3m from camera) that are separated by depth discontinuities.

The user selects an object by:

- **v1:** pressing `1`, `2`, `3`, ... on the keyboard for the corresponding highlighted object
- **v2 (later):** tapping a tablet showing thumbnails of the segmented scene
- **v3 (later):** pointing at the object and letting hand-pose detection pick it

### 4. Fine calibration on the selected object (~1 second)

Once an object is picked, the projector throws a localized denser cobblestone pattern only on that object's bounding region. This gets us surface-accurate calibration (where effects land *on* the object instead of next to it).

### 5. Effect rendering

Pattern fades out, effect fades in. The selected object's silhouette drives a particle/light effect:

- Fire (default v1)
- Liquid flow
- Lightning crackle
- Soft glow / aura
- Color shift based on motion
- More over time

Effect tracks the object as it moves, at ~60fps, with <80ms end-to-end latency.

### 6. Reselect / recalibrate

- `C` re-runs full calibration (if camera or projector moved)
- `R` re-runs object selection (if you want to switch from the guitar to your hand)
- `E` cycles through effects
- `Esc` returns to idle

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LiveTracking app                          │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Capture    │───▶│  Calibrate  │───▶│  Segment            │  │
│  │  pyrealsense│    │  OpenCV     │    │  depth-cluster      │  │
│  │  D455 RGB+D │    │  homography │    │  + optional SAM2    │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│         │                                          │             │
│         │                                          ▼             │
│         │                                ┌─────────────────────┐ │
│         │                                │  Object picker UI   │ │
│         │                                │  PyQt / projected   │ │
│         │                                └─────────────────────┘ │
│         │                                          │             │
│         ▼                                          ▼             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   Render pipeline (shader-based, OpenGL or Pygame+GLSL)   │   │
│  │   - Cobblestone pattern generator (calibration)           │   │
│  │   - Idle animation                                        │   │
│  │   - Object highlight glows                                │   │
│  │   - Effect compositor (fire, liquid, lightning, glow)     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│                    ┌────────────────┐                            │
│                    │  Window output │                            │
│                    │  HDMI fullscreen│                            │
│                    └────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ HDMI
                      ┌──────────────┐
                      │   Projector   │  (Dangbei MP1 MAX
                      │               │   or JMGO N3 Ultimate)
                      └──────────────┘
```

## Stack decision: Python prototype first, TouchDesigner later (maybe)

**Originally planned:** TouchDesigner all the way.

**Why we're starting in Python for the prototype:**

- TD requires graphical patching — not amenable to "Claude Code, build this in an evening"
- Python + OpenCV + pyrealsense2 + ModernGL (for shader effects) gets us a working end-to-end loop in a single git repo that any AI agent can iterate on
- Same RTX 5090 underneath; latency budget is fine
- If we hit a perf wall, we port the hot path to TD or to a C++/CUDA module

**Hot path latency budget (Python prototype):**

| Stage | Target | Notes |
|---|---|---|
| RealSense capture (60fps) | ~16 ms | hardware limit |
| Depth + RGB to GPU texture | ~3 ms | pyrealsense2 + CuPy or direct OpenGL upload |
| Segmentation (depth-cluster) | ~5 ms | OpenCV connected components |
| Effect shader pass | ~5 ms | ModernGL fragment shader on the 5090 |
| Window flip | ~16 ms | vsync at 60fps; could disable for lower latency |
| Projector HDMI lag (JMGO) | ~1 ms | 1ms low-latency mode |
| **Total** | **~46 ms** | well under 80ms threshold |

Dangbei adds 12-35ms over JMGO; still within budget.

## Build phases (revised, post-cobblestone)

### Phase 0 — Tonight, no projector yet

Build the software pipeline running against the camera + a monitor as a "virtual projector." Validates everything except actual projection alignment.

- [ ] Repo scaffold (`src/`, requirements, entry point)
- [ ] Cobblestone pattern generator (animated, 8×6 hidden grid, color-encoded positions)
- [ ] Idle animation
- [ ] RealSense capture loop
- [ ] Depth-cluster segmentation
- [ ] Object highlight rendering (glows around segmented objects)
- [ ] Keyboard selection (1, 2, 3...)
- [ ] Effect renderer: fire (default), glow, color shift
- [ ] Hotkey loop (C, R, E, Esc)
- [ ] Diagnostic overlay (FPS, latency, object count)

### Phase 1 — Tuesday + projector

- [ ] Mount projector + RealSense rigidly together
- [ ] Run actual structured-light calibration end-to-end
- [ ] Measure real latency
- [ ] Tune cobblestone palette + grid density for the projector's resolution
- [ ] First "fire on white guitar" demo recording

### Phase 2 — Effects library

- Fire, liquid, lightning, glow, color shift, motion trails
- Per-effect parameter UI

### Phase 3 — Object selection upgrades

- Tablet UI showing depth-segmented scene + tap to pick
- Pointing / hand-pose selection
- Multi-object simultaneous effects

### Phase 4 — Production stack decision

If Python perf is fine: stay in Python, polish.
If we need more headroom: port hot loop to TouchDesigner or a custom C++/CUDA module.

## Risks (updated)

- **Cobblestone decoding speed**: photo-real irregular cobblestones may decode slowly. Mitigation: stylized stones laid out in a hidden grid (described above). Real-world cobble photos can be a "theme" applied to the same grid, not the underlying math.
- **Glossy white guitar IR dropouts**: same as before, hasn't changed.
- **Projector input lag**: solved by JMGO's 1ms mode. Dangbei works for prototyping.
- **PyQt vs projected UI**: doing the highlight UI *via the projector itself* requires the calibration to already be working. Chicken-and-egg solved by using a coarse "any flat surface" homography for the highlight pass, then refining after object selection.

## Open questions for Ted (parked, not blocking the prototype)

- Cobblestone texture style — photo-real, low-poly stylized, or historical-NYC photographs as a theme?
- Should "selection" eventually accept voice ("the guitar")?
- Cobblestone Labs collaboration — do they want their physical space to use this calibration system at install time?
- TimeWalk integration — projected historical scenes on physical models?

## Done definition for tonight

- `python -m livetracking` launches a window
- Window shows animated cobblestone idle pattern
- Pressing `C` runs a 3-second calibration cycle (printed timings + saved homography)
- Pressing `S` enters object-selection mode (highlights segmented foreground objects with colored glows)
- Pressing `1`/`2`/`3` selects an object
- Pressing `E` cycles fire / glow / colorshift effects on the selected object
- Pressing `Esc` returns to idle
- All running at 30+ fps with FPS overlay visible

We won't have a real projector aimed at a real object until Tuesday, but the pipeline will be done.

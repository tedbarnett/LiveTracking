# LiveTracking — Build Plan

## Goal

Hold any object (white guitar, hand, prop) in front of the projector. RealSense depth camera isolates the object. TouchDesigner renders an effect (fire, glow, particles) shaped to the silhouette. Projector paints the effect back onto the object in real-time.

## Success Criteria

- Effect visibly tracks the object as it moves at human speed
- End-to-end latency under 80ms (effect feels "attached" not "chasing")
- Works under normal room lighting (not pitch-dark hack)
- Switchable effects (fire, water, lightning, glow) — at least 3 variants
- Setup takes under 10 minutes from cold boot

## Phase 1: Setup Verification

Done with what's already on the PC.

- [x] RealSense D455 enumerates (Status: OK on both Depth and RGB nodes)
- [x] RealSense SDK 2.0 installed
- [x] TouchDesigner installed with full license
- [x] Projector: Dangbei MP1 MAX confirmed (HDMI 2.1, Game Mode, ~12-35ms lag)
- [ ] **Action:** open `realsense-viewer.exe`, confirm live depth + RGB feeds
- [ ] **Action:** in TouchDesigner, drop a `Realsense TOP`, confirm the D455 appears in its dropdown and produces live depth output
- [ ] **Action:** enable Game Mode on the Dangbei MP1 MAX, confirm HDMI (not wireless) connection

## Phase 2: MVP — Silhouette + Fire

The "I hold a white guitar and it lights up" demo. Office setup: Dangbei MP1 MAX on shelf, projecting across to couch wall, white guitar on couch. Couch and framed maps behind are far enough back that depth thresholding excludes them.

### 2a. Projector physical setup

- Mount projector (ceiling, table, or temporary tripod)
- Aim it at a flat wall or large board
- Aim the RealSense from roughly the same axis as the projector (parallax-minimizing — they don't need to be coincident but closer is better)
- Confirm "Game Mode" / ALLM is enabled on the JMGO N3
- Set projector to 1080p input (4K is overkill for this and adds processing — we can revisit)

### 2b. Depth → mask in TouchDesigner

- `Realsense TOP` → depth stream output
- `Threshold TOP` to isolate objects within ~0.5m to ~2m from camera (background is wall, foreground is empty air)
- `Blur TOP` for soft edges
- Output: a binary silhouette mask

### 2c. Effect: fire on silhouette

- `Noise TOP` animated upward for flame motion
- `Composite TOP` to mask the noise with the silhouette
- Hue/luminance ramp to make it look like fire (yellow core, orange middle, red edge, transparent outside)
- Send to `Window COMP` set to the projector's display

### 2d. Calibration

- Project a checkerboard pattern
- Capture the projection in the RealSense RGB stream
- Compute the homography between RealSense view and projector output
- Apply to silhouette before compositing — this is what makes the fire land *on the guitar* instead of next to it

### 2e. Verification

- Hold the white guitar in the projection field
- Move it slowly, then quickly
- Effect should track without obvious lag
- Take a video for review

## Phase 3: Effect Library

Once Phase 2 works, build a switchable library:

- Fire (Phase 2)
- Water/liquid flow (downward noise + blue ramp)
- Lightning (Voronoi crackle textures)
- Glow/aura (simple bloom around silhouette)
- Color shift (paint the object different colors based on motion)

UI: a simple TouchDesigner panel with effect buttons, or MIDI/OSC trigger from a phone.

## Phase 4: Hands-Free Interaction

- Detect motion vectors from depth stream (object moving fast → spawn extra particles)
- Hand pose detection (open hand → glow, fist → fire) — uses RealSense's body tracking or MediaPipe
- Multi-object support (track 2-3 objects independently)

## Phase 5: Demo / Show Mode

- Auto-start on boot
- Cycle through effects on a timer
- Looks good even with no one in front of it (idle animation)

## Known Risks

- **Glossy white guitar:** RealSense D455's IR stereo may have dropouts on glossy lacquer surfaces. Mitigation: depth filling/inpainting in TD, or matte spray (not happening on this guitar).
- **Office wall has framed glass maps:** they would cause IR reflections and projection hot spots, but they're behind the couch which is behind the depth threshold, so they should be excluded from the tracking mask. Test will confirm.
- **Ambient light:** Dangbei is 3100 lumens. Side lamp in office reduces contrast. Lights-off testing for best results until white-wall room (1-2 months out).
- **RealSense placement:** if camera and projector are too far apart, parallax means the effect "leans" off the object as you move toward/away from the camera. Mount the D455 on top of the Dangbei (it has a flat top).
- **TD network performance:** if the depth → mask → effect chain doesn't hit 60fps, drop to 30fps but keep latency low. Don't accept frame buffering.

## References

- TouchDesigner RealSense TOP docs: https://docs.derivative.ca/Realsense_TOP
- Intel RealSense Examples: https://github.com/IntelRealSense/librealsense/tree/master/examples
- Projection mapping fundamentals (TD community): https://docs.derivative.ca/Projection_Mapping

## Open Questions for Ted

- Mounting strategy for the projector — ceiling? Tripod? Permanent vs portable?
- Demo venue — is this for the Cobblestone Labs Chelsea studio long-term, or just personal experiments first?
- TimeWalk integration — could this be a projected interaction layer for the historical scenes (e.g. project a 1664 building outline onto a physical model)? Worth considering as a stretch goal.

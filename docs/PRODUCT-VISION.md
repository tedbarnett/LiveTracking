# LiveTracking — Product Vision: Cobblestone

*Last updated 2026-05-24 by Helm. Companion to `SPEC.md` and `CALIBRATION-MATH.md`.*

## The product in one paragraph

A projection mapping system you can drop into any room — nightclub, gallery, wedding, conference, corporate event — and have it lit up and tracking in under ten minutes. No "fussy calibration." No measuring tapes. No bracket-engineering the camera to the projector. Plop down a projector. Plop down a depth camera somewhere with overlapping field of view. Press one button. Cobblestones light up for five seconds, the geometry solves itself, the system starts painting effects onto whatever's in the room.

Codename: **Cobblestone**, since the calibration pattern is the brand and the metaphor.

## Why this matters

The market for projection mapping at events is real and currently underserved.

Today, projection mapping a venue is a $5K-$50K job. A specialist shows up with TouchDesigner or Resolume, takes 4-8 hours measuring, calibrating, masking projection zones by hand, and the result is locked to that exact projector position. Move the projector, redo the calibration. The art is gorgeous, the workflow is brutal.

Most venues never get projection mapping because the install cost can't amortize over a single night. Cobblestone is the system that gets it under that threshold — ten-minute setup, recoverable from chaos, runs autonomously through the show.

## The reframing: stop fighting parallax, let it work for you

Conventional projection mapping minimizes the camera-projector baseline by physically strapping them together. That's the wrong instinct for our use case.

Instead:

- **Camera goes wherever the operator can see the action best** (ceiling-mounted bird's-eye, tripod near the dance floor, behind the DJ booth, opposite wall, doesn't matter)
- **Projector goes wherever the throw distance works**
- **They don't need to be near each other.** They just need overlapping fields of view.

The math doesn't care about baseline. Once we know `R` and `T` between the two devices, a 3m baseline computes identically to a 10cm one. What's needed isn't a small baseline — it's an **automatic measurement of whatever baseline you happen to have**.

That measurement is what the cobblestone pattern provides, in one 5-second capture.

## Operator's experience (the whole product)

### Setup (T-30 minutes)

1. Carry in a projector. Set it on a table, ceiling mount, tripod, whatever you've got.
2. Carry in a RealSense camera. Set it somewhere with a view of where the action will be. Tripod, shelf, ceiling clip, your call.
3. Plug both into the controller PC over USB.
4. Open the Cobblestone app on a tablet. Both devices show "ready."

### Calibration (T-25 minutes, takes 10 seconds)

5. Tablet shows a live camera preview. Projector throws a faint identifying pattern. Operator confirms in the preview that "yes, the camera can see roughly where the projector is throwing."
6. Tap "Calibrate."
7. Projector lights up with full cobblestone pattern for ~5 seconds. Animated, decoded by camera, geometry solved.
8. Status indicator: green "Locked."

### Effect design (T-20 minutes)

9. Tablet shows the segmented scene — the system's view of "objects" and "zones." Walls, floor, furniture, people-shaped blobs.
10. Operator drags effects from a library onto zones / object types.
    - *"Color wash on the floor."*
    - *"Fire trail on anything that moves in this region."*
    - *"Spotlight follows the DJ."*
    - *"Lightning on the back wall on every beat."*
11. Effects preview live in the tablet view as they're authored.

### The show (T-0 onward)

12. Operator taps "Run."
13. System operates autonomously. Object tracking, drift detection, recalibration on physical disturbance, beat sync (if a music input is plugged in).
14. Operator monitors from the tablet. Can intervene: pause, swap effects, force recalibration, fine-tune offsets, kill switch.
15. End of show: tap "Save show" to capture the configuration. Loadable next time.

## What makes it work at a nightclub specifically

**Fog and haze are helpful.** Smoke means projector beams are visible mid-air, giving more depth points for triangulation. Calibration accuracy goes up, not down.

**Moving people don't break calibration.** It's depth-aware (full 3D). People walking through during calibration just appear as foreground objects at varying depths — additional samples, not occlusions.

**Bumped equipment self-heals.** Drift detector runs continuously: every ~30 seconds, the system briefly flashes a sparse subset of cobblestones during the effect pass and verifies they land where the calibration predicts. If drift > 5 px, silent automatic recalibration starts. Effects continue running on the last-known calibration during the few seconds the new one converges. Operator sees a yellow "Recalibrating" pip on the tablet but the show doesn't stop.

**Multi-projector / multi-camera.** The geometry math doesn't care how many devices you have. Each camera-projector pair gets its own `R`, `T`. Calibrate them in parallel from one tap. Large venues become tractable.

**Operator-friendly recovery.** If something goes sideways — wrong calibration, occluded camera, projector unplugged — the operator gets a clear notification and a one-tap recovery path. No looking at log files mid-show.

## The detection stack: four layers

Cobblestone segments the live scene into four conceptual layers, each receiving different effect treatment.

### Layer 1 — Static scenery

Captured during calibration as the depth-distant surface. Walls, ceiling, mounted furniture, stage. Receives slow ambient effects: color washes, gentle pattern projection, transitions tied to music BPM. Doesn't change frame-to-frame.

### Layer 2 — Tracked objects

Foreground depth segmentation finds discrete blobs in the 0.4m-3m range from the camera. Each blob is tracked frame-to-frame (Kalman filter or BYTETrack). Examples: instruments, props, the DJ's laptop, a sculpture on a plinth, the cocktail cart.

Operator assigns effects per object type. Effect follows the object as it moves.

### Layer 3 — People

Treated separately from props because:

- RealSense supports skeleton tracking natively for up to a few people (D455 has it in SDK 2.x).
- Pose-aware effects ("paint a halo on heads," "make hands trail particles") only work with skeleton data, not raw blobs.
- Privacy story is cleaner — we explicitly do *not* do facial recognition, just per-frame skeleton tracking. No data stored after the frame is rendered.

For higher people-counts (crowd scenes), fall back to YOLOv8 pose estimation running on the GPU.

### Layer 4 — Operator picks

Operator taps an object/zone on the tablet and assigns a custom effect outside the auto-rules. Used for highlights ("paint the bride blue when she walks down the aisle," "spotlight the bottle service table when champagne arrives").

## Tablet control surface

Three views, three panels.

**Scene panel.** Live thumbnail of the segmented scene, with detected objects color-coded. Operator taps to select an object or drags to define a zone.

**Effect panel.** Grid of effects (fire, lightning, glitter, color wash, smoke trails, dot pattern, custom shaders, image/video texture). Drag onto a scene element to assign.

**Status panel.** Calibration health (Locked / Drifting / Recalibrating with timer), FPS, latency overlay, projector + camera connection status. One emergency button per device (kill, reset, recalibrate).

Connection is WebSocket from tablet to controller PC. Latency from tap to projection update: target under 100ms.

## Hardware reference design (v1)

- **Controller PC:** any modern Windows or Linux box with an RTX-class GPU. RTX 5090 = comfortable headroom for 4K@60fps with effects. RTX 4070-ish = sufficient for 1080p@60fps. Mini-PC form factor at the venue.
- **Depth camera:** Intel RealSense D455 (current pick). Future: D456 when it lands; Microsoft Azure Kinect is discontinued but works; for outdoor or very large spaces, consider stereo cameras.
- **Projector:** any HDMI-input projector. Recommended class: 3000+ lumens for venues with house light, 5000+ for fully lit rooms or large throws. Low input lag (Game Mode, ALLM, VRR) is the single biggest spec for feel.
- **Tablet:** any iPad or Android tablet running the Cobblestone client. Web-based to start, native later.
- **Optional:** beat detector audio input (3.5mm jack from the venue's mixer feed), for music-synced effects.

**Cost target for a venue install kit:** under $5K hardware all-in. Software is the value.

## Hardware avoidance: what we explicitly don't need

- No measuring tape
- No mounting bracket to strap camera to projector
- No checkerboard calibration target
- No pre-rendered projection masks per venue
- No tracking markers on objects or people
- No specialized projector

This is the entire "fussy calibration" we're eliminating.

## Multi-device topology

```
                           ┌─────────────────┐
                           │  Tablet (web)   │
                           │   Operator UI   │
                           └────────┬────────┘
                                    │ WebSocket
                                    ▼
┌──────────────────────────────────────────────────────────┐
│                  Controller PC (Cobblestone app)         │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Camera 1 │  │ Camera 2 │  │ Audio in │  │  Show    │  │
│  │  D455    │  │  D455    │  │  3.5mm   │  │  state   │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       └────────────┴─────────────┘             │        │
│                    ▼                            │        │
│           ┌─────────────────┐                  │        │
│           │  Scene model    │◄─────────────────┘        │
│           │  (3D + objects) │                            │
│           └────────┬────────┘                            │
│                    │                                     │
│                    ▼                                     │
│           ┌─────────────────┐                            │
│           │ Effect renderer │                            │
│           │ (per projector) │                            │
│           └────────┬────────┘                            │
│                    │                                     │
└────────────────────┼─────────────────────────────────────┘
                     │ HDMI x N
                     ▼
              ┌──────────────┐
              │ Projector 1  │
              │ Projector 2  │
              │ Projector N  │
              └──────────────┘
```

Cameras and projectors are independent. The controller fuses cameras into a unified 3D scene model and renders the right slice of that scene to each projector's frustum.

## Roadmap from where we are now

### Phase 0 (tonight, 2026-05-24)
Camera-space prototype on the monitor, no projector. Validates the segmentation, effects, UI loop. Status: in progress (Claude Code building).

### Phase 1 (Tuesday-Sunday, 2026-05-26 to 06-01)
JMGO arrives. First real projection on white guitar. First cobblestone calibration capture. Validate the math from `CALIBRATION-MATH.md` against a real rig.

### Phase 2 (June 2026)
Decouple camera from projector. Demonstrate calibration at 1m, 3m, and 6m baselines. Add the drift detector.

### Phase 3 (Q3 2026)
Tablet control surface. Object tracking (frame-to-frame, not just per-frame seg). Skeleton tracking. Beat sync.

### Phase 4 (Q3-Q4 2026)
Multi-projector / multi-camera. Effect-zone authoring. Show save/load.

### Phase 5 (late 2026)
**First real venue install.** Candidates: Cobblestone Labs Chelsea studio (relationship installed), a friend's loft party (low-stakes test), a small bar with a mural wall (paid pilot).

### Phase 6 (2027)
Productize. Hardware bundle SKU. Software license tier. Operator certification (?). Resellers.

## Business shape (initial thoughts, not committed)

- **License the software** to AV companies who already do event production. They handle install/operations; we get a per-show or per-month fee.
- **Sell turnkey kits** with controller PC + camera + projector, pre-configured. $5-8K bundle. Margin on the bundle is meh; the recurring software side is the play.
- **Service tier** for high-end installs: museums, corporate showrooms, hotel lobbies. Done with a Cobblestone certified integrator. $50-200K per install, recurring support contract.
- **Cobblestone Labs as the first lighthouse customer / showroom.** Their Chelsea immersive room demos this system live for prospective clients. Win-win — they get marquee tech, we get a referenceable install in a high-traffic AV space.

The TimeWalk angle is real here too: projection-mapped historical scenes onto physical models, projected onto buildings during walking tours, projection installs at museums showing historic city overlays. A specific vertical inside the same software platform.

## Naming

Codename **Cobblestone** is sticky for several reasons:

- The calibration pattern is the brand
- Ties to Ted's existing Cobblestone Labs investment
- The product literally "throws cobblestones, then paints the world" — that's the tagline
- Historical resonance with 1664 Manhattan / TimeWalk
- One-word, distinct, hasn't been trademarked in this space (to verify before commitment)

Alternatives if Cobblestone is taken: **Beacon**, **Lighthouse** (ties to Helm), **Cairn**, **Klieg**.

## Open questions for Ted

- Cobblestone Labs partnership shape: do Craig + Abby want to be the go-to-market vehicle? Or is Cobblestone-the-software an arms-length licensor to them?
- TimeWalk integration: is "projection-mapped historic Manhattan on a physical model" a real product or a stretch demo?
- Capital required: this needs probably $200-500K to get from Phase 5 demo to Phase 6 productized SKU. Bootstrap with Cobblestone Labs revenue, or take outside capital, or self-fund?
- Founding team shape: who builds the venue partnerships? You're the inventor/CTO archetype here; we'd need a sales/biz lead for events vertical eventually.

## TL;DR

We're building a self-calibrating projection mapping system that drops into any venue in ten minutes, survives the chaos of a live event, and replaces a $10K specialist install with a $5K box and a tap on a tablet. Codename Cobblestone. The cobblestone pattern is both the calibration mechanism and the brand. LiveTracking is the prototype platform; Cobblestone is what we ship.

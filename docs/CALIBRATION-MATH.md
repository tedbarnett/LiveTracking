# LiveTracking — Calibration Math

*Last updated 2026-05-24 by Helm. Companion to `SPEC.md`.*

## The problem in one sentence

The RealSense camera and the projector don't share the same lens, so they see the world from slightly different angles. Without correcting for that, the projected effect lands offset from the object — by a number of pixels that **grows or shrinks with the object's depth**. A single 2D homography cannot fix this. We need a full 3D calibration.

## Two issues, often confused

### 1. Pixel correspondence (2D, easy)

> "When the camera sees pixel `(x_c, y_c)`, which projector pixel `(x_p, y_p)` paints that same spot?"

For a flat surface at one known depth, this is a single homography. 8 numbers, solvable with 4+ point correspondences. This is what most "projection mapping" tutorials cover.

### 2. Depth-dependent parallax (3D, the real problem)

> "Even with perfect calibration at depth = 1m, an object at depth = 2m projects to a different projector pixel because the projector and camera see it from different angles."

A flat wall at one depth = problem 1 only. A chair sticking out of a wall = problem 2. Anything held in your hand at varying distances = absolutely problem 2.

## How bad is the parallax in practice

```
offset_pixels ≈ baseline × focal_length / depth
```

Worked example for the JMGO + D455 rig:

- Baseline (projector lens ↔ D455): ~10 cm if mounted close
- Projector focal length: roughly equivalent to ~1500 pixels at 1080p
- Object depth: 1.5 m

`offset ≈ 0.10 × 1500 / 1.5 = 100 pixels`

100 pixels of offset on a 1080p projection is obvious — the glow would visibly trail off the chair onto the wall behind. Even with a 5 cm baseline, you'd see ~50 px offset. Parallax is not a "rounding error" problem.

## The textbook solution: full 3D calibration

Three components.

### 1. Intrinsic calibration (each device, one-time)

What it is: focal length, principal point, lens distortion. Encoded as a `K` matrix and distortion coefficients.

**Camera:** RealSense ships factory-calibrated. We read `K_cam` and distortion from the SDK on launch. Zero work.

**Projector:** a projector is a "camera in reverse." We treat it as a virtual camera that captures whatever we tell it to display. To get `K_proj`:
- Display a known pattern (cobblestones, in our case)
- Camera observes that pattern projected onto a flat surface, at multiple positions / distances
- For each captured frame, we have the (projector_pixel, camera_pixel) correspondences. The cobblestones encode their own position so decoding is one-shot.
- Run OpenCV's `calibrateCamera` on the projector_pixel side using the world points from the camera+depth side.

### 2. Extrinsic calibration (relative pose, one-time per rig)

What it is: the 3D rotation `R` and translation `T` that take a point from camera coordinates to projector coordinates.

After intrinsic calibration, we already have many `(world_3D_point, projector_pixel)` pairs from the cobblestone capture. World points come from `K_cam_inverse · (camera_pixel) × depth`. Projector pixels come from decoding stones.

Feed all of those pairs into `cv2.solvePnP(world_points, projector_pixels, K_proj, dist_proj)`. Returns `R` and `T`. Done.

We can do steps 1 and 2 in the same calibration capture — the cobblestone pattern hits multiple depths naturally because the scene has 3D depth (chair pokes out, wall is back), giving us the depth variance we need.

### 3. Per-frame projection (every frame, fast)

Runtime, 60 fps:

```python
# RealSense gives us depth for every camera pixel
for each camera_pixel (x_c, y_c):
    depth = depth_frame[y_c, x_c]

    # Backproject to 3D in camera space
    point_3D_cam = K_cam_inv @ [x_c, y_c, 1] * depth

    # Transform to projector space
    point_3D_proj = R @ point_3D_cam + T

    # Project to projector pixels
    point_2D_proj = K_proj @ point_3D_proj
    x_p, y_p = point_2D_proj[:2] / point_2D_proj[2]

    # Now (x_c, y_c) ↔ (x_p, y_p) for THIS depth
    # → write the effect color for object at (x_c, y_c) into projector pixel (x_p, y_p)
```

This is one matrix multiply per pixel. On the RTX 5090 as a fragment shader, this is sub-millisecond for a 1080p frame. Trivial.

## Why our cobblestone pattern is load-bearing here

Each cobblestone has a **known projector-pixel location** because we drew it. When the camera sees that stone projected onto something in the world, it observes the same stone at some camera-pixel location, at some depth (from RealSense).

That gives us a triple per stone:

```
(projector_pixel_xy, camera_pixel_xy, depth_at_that_camera_pixel)
```

Converting `(camera_pixel_xy, depth)` → 3D world point via `K_cam`, each stone gives us:

```
(world_3D_point, projector_pixel_xy)
```

A few hundred stones across the calibration capture → one call to `solvePnP` → `R`, `T`, and refined `K_proj`. The pattern doesn't just help with 2D correspondence — it's the engine of the entire 3D calibration.

## Hidden-grid layout (why it matters for decoding)

In `SPEC.md` we said the cobblestones sit on a hidden 8×6 grid even though they look irregular. Two reasons:

1. **Fast decoding.** Each stone's cell is at a known projector-pixel range. The decoder knows roughly where to look for each stone.
2. **Color encoding is grid-indexed.** Stone at `(row=3, col=5)` gets a specific color from the De Bruijn-style palette. The decoder sees a color, looks it up, and instantly knows which cell of the grid it just decoded — without having to figure out the layout from scratch.

Without the hidden grid, decoding a freeform cobblestone field is closer to a SLAM problem — slow and brittle.

## Edge cases we'll have to handle

### Stones falling on holes in the depth map
RealSense gets confused by glossy surfaces, dark surfaces, and small features. Some stones will land on pixels with depth = 0 (unknown). We discard those from the calibration solve — there's enough redundancy in a few hundred stones that losing 10% doesn't matter.

### Stones falling on the projector's own shadow
The projector can't paint where it can't see. If the chair occludes some scene area from the projector, those stones don't appear. The decoder skips them automatically — they just don't show up in the camera frame.

### Rig drift over time
Vibration, thermal expansion, someone bumping the table — these can shift the camera-projector geometry by millimeters. Plan: a "calibration drift detector" — every few minutes, briefly flash a sparse subset of stones (say 12 of them) during the effect pass. Verify they land where the calibration predicts. If drift > 5 px, prompt the user to recalibrate. Cheap insurance.

### Per-frame depth changes
The 3D calibration is per-rig (one-time). Per-frame work is just the reprojection lookup. So if the chair moves, no recalibration needed — the reprojection automatically follows because depth changes per frame.

## Implementation order

For LiveTracking:

1. **Tonight (camera-space only):** segment chair, render effect in camera coordinates, validate pipeline on monitor.
2. **Tuesday + projector arrives:** mount D455 rigidly to projector (gaffer tape is fine for v1). Capture intrinsics for projector via factory specs as a starting guess.
3. **Tuesday evening:** run cobblestone calibration capture → solvePnP → save `calibration.json` with `R`, `T`, `K_proj`.
4. **Wednesday:** add the per-frame reprojection step. Same pipeline as tonight, but the output is reprojected to projector space before display.
5. **Later:** drift detector, multi-projector support.

## Practical recommendation for the JMGO mount

- The JMGO has 3-in-1 lens shift + AI gimbal. The lens center moves when the gimbal moves. **The calibration is only valid for one gimbal position.** Lock the gimbal before calibrating, or re-calibrate after each adjustment.
- Mount the D455 directly on top of the projector body, centered over the lens horizontally. The smaller the physical baseline, the less the depth-dependent parallax to begin with.
- A simple bracket from a cold shoe + tripod-thread adapter would work. Cost: ~$20 on Amazon, or 3D-print one in a couple hours.
- For tonight / pre-Tuesday: gaffer tape and a steady surface is sufficient to validate the calibration math.

## TL;DR

- Parallax is depth-dependent. Single-homography solutions only work for flat surfaces, not 3D objects.
- Solution: full 3D calibration (intrinsics + extrinsics) + per-frame reprojection using RealSense depth.
- The cobblestone pattern gives us all the data we need in one 3-second capture.
- Runtime cost is one matrix multiply per pixel — trivial on GPU.
- Practical hygiene: rigid mount, lock the gimbal, re-calibrate after any physical change, periodic drift checks.

# Next Milestone — General Object Selection

## Product goal

Turn LiveTracking from a post-it calibration demo into a general projector-area object selection tool.

The next demo should let Ted put arbitrary objects inside the projector rectangle — guitar, hand, box, bottle, book, prop — and have the system outline all selectable objects, label them, let Ted cycle through them, and project an effect onto the selected object.

## Why this replaces the post-it-first approach

The closed-loop post-it demo proved that the camera/projector feedback loop works, but it is not the right main loop for the product. It is slow because it searches target-by-target by projecting a mark, capturing a camera diff, correcting, and repeating.

The product needs a fast live loop:

```text
calibrate once -> crop to projector area -> segment all objects in one pass -> track IDs -> project outlines/effects
```

## Success criteria

- Detect 1-10 physical objects inside the projector rectangle.
- Update object list at 15 FPS minimum; target 30-60 FPS on PC-5090.
- Give each object a persistent ID while it remains visible.
- Project outlines and small labels onto detected objects.
- `Tab` cycles selected object.
- `Shift+Tab` cycles backward.
- `Enter` locks selected object.
- `E` cycles effect.
- `R` rescans objects.
- `C` recalibrates projector/camera mapping.
- Selected object receives a projected fire/glow/color effect.
- Works without assuming post-it notes.

## Primary algorithm: depth-first segmentation

Default live object discovery should use RealSense depth, not projector-camera closed-loop search.

1. Capture RGB + depth.
2. Restrict analysis to the calibrated projector rectangle.
3. Compare current depth against a captured background/wall depth model.
4. Keep pixels closer than the background by `min_depth_delta_mm`.
5. Clean mask using morphology.
6. Run connected components.
7. Filter by area and bounding-box size.
8. Track objects frame-to-frame using centroid, bounding box overlap, area, and depth.

This should be fast because it detects all candidate objects in one frame, rather than searching for each target through the projector.

## Secondary algorithm: flat-surface visual detection

Flat things stuck to the wall — post-it notes, labels, stickers, paper, tape — have little or no depth separation. They need a slower RGB/edge/color path.

Treat this as a fallback/debug mode, not the main product loop.

## Object UI

Project back onto the scene:

- thin contour around every detected object
- small object number near centroid
- pulsing/thicker outline for selected object
- effect only on selected object mask

## Recommended module layout

```text
src/livetracking/
  capture/
    realsense_capture.py
  calibration/
    projector_roi.py
    homography.py
    background_model.py
  segmentation/
    depth_segmenter.py
    rgb_flat_segmenter.py
    object_tracker.py
  rendering/
    projector_overlay.py
    effects.py
  app/
    live_select.py
```

## Immediate implementation plan

1. Add depth segmentation module.
2. Add object tracker module.
3. Add diagnostic script for RealSense + display sanity checks.
4. Add config defaults for object discovery.
5. Keep post-it closed-loop code as a calibration/debug experiment.
6. Build `python -m livetracking.app.live_select` as the new MVP entry point.

## Definition of done for this milestone

A successful run should look like this:

1. Launch app.
2. Press `B` to capture empty background.
3. Put several objects into projector field.
4. App outlines all objects and labels them 1-N.
5. Press `Tab` to cycle selection.
6. Selected object gets a different projected highlight.
7. Press `E` to cycle effects.
8. App stays interactive at 15+ FPS.

"""Calibration submodule for LiveTracking.

Working algorithms:

2026-05-24 night session (dark room, fixed exposure):
- closed_loop_search: per-target iterative search that finds the projector
  position for each detected target by closed-loop diff measurement. The
  algorithm that finally landed 3/3 + signs on 3 post-its with the Kodak.
- edge_detect: Otsu + Canny detection of projection rectangle + post-its.
  Works in dark rooms with high contrast between projection and wall.

2026-05-25 morning session (daylight + ambient):
- diff_detect: ambient-independent post-it detection via projector
  differencing. Don't look at the camera frame directly - look at what
  the projector adds to it. The detector your eye uses.

Key learnings:
- Use threshold=30 on the diff (not 15) so scene-wide noise doesn't connect
  into one giant fake blob via morphology.
- Lock RealSense auto-exposure (rs.option.enable_auto_exposure=0,
  rs.option.exposure=150) so the camera doesn't compensate for projector
  brightness changes between baseline and lit frames.
- DO NOT rely on absolute pixel thresholds for detection. Otsu, Canny,
  fixed saturation cutoffs - all brittle to ambient. The morning sun broke
  every one of these. Use what the projector ADDS to the scene
  (differencing), then find the brightest patches inside that added light.
  This is what your eye does and what survives any lighting.
- Future direction (Ted's idea, 2026-05-25): project the camera frame back
  onto the wall and minimize image-registration error via gradient descent.
  Yields a continuous projector<->camera homography that eliminates
  per-target detection entirely. Likely the right v2 architecture.
"""

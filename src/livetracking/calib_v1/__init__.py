"""Calibration submodule for LiveTracking.

Working algorithms from the 2026-05-24 night session:
- closed_loop_search: per-target iterative search that finds the projector
  position for each detected target by closed-loop diff measurement. The
  algorithm that finally landed 3/3 + signs on 3 post-its with the Kodak.
- edge_detect: edge-based detection of the projection rectangle + post-its.

Key learnings encoded:
- Use threshold=30 on the diff (not 15) so scene-wide noise doesn't connect
  into one giant fake blob via morphology.
- Lock RealSense auto-exposure (rs.option.enable_auto_exposure=0,
  rs.option.exposure=150) so the camera doesn't compensate for projector
  brightness changes between baseline and lit frames.
"""

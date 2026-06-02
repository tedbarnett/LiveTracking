"""Pure-numpy depth-probe helpers.

Extracted from scripts/calibrate_parallax.py so the probing math can be
unit-tested without pyrealsense2 / pygame in the import chain.

A "centerbox" is a 2*half x 2*half pixel window at the camera frame center.
We use it during parallax NEAR-calibration to read the depth of whatever
the operator placed at the camera crosshair.

Two probes:
    - median_depth_in_centerbox: robust to noise but gets swamped when the
      foreground target is smaller than the box (the bodhran-on-couch case).
    - nearest_depth_in_centerbox: nearest-decile percentile, picks up small
      foreground objects without falling for true single-pixel noise spikes.

Both return 0.0 when the centerbox has too few valid pixels (depth==0 means
"invalid" in RealSense's convention).
"""
from __future__ import annotations

import numpy as np

# Minimum valid-pixel count before we trust the probe; below this we return
# 0.0 instead of a noisy single-pixel reading.
_MIN_VALID_PIXELS = 50


def _centerbox(depth_m: np.ndarray, half: int) -> np.ndarray:
    """Return the valid (depth > 0) values inside a 2*half centerbox."""
    h, w = depth_m.shape[:2]
    cx, cy = w // 2, h // 2
    win = depth_m[max(0, cy - half):cy + half,
                  max(0, cx - half):cx + half]
    return win[win > 0]


def median_depth_in_centerbox(depth_m: np.ndarray, half: int = 60) -> float:
    """Median depth (meters) in a 2*half centerbox. 0.0 if too few valid px."""
    vals = _centerbox(depth_m, half)
    if vals.size < _MIN_VALID_PIXELS:
        return 0.0
    return float(np.median(vals))


def nearest_depth_in_centerbox(depth_m: np.ndarray, half: int = 60,
                                pct: float = 10.0) -> float:
    """`pct`-th percentile of valid depths in a 2*half centerbox.

    Use this when the operator places a NEAR target in front of a deeper
    background: the median is pulled toward whichever surface owns more
    pixels, so a small foreground object centered in the box gets ignored.
    Sampling the nearest decile picks up the foreground without being
    dominated by a single noisy spike (which a true min() would catch).

    Args:
        depth_m: HxW float32 depth map in meters (0 = invalid).
        half:    Half-width of the centerbox in pixels.
        pct:     Percentile to return (0-100). 10.0 -> nearest 10%.
                 0.0 -> true minimum (use with caution: D455 returns noise
                 spikes at 0.01-0.1 m even when nothing is that close).

    Returns:
        Depth in meters, or 0.0 if fewer than _MIN_VALID_PIXELS valid pixels
        in the centerbox.
    """
    vals = _centerbox(depth_m, half)
    if vals.size < _MIN_VALID_PIXELS:
        return 0.0
    return float(np.percentile(vals, pct))

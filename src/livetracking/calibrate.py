"""Calibration: structured-light cobblestone decoder.

Phase 0 stub. We project the calibration cobblestone for ~3 seconds,
capture frames, and write a placeholder homography to disk.

Phase 1 replaces this with a real per-depth-slice solver:
  for each captured frame, locate each stone's centroid in camera coords
  by color-matching to the encoded palette, then RANSAC a homography
  between projector grid coords and camera image coords.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np


DEFAULT_HOMOGRAPHY_PATH = Path("calibration/homography.json")
CALIBRATION_DURATION_S = 3.0
TARGET_FRAMES = 90


@dataclass
class CalibrationResult:
    duration_s: float
    frames_captured: int
    target_frames: int
    homography: List[List[float]]   # 3x3
    placeholder: bool
    timings_ms: dict


def _identity_homography() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def run_calibration(
    capture_fn: Callable[[], object],
    save_path: Path = DEFAULT_HOMOGRAPHY_PATH,
    duration_s: float = CALIBRATION_DURATION_S,
    on_progress: Optional[Callable[[float, int], None]] = None,
) -> CalibrationResult:
    """Drive a calibration cycle.

    capture_fn: a callable returning the latest captured frame (Frame instance).
                We only need it to consume frames — the caller is responsible
                for projecting the calibration pattern during the cycle.
    """
    t_start = time.perf_counter()
    frames: list = []
    t_first = None
    t_last = None
    target_end = t_start + duration_s
    while time.perf_counter() < target_end:
        frame = capture_fn()
        if frame is None:
            continue
        if t_first is None:
            t_first = time.perf_counter()
        t_last = time.perf_counter()
        frames.append(frame)
        if on_progress is not None:
            elapsed = time.perf_counter() - t_start
            on_progress(elapsed / duration_s, len(frames))

    t_capture_end = time.perf_counter()

    # Phase 0: placeholder homography (identity). Phase 1 replaces with solver.
    H = _identity_homography()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "homography": H.tolist(),
        "placeholder": True,
        "frames_captured": len(frames),
        "captured_at": time.time(),
    }
    save_path.write_text(json.dumps(payload, indent=2))

    t_save_end = time.perf_counter()

    timings_ms = {
        "capture_ms": (t_capture_end - t_start) * 1000.0,
        "save_ms": (t_save_end - t_capture_end) * 1000.0,
        "total_ms": (t_save_end - t_start) * 1000.0,
        "frame_interval_ms": ((t_last - t_first) / max(1, len(frames) - 1)) * 1000.0
            if len(frames) > 1 and t_first is not None and t_last is not None else 0.0,
    }

    print(f"[calibrate] captured {len(frames)} frames in "
          f"{timings_ms['capture_ms']:.1f} ms "
          f"(target {TARGET_FRAMES}, duration {duration_s:.1f}s)")
    print(f"[calibrate] frame interval: {timings_ms['frame_interval_ms']:.1f} ms "
          f"-> {1000.0 / max(0.001, timings_ms['frame_interval_ms']):.1f} fps")
    print(f"[calibrate] wrote placeholder homography to {save_path}")

    return CalibrationResult(
        duration_s=duration_s,
        frames_captured=len(frames),
        target_frames=TARGET_FRAMES,
        homography=H.tolist(),
        placeholder=True,
        timings_ms=timings_ms,
    )


def load_homography(path: Path = DEFAULT_HOMOGRAPHY_PATH) -> np.ndarray:
    if not path.exists():
        return _identity_homography()
    data = json.loads(path.read_text())
    return np.asarray(data["homography"], dtype=np.float64)

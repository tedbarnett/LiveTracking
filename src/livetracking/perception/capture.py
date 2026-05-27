"""RealSense D455 capture wrapper.

Acquires aligned (color BGR, depth meters) frames with a manual exposure lock.
Exposure lock is non-negotiable for projection-mapping work: auto-exposure
breaks every differencing-based detector when the projector fires bright
light. See ``computer-vision/projection-mapping`` skill, rule #4.

Only ONE process on this host can hold the D455. Close the Intel RealSense
Viewer / Windows Camera app first if init fails.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from livetracking.paths import RS_EXPOSURE, RS_FPS, RS_HEIGHT, RS_WIDTH


@dataclass
class Frame:
    color: np.ndarray   # (H, W, 3) uint8 BGR
    depth_m: np.ndarray # (H, W)    float32 meters (0 = invalid)
    timestamp: float


class RealSenseCapture:
    """Aligned color + depth from a D455 with manual exposure lock."""

    def __init__(
        self,
        width: int = RS_WIDTH,
        height: int = RS_HEIGHT,
        fps: int = RS_FPS,
        exposure: int = RS_EXPOSURE,
        serial: Optional[str] = None,
        warmup_frames: int = 10,
    ):
        import pyrealsense2 as rs  # local import — heavy DLL load

        self.width = width
        self.height = height
        self.fps = fps
        self._rs = rs

        ctx = rs.context()
        devices = list(ctx.query_devices())
        if not devices:
            raise RuntimeError(
                "No RealSense device found. Plug in the D455 and close any "
                "process that may be holding it (RealSense Viewer, Camera app)."
            )

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self._pipeline.start(cfg)

        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())
        self._align = rs.align(rs.stream.color)

        # Manual exposure on the color sensor (sensors[1] is RGB on D455).
        try:
            sensor = profile.get_device().query_sensors()[1]
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, exposure)
        except Exception as e:
            print(f"[capture] exposure lock failed (continuing): {e}")

        for _ in range(warmup_frames):
            self._pipeline.wait_for_frames()

    def read(self) -> Frame:
        frames = self._align.process(self._pipeline.wait_for_frames())
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        color = np.asanyarray(cf.get_data())
        depth_m = np.asanyarray(df.get_data()).astype(np.float32) * self._depth_scale
        return Frame(color=color, depth_m=depth_m, timestamp=time.time())

    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    def __enter__(self) -> "RealSenseCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

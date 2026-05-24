"""Frame capture: RealSense D455 with webcam + synthetic fallback."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Frame:
    color: np.ndarray          # (H, W, 3) uint8 BGR
    depth_m: np.ndarray        # (H, W)    float32 meters (0 = no data)
    timestamp: float           # seconds since epoch
    source: str                # "realsense" | "webcam" | "synthetic"


class FrameSource:
    """Unified camera interface. Tries RealSense first, then webcam, then synthetic."""

    def __init__(
        self,
        prefer_realsense: bool = True,
        width: int = 848,
        height: int = 480,
        fps: int = 30,
        serial: Optional[str] = None,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.source = "synthetic"
        self._pipeline = None
        self._align = None
        self._depth_scale = 1.0
        self._cap = None
        self._t0 = time.time()

        if prefer_realsense and self._try_realsense(serial):
            self.source = "realsense"
            return
        if self._try_webcam():
            self.source = "webcam"
            return
        # synthetic fallback always succeeds

    def _try_realsense(self, serial: Optional[str]) -> bool:
        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError:
            return False
        try:
            ctx = rs.context()
            devices = list(ctx.query_devices())
            if not devices:
                return False
            pipeline = rs.pipeline()
            config = rs.config()
            if serial:
                config.enable_device(serial)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            profile = pipeline.start(config)
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())
            self._align = rs.align(rs.stream.color)
            self._pipeline = pipeline
            self._rs = rs
            return True
        except Exception as e:
            print(f"[capture] RealSense init failed: {e}")
            try:
                if self._pipeline is not None:
                    self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
            return False

    def _try_webcam(self) -> bool:
        try:
            import cv2
        except ImportError:
            return False
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        ok, _ = cap.read()
        if not ok:
            cap.release()
            return False
        self._cap = cap
        return True

    def read(self) -> Frame:
        if self.source == "realsense":
            return self._read_realsense()
        if self.source == "webcam":
            return self._read_webcam()
        return self._read_synthetic()

    def _read_realsense(self) -> Frame:
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        color = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * self._depth_scale
        return Frame(color=color, depth_m=depth_m, timestamp=time.time(), source="realsense")

    def _read_webcam(self) -> Frame:
        ok, color = self._cap.read()
        if not ok:
            return self._read_synthetic()
        if color.shape[1] != self.width or color.shape[0] != self.height:
            import cv2
            color = cv2.resize(color, (self.width, self.height))
        depth_m = self._fake_depth_from_color(color)
        return Frame(color=color, depth_m=depth_m, timestamp=time.time(), source="webcam")

    def _read_synthetic(self) -> Frame:
        t = time.time() - self._t0
        h, w = self.height, self.width
        # Cache the static gradient + grid once; per-frame work is just
        # stamping moving "objects" onto a copy.
        if getattr(self, "_synth_bg", None) is None:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            r = (xx / w * 200 + 30).astype(np.uint8)
            g = (yy / h * 200 + 30).astype(np.uint8)
            b = np.full_like(r, 80)
            self._synth_bg = np.stack([b, g, r], axis=-1)
            self._synth_xx = xx
            self._synth_yy = yy
        color = self._synth_bg.copy()
        depth_m = np.full((h, w), 5.0, dtype=np.float32)
        # two moving "objects" — small enough that boolean masking is cheap.
        cx1 = int(w * 0.35 + math.sin(t * 0.7) * w * 0.1)
        cy1 = int(h * 0.5 + math.cos(t * 0.5) * h * 0.1)
        cx2 = int(w * 0.7 + math.sin(t * 1.1) * w * 0.05)
        cy2 = int(h * 0.4 + math.cos(t * 0.9) * h * 0.08)
        for (cx, cy, rad, dist, col) in [
            (cx1, cy1, 90, 1.2, (240, 220, 200)),
            (cx2, cy2, 60, 0.9, (180, 230, 240)),
        ]:
            # Local bbox only — much faster than whole-image distance check.
            x0 = max(0, cx - rad); x1 = min(w, cx + rad)
            y0 = max(0, cy - rad); y1 = min(h, cy + rad)
            if x1 <= x0 or y1 <= y0:
                continue
            xs = self._synth_xx[y0:y1, x0:x1]
            ys = self._synth_yy[y0:y1, x0:x1]
            mask = (xs - cx) ** 2 + (ys - cy) ** 2 < rad * rad
            depth_m[y0:y1, x0:x1][mask] = dist
            color[y0:y1, x0:x1][mask] = col
        return Frame(color=color, depth_m=depth_m, timestamp=time.time(), source="synthetic")

    @staticmethod
    def _fake_depth_from_color(color: np.ndarray) -> np.ndarray:
        # Crude depth heuristic for webcam: darker = closer.
        import cv2
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return 0.5 + (1.0 - gray) * 2.5  # 0.5..3.0m

    def close(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

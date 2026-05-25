"""LiveTracking hardware diagnostics.

Run from the repository root:

    python scripts/diagnose_hardware.py

This script checks whether OpenCV, pyrealsense2, and the D455 are available. If a
RealSense camera is connected, it captures a few frames and writes diagnostic PNGs
to diagnostics/latest/.
"""

from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np


def main() -> int:
    output_dir = Path("diagnostics/latest")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("LiveTracking hardware diagnostics")
    print(f"OpenCV: {cv2.__version__}")

    try:
        import pyrealsense2 as rs
    except Exception as exc:  # pragma: no cover - depends on local hardware
        print(f"pyrealsense2 import failed: {exc}")
        print("Install Intel RealSense SDK 2.0 and pyrealsense2, or use webcam fallback mode.")
        return 1

    context = rs.context()
    devices = context.query_devices()
    print(f"RealSense devices: {len(devices)}")
    for index, device in enumerate(devices):
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        firmware = device.get_info(rs.camera_info.firmware_version)
        print(f"  {index}: {name} serial={serial} firmware={firmware}")

    if len(devices) == 0:
        print("No RealSense camera detected.")
        return 2

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)

    profile = pipeline.start(config)
    colorizer = rs.colorizer()

    try:
        # Try to lock exposure. Some cameras/sensors may not support every option.
        for sensor in profile.get_device().query_sensors():
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                print(f"Locked auto exposure off for {sensor.get_info(rs.camera_info.name)}")

        start = time.perf_counter()
        frames_seen = 0
        color_image = None
        depth_image = None
        depth_preview = None

        while frames_seen < 90:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            frames_seen += 1
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_preview = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        elapsed = time.perf_counter() - start
        fps = frames_seen / elapsed if elapsed else 0.0
        print(f"Captured {frames_seen} frames in {elapsed:.2f}s = {fps:.1f} FPS")

        if color_image is not None:
            cv2.imwrite(str(output_dir / "color.png"), color_image)
        if depth_image is not None:
            np.save(output_dir / "depth_mm.npy", depth_image)
        if depth_preview is not None:
            cv2.imwrite(str(output_dir / "depth_preview.png"), depth_preview)

        print(f"Wrote diagnostics to {output_dir}")
        return 0
    finally:
        pipeline.stop()


if __name__ == "__main__":
    raise SystemExit(main())

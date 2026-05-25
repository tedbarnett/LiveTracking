"""Depth-first live object selection demo.

This is a fast MVP loop for general object discovery. It uses RealSense depth to
segment all foreground objects in one pass, assigns persistent IDs, and lets the
operator cycle selection with Tab.

Run:

    python -m livetracking.app.live_select_demo

Controls:

    B      capture background depth
    Tab    cycle selected object
    Esc    quit
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from livetracking.segmentation import DepthSegmenter, ObjectTracker


class RealSenseFrames:
    def __init__(self) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
        profile = self.pipeline.start(config)

        for sensor in profile.get_device().query_sensors():
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 0)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("Missing RealSense depth or color frame")
        depth_mm = np.asanyarray(depth_frame.get_data())
        color_bgr = np.asanyarray(color_frame.get_data())
        return color_bgr, depth_mm

    def stop(self) -> None:
        self.pipeline.stop()


def draw_tracks(image: np.ndarray, tracks) -> np.ndarray:
    overlay = image.copy()
    for track in tracks:
        x, y, w, h = track.bbox
        thickness = 4 if track.selected else 2
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), thickness)
        cx, cy = track.centroid
        cv2.putText(
            overlay,
            str(track.track_id),
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def main() -> int:
    capture = RealSenseFrames()
    segmenter = DepthSegmenter()
    tracker = ObjectTracker()
    background_ready = False
    last_time = time.perf_counter()
    fps = 0.0

    print("Controls: B=capture background, Tab=cycle object, Esc=quit")

    try:
        while True:
            color, depth = capture.read()
            if background_ready:
                detections = segmenter.segment(depth)
                tracks = tracker.update(detections)
            else:
                tracks = []

            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            view = draw_tracks(color, tracks)
            status = "B: capture background" if not background_ready else f"objects={len(tracks)} fps={fps:.1f}"
            cv2.putText(view, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.imshow("LiveTracking general object selection", view)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("b"), ord("B")):
                segmenter.set_background(depth)
                background_ready = True
                print("Captured background depth model")
            elif key == 9:
                selected = tracker.cycle_selection(1)
                if selected:
                    print(f"Selected object {selected.track_id}")

        return 0
    finally:
        capture.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())

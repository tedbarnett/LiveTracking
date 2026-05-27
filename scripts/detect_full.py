"""Full pipeline single-shot CLI.

Captures one (color, depth) frame, runs Stage 1 + DINO + SAM 2 + the tracker,
then dumps an annotated overlay showing the final per-object masks with
numbered colored outlines.

Output: scripts/out/full_overlay.png + scripts/out/full_summary.json
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from livetracking.paths import SCRIPT_OUT_DIR, describe
from livetracking.perception.capture import RealSenseCapture
from livetracking.perception.footprint import (
    footprint_corners_in_camera,
    load_homography,
)
from livetracking.perception.pipeline import Pipeline, PipelineConfig


def main() -> int:
    print(f"[full] config: {describe()}")
    H, meta = load_homography()
    PW, PH = int(meta["proj_w"]), int(meta["proj_h"])

    cap = RealSenseCapture()
    cw, ch = cap.size()
    try:
        for _ in range(5):
            cap.read()
        frame = cap.read()
    finally:
        cap.close()

    cfg = PipelineConfig(proj_w=PW, proj_h=PH)
    pipeline = Pipeline(H, cw, ch, cfg)
    objects = pipeline.step(frame.color, frame.depth_m)
    print(f"[full] timings: {pipeline.last_timings_ms}")
    print(f"[full] {len(objects)} tracked objects:")
    for o in objects:
        print(f"  #{o.object_id:2d} {o.name!r:25s} "
              f"score={o.label_score:.2f} depth={o.median_depth_m:.2f} m  "
              f"cam_centroid=({o.centroid_cam[0]:.0f},{o.centroid_cam[1]:.0f})  "
              f"proj_mask={'yes' if o.proj_mask is not None else 'NO'}")

    # ---- overlay ----
    fp_corners = footprint_corners_in_camera(H, PW, PH)
    overlay = frame.color.copy()
    outside = (pipeline.footprint == 0)
    overlay[outside] = (overlay[outside] * 0.45).astype(np.uint8)
    cv2.polylines(overlay, [fp_corners.astype(np.int32)], True, (0, 255, 255), 2)
    for o in objects:
        col = o.color_rgb
        contours, _ = cv2.findContours(o.cam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, col, 3)
        cx, cy = int(o.centroid_cam[0]), int(o.centroid_cam[1])
        cv2.circle(overlay, (cx, cy), 22, (0, 0, 0), -1)
        cv2.circle(overlay, (cx, cy), 22, col, 2)
        cv2.putText(overlay, str(o.object_id), (cx - 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(overlay, o.name, (cx + 26, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "full_overlay.png"), overlay)

    with open(os.path.join(SCRIPT_OUT_DIR, "full_summary.json"), "w") as f:
        json.dump({
            "timings_ms": pipeline.last_timings_ms,
            "objects": [
                {
                    "id": o.object_id,
                    "name": o.name,
                    "label_score": round(o.label_score, 3),
                    "depth_m": round(o.median_depth_m, 3),
                    "centroid_cam": list(o.centroid_cam),
                    "bbox_cam": list(o.bbox_cam),
                    "has_proj_mask": o.proj_mask is not None,
                } for o in objects
            ],
        }, f, indent=2)
    print(f"[full] wrote full_overlay.png + full_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

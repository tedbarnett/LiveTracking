"""Smoke-test: load SAM 2 + Grounding DINO on GPU and run them on the
latest captured Stage-1 image (scripts/out/det_color.png).

Verifies:
  - Both models load on CUDA (sm_120, RTX 5090).
  - SAM 2 returns a mask given a point prompt at each Stage-1 blob centroid.
  - Grounding DINO returns labels for a generic prompt.

Writes:
  - scripts/out/sam_overlay.png — SAM masks tinted onto the color image.
  - scripts/out/dino_overlay.png — DINO bboxes + labels drawn.
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

from livetracking.paths import SCRIPT_OUT_DIR
from livetracking.perception.recognize import DEFAULT_DINO_PROMPT, Recognizer
from livetracking.perception.types import Blob


def main() -> int:
    color_path = os.path.join(SCRIPT_OUT_DIR, "det_color.png")
    summary_path = os.path.join(SCRIPT_OUT_DIR, "det_summary.json")
    if not os.path.exists(color_path) or not os.path.exists(summary_path):
        print(f"[smoke] missing inputs — run scripts/detect_geometry.py first.")
        return 2

    color = cv2.imread(color_path)
    with open(summary_path) as f:
        summary = json.load(f)
    print(f"[smoke] image {color.shape}, {len(summary['blobs'])} Stage-1 blobs")

    rec = Recognizer()  # downloads checkpoints on first call

    # ---- Grounding DINO over the whole image ----
    dino = rec.label_image(color, prompt=DEFAULT_DINO_PROMPT)
    print(f"[smoke] DINO detected {len(dino)} candidates:")
    for d in dino:
        print(f"  - {d['label']!r:25s} score={d['score']:.2f}  bbox={d['bbox']}")

    dino_vis = color.copy()
    for d in dino:
        x, y, w, h = d["bbox"]
        cv2.rectangle(dino_vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(dino_vis, f"{d['label']} {d['score']:.2f}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "dino_overlay.png"), dino_vis)

    # ---- SAM 2 point-prompted by Stage-1 centroids ----
    fake_blobs = [
        Blob(
            blob_id=b["id"],
            cam_mask=np.zeros(color.shape[:2], np.uint8),
            centroid_cam=tuple(b["centroid_cam"]),
            bbox_cam=tuple(b["bbox_cam"]),
            area_px=b["area_px"],
            median_depth_m=b["median_depth_m"],
        )
        for b in summary["blobs"]
    ]
    sam_out = rec.segment_with_points(color, [b.centroid_cam for b in fake_blobs])
    print(f"[smoke] SAM2 returned {len(sam_out)} masks:")
    sam_vis = color.copy()
    palette = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80),
        (255, 80, 255), (80, 255, 255), (255, 160, 80), (160, 80, 255),
    ]
    for i, (mask, score) in enumerate(sam_out):
        col = palette[i % len(palette)]
        print(f"  blob #{fake_blobs[i].blob_id} centroid={fake_blobs[i].centroid_cam} "
              f"sam_score={score:.3f} mask_area={int((mask>0).sum())}")
        tint = np.zeros_like(sam_vis)
        tint[mask > 0] = col
        sam_vis = cv2.addWeighted(sam_vis, 1.0, tint, 0.45, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(sam_vis, contours, -1, col, 2)
        cx, cy = int(fake_blobs[i].centroid_cam[0]), int(fake_blobs[i].centroid_cam[1])
        cv2.circle(sam_vis, (cx, cy), 6, (255, 255, 255), -1)
    cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "sam_overlay.png"), sam_vis)
    print(f"[smoke] wrote sam_overlay.png + dino_overlay.png to {SCRIPT_OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

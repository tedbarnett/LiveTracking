"""Vision-truth check: run DINO on one fresh camera frame and dump labels."""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from livetracking.paths import SCRIPT_OUT_DIR
from livetracking.perception.capture import RealSenseCapture
from livetracking.perception.recognize import Recognizer, DEFAULT_DINO_PROMPT


def main():
    cap = RealSenseCapture()
    try:
        for _ in range(8):
            cap.read()
        shot = cap.read()
    finally:
        cap.close()
    img = shot.color
    cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "dino_probe_input.png"), img)

    rec = Recognizer()
    # Lower thresholds to see everything
    dets = rec.label_image(img, prompt=DEFAULT_DINO_PROMPT,
                           box_threshold=0.20, text_threshold=0.20)
    # Sort by score desc
    dets.sort(key=lambda d: -d["score"])
    print(f"[probe] DINO returned {len(dets)} detections")
    vis = img.copy()
    for i, d in enumerate(dets[:25]):
        x, y, w, h = d["bbox"]
        color = (0, 255, 0) if d["score"] > 0.4 else (0, 200, 200)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = f"{i}:{d['label'][:20]}={d['score']:.2f}"
        cv2.putText(vis, label, (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        print(f"  #{i}: score={d['score']:.3f} label={d['label']!r:35s} "
              f"bbox=({x},{y},{w},{h})")
    cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "dino_probe_dets.png"), vis)
    with open(os.path.join(SCRIPT_OUT_DIR, "dino_probe.json"), "w") as f:
        json.dump(dets, f, indent=2)
    print(f"[probe] wrote {SCRIPT_OUT_DIR}\\dino_probe_dets.png")


if __name__ == "__main__":
    main()

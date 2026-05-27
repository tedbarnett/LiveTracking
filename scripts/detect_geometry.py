"""Stage 1 CLI: single-shot foreground detection inside projector footprint.

Captures one (color, depth) frame, loads the saved homography, runs the
depth-plane Stage-1 detector, and dumps every diagnostic image to scripts/out/:

  det_color.png            — raw color frame
  det_depth_colormap.png   — colorized depth
  det_footprint_mask.png   — projector footprint in camera space
  det_wall_band.png        — band used for plane fit
  det_foreground_raw.png   — pre-morph foreground mask
  det_foreground_clean.png — post-morph foreground mask
  det_blobs_overlay.png    — numbered + outlined blobs over color (THE money image)
  det_summary.json         — blob list with id, centroid, area, depth

Run after `calibrate_homography.py`. The skill's diagnostic-dump rule:
every geometric stage writes a debug image so silent failures are visible.
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
    footprint_mask_in_camera,
    load_homography,
)
from livetracking.perception.geometry import GeometryParams, detect_blobs


# Approximate D455 color-stream intrinsics at 848x480 (close enough for
# plane-fit purposes; for production read these from rs.video_stream_profile).
INTRINSICS_848x480 = (615.0, 615.0, 424.0, 240.0)  # fx, fy, cx, cy


def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    """Convert metric depth to a viridis-ish 8-bit BGR image."""
    valid = depth_m > 0
    if not np.any(valid):
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth_m[valid], [2, 98])
    norm = np.clip((depth_m - lo) / max(1e-6, hi - lo), 0, 1)
    norm[~valid] = 0
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)


def draw_blobs_overlay(color: np.ndarray, blobs, footprint_mask: np.ndarray,
                       fp_corners: np.ndarray) -> np.ndarray:
    out = color.copy()
    # Tint pixels OUTSIDE the footprint darker so the rectangle is obvious
    outside = (footprint_mask == 0)
    out[outside] = (out[outside] * 0.45).astype(np.uint8)
    # Outline the footprint
    cv2.polylines(
        out, [fp_corners.astype(np.int32)], True, (0, 255, 255), 2
    )
    # Palette
    palette = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80),
        (255, 80, 255), (80, 255, 255), (255, 160, 80), (160, 80, 255),
    ]
    for b in blobs:
        col = palette[(b.blob_id - 1) % len(palette)]
        # contour
        contours, _ = cv2.findContours(b.cam_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, col, 3)
        # number at centroid
        cx, cy = int(b.centroid_cam[0]), int(b.centroid_cam[1])
        cv2.circle(out, (cx, cy), 22, (0, 0, 0), -1)
        cv2.circle(out, (cx, cy), 22, col, 2)
        cv2.putText(out, str(b.blob_id), (cx - 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def main() -> int:
    print(f"[detect] config: {describe()}")
    H, meta = load_homography()
    PW = int(meta.get("proj_w", 3840))
    PH = int(meta.get("proj_h", 2160))
    print(f"[detect] loaded H, projector {PW}x{PH}, {meta.get('n_correspondences')} dots")

    cap = RealSenseCapture()
    try:
        for _ in range(5):
            cap.read()
        frame = cap.read()
    finally:
        cap.close()

    cw = frame.color.shape[1]
    ch = frame.color.shape[0]
    fp_mask = footprint_mask_in_camera(H, PW, PH, cw, ch)
    fp_corners = footprint_corners_in_camera(H, PW, PH)
    print(f"[detect] footprint area = {int(fp_mask.sum())} px / {cw*ch} total "
          f"({100.0*fp_mask.sum()/(cw*ch):.1f}%)")

    blobs, dbg = detect_blobs(
        frame.color, frame.depth_m, fp_mask, INTRINSICS_848x480, GeometryParams()
    )
    plane = dbg["plane"]
    print(f"[detect] plane = [{plane[0]:.3f}, {plane[1]:.3f}, {plane[2]:.3f}, {plane[3]:.3f}]  "
          f"inlier_frac={dbg['inlier_frac']:.2f}")
    print(f"[detect] raw components = {dbg['n_components_raw']}, "
          f"surviving blobs = {len(blobs)}")
    for b in blobs:
        print(f"  #{b.blob_id} area={b.area_px:6d} px  "
              f"centroid=({b.centroid_cam[0]:.0f},{b.centroid_cam[1]:.0f})  "
              f"depth={b.median_depth_m:.2f} m")

    # ---- diagnostic dumps ----
    cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "det_color.png"), frame.color)
    cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "det_depth_colormap.png"),
                colorize_depth(frame.depth_m))
    cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "det_footprint_mask.png"), fp_mask * 255)
    cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "det_wall_band.png"),
                dbg["wall_band_mask"] * 255)
    cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "det_foreground_raw.png"),
                dbg["foreground_mask_raw"])
    cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "det_foreground_clean.png"),
                dbg["foreground_mask_clean"])
    overlay = draw_blobs_overlay(frame.color, blobs, fp_mask, fp_corners)
    cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "det_blobs_overlay.png"), overlay)

    summary = {
        "n_blobs": len(blobs),
        "plane": plane,
        "inlier_frac": dbg["inlier_frac"],
        "blobs": [
            {
                "id": b.blob_id,
                "centroid_cam": list(b.centroid_cam),
                "bbox_cam": list(b.bbox_cam),
                "area_px": b.area_px,
                "median_depth_m": round(b.median_depth_m, 3),
            } for b in blobs
        ],
    }
    with open(os.path.join(SCRIPT_OUT_DIR, "det_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[detect] wrote diagnostic PNGs and det_summary.json to {SCRIPT_OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

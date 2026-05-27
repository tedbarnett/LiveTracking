"""Camera <-> projector homography calibration via projected white dots.

Projects a 3x3 grid of bright white dots at known projector positions, finds
each in the camera via a baseline diff, and solves for H with RANSAC.

Writes:
  - runtime/calibration/H.npy    (3x3 homography, camera px -> projector px)
  - runtime/calibration/calib.json
  - scripts/out/calib_dots.png   (visualization of detected dot centers)
  - scripts/out/calib_baseline.png

Settings come from `computer-vision/projection-mapping` skill:
  - exposure locked at LIVETRACKING_RS_EXPOSURE (default 700)
  - diff threshold > 30
  - reject dots within 8 px of the camera frame edge
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

# Allow running this script directly (without `pip install -e .`).
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from livetracking.paths import DISPLAY_INDEX, SCRIPT_OUT_DIR, describe
from livetracking.perception.capture import RealSenseCapture
from livetracking.perception.footprint import save_homography


# Dot radius in *projector* pixels. At 3840x2160 a radius of 180 gives a dot
# ~5% of frame width, which is reliably detectable in the camera even with
# room lighting and the projector throwing color. Older code used 70 — too
# small at 4K.
DOT_RADIUS_PROJ_PX = 180
DOT_GRID = [(fx, fy) for fy in (0.2, 0.5, 0.8) for fx in (0.2, 0.5, 0.8)]
DIFF_THRESHOLD = 30
EDGE_REJECT_PX = 8


def pick_projector_display(pygame):
    """Pick the projector display by index env override or biggest-display heuristic."""
    pygame.display.init()
    sizes = pygame.display.get_desktop_sizes()
    if not sizes:
        raise RuntimeError("pygame found 0 displays.")
    if DISPLAY_INDEX is not None:
        if not (0 <= DISPLAY_INDEX < len(sizes)):
            raise RuntimeError(
                f"LIVETRACKING_DISPLAY_INDEX={DISPLAY_INDEX} out of range "
                f"(found {len(sizes)} display(s))."
            )
        idx = DISPLAY_INDEX
    else:
        idx = max(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1])
    return idx, sizes[idx]


def main() -> int:
    print(f"[calib] config: {describe()}")
    import pygame
    pygame.init()
    pi, (PW, PH) = pick_projector_display(pygame)
    print(f"[calib] projector = display {pi}, {PW}x{PH}")
    screen = pygame.display.set_mode((PW, PH), pygame.NOFRAME, display=pi)

    def show_dot(xy):
        screen.fill((0, 0, 0))
        if xy is not None:
            pygame.draw.circle(screen, (255, 255, 255),
                               (int(xy[0]), int(xy[1])), DOT_RADIUS_PROJ_PX)
        pygame.display.flip()
        for _ in pygame.event.get():
            pass

    cap = RealSenseCapture()
    cw, ch = cap.size()
    try:
        # baseline (black)
        show_dot(None)
        time.sleep(0.8)
        for _ in range(5):
            cap.read()
        baseline = cap.read()
        gblk = cv2.cvtColor(baseline.color, cv2.COLOR_BGR2GRAY).astype(np.int16)
        cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "calib_baseline.png"), baseline.color)

        cam_pts, proj_pts = [], []
        dotvis = baseline.color.copy()
        for i, (fx_frac, fy_frac) in enumerate(DOT_GRID):
            proj_xy = (fx_frac * PW, fy_frac * PH)
            show_dot(proj_xy)
            time.sleep(0.25)
            for _ in range(3):
                cap.read()
            shot = cap.read()
            d = cv2.cvtColor(shot.color, cv2.COLOR_BGR2GRAY).astype(np.int16) - gblk
            d = np.clip(d, 0, 255).astype(np.uint8)
            d = cv2.GaussianBlur(d, (0, 0), 5)
            _, mx, _, loc = cv2.minMaxLoc(d)
            lx, ly = int(loc[0]), int(loc[1])
            mx_i = int(mx)
            on_border = (
                lx < EDGE_REJECT_PX or ly < EDGE_REJECT_PX
                or lx > cw - EDGE_REJECT_PX or ly > ch - EDGE_REJECT_PX
            )
            ok = (mx_i > DIFF_THRESHOLD) and not on_border
            tag = "OK " if ok else "MISS"
            print(f"[calib] dot {i} proj=({proj_xy[0]:.0f},{proj_xy[1]:.0f}) "
                  f"maxdiff={mx_i:3d} cam=({lx},{ly}) {tag}")
            if ok:
                cam_pts.append([lx, ly])
                proj_pts.append([proj_xy[0], proj_xy[1]])
                cv2.circle(dotvis, (lx, ly), 8, (0, 255, 0), 2)
                cv2.putText(dotvis, str(i), (lx + 10, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        show_dot(None)

        if len(cam_pts) < 4:
            print(f"[calib] FAILED — only {len(cam_pts)} dots detected, need >= 4.")
            cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "calib_dots.png"), dotvis)
            return 2

        H, inliers = cv2.findHomography(
            np.array(cam_pts, np.float32),
            np.array(proj_pts, np.float32),
            cv2.RANSAC, 15.0,    # 15 px tolerates parallax across wall vs sofa cushion
        )
        n_in = int(inliers.sum()) if inliers is not None else 0
        det = float(np.linalg.det(H[:2, :2]))
        print(f"[calib] H solved with {n_in}/{len(cam_pts)} RANSAC inliers, "
              f"det(H[:2,:2])={det:.4f}")
        if abs(det) < 0.01:
            print("[calib] WARNING: near-degenerate homography (det too small).")

        save_homography(
            H,
            proj_size=(PW, PH),
            cam_size=(cw, ch),
            n_correspondences=len(cam_pts),
            extra={
                "ransac_inliers": n_in,
                "det_2x2": round(det, 4),
                "dot_grid": [list(p) for p in DOT_GRID],
            },
        )
        cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "calib_dots.png"), dotvis)
        print(f"[calib] saved homography + meta + {os.path.join(SCRIPT_OUT_DIR, 'calib_dots.png')}")
        return 0
    finally:
        cap.close()
        try:
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

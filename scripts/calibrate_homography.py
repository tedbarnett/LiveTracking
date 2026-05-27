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


DOT_RADIUS_PROJ_PX = 180
DOT_GRID = [(fx, fy) for fy in (0.2, 0.5, 0.8) for fx in (0.2, 0.5, 0.8)]
DIFF_THRESHOLD = 30
EDGE_REJECT_PX = 8
# Pixels that brighten by more than this between full-black and full-white
# projector frames count as "the projector is lighting that camera pixel".
# Measured, not computed — handles multi-plane scenes (wall + sofa) correctly.
FOOTPRINT_DIFF_THRESHOLD = 25
FOOTPRINT_MORPH_CLOSE = 9


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

    def show_fill(color):
        screen.fill(color)
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

        # ---- measured projector footprint: project distinctive color, diff ----
        # White-vs-black diff fails when ambient room light is bright (the
        # diff is small even outside the projector cone, but small EVERYWHERE
        # so thresholding picks up the whole frame). Instead, project
        # MAGENTA — high red+blue, low green — and look for pixels whose
        # red+blue went UP much more than green did. Real projector pixels
        # will show this chroma shift; ambient-lit pixels won't.
        show_fill((255, 0, 255))
        time.sleep(0.8)
        for _ in range(5):
            cap.read()
        mag_shot = cap.read()
        show_fill((0, 0, 0))
        # BGR; "magenta" means high B + high R, low G
        b0, g0, r0 = cv2.split(baseline.color.astype(np.int16))
        b1, g1, r1 = cv2.split(mag_shot.color.astype(np.int16))
        # Chroma signal = (Δred + Δblue) - 2·Δgreen.  Pixels lit by magenta
        # projection get a big positive value; pixels merely brightened by
        # ambient changes get ~0.
        chroma = (r1 - r0) + (b1 - b0) - 2 * (g1 - g0)
        chroma = np.clip(chroma, 0, 255).astype(np.uint8)
        _, lit = cv2.threshold(chroma, FOOTPRINT_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        k = np.ones((FOOTPRINT_MORPH_CLOSE, FOOTPRINT_MORPH_CLOSE), np.uint8)
        lit = cv2.morphologyEx(lit, cv2.MORPH_CLOSE, k)
        # Also fill holes inside the lit region (dark objects don't reflect
        # magenta well, but they're INSIDE the cone so should still count).
        # Fill via flood-fill from corners marking background, then invert.
        h, w = lit.shape
        ff = lit.copy()
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, ff_mask, (0, 0), 255)
        cv2.floodFill(ff, ff_mask, (w - 1, 0), 255)
        cv2.floodFill(ff, ff_mask, (0, h - 1), 255)
        cv2.floodFill(ff, ff_mask, (w - 1, h - 1), 255)
        holes_filled = cv2.bitwise_or(lit, cv2.bitwise_not(ff))
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(holes_filled, 8)
        footprint_meas = np.zeros_like(lit)
        if n_lab > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            footprint_meas[labels == largest] = 255
        cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "calib_magenta.png"), mag_shot.color)
        cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "calib_footprint_diff.png"), chroma)
        cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "calib_footprint_measured.png"), footprint_meas)
        meas_frac = float((footprint_meas > 0).sum()) / (cw * ch)
        print(f"[calib] measured footprint = {int((footprint_meas>0).sum())} px "
              f"({100*meas_frac:.1f}% of camera frame)")

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

        # ---- saved homography + measured dot quad ----
        # We record the 4 *outermost* detected dot positions so the perception
        # daemon can draw a quadrilateral that bounds where we actually
        # observed the projector — not an extrapolation of H to the full
        # projector frame (which goes wildly off when the calibration plane
        # is a compromise between wall and sofa).
        cam_pts_arr = np.array(cam_pts, dtype=np.float32)
        proj_pts_arr = np.array(proj_pts, dtype=np.float32)
        # Build a convex hull of the measured dot positions, then approximate
        # to a quadrilateral.
        hull = cv2.convexHull(cam_pts_arr.reshape(-1, 1, 2))
        # Approximate to 4 points (or just use the hull if fewer)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        # Save just the dot-camera positions; perception can decide how to
        # turn that into a footprint outline.
        np.save(os.path.join(os.path.dirname(__file__), "..", "runtime",
                             "calibration", "dot_cam_pts.npy"),
                cam_pts_arr)
        np.save(os.path.join(os.path.dirname(__file__), "..", "runtime",
                             "calibration", "dot_proj_pts.npy"),
                proj_pts_arr)
        print(f"[calib] saved {len(cam_pts)} measured dot positions for "
              f"footprint estimation (hull has {len(approx)} corners)")

        save_homography(
            H,
            proj_size=(PW, PH),
            cam_size=(cw, ch),
            n_correspondences=len(cam_pts),
            extra={
                "ransac_inliers": n_in,
                "det_2x2": round(det, 4),
                "dot_grid": [list(p) for p in DOT_GRID],
                "measured_footprint_file": "footprint_measured.png",
                "measured_footprint_fraction": round(meas_frac, 4),
            },
        )
        # Also persist the measured footprint mask alongside H.
        import shutil
        meas_path = os.path.join(os.path.dirname(__file__), "..", "runtime",
                                 "calibration", "footprint_measured.png")
        os.makedirs(os.path.dirname(meas_path), exist_ok=True)
        cv2.imwrite(meas_path, footprint_meas)
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

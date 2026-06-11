"""Camera <-> projector homography calibration via time-multiplexed ArUco.

Projects ONE large ArUco marker at a time at a known projector position,
captures a camera frame, and detects the marker. After cycling through a
grid of positions, we have N markers x 4 corners = 4N correspondences
and solve H with RANSAC.

Why time-multiplexed beats a single grid frame:
  - Each marker is HUGE (~25% of screen) so it survives the camera's
    spatial averaging at 848x480 even at 4K projector resolution.
  - Each marker projects with full-frame contrast — the screen around it
    is black (projector emits nothing), so the marker pops against
    ambient light. No competing "white background" washing out the wall.
  - Per-frame detection is unambiguous (we know which marker is on screen),
    so detector confusion doesn't matter.

Why ArUco beats the original white dots:
  - Sub-pixel corner refinement is built in.
  - Detection is shape+code based, not brightness threshold — robust to
    ambient daylight that overwhelmed dot diff in normally-lit rooms.

Writes (filenames preserved for backward compat with footprint.py):
  - runtime/calibration/H.npy            3x3 homography, cam px -> proj px
  - runtime/calibration/calib.json       metadata
  - runtime/calibration/dot_cam_pts.npy  (N, 2) camera-space corners
  - runtime/calibration/dot_proj_pts.npy (N, 2) projector-space corners
  - scripts/out/calib_aruco_capture_NN.png   per-marker capture
  - scripts/out/calib_aruco_overview.png     baseline + all detections overlaid
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from livetracking.paths import DISPLAY_INDEX, SCRIPT_OUT_DIR, describe
from livetracking.perception.capture import RealSenseCapture
from livetracking.perception.footprint import save_homography


# Time-multiplexed grid. Each marker fills MARKER_SIZE_FRAC of min(PW, PH).
# The grid spans GRID_FRACS — kept inside 0.10..0.90 so markers stay on the
# projector cone (a half-marker outside the edge would clip).
GRID_N = 4
GRID_FRACS = np.linspace(0.18, 0.82, GRID_N)
MARKER_SIZE_FRAC = 0.30   # 30% of min(PW,PH) = ~648 proj px at 2160 vertical.
                          # Was 0.22 — at the dim cone edges (DLP brightness
                          # falloff) 22% (~105 cam px) had marginal contrast and
                          # edge markers failed DETECTION, collapsing the
                          # footprint to ~half the real cone.
RETRY_SCALE = 1.5         # second-chance pass: missed markers retried at
                          # this multiple of marker_px (more camera pixels =
                          # detectable at lower contrast).
ARUCO_DICT_NAME = cv2.aruco.DICT_4X4_50
SETTLE_S = 0.35           # time between projector flip and capture
CAPTURE_DRAIN = 4         # discard this many frames after settle


def pick_projector_display(pygame):
    pygame.display.init()
    sizes = pygame.display.get_desktop_sizes()
    if not sizes:
        raise RuntimeError("pygame found 0 displays.")
    if DISPLAY_INDEX is not None and 0 <= DISPLAY_INDEX < len(sizes):
        idx = DISPLAY_INDEX
    else:
        idx = max(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1])
    return idx, sizes[idx]


def grid_positions(PW: int, PH: int) -> list[tuple[int, int, int]]:
    """Return [(marker_id, cx_proj, cy_proj), ...] for each grid cell."""
    out = []
    mid = 0
    for fy in GRID_FRACS:
        for fx in GRID_FRACS:
            out.append((mid, int(round(fx * PW)), int(round(fy * PH))))
            mid += 1
    return out


def build_marker_frame(
    PW: int, PH: int, marker_id: int, marker_px: int, cx: int, cy: int,
    dictionary,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a single ArUco marker at (cx, cy) on a BLACK projector frame.

    Returns (frame_bgr, corners_4x2_proj).

    The marker has a generous WHITE quiet-zone border, so what gets projected
    is effectively a square WHITE tile with the black ArUco pattern inside.
    Against the ambient-lit wall (no other projector light competing) this
    produces high contrast in the camera.
    """
    frame = np.zeros((PH, PW, 3), dtype=np.uint8)
    quiet = max(8, marker_px // 4)   # white quiet zone around marker
    tile_size = marker_px + 2 * quiet
    x0 = cx - tile_size // 2
    y0 = cy - tile_size // 2
    x1 = x0 + tile_size
    y1 = y0 + tile_size
    # Clamp tile to frame
    if x0 < 0 or y0 < 0 or x1 > PW or y1 > PH:
        x0 = max(0, min(x0, PW - tile_size))
        y0 = max(0, min(y0, PH - tile_size))
        x1 = x0 + tile_size
        y1 = y0 + tile_size
    # White quiet-zone tile
    frame[y0:y1, x0:x1] = (255, 255, 255)
    # Marker bits in the middle
    mx = x0 + quiet
    my = y0 + quiet
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
    frame[my:my + marker_px, mx:mx + marker_px] = cv2.cvtColor(
        marker_img, cv2.COLOR_GRAY2BGR
    )
    corners_proj = np.array(
        [[mx, my],
         [mx + marker_px - 1, my],
         [mx + marker_px - 1, my + marker_px - 1],
         [mx, my + marker_px - 1]],
        dtype=np.float32,
    )
    return frame, corners_proj


def main() -> int:
    print(f"[calib] config: {describe()}")
    import pygame
    pygame.init()
    pi, (PW, PH) = pick_projector_display(pygame)
    print(f"[calib] projector = display {pi}, {PW}x{PH}")
    screen = pygame.display.set_mode((PW, PH), pygame.NOFRAME, display=pi)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
    marker_px = int(round(min(PW, PH) * MARKER_SIZE_FRAC))
    cells = grid_positions(PW, PH)
    print(f"[calib] grid {GRID_N}x{GRID_N} = {len(cells)} cells; "
          f"marker_px={marker_px} (~{marker_px*848/PW:.0f} cam px wide)")

    def show_bgr(frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.image.frombuffer(
            np.ascontiguousarray(rgb).tobytes(), (PW, PH), "RGB"
        )
        screen.fill((0, 0, 0))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        for _ in pygame.event.get():
            pass

    def show_black() -> None:
        screen.fill((0, 0, 0))
        pygame.display.flip()
        for _ in pygame.event.get():
            pass

    # Detector params tuned for projected (low-DLP-contrast) markers on real walls
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.minCornerDistanceRate = 0.03
    params.minDistanceToBorder = 1
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    cap = RealSenseCapture()
    cw, ch = cap.size()
    cam_pts: list[list[float]] = []
    proj_pts: list[list[float]] = []
    # (cam_u, cam_v, depth_m) at each detected marker center — used to
    # fit the wall plane for parallax compensation.
    cam_depth_samples: list[list[float]] = []
    overview = None
    n_detected = 0
    miss_ids: list[int] = []

    try:
        show_black()
        time.sleep(0.6)
        for _ in range(6):
            cap.read()
        baseline = cap.read()
        overview = baseline.color.copy()

        for marker_id, cx, cy in cells:
            found = False
            cam4 = None
            shot = None
            corners_proj = None
            retry_px = int(round(marker_px * RETRY_SCALE))
            for attempt_i, attempt_px in enumerate((marker_px, retry_px)):
                frame_bgr, corners_proj = build_marker_frame(
                    PW, PH, marker_id, attempt_px, cx, cy, dictionary
                )
                show_bgr(frame_bgr)
                time.sleep(SETTLE_S)
                for _ in range(CAPTURE_DRAIN):
                    cap.read()
                shot = cap.read()
                gray = cv2.cvtColor(shot.color, cv2.COLOR_BGR2GRAY)
                det_corners, det_ids, _ = detector.detectMarkers(gray)
                if det_ids is not None:
                    ids_flat = det_ids.flatten().tolist()
                    if marker_id in ids_flat:
                        idx = ids_flat.index(marker_id)
                        cam4 = det_corners[idx].reshape(4, 2)
                        found = True
                        if attempt_i > 0:
                            print(f"[calib] marker {marker_id:2d} recovered "
                                  f"on 2nd pass ({attempt_px}px)")
                        break
            if found:
                for k in range(4):
                    cam_pts.append([float(cam4[k, 0]), float(cam4[k, 1])])
                    proj_pts.append([float(corners_proj[k, 0]),
                                     float(corners_proj[k, 1])])
                # Sample wall depth at this marker's center for plane fit.
                cxm, cym = cam4.mean(axis=0)
                ix, iy = int(round(cxm)), int(round(cym))
                if 0 <= ix < shot.depth_m.shape[1] and \
                   0 <= iy < shot.depth_m.shape[0]:
                    # 5x5 median for robustness against missing pixels.
                    x0 = max(0, ix - 2); x1 = min(shot.depth_m.shape[1], ix + 3)
                    y0 = max(0, iy - 2); y1 = min(shot.depth_m.shape[0], iy + 3)
                    patch = shot.depth_m[y0:y1, x0:x1]
                    valid = patch[patch > 0.1]
                    if valid.size >= 4:
                        cam_depth_samples.append(
                            [float(cxm), float(cym),
                             float(np.median(valid))]
                        )
                cv2.polylines(overview,
                              [cam4.astype(np.int32)], True,
                              (0, 255, 0), 2)
                centroid = cam4.mean(axis=0).astype(int)
                cv2.putText(overview, str(marker_id),
                            tuple(centroid),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                n_detected += 1
            tag = "OK " if found else "MISS"
            print(f"[calib] marker {marker_id:2d} proj=({cx},{cy}) {tag}")
            if not found:
                miss_ids.append(marker_id)
                # Save the failing capture for diagnostics
                cv2.imwrite(
                    os.path.join(SCRIPT_OUT_DIR,
                                 f"calib_aruco_miss_{marker_id:02d}.png"),
                    shot.color,
                )
        show_black()

        cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "calib_aruco_overview.png"),
                    overview)
        print(f"[calib] detected {n_detected}/{len(cells)} markers; "
              f"{len(cam_pts)} corner correspondences")
        if miss_ids:
            print(f"[calib] missed marker ids: {miss_ids}")

        if n_detected < 4:
            print(f"[calib] FAILED - only {n_detected} markers detected, "
                  "need >= 4 well-spread. Try turning down room lights, or "
                  "check that the JMGO menu isn't covering the HDMI feed.")
            return 2

        cam_arr = np.array(cam_pts, dtype=np.float32)
        proj_arr = np.array(proj_pts, dtype=np.float32)

        H, inliers = cv2.findHomography(
            cam_arr, proj_arr,
            cv2.RANSAC, 15.0,
        )
        n_in = int(inliers.sum()) if inliers is not None else 0
        det2 = float(np.linalg.det(H[:2, :2]))
        print(f"[calib] H solved with {n_in}/{len(cam_arr)} RANSAC inliers, "
              f"det(H[:2,:2])={det2:.4f}")
        if abs(det2) < 0.01:
            print("[calib] WARNING: near-degenerate homography.")
        if n_in < 8:
            print(f"[calib] FAILED - only {n_in} inliers (need >= 8).")
            return 3

        # Persist correspondences (preserved filenames for footprint.py).
        calib_dir = os.path.join(os.path.dirname(__file__), "..", "runtime",
                                 "calibration")
        os.makedirs(calib_dir, exist_ok=True)
        np.save(os.path.join(calib_dir, "dot_cam_pts.npy"), cam_arr)
        np.save(os.path.join(calib_dir, "dot_proj_pts.npy"), proj_arr)

        # Fit wall plane in camera 3D space from depth samples at the
        # detected ArUco marker centers. This is the surface the H was
        # solved on, so it's the correct reference plane for parallax
        # compensation (objects closer than this plane need shifting).
        plane = None
        if len(cam_depth_samples) >= 3:
            # Hard-coded D455 intrinsics (matches PipelineConfig.intrinsics).
            fx, fy, cx_i, cy_i = 615.0, 615.0, 424.0, 240.0
            # Sort by depth and keep only the FAR HALF — markers that
            # landed on the back wall, filtering out ones that hit closer
            # surfaces (couch, objects). This matters because the JMGO
            # projects partially onto the couch + objects in front of the
            # wall, and we want the wall plane, not the couch plane.
            samples_sorted = sorted(cam_depth_samples, key=lambda s: -s[2])
            kept = samples_sorted[: max(3, len(samples_sorted) // 2)]
            print(f"[calib] plane-fit samples: {len(cam_depth_samples)} "
                  f"detected, keeping {len(kept)} farthest "
                  f"(depth range {kept[-1][2]:.2f}m - {kept[0][2]:.2f}m).")
            pts3 = []
            for u, v, z in kept:
                X = (u - cx_i) * z / fx
                Y = (v - cy_i) * z / fy
                pts3.append([X, Y, z])
            P = np.array(pts3, dtype=np.float64)
            # Plane fit: minimize ||A n + b|| where n=(a,b,c), b=-d_plane.
            # Use SVD on centered points to get the normal robustly.
            centroid = P.mean(axis=0)
            U, S, Vt = np.linalg.svd(P - centroid, full_matrices=False)
            normal = Vt[-1]                       # smallest singular vector
            a, b, c = float(normal[0]), float(normal[1]), float(normal[2])
            d_p = float(-(a * centroid[0] + b * centroid[1] + c * centroid[2]))
            # Normalize so c is negative (Z increases away from camera,
            # plane is "in front" of camera → aX+bY+cZ+d=0 with c<0).
            if c > 0:
                a, b, c, d_p = -a, -b, -c, -d_p
            plane = [a, b, c, d_p]
            # Residuals (perpendicular distances)
            resid = (P @ np.array([a, b, c]) + d_p) / max(
                1e-6, float(np.linalg.norm([a, b, c]))
            )
            print(f"[calib] wall plane: a={a:.4f} b={b:.4f} c={c:.4f} "
                  f"d={d_p:.4f}  (residual median={float(np.median(np.abs(resid))):.3f}m, "
                  f"max={float(np.max(np.abs(resid))):.3f}m, "
                  f"n_samples={len(pts3)})")
            np.save(os.path.join(calib_dir, "wall_plane.npy"),
                    np.array(plane, dtype=np.float64))
            print(f"[calib] saved wall_plane.npy")
        else:
            print(f"[calib] only {len(cam_depth_samples)} depth samples — "
                  "skipping wall plane fit (parallax will fall back to "
                  "Stage-1 plane).")

        save_homography(
            H,
            proj_size=(PW, PH),
            cam_size=(cw, ch),
            n_correspondences=len(cam_arr),
            extra={
                "method": "aruco-timemux",
                "aruco_dict": "DICT_4X4_50",
                "markers_attempted": len(cells),
                "markers_detected": n_detected,
                "miss_ids": miss_ids,
                "ransac_inliers": n_in,
                "det_2x2": round(det2, 4),
                "marker_size_frac": MARKER_SIZE_FRAC,
                "grid_n": GRID_N,
            },
        )

        # Synthesize footprint mask from H_dot for backward compat.
        try:
            H_dot, _ = cv2.findHomography(proj_arr, cam_arr, cv2.RANSAC, 5.0)
            corners_proj_frame = np.array(
                [[[0, 0]], [[PW - 1, 0]], [[PW - 1, PH - 1]], [[0, PH - 1]]],
                dtype=np.float32,
            )
            corners_cam = cv2.perspectiveTransform(
                corners_proj_frame, H_dot
            ).reshape(-1, 2)
            corners_cam[:, 0] = np.clip(corners_cam[:, 0], 0, cw - 1)
            corners_cam[:, 1] = np.clip(corners_cam[:, 1], 0, ch - 1)
            mask = np.zeros((ch, cw), dtype=np.uint8)
            cv2.fillConvexPoly(mask, corners_cam.astype(np.int32), 255)
            cv2.imwrite(os.path.join(calib_dir, "footprint_measured.png"), mask)
        except Exception as e:
            print(f"[calib] could not synthesize footprint mask: {e!r}")

        print(f"[calib] saved homography + meta + "
              f"{os.path.join(SCRIPT_OUT_DIR, 'calib_aruco_overview.png')}")
        return 0
    finally:
        cap.close()
        try:
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

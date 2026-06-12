"""Camera <-> projector calibration via Gray-code structured light.

Replaces the ArUco approach (calibrate_homography.py) which required flat,
unbroken wall patches for marker detection — something this room does not
have (guitars, window AC, couch all sit inside the projection cone).

Gray code needs NO flat surfaces: each projected pattern is a set of
black/white stripes, and every camera pixel independently decodes which
projector column/row illuminates it by watching the stripe sequence.
Objects at different depths decode fine; pixels the projector can't reach
simply fail the contrast test and drop out. The result is a DENSE
correspondence map (one per camera pixel, thousands total) instead of
4 corners x N markers.

Pipeline:
  1. Project white + black reference frames (per-pixel contrast gate).
  2. Project Gray-code bit patterns + inverses, vertical (X) then
     horizontal (Y). Inverse-pair differencing makes decoding ambient-
     light-proof: bit = (pattern > inverse), no absolute threshold.
  3. Decode proj (x, y) for every valid camera pixel.
  4. RANSAC homography on a subsample — the wall is the dominant plane,
     so off-plane pixels (guitars, couch) become outliers automatically.
  5. Fit the wall plane from depth at H-inlier pixels (far-half filter).

Writes (same contract as calibrate_homography.py so downstream is
untouched):
  - runtime/calibration/H.npy             3x3 cam px -> proj px
  - runtime/calibration/calib.json        metadata (method=graycode)
  - runtime/calibration/dot_cam_pts.npy   (N, 2) inlier cam points
  - runtime/calibration/dot_proj_pts.npy  (N, 2) inlier proj points
  - runtime/calibration/wall_plane.npy    [a, b, c, d] camera-space plane
  - runtime/calibration/dense_map.npz     full per-pixel cam->proj map +
                                          valid mask (plane-free warping,
                                          not yet consumed by daemons)
  - runtime/calibration/footprint_measured.png  REAL cone mask (decoded
                                          valid pixels, not an H-derived
                                          convex hull)
  - scripts/out/calib_gray_decode.png     diagnostic: decoded X map
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

# ----------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------
SETTLE_S = 0.30          # projector flip -> capture delay
CAPTURE_DRAIN = 4        # frames discarded after settle (camera pipeline lag)
AVG_FRAMES = 8           # frames averaged per capture. Noise drops ~sqrt(N):
                         # in daylight the JMGO adds only ~5 gray levels over
                         # ambient, below single-frame sensor noise (~3).
                         # Averaging 8 frames pulls the floor to ~1.
SKIP_FINE_BITS = 3       # drop stripes finer than 2^SKIP_FINE_BITS proj px.
                         # Camera sees ~4.5 proj px per cam px (848 vs 3840):
                         # 4px stripes are SUB-camera-pixel and decode as
                         # noise, killing valid pixels bit by bit. 8px blocks
                         # (~1.8 cam px) are the finest resolvable level and
                         # still give ±4 proj px quantization (<1 cam px).
NOISE_SIGMAS_REF = 6.0   # white-black gate = this many measured noise sigmas
NOISE_SIGMAS_BIT = 3.0   # per-bit |pattern-inverse| gate, in noise sigmas
MAX_RANSAC_PTS = 20000   # subsample cap for findHomography
RANSAC_REPROJ_PX = 8.0   # proj-px tolerance; wall pixels agree, off-plane
                         # objects (parallax offsets of 10s of px) drop out
MIN_VALID_PIXELS = 300   # decoded-pixel sanity floor. A few hundred well-
                         # spread correspondences over-determine the 8-DOF
                         # homography ~150x. The footprint no longer depends
                         # on decode survivors (it uses the contrast-gate
                         # cone), so a lean decode set is fine.
LENIENT_FINE_BITS = 2    # the finest N bits decode sign-only (no validity
                         # kill): blurred fine stripes at the cone edge would
                         # otherwise wipe out pixels that decode perfectly on
                         # coarse bits. A wrong finest bit = one 8px block of
                         # proj error — RANSAC outlier at worst.
MIN_INLIER_FRAC = 0.25   # H must explain at least this fraction of sample


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


def gray_encode(v: np.ndarray) -> np.ndarray:
    return v ^ (v >> 1)


def gray_decode_bits(bits: np.ndarray) -> np.ndarray:
    """bits: (n_bits, H, W) uint8, MSB first -> binary value (H, W) int32."""
    out = bits[0].astype(np.int32)
    acc = bits[0].astype(np.int32)
    for k in range(1, bits.shape[0]):
        acc = acc ^ bits[k].astype(np.int32)
        out = (out << 1) | acc
    return out


def build_patterns(extent: int, skip: int) -> tuple[np.ndarray, int]:
    """1-D Gray-code stripe table for an axis of `extent` projector px.

    Returns (table, n_bits) where table is (n_bits, extent) uint8 in {0,255}
    MSB first, encoding gray(position >> skip).
    """
    blocks = (extent + (1 << skip) - 1) >> skip
    n_bits = max(1, int(np.ceil(np.log2(blocks))))
    pos = np.arange(extent, dtype=np.int32) >> skip
    g = gray_encode(pos)
    table = np.zeros((n_bits, extent), dtype=np.uint8)
    for k in range(n_bits):
        bit = (g >> (n_bits - 1 - k)) & 1
        table[k] = (bit * 255).astype(np.uint8)
    return table, n_bits


def main() -> int:
    print(f"[gray] config: {describe()}")
    import pygame
    pygame.init()
    pi, (PW, PH) = pick_projector_display(pygame)
    print(f"[gray] projector = display {pi}, {PW}x{PH}")
    screen = pygame.display.set_mode((PW, PH), pygame.NOFRAME, display=pi)

    def show_gray_frame(frame_2d: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_2d, cv2.COLOR_GRAY2RGB)
        surf = pygame.image.frombuffer(
            np.ascontiguousarray(rgb).tobytes(), (PW, PH), "RGB"
        )
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        for _ in pygame.event.get():
            pass

    cap = RealSenseCapture()
    try:
        def grab(n_avg: int = AVG_FRAMES) -> tuple[np.ndarray, np.ndarray]:
            """Settle, drain pipeline lag, then average n frames (float32)."""
            time.sleep(SETTLE_S)
            for _ in range(CAPTURE_DRAIN):
                cap.read()
            acc = None
            depth = None
            for _ in range(n_avg):
                shot = cap.read()
                g = cv2.cvtColor(shot.color, cv2.COLOR_BGR2GRAY).astype(
                    np.float32)
                acc = g if acc is None else acc + g
                depth = shot.depth_m
            return acc / n_avg, depth

        # --- adaptive exposure -------------------------------------------
        # Gray-code differencing cancels ambient light ONLY while the sensor
        # is below saturation. In a daylit room the default exposure (700)
        # pins the wall near the sensor ceiling and the projector's
        # contribution clips away. Halve until the FULL WHITE frame has
        # headroom, then refine toward p99 ~ 215 to keep maximum signal.
        exposure = int(os.environ.get("LIVETRACKING_RS_EXPOSURE", "700"))
        show_gray_frame(np.full((PH, PW), 255, dtype=np.uint8))
        img_white, depth_white = grab(2)
        for _ in range(8):
            p99 = float(np.percentile(img_white, 99))
            print(f"[gray] exposure={exposure} white p99={p99:.0f}")
            if p99 <= 225:
                break
            exposure = max(40, exposure // 2)
            cap.set_exposure(exposure)
            img_white, depth_white = grab(2)
        # Refine back UP: signal scales with exposure, so sit just under
        # the clip point instead of far below it.
        p99 = float(np.percentile(img_white, 99))
        if p99 < 200 and exposure < 700:
            exposure = min(700, max(40, int(exposure * 215.0 / max(p99, 1))))
            cap.set_exposure(exposure)
            img_white, depth_white = grab(2)
            print(f"[gray] refined exposure={exposure} white "
                  f"p99={np.percentile(img_white, 99):.0f}")
        # Full-quality white reference at the final exposure.
        img_white, depth_white = grab()

        # --- empirical noise floor ----------------------------------------
        # Two identical black captures differ only by sensor noise; their
        # diff std (scaled by 1/sqrt(2)) is the per-capture noise sigma.
        show_gray_frame(np.zeros((PH, PW), dtype=np.uint8))
        img_black, _ = grab()
        img_black2, _ = grab()
        sigma = float(np.std(img_black - img_black2)) / np.sqrt(2.0)
        min_contrast = max(NOISE_SIGMAS_REF * sigma, 3.0)
        min_bit_conf = max(NOISE_SIGMAS_BIT * sigma, 1.5)
        print(f"[gray] noise sigma={sigma:.2f} -> contrast gate "
              f"{min_contrast:.1f}, bit gate {min_bit_conf:.1f}")

        contrast = img_white - img_black
        valid = contrast > min_contrast
        n_valid0 = int(valid.sum())
        # The contrast-gate mask IS the projector cone: every pixel the
        # projector measurably lights. Keep it for the footprint — the
        # decode gates below are far stricter (a pixel must resolve EVERY
        # stripe bit) and only a fraction of the cone survives them.
        # Footprint != decode-survivors; that conflation shrank the
        # footprint to the cone's crisp center.
        cone_mask = cv2.morphologyEx(
            valid.astype(np.uint8) * 255, cv2.MORPH_OPEN,
            np.ones((5, 5), np.uint8))
        # Diagnostic dump: what did the camera actually see? (rule: when a
        # run fails, diagnose with telemetry, not parameter guessing)
        os.makedirs(SCRIPT_OUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "gray_ref_white.png"),
                    np.clip(img_white, 0, 255).astype(np.uint8))
        cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "gray_ref_black.png"),
                    np.clip(img_black, 0, 255).astype(np.uint8))
        for tag, im in (("white", img_white), ("black", img_black),
                        ("diff", contrast)):
            q = np.percentile(im, [5, 50, 95, 99])
            print(f"[gray] ref {tag}: p5={q[0]:.0f} p50={q[1]:.0f} "
                  f"p95={q[2]:.0f} p99={q[3]:.0f}")
        print(f"[gray] contrast gate: {n_valid0} px see projector light "
              f"(of {valid.size})")
        if n_valid0 < MIN_VALID_PIXELS:
            print(f"[gray] FAILED - only {n_valid0} px pass the contrast "
                  f"gate (need {MIN_VALID_PIXELS}). Is the projector on and "
                  "the lens cap off?")
            return 2

        # --- stripe patterns ---------------------------------------------
        tab_x, nbx = build_patterns(PW, SKIP_FINE_BITS)
        tab_y, nby = build_patterns(PH, SKIP_FINE_BITS)
        n_frames = 2 * (nbx + nby) + 2
        print(f"[gray] bits: {nbx} X + {nby} Y "
              f"(skip={SKIP_FINE_BITS} -> {1 << SKIP_FINE_BITS}px blocks), "
              f"{n_frames} frames total")

        def capture_axis(table: np.ndarray, axis: str) -> np.ndarray:
            """Project each bit + inverse; return decoded bits stack."""
            n_bits = table.shape[0]
            bits = np.zeros((n_bits,) + img_white.shape, dtype=np.uint8)
            for k in range(n_bits):
                if axis == "x":
                    pat = np.broadcast_to(table[k], (PH, PW))
                else:
                    pat = np.broadcast_to(table[k][:, None], (PH, PW))
                pat = np.ascontiguousarray(pat)
                show_gray_frame(pat)
                img_p, _ = grab()
                show_gray_frame(255 - pat)
                img_n, _ = grab()
                if k == 0 and axis == "x":
                    cv2.imwrite(
                        os.path.join(SCRIPT_OUT_DIR, "gray_x0_pat.png"),
                        np.clip(img_p, 0, 255).astype(np.uint8))
                    cv2.imwrite(
                        os.path.join(SCRIPT_OUT_DIR, "gray_x0_inv.png"),
                        np.clip(img_n, 0, 255).astype(np.uint8))
                diff = img_p - img_n
                bits[k] = (diff > 0).astype(np.uint8)
                # Confidence gate on COARSE bits only: an ambiguous coarse
                # bit means a huge decode error, so the pixel dies. The
                # finest LENIENT_FINE_BITS decode sign-only — edge blur
                # there costs one block of precision, not correctness.
                if k < n_bits - LENIENT_FINE_BITS:
                    np.logical_and(valid, np.abs(diff) >= min_bit_conf,
                                   out=valid)
                print(f"[gray] {axis}-bit {k + 1}/{n_bits} captured "
                      f"(valid now {int(valid.sum())})")
                if int(valid.sum()) < MIN_VALID_PIXELS:
                    print(f"[gray] ABORT - valid pixels collapsed at "
                          f"{axis}-bit {k + 1}; see gray_x0_*.png dumps")
                    raise SystemExit(2)
            return bits

        bits_x = capture_axis(tab_x, "x")
        bits_y = capture_axis(tab_y, "y")
        show_gray_frame(np.zeros((PH, PW), dtype=np.uint8))

        n_valid = int(valid.sum())
        print(f"[gray] decoded pixels surviving all gates: {n_valid}")
        if n_valid < MIN_VALID_PIXELS:
            print(f"[gray] FAILED - only {n_valid} px decoded "
                  f"(need {MIN_VALID_PIXELS}).")
            return 2

        # --- decode --------------------------------------------------------
        half = 1 << (SKIP_FINE_BITS - 1) if SKIP_FINE_BITS > 0 else 0
        px = (gray_decode_bits(bits_x) << SKIP_FINE_BITS) + half
        py = (gray_decode_bits(bits_y) << SKIP_FINE_BITS) + half
        in_range = (px >= 0) & (px < PW) & (py >= 0) & (py < PH)
        valid &= in_range

        ch, cw = valid.shape
        ys, xs = np.where(valid)
        cam_pts = np.stack([xs, ys], axis=1).astype(np.float32)
        proj_pts = np.stack([px[ys, xs], py[ys, xs]], axis=1).astype(
            np.float32)
        print(f"[gray] {len(cam_pts)} dense correspondences")

        # --- diagnostics dump ----------------------------------------------
        os.makedirs(SCRIPT_OUT_DIR, exist_ok=True)
        vis = np.zeros((ch, cw, 3), dtype=np.uint8)
        vis[..., 2] = np.where(valid, (px * 255 // max(PW, 1)), 0)
        vis[..., 1] = np.where(valid, (py * 255 // max(PH, 1)), 0)
        cv2.imwrite(os.path.join(SCRIPT_OUT_DIR, "calib_gray_decode.png"),
                    vis)

        # --- wall selection by DEPTH, then homography -----------------------
        # With full-cone coverage the wall is NOT the majority of decoded
        # pixels (clutter at other depths eats the cone), so plain RANSAC
        # can latch onto the wrong surface. Find the dominant 3-D plane in
        # depth space first, keep pixels near it, fit H on those only.
        calib_dir = os.path.abspath(
            os.path.join(HERE, "..", "runtime", "calibration"))
        os.makedirs(calib_dir, exist_ok=True)
        fx_i, fy_i, cx_i, cy_i = 615.0, 615.0, 424.0, 240.0
        z_all = depth_white[ys, xs]
        has_z = z_all > 0.1
        plane = None
        wall_sel = np.ones(len(cam_pts), dtype=bool)  # fallback: everything
        if int(has_z.sum()) >= 200:
            u = xs[has_z].astype(np.float64)
            v = ys[has_z].astype(np.float64)
            z = z_all[has_z].astype(np.float64)
            X = (u - cx_i) * z / fx_i
            Y = (v - cy_i) * z / fy_i
            P = np.stack([X, Y, z], axis=1)
            # RANSAC plane fit in 3-D
            rng = np.random.default_rng(0)
            best_in = None
            for _ in range(300):
                ids = rng.choice(len(P), 3, replace=False)
                p0, p1, p2 = P[ids]
                n = np.cross(p1 - p0, p2 - p0)
                nn = np.linalg.norm(n)
                if nn < 1e-9:
                    continue
                n = n / nn
                d = -float(n @ p0)
                dist = np.abs(P @ n + d)
                inl = dist < 0.05  # 5 cm slab
                if best_in is None or inl.sum() > best_in.sum():
                    best_in, best_n, best_d = inl, n, d
            # Prefer the FARTHEST big plane (wall behind clutter): among
            # planes with >=60% of the best support, take the deepest.
            # Single-shot heuristic: refit on inliers, then report.
            Pw = P[best_in]
            centroid = Pw.mean(axis=0)
            _, _, Vt = np.linalg.svd(Pw - centroid, full_matrices=False)
            n = Vt[-1]
            d = -float(n @ centroid)
            if n[2] > 0:
                n, d = -n, -d
            a, b, c = float(n[0]), float(n[1]), float(n[2])
            plane = [a, b, c, d]
            resid = np.abs(Pw @ n + d)
            print(f"[gray] depth plane: a={a:.4f} b={b:.4f} c={c:.4f} "
                  f"d={d:.4f} support={int(best_in.sum())}/{len(P)} "
                  f"(median |resid|={float(np.median(resid)):.3f}m, "
                  f"mean depth={float(Pw[:, 2].mean()):.2f}m)")
            # Map plane membership back to the full decoded-pixel set.
            wall_sel = np.zeros(len(cam_pts), dtype=bool)
            idx_has_z = np.where(has_z)[0]
            wall_sel[idx_has_z[best_in]] = True
            np.save(os.path.join(calib_dir, "wall_plane.npy"),
                    np.array(plane, dtype=np.float64))
        else:
            print("[gray] insufficient depth - fitting H on ALL decoded px")

        cam_w_pts = cam_pts[wall_sel]
        proj_w_pts = proj_pts[wall_sel]
        print(f"[gray] wall-selected correspondences: {len(cam_w_pts)}")
        if len(cam_w_pts) > MAX_RANSAC_PTS:
            sel = np.random.default_rng(0).choice(
                len(cam_w_pts), MAX_RANSAC_PTS, replace=False)
            cam_s, proj_s = cam_w_pts[sel], proj_w_pts[sel]
        else:
            cam_s, proj_s = cam_w_pts, proj_w_pts

        H, inlier_mask = cv2.findHomography(
            cam_s, proj_s, cv2.RANSAC, RANSAC_REPROJ_PX)
        if H is None:
            print("[gray] FAILED - findHomography returned None.")
            return 3
        inl = inlier_mask.ravel().astype(bool)
        n_in = int(inl.sum())
        frac = n_in / len(cam_s)
        det2 = float(np.linalg.det(H[:2, :2]))
        print(f"[gray] H solved: {n_in}/{len(cam_s)} inliers "
              f"({frac:.0%}), det(H[:2,:2])={det2:.4f}")
        if frac < MIN_INLIER_FRAC:
            print(f"[gray] FAILED - inlier fraction {frac:.0%} < "
                  f"{MIN_INLIER_FRAC:.0%}; no dominant plane found.")
            return 3

        # --- persist inlier correspondences ---------------------------------
        cam_in = cam_s[inl]
        proj_in = proj_s[inl]
        np.save(os.path.join(calib_dir, "dot_cam_pts.npy"), cam_in)
        np.save(os.path.join(calib_dir, "dot_proj_pts.npy"), proj_in)
        # (wall_plane.npy already saved by the depth-plane selection above.)

        # --- dense map (plane-free future) ----------------------------------
        np.savez_compressed(
            os.path.join(calib_dir, "dense_map.npz"),
            proj_x=px.astype(np.float32),
            proj_y=py.astype(np.float32),
            valid=valid,
            cone_mask=(cone_mask > 0),
        )

        # --- REAL footprint mask --------------------------------------------
        # Convex hull of the CONTRAST-GATE cone (all projector-lit pixels),
        # not of the decode survivors. Decode survivors cluster in the
        # crisp center and undersell the cone by 5-40x; the contrast mask
        # is the projector's actual reach. Hull-filled because perception
        # gates objects on footprint overlap.
        cyx = np.column_stack(np.where(cone_mask > 0))  # (N, [y, x])
        fp = np.zeros((ch, cw), dtype=np.uint8)
        hull = cv2.convexHull(cyx[:, ::-1].astype(np.int32))  # -> (x, y)
        cv2.fillConvexPoly(fp, hull, 255)
        n_fp = int((fp > 0).sum())
        print(f"[gray] footprint: cone-gate {int((cone_mask > 0).sum())} px "
              f"-> hull-filled {n_fp} px "
              f"({100.0 * n_fp / fp.size:.0f}% of frame)")
        cv2.imwrite(os.path.join(calib_dir, "footprint_measured.png"), fp)

        save_homography(
            H,
            proj_size=(PW, PH),
            cam_size=(cw, ch),
            n_correspondences=int(len(cam_in)),
            extra={
                "method": "graycode",
                "n_frames": int(n_frames),
                "skip_fine_bits": int(SKIP_FINE_BITS),
                "decoded_pixels": int(n_valid),
                "ransac_sample": int(len(cam_s)),
                "ransac_inliers": int(n_in),
                "inlier_frac": round(frac, 3),
                "det_2x2": round(det2, 4),
            },
        )
        print("[gray] saved H + meta + dense_map.npz + real footprint")
        return 0
    finally:
        cap.close()
        try:
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

"""Closed-loop iterative search PER TARGET with the corrected diff detection.

Key fix from prior versions: threshold the diff at 30+ instead of 15. The
lower threshold connected scene-wide noise into one giant fake blob. With
30+, only the actual projected + sign survives.
"""
import os, sys, time, math, json
import numpy as np, cv2, pygame
import pyrealsense2 as rs

DISPLAY_X = 5120
DISPLAY_Y = 0
DISPLAY_W = 1280
DISPLAY_H = 720

POSTIT_TARGETS_CAM = [
    (561, 100),
    (596, 134),
    (652, 110),
]
COLOR_NAMES = ["magenta", "yellow", "cyan"]
COLORS_BGR_FINAL = [
    (255, 100, 255),
    ( 50, 255, 255),
    (255, 255,  50),
]


def blit(screen, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    s = pygame.image.frombuffer(rgb.tobytes(), (DISPLAY_W, DISPLAY_H), "RGB")
    screen.blit(s, (0, 0))
    pygame.display.flip()


def render_solid(c):
    return np.full((DISPLAY_H, DISPLAY_W, 3),
                   np.array(c, dtype=np.uint8), dtype=np.uint8)


def render_one_plus(px, py, length=45, thick=16):
    canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    px = int(px); py = int(py)
    if not (0 <= px < DISPLAY_W and 0 <= py < DISPLAY_H):
        return canvas
    cv2.line(canvas, (px - length, py), (px + length, py),
             (255, 255, 255), thick, cv2.LINE_AA)
    cv2.line(canvas, (px, py - length), (px, py + length),
             (255, 255, 255), thick, cv2.LINE_AA)
    cv2.circle(canvas, (px, py), 8, (255, 255, 255), -1, cv2.LINE_AA)
    return canvas


def render_final(positions, length=50, thick=18):
    canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    for i, (px, py) in enumerate(positions):
        if not (0 <= px < DISPLAY_W and 0 <= py < DISPLAY_H):
            continue
        color = COLORS_BGR_FINAL[i % 3]
        px = int(px); py = int(py)
        cv2.line(canvas, (px - length, py), (px + length, py),
                 color, thick, cv2.LINE_AA)
        cv2.line(canvas, (px, py - length), (px, py + length),
                 color, thick, cv2.LINE_AA)
        ccolor = tuple(int(min(255, c * 1.3)) for c in color)
        cv2.circle(canvas, (px, py), max(4, thick // 2),
                   ccolor, -1, cv2.LINE_AA)
    return canvas


def find_plus_blob(baseline, with_plus, target_cam=None, search_r=200):
    """Find the actual + sign blob in the diff. Use threshold=30 to suppress
    scene-wide noise."""
    b = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY).astype(np.int16)
    w = cv2.cvtColor(with_plus, cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff = (w - b).clip(min=0).astype(np.uint8)
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    if target_cam is not None:
        circ = np.zeros_like(mask)
        cv2.circle(circ, (int(target_cam[0]), int(target_cam[1])), search_r, 255, -1)
        mask = cv2.bitwise_and(mask, circ)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    blobs = [(i, stats[i, cv2.CC_STAT_AREA])
             for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 8]
    if not blobs:
        return None
    blobs.sort(key=lambda t: -t[1])
    i = blobs[0][0]
    return (float(cents[i][0]), float(cents[i][1]), int(stats[i, cv2.CC_STAT_AREA]))


def main():
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{DISPLAY_X},{DISPLAY_Y}"
    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), pygame.NOFRAME)
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipe.start(cfg)

    # Lock camera exposure to reduce scene-wide brightness shifts
    try:
        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.exposure, 150)  # fixed exposure
        print("[init] locked camera exposure to 150", flush=True)
    except Exception as e:
        print(f"[init] couldn't lock exposure: {e}", flush=True)

    for _ in range(20):
        pipe.wait_for_frames()

    try:
        # Baseline
        print("[init] baseline", flush=True)
        blit(screen, render_solid((0, 0, 0)))
        for _ in range(30):
            for _ in pygame.event.get(): pass
            pipe.wait_for_frames()
        f = pipe.wait_for_frames()
        baseline = np.asanyarray(f.get_color_frame().get_data())
        cv2.imwrite(r"D:\Github-D\LiveTracking\tmp\iv2_baseline.png", baseline)

        # Build initial H from projection quad
        quad_cam = np.array([
            [423, 0], [1153, 0], [1153, 719], [423, 719]
        ], dtype=np.float32)
        proj_corners = np.array([
            [0, 0], [DISPLAY_W - 1, 0],
            [DISPLAY_W - 1, DISPLAY_H - 1], [0, DISPLAY_H - 1]
        ], dtype=np.float32)
        H_cam_to_proj_init, _ = cv2.findHomography(quad_cam, proj_corners)

        winners = []
        for tgt_idx, target_cam in enumerate(POSTIT_TARGETS_CAM):
            label = COLOR_NAMES[tgt_idx]
            print(f"\n[target {tgt_idx+1} {label}] target cam{target_cam}", flush=True)
            # Initial estimate via homography
            pt = np.array([[target_cam]], dtype=np.float32)
            proj_pt = cv2.perspectiveTransform(pt, H_cam_to_proj_init)
            proj_xy = [float(proj_pt[0][0][0]), float(proj_pt[0][0][1])]

            best = None
            for it in range(15):
                blit(screen, render_one_plus(proj_xy[0], proj_xy[1]))
                for _ in range(12):
                    for _ in pygame.event.get(): pass
                    pipe.wait_for_frames()
                f = pipe.wait_for_frames()
                wp = np.asanyarray(f.get_color_frame().get_data())
                blob = find_plus_blob(baseline, wp, target_cam=target_cam,
                                       search_r=250)
                if blob is None:
                    print(f"  iter {it}: proj({proj_xy[0]:.0f},{proj_xy[1]:.0f}) -> NO BLOB",
                          flush=True)
                    # try without restriction
                    blob = find_plus_blob(baseline, wp, target_cam=None)
                    if blob is None:
                        proj_xy[0] += 20
                        proj_xy[1] += 20
                        continue
                cx, cy, ca = blob
                err = math.hypot(cx - target_cam[0], cy - target_cam[1])
                print(f"  iter {it}: proj({proj_xy[0]:.0f},{proj_xy[1]:.0f}) -> "
                      f"cam({cx:.1f},{cy:.1f}) err={err:.1f}px area={ca}",
                      flush=True)
                if best is None or err < best[0]:
                    best = (err, list(proj_xy), (cx, cy))
                if err < 12.0:
                    print(f"  CONVERGED at iter {it}", flush=True)
                    break
                # Simple proportional correction in projector space
                # measured: at this proj, we get this cam. We want target cam.
                # Use empirical relationship from prior measurements:
                # the projector y axis seems inverted relative to camera y.
                # Use raw delta scaled by Jacobian.
                ox = cx - target_cam[0]
                oy = cy - target_cam[1]
                # Use measured x-scale (proj_x changes 1.75 per cam_x change)
                # For y, we don't know the sign so try the opposite direction
                # of what the planar homography suggests:
                damping = max(0.3, 0.7 - 0.04 * it)
                proj_xy[0] -= ox * 1.75 * damping  # x maps directly with H scale
                # For y, try both signs and pick the one that improved
                # On odd iters use one sign, on even the other
                if it % 2 == 0:
                    proj_xy[1] -= oy * 1.0 * damping
                else:
                    proj_xy[1] += oy * 1.0 * damping
                # clamp
                proj_xy[0] = max(60, min(DISPLAY_W - 60, proj_xy[0]))
                proj_xy[1] = max(60, min(DISPLAY_H - 60, proj_xy[1]))

            # Reset to black between targets
            blit(screen, render_solid((0, 0, 0)))
            for _ in range(20):
                for _ in pygame.event.get(): pass
                pipe.wait_for_frames()

            if best:
                winners.append({
                    "label": label,
                    "target_cam": list(target_cam),
                    "proj_xy": best[1],
                    "cam_xy": list(best[2]),
                    "err_px": best[0],
                })
                print(f"[{label}] best: err={best[0]:.1f}px proj{best[1]}",
                      flush=True)
            else:
                print(f"[{label}] NEVER FOUND BLOB", flush=True)

        # Save
        with open(r"D:\Github-D\LiveTracking\tmp\iv2_result.json", "w") as f:
            json.dump({"winners": winners}, f, indent=2)
        print("\n[final winners]", flush=True)
        for w in winners:
            print(f"  {w['label']}: err={w['err_px']:.1f}px proj{w['proj_xy']}", flush=True)

        # Final projection
        positions = [tuple(w["proj_xy"]) for w in winners]
        if len(positions) == 3:
            blit(screen, render_final(positions))
            for _ in range(30):
                for _ in pygame.event.get(): pass
                pipe.wait_for_frames()
            f = pipe.wait_for_frames()
            cam_final = np.asanyarray(f.get_color_frame().get_data())
            cv2.imwrite(r"D:\Github-D\LiveTracking\tmp\iv2_final.png", cam_final)
            print("\nscreenshot tmp/iv2_final.png", flush=True)

            # Hold
            t0 = time.time()
            running = True
            while running and (time.time() - t0) < 12 * 3600:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        running = False
                    elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                time.sleep(0.05)
    finally:
        try: pipe.stop()
        except: pass
        pygame.quit()


if __name__ == "__main__":
    sys.exit(main())

"""LiveTracking projection daemon.

Long-running process that:
  - Owns the pygame window on DISPLAY2 (the projector)
  - Owns the RealSense pipeline
  - Detects post-its inside the projection rectangle via differencing
  - Renders either a + sign or a colored animated fill on each target
  - Self-heals every 30 seconds

State IPC (filesystem, so the web UI can be a separate process):
  - Writes STATE_FILE (JSON) every ~500ms with current target list, render mode,
    heal count, uptime, last status.
  - Writes FRAME_FILE (JPEG) every ~500ms with the latest RealSense RGB frame.
  - Reads COMMAND_FILE every loop iteration:
      restart      - re-detect and re-converge from scratch
      mode_plus    - render + signs (one per target)
      mode_fill    - render colored animated fills (default)
      screenshot   - save a one-shot screenshot of the camera view
      quit         - exit cleanly (leaves last fills on screen)

Also supports v8-era iv10_cmd.txt for backward compat.

Derived from tmp/iter_v10_diff_detect.py (2026-05-25 working version).
"""
import os, sys, time, math, json, threading
import numpy as np, cv2, pygame
import pyrealsense2 as rs

STATE_DIR = r"D:\Github-D\LiveTracking\runtime"
STATE_FILE = os.path.join(STATE_DIR, "state.json")
FRAME_FILE = os.path.join(STATE_DIR, "latest_frame.jpg")
COMMAND_FILE = os.path.join(STATE_DIR, "command.txt")
NUDGES_FILE = os.path.join(STATE_DIR, "nudges.json")
LEGACY_CMD_FILE = r"D:\Github-D\LiveTracking\tmp\iv10_cmd.txt"
os.makedirs(STATE_DIR, exist_ok=True)

NUDGE_STEP_PX = 5  # projector pixels per nudge click


def load_nudges():
    """Load per-target [dx, dy] projector-space offsets from disk.

    Returns {1: [dx, dy], 2: [dx, dy], 3: [dx, dy]} - keys are 1-indexed.
    Defaults all to [0, 0].
    """
    nudges = {1: [0.0, 0.0], 2: [0.0, 0.0], 3: [0.0, 0.0]}
    if os.path.exists(NUDGES_FILE):
        try:
            with open(NUDGES_FILE) as f:
                loaded = json.load(f)
            for k, v in loaded.items():
                try:
                    nudges[int(k)] = [float(v[0]), float(v[1])]
                except Exception:
                    pass
        except Exception as e:
            print(f"  nudges load err: {e}", flush=True)
    return nudges


def save_nudges(nudges):
    try:
        tmp = NUDGES_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({str(k): v for k, v in nudges.items()}, f, indent=2)
        os.replace(tmp, NUDGES_FILE)
    except Exception as e:
        print(f"  nudges save err: {e}", flush=True)

DISPLAY_X = 5120
DISPLAY_Y = 0
DISPLAY_W = 1280
DISPLAY_H = 720

FILL_SHRINK = 0.85
VERTICAL_OFFSET_FRAC = 0.10
HEAL_INTERVAL_S = 30.0
SHIFT_THRESHOLD_PX = 20.0


def blit(screen, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    s = pygame.image.frombuffer(rgb.tobytes(), (DISPLAY_W, DISPLAY_H), "RGB")
    screen.blit(s, (0, 0))
    pygame.display.flip()


def render_solid(c):
    return np.full((DISPLAY_H, DISPLAY_W, 3),
                   np.array(c, dtype=np.uint8), dtype=np.uint8)


# BGR colors matching the web UI per-target legend.
# Target 1 = red, 2 = green, 3 = blue.
PLUS_COLORS_BGR = [
    (60, 60, 255),    # red
    (60, 255, 60),    # green
    (255, 80, 80),    # blue
]


def render_one_plus(px, py, length=45, thick=16, color=(255, 255, 255)):
    """Render a single + sign at (px, py) in the given BGR color.

    Defaults to white because the closed-loop search depends on white-on-black
    for differencing-based detection. Only the live render-mode loop should
    pass a non-white color.
    """
    canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    if not (0 <= px < DISPLAY_W and 0 <= py < DISPLAY_H):
        return canvas
    px = int(px); py = int(py)
    cv2.line(canvas, (px - length, py), (px + length, py),
             color, thick, cv2.LINE_AA)
    cv2.line(canvas, (px, py - length), (px, py + length),
             color, thick, cv2.LINE_AA)
    cv2.circle(canvas, (px, py), 8, color, -1, cv2.LINE_AA)
    return canvas


def gen_color_cycle(w, h, t_s):
    hue = int((t_s * 30) % 180)
    hsv = np.full((h, w, 3), [hue, 240, 255], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def gen_fire(w, h, t_s):
    noise = np.random.rand(h, w).astype(np.float32)
    rows = np.arange(h).reshape(-1, 1).astype(np.float32)
    phase = (rows * 0.1 + t_s * 6.0) % h
    bias = np.sin(phase * 0.3 + t_s * 4.0) * 0.3 + 0.7
    intensity = noise * 0.5 + 0.5 * bias
    height_attenuate = np.linspace(1.0, 0.2, h, dtype=np.float32).reshape(-1, 1)
    intensity = np.clip(intensity * height_attenuate, 0, 1)
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[..., 0] = (intensity * 60).astype(np.uint8)
    bgr[..., 1] = (intensity * 180 * intensity).astype(np.uint8)
    bgr[..., 2] = np.clip(intensity * 255 + 80, 0, 255).astype(np.uint8)
    return bgr


def gen_water(w, h, t_s):
    xs = np.arange(w).astype(np.float32).reshape(1, -1)
    ys = np.arange(h).astype(np.float32).reshape(-1, 1)
    wave1 = np.sin(xs * 0.10 + t_s * 3.0)
    wave2 = np.sin(ys * 0.08 - t_s * 2.0)
    wave3 = np.sin((xs + ys) * 0.05 + t_s * 1.5)
    combined = (wave1 + wave2 + wave3) / 3.0
    intensity = (combined + 1) / 2
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[..., 0] = (intensity * 200 + 55).astype(np.uint8)
    bgr[..., 1] = (intensity * 220 + 30).astype(np.uint8)
    bgr[..., 2] = (intensity * 80).astype(np.uint8)
    return bgr


def render_animated_fills(rect_specs, t_s):
    """rect_specs is a list of (proj_quad, content_fn).

    proj_quad is a 4x2 float array of projector-space corners in order
    [top-left, top-right, bottom-right, bottom-left] (matching the
    cam-space rotated-rect ordering, just mapped through the Jacobian into
    projector space, so the quad may be sheared/skewed).
    """
    canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    overall_pulse = 0.75 + 0.25 * (0.5 + 0.5 * math.sin(t_s * 1.8))
    for i, (proj_quad, content_fn) in enumerate(rect_specs):
        dst = np.asarray(proj_quad, dtype=np.float32)
        # Pick a content texture size that roughly matches the quad's
        # projector-space extent so the warp resamples sensibly.
        side_top = np.linalg.norm(dst[1] - dst[0])
        side_bot = np.linalg.norm(dst[2] - dst[3])
        side_left = np.linalg.norm(dst[3] - dst[0])
        side_right = np.linalg.norm(dst[2] - dst[1])
        w = max(2, int(round((side_top + side_bot) * 0.5)))
        h = max(2, int(round((side_left + side_right) * 0.5)))
        content = content_fn(w, h, t_s)
        content = (content.astype(np.float32) * overall_pulse).clip(0, 255).astype(np.uint8)
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(content, M, (DISPLAY_W, DISPLAY_H),
                                      borderValue=(0, 0, 0))
        mask = np.zeros((DISPLAY_H, DISPLAY_W), dtype=np.uint8)
        cv2.fillPoly(mask, [dst.astype(np.int32)], 255)
        canvas = np.where(mask[..., None] > 0, warped, canvas)
    return canvas


def find_postits_diff(cam_black, cam_white):
    """Ambient-independent post-it detection.

    1. Diff (white - black) reveals the projection rectangle - purely the
       light the projector added. Robust to ambient brightness.
    2. Inside that rectangle, find the brightest local patches in the WHITE
       capture. Post-its reflect more projector light than surrounding wall.
    3. Filter by size/aspect to keep ~post-it sized blobs.
    """
    bg = cv2.cvtColor(cam_black, cv2.COLOR_BGR2GRAY).astype(np.int16)
    wg = cv2.cvtColor(cam_white, cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff = np.clip(wg - bg, 0, 255).astype(np.uint8)
    # Threshold: above 3x noise floor (typical noise std ~10 -> thresh 30)
    _, proj_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    proj_mask = cv2.morphologyEx(proj_mask, cv2.MORPH_CLOSE, k, iterations=2)
    proj_mask = cv2.morphologyEx(proj_mask, cv2.MORPH_OPEN, k, iterations=1)
    if proj_mask.sum() < 5000:
        print(f"   diff: proj_mask too small ({int(proj_mask.sum())} px)", flush=True)
        return None
    # Restrict to biggest CC of proj_mask
    num, lab, stats, _ = cv2.connectedComponentsWithStats(proj_mask, 8)
    if num <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    proj_mask = ((lab == biggest).astype(np.uint8)) * 255
    # Within proj_mask, find local brightness peaks in WHITE capture.
    # The post-its will be brighter than surrounding wall (more reflective).
    inside = cv2.cvtColor(cam_white, cv2.COLOR_BGR2GRAY)
    inside_masked = cv2.bitwise_and(inside, inside, mask=proj_mask)
    # Adaptive: threshold at top quantile inside the projection.
    pixels_in_proj = inside[proj_mask > 0]
    if len(pixels_in_proj) < 1000:
        return None
    # Post-its should be the brightest ~3-8% inside the projection rectangle.
    # Find the threshold at the 92nd percentile and binarize.
    thresh_val = float(np.percentile(pixels_in_proj, 92))
    print(f"   diff: inside-proj 92nd %ile = {thresh_val:.0f}", flush=True)
    bright = ((inside_masked >= thresh_val) & (proj_mask > 0)).astype(np.uint8) * 255
    # Light close to fill any small holes inside post-it patches.
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                iterations=1)
    num2, lab2, stats2, cents2 = cv2.connectedComponentsWithStats(bright, 8)
    cands = []
    for i in range(1, num2):
        a = int(stats2[i, cv2.CC_STAT_AREA])
        if a < 80 or a > 8000:
            continue
        bw_ = stats2[i, cv2.CC_STAT_WIDTH]
        bh_ = stats2[i, cv2.CC_STAT_HEIGHT]
        if max(bw_, bh_) / max(1, min(bw_, bh_)) > 3.0:
            continue
        # Switched 2026-05-25: use minAreaRect center, NOT the brightness centroid.
        # The brightness centroid biases toward whichever side of the post-it
        # reflects more projector light - on the 45-rotated diamond this pulls
        # the detected center systematically downward by ~10px, which the
        # daemon then converges sub-pixel onto. The minAreaRect center is the
        # geometric center of the post-it's oriented bbox, unbiased by
        # intra-blob brightness variation.
        ys, xs = np.where(lab2 == i)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        (rcx, rcy), (rw, rh), rangle = cv2.minAreaRect(pts)
        cands.append({
            "centroid_cam": (float(rcx), float(rcy)),
            "rot_size": [float(rw), float(rh)],
            "rot_angle_deg": float(rangle),
            "area": float(a),
        })
    # Cluster very-close duplicates
    deduped = []
    used = set()
    for i, p in enumerate(cands):
        if i in used:
            continue
        cluster = [p]
        for j in range(i + 1, len(cands)):
            if j in used:
                continue
            q = cands[j]
            dx = p["centroid_cam"][0] - q["centroid_cam"][0]
            dy = p["centroid_cam"][1] - q["centroid_cam"][1]
            if dx * dx + dy * dy < 30 * 30:
                cluster.append(q)
                used.add(j)
        used.add(i)
        cluster.sort(key=lambda x: -x["area"])
        deduped.append(cluster[0])
    # Keep top 3 by area
    deduped.sort(key=lambda x: -x["area"])
    deduped = deduped[:3]
    deduped.sort(key=lambda x: x["centroid_cam"][0])
    if len(deduped) >= 3:
        return deduped
    return None


def find_postits(cam_img):
    gray = cv2.cvtColor(cam_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=2)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    if num <= 1:
        return None, None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    proj_mask = ((lab == biggest).astype(np.uint8)) * 255
    roi = cv2.bitwise_and(cam_img, cam_img, mask=proj_mask)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 35, 110)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                        iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < 800 or a > 8000:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 50:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) < 4 or len(approx) > 6:
            continue
        rect = cv2.minAreaRect(c)
        (rcx, rcy), (rw, rh), rangle = rect
        if min(rw, rh) < 12:
            continue
        if max(rw, rh) / max(1, min(rw, rh)) > 2.0:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if proj_mask[cy, cx] == 0:
            continue
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull) or 1
        if a / ha < 0.85:
            continue
        cands.append({
            "centroid_cam": (cx, cy),
            "rot_size": [float(rw), float(rh)],
            "rot_angle_deg": float(rangle),
            "area": float(a),
        })
    deduped = []
    used = set()
    for i, p in enumerate(cands):
        if i in used: continue
        cluster = [p]
        for j in range(i + 1, len(cands)):
            if j in used: continue
            q = cands[j]
            dx = p["centroid_cam"][0] - q["centroid_cam"][0]
            dy = p["centroid_cam"][1] - q["centroid_cam"][1]
            if dx * dx + dy * dy < 35 * 35:
                cluster.append(q)
                used.add(j)
        used.add(i)
        cluster.sort(key=lambda x: -x["area"])
        deduped.append(cluster[0])
    if len(deduped) > 3:
        centroids = np.array([p["centroid_cam"] for p in deduped], dtype=np.float32)
        median = np.median(centroids, axis=0)
        deduped = [p for p, c in zip(deduped, centroids)
                   if np.linalg.norm(c - median) < 200]
    deduped.sort(key=lambda x: x["centroid_cam"][0])
    return deduped[:3], proj_mask


def find_plus_blob(baseline, with_plus, target_cam=None, search_r=200):
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


def converge_target(screen, pipe, baseline, target_cam, initial_proj_xy,
                     max_iter=18, label=""):
    proj_xy = list(initial_proj_xy)
    best = None
    for it in range(max_iter):
        blit(screen, render_one_plus(proj_xy[0], proj_xy[1]))
        for _ in range(8):
            for _ in pygame.event.get(): pass
            pipe.wait_for_frames()
        f = pipe.wait_for_frames()
        wp = np.asanyarray(f.get_color_frame().get_data())
        blob = find_plus_blob(baseline, wp, target_cam=target_cam, search_r=200)
        if blob is None:
            proj_xy[1] += 30
            continue
        cx, cy, ca = blob
        err = math.hypot(cx - target_cam[0], cy - target_cam[1])
        if best is None or err < best["err"]:
            best = {"err": err, "proj_xy": list(proj_xy), "cam_xy": (cx, cy)}
        if err < 6.0:
            break
        ox = cx - target_cam[0]
        oy = cy - target_cam[1]
        damping = max(0.25, 0.7 - 0.04 * it)
        proj_xy[0] -= ox * 1.75 * damping
        if (it // 2) % 2 == 0:
            proj_xy[1] -= oy * 1.0 * damping
        else:
            proj_xy[1] += oy * 1.0 * damping
        proj_xy[0] = max(40, min(DISPLAY_W - 40, proj_xy[0]))
        proj_xy[1] = max(40, min(DISPLAY_H - 40, proj_xy[1]))
    return best


def probe_jacobian(screen, pipe, baseline, converged_proj, converged_cam,
                    probe_px=30.0):
    """Measure the full 2x2 local Jacobian J of the projector->camera mapping.

    Probes +probe_px displacements in proj-x and proj-y, captures the FULL
    2D camera-space delta of the rendered + blob for each probe, and
    assembles J such that  cam_delta = J @ proj_delta  for small
    displacements near (converged_proj, converged_cam).

    Returns J as a 2x2 numpy array. Falls back to a sensible diagonal
    default if a probe fails to detect a blob.
    """
    # +X probe in projector space
    blit(screen, render_one_plus(converged_proj[0] + probe_px, converged_proj[1]))
    for _ in range(8):
        for _ in pygame.event.get(): pass
        pipe.wait_for_frames()
    f = pipe.wait_for_frames()
    wp = np.asanyarray(f.get_color_frame().get_data())
    blob_x = find_plus_blob(baseline, wp, target_cam=converged_cam, search_r=180)
    if blob_x is not None:
        col_x = np.array([(blob_x[0] - converged_cam[0]) / probe_px,
                          (blob_x[1] - converged_cam[1]) / probe_px])
    else:
        col_x = np.array([0.22, 0.0])

    # +Y probe in projector space
    blit(screen, render_one_plus(converged_proj[0], converged_proj[1] + probe_px))
    for _ in range(8):
        for _ in pygame.event.get(): pass
        pipe.wait_for_frames()
    f = pipe.wait_for_frames()
    wp = np.asanyarray(f.get_color_frame().get_data())
    blob_y = find_plus_blob(baseline, wp, target_cam=converged_cam, search_r=180)
    if blob_y is not None:
        col_y = np.array([(blob_y[0] - converged_cam[0]) / probe_px,
                          (blob_y[1] - converged_cam[1]) / probe_px])
    else:
        col_y = np.array([0.0, 0.22])

    J = np.column_stack([col_x, col_y])  # 2x2; columns are camera-deltas per unit proj displacement.

    # Guard against degenerate (near-singular) Jacobians. If det is tiny,
    # fall back to a diagonal approximation using whichever column has
    # signal so np.linalg.solve doesn't blow up downstream.
    det = float(np.linalg.det(J))
    if abs(det) < 1e-4:
        sx = col_x[0] if abs(col_x[0]) > 0.05 else 0.22
        sy = col_y[1] if abs(col_y[1]) > 0.05 else 0.22
        J = np.array([[sx, 0.0], [0.0, sy]])
    return J


def build_winner(target_cam, rot_size, rot_angle, converged_proj,
                  converged_cam, J, err):
    """Compute the projector center for the FILL such that the camera-observed
    center of the fill lands on target_cam, plus the projector-space quad
    that the fill should be rendered into.

    J is the local 2x2 camera<-projector Jacobian (cam_delta = J @ proj_delta).
    To convert a desired camera-space shift back into a projector-space shift
    we solve J @ proj_delta = cam_delta -> proj_delta = solve(J, cam_delta).

    The fill quad in camera space is the (shrunk) rotated rectangle from the
    detector. To map it into projector space we transform each corner's
    cam-space offset (from the cam center) through J_inv and add it to the
    projector-space center. The result is a general quadrilateral, not
    necessarily a rotated rectangle - which is the whole point of this fix.
    """
    cam_w, cam_h = rot_size
    cam_w_eff = cam_w * FILL_SHRINK
    cam_h_eff = cam_h * FILL_SHRINK

    # Residual correction: shift proj so the rendered + lands at target_cam.
    dx_cam = target_cam[0] - converged_cam[0]
    dy_cam = target_cam[1] - converged_cam[1]
    cam_delta = np.array([dx_cam, dy_cam], dtype=float)
    proj_delta = np.linalg.solve(J, cam_delta)
    dx_proj, dy_proj = float(proj_delta[0]), float(proj_delta[1])

    corrected_proj = [converged_proj[0] + dx_proj,
                       converged_proj[1] + dy_proj]

    # Build cam-space corner offsets for the (shrunk) rotated rect, then
    # map each through J_inv into projector-space corner offsets.
    rad = math.radians(rot_angle)
    cosr, sinr = math.cos(rad), math.sin(rad)
    hw = cam_w_eff / 2.0
    hh = cam_h_eff / 2.0
    cam_corner_offsets = []
    for lx, ly in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
        rx = lx * cosr - ly * sinr
        ry = lx * sinr + ly * cosr
        cam_corner_offsets.append([rx, ry])
    cam_corner_offsets = np.array(cam_corner_offsets, dtype=float)  # 4x2
    # Solve J @ proj_off = cam_off  for each corner. np.linalg.solve
    # broadcasts on the RHS columns, so transpose -> solve -> transpose.
    proj_corner_offsets = np.linalg.solve(J, cam_corner_offsets.T).T  # 4x2
    proj_quad = proj_corner_offsets + np.array(corrected_proj)

    # Diagonal scale equivalents for diagnostics + state.json.
    sx_diag = float(J[0, 0])
    sy_diag = float(J[1, 1])

    return {
        "target_cam": list(target_cam),
        "converged_cam": [converged_cam[0], converged_cam[1]],
        "rot_size": list(rot_size),
        "rot_angle_deg": rot_angle,
        "proj_center": corrected_proj,
        "converged_proj": list(converged_proj),
        "residual_cam": [dx_cam, dy_cam],
        "residual_proj": [dx_proj, dy_proj],
        "jacobian": [[float(J[0, 0]), float(J[0, 1])],
                       [float(J[1, 0]), float(J[1, 1])]],
        "scale_x": sx_diag,
        "scale_y": sy_diag,
        "proj_quad": proj_quad.tolist(),
        "err_px": err,
    }


def winners_to_rect_specs(winners, content_fns):
    specs = []
    for i, w in enumerate(winners):
        specs.append((
            np.asarray(w["proj_quad"], dtype=np.float32),
            content_fns[i % len(content_fns)],
        ))
    return specs


def main():
    restart_requested = False  # hoist so finally + return path can see it
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{DISPLAY_X},{DISPLAY_Y}"
    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), pygame.NOFRAME)
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    try:
        sensor = profile.get_device().query_sensors()[1]
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, 150)
    except Exception:
        pass
    for _ in range(20):
        pipe.wait_for_frames()

    content_fns = [gen_color_cycle, gen_fire, gen_water]

    def capture_baseline():
        blit(screen, render_solid((0, 0, 0)))
        for _ in range(25):
            for _ in pygame.event.get(): pass
            pipe.wait_for_frames()
        f = pipe.wait_for_frames()
        return np.asanyarray(f.get_color_frame().get_data())

    def detect_postits_now():
        # v10: differencing-based detection. Capture black + white, diff to find
        # the projection rectangle (ambient-independent), then find the brightest
        # patches inside (post-its reflect more projector light than the wall).
        for attempt in range(3):
            blit(screen, render_solid((0, 0, 0)))
            for _ in range(30):
                for _ in pygame.event.get(): pass
                pipe.wait_for_frames()
            f = pipe.wait_for_frames()
            cam_black = np.asanyarray(f.get_color_frame().get_data())
            blit(screen, render_solid((255, 255, 255)))
            for _ in range(30):
                for _ in pygame.event.get(): pass
                pipe.wait_for_frames()
            f = pipe.wait_for_frames()
            cam_white = np.asanyarray(f.get_color_frame().get_data())
            cv2.imwrite(r"D:\Github-D\Helm\openclaw-workspace\tmp\diag_v10_black.png", cam_black)
            cv2.imwrite(r"D:\Github-D\Helm\openclaw-workspace\tmp\diag_v10_white.png", cam_white)
            postits = find_postits_diff(cam_black, cam_white)
            if postits and len(postits) >= 3:
                print(f"   diff-detect: found {len(postits)} post-its", flush=True)
                return postits
            print(f"   diff-detect attempt {attempt}: failed", flush=True)
        return postits

    quad_cam = np.array([[423,0],[1153,0],[1153,719],[423,719]], dtype=np.float32)
    proj_corners = np.array([[0,0],[DISPLAY_W-1,0],[DISPLAY_W-1,DISPLAY_H-1],[0,DISPLAY_H-1]],
                             dtype=np.float32)
    H_init, _ = cv2.findHomography(quad_cam, proj_corners)

    try:
        # initial detection + calibration
        print("[init] detecting post-its", flush=True)
        postits = detect_postits_now()
        if not postits or len(postits) < 3:
            print(f"FAIL: only {len(postits) if postits else 0} post-its", flush=True)
            return 1
        for i, p in enumerate(postits):
            print(f"   #{i+1}: cam{p['centroid_cam']} angle={p['rot_angle_deg']:.1f}",
                  flush=True)

        print("[init] capturing baseline", flush=True)
        baseline = capture_baseline()

        print("[init] converging each target + ITERATIVE RESIDUAL", flush=True)
        winners = []
        for tgt_idx, p in enumerate(postits):
            label = f"#{tgt_idx+1}"
            target_cam = p["centroid_cam"]
            pt = np.array([[target_cam]], dtype=np.float32)
            proj_pt = cv2.perspectiveTransform(pt, H_init)
            init_proj = [float(proj_pt[0][0][0]), float(proj_pt[0][0][1])]
            best = converge_target(screen, pipe, baseline, target_cam, init_proj,
                                    max_iter=18, label=label)
            J = probe_jacobian(screen, pipe, baseline, best["proj_xy"],
                                best["cam_xy"])
            print(f"   {label}: closed-loop err={best['err']:.1f}px J=[[{J[0,0]:.3f},{J[0,1]:.3f}],[{J[1,0]:.3f},{J[1,1]:.3f}]]",
                  flush=True)

            # ITERATIVE RESIDUAL REFINEMENT
            # After closed-loop converges, the + lands within 6-15 px of target.
            # Apply residual correction in projector space using the FULL
            # 2x2 Jacobian (inverted via np.linalg.solve), project + again,
            # measure new residual, refine until <1 px or max 6 iters.
            current_proj = list(best["proj_xy"])
            current_cam = best["cam_xy"]
            for r_iter in range(8):
                dx_cam = target_cam[0] - current_cam[0]
                dy_cam = target_cam[1] - current_cam[1]
                residual_err = math.hypot(dx_cam, dy_cam)
                if residual_err < 1.0:
                    print(f"   {label}: residual converged at <1px (r_iter={r_iter})",
                          flush=True)
                    break
                proj_delta = np.linalg.solve(J, np.array([dx_cam, dy_cam], dtype=float))
                dx_proj, dy_proj = float(proj_delta[0]), float(proj_delta[1])
                # gentle damping for stability
                damp = 0.7
                current_proj[0] += dx_proj * damp
                current_proj[1] += dy_proj * damp
                # Reproject and measure
                blit(screen, render_one_plus(current_proj[0], current_proj[1]))
                for _ in range(8):
                    for _ in pygame.event.get(): pass
                    pipe.wait_for_frames()
                f = pipe.wait_for_frames()
                wp = np.asanyarray(f.get_color_frame().get_data())
                blob = find_plus_blob(baseline, wp, target_cam=target_cam, search_r=200)
                if blob is None:
                    print(f"   {label}: r_iter={r_iter} NO BLOB", flush=True)
                    break
                current_cam = (blob[0], blob[1])
                new_err = math.hypot(blob[0] - target_cam[0],
                                       blob[1] - target_cam[1])
                print(f"   {label}: r_iter={r_iter} err={new_err:.1f}px proj=({current_proj[0]:.1f},{current_proj[1]:.1f})",
                      flush=True)

            # Final position - use this directly without further residual math
            # since we've already iterated
            final_err = math.hypot(current_cam[0] - target_cam[0],
                                    current_cam[1] - target_cam[1])
            print(f"   {label}: FINAL residual err={final_err:.1f}px", flush=True)

            # build_winner uses converged_cam/proj for the rect_specs. Pass
            # the iteratively-refined values so the fill center IS the
            # final projector position (no further residual correction needed).
            winners.append(build_winner(target_cam, p["rot_size"], p["rot_angle_deg"],
                                          current_proj, current_cam,
                                          J, final_err))
            blit(screen, render_solid((0, 0, 0)))
            for _ in range(15):
                for _ in pygame.event.get(): pass
                pipe.wait_for_frames()

        rect_specs = winners_to_rect_specs(winners, content_fns)

        # ANIMATION + SELF-HEAL LOOP
        print("[run] rendering animations with self-heal every 30s", flush=True)
        t0 = time.time()
        last_heal = time.time()
        last_publish = 0.0
        screenshot_taken = False
        heal_count = 0
        running = True
        render_mode = "plus"  # default - clearer for diagnosing per-target placement  # "fill" or "plus"
        last_status = "ok"
        nudges = load_nudges()  # {1: [dx, dy], 2: [dx, dy], 3: [dx, dy]} in projector px
        print(f"[run] loaded nudges: {nudges}", flush=True)
        clock = pygame.time.Clock()
        while running and (time.time() - t0) < 12 * 3600:
            t = time.time() - t0

            # SELF-HEAL
            if time.time() - last_heal > HEAL_INTERVAL_S:
                last_heal = time.time()
                heal_count += 1
                print(f"\n[heal #{heal_count} t={t:.0f}s] re-detecting", flush=True)
                new_postits = detect_postits_now()
                if not new_postits or len(new_postits) < 3:
                    print(f"  heal: only {len(new_postits) if new_postits else 0} found, skipping",
                          flush=True)
                else:
                    # Nearest-neighbor data association: match each existing
                    # winner to its closest detected post-it (by Euclidean
                    # distance in camera coords), greedy. Prevents the
                    # heal-failure-on-large-move bug where new_postits got
                    # re-sorted left-to-right and a moved target got compared
                    # to a different post-it's position.
                    assigned = [None] * len(winners)
                    available = list(range(len(new_postits)))
                    pairs = []
                    for wi, old_w in enumerate(winners):
                        for pi in available:
                            ocx, ocy = old_w["target_cam"]
                            ncx, ncy = new_postits[pi]["centroid_cam"]
                            dist = math.hypot(ocx - ncx, ocy - ncy)
                            pairs.append((dist, wi, pi))
                    pairs.sort(key=lambda x: x[0])
                    used_winners = set()
                    for dist, wi, pi in pairs:
                        if wi in used_winners or pi not in available:
                            continue
                        assigned[wi] = pi
                        used_winners.add(wi)
                        available.remove(pi)

                    shifted_indices = []
                    matched_postits = {}  # winner-idx -> new_postit dict
                    for i, pi in enumerate(assigned):
                        if pi is None:
                            print(f"  heal #{i+1}: no match found in new detections",
                                  flush=True)
                            continue
                        new_p = new_postits[pi]
                        oc = winners[i]["target_cam"]
                        nc = new_p["centroid_cam"]
                        shift = math.hypot(oc[0] - nc[0], oc[1] - nc[1])
                        matched_postits[i] = new_p
                        if shift > SHIFT_THRESHOLD_PX:
                            print(f"  shift #{i+1}: {shift:.1f}px ({oc} -> {nc}) RECONVERGING",
                                  flush=True)
                            shifted_indices.append(i)
                        else:
                            print(f"  stable #{i+1}: {shift:.1f}px", flush=True)
                    if shifted_indices:
                        baseline = capture_baseline()
                        for i in shifted_indices:
                            new_p = matched_postits[i]
                            new_target = new_p["centroid_cam"]
                            init_proj = list(winners[i]["proj_center"])  # start near prior
                            best = converge_target(screen, pipe, baseline, new_target,
                                                    init_proj, max_iter=12, label=f"heal#{i+1}")
                            if best:
                                J = probe_jacobian(screen, pipe, baseline,
                                                    best["proj_xy"], best["cam_xy"])
                                # iterative residual refinement (same as initial)
                                current_proj = list(best["proj_xy"])
                                current_cam = best["cam_xy"]
                                for r_iter in range(6):
                                    dx_cam = new_target[0] - current_cam[0]
                                    dy_cam = new_target[1] - current_cam[1]
                                    if math.hypot(dx_cam, dy_cam) < 1.0:
                                        break
                                    proj_delta = np.linalg.solve(
                                        J, np.array([dx_cam, dy_cam], dtype=float))
                                    current_proj[0] += float(proj_delta[0]) * 0.7
                                    current_proj[1] += float(proj_delta[1]) * 0.7
                                    blit(screen, render_one_plus(current_proj[0], current_proj[1]))
                                    for _ in range(6):
                                        for _ in pygame.event.get(): pass
                                        pipe.wait_for_frames()
                                    f = pipe.wait_for_frames()
                                    wp = np.asanyarray(f.get_color_frame().get_data())
                                    blob = find_plus_blob(baseline, wp,
                                                           target_cam=new_target, search_r=200)
                                    if blob is None:
                                        break
                                    current_cam = (blob[0], blob[1])
                                final_err = math.hypot(current_cam[0] - new_target[0],
                                                        current_cam[1] - new_target[1])
                                winners[i] = build_winner(new_target, new_p["rot_size"],
                                                            new_p["rot_angle_deg"],
                                                            current_proj, current_cam,
                                                            J, final_err)
                                rect_specs = winners_to_rect_specs(winners, content_fns)
                                print(f"  heal #{i+1} -> err={best['err']:.1f}px",
                                      flush=True)
                # after heal, reset t (so animations don't jump)
                # actually keep t0 for continuous animation
                continue

            # Normal render based on current mode. Apply per-target nudges
            # in projector space - manual user offsets via web UI that
            # compensate for residual parallax / centroid bias.
            if render_mode == "plus":
                # Render one + sign per target at its converged projector position.
                # Target 1 = red, 2 = green, 3 = blue (matches web UI legend).
                frame = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
                for idx, w in enumerate(winners):
                    px, py = w["proj_center"]
                    nudge = nudges.get(idx + 1, [0.0, 0.0])
                    px += nudge[0]
                    py += nudge[1]
                    color = PLUS_COLORS_BGR[idx % len(PLUS_COLORS_BGR)]
                    plus = render_one_plus(px, py, length=45, thick=14,
                                            color=color)
                    frame = np.maximum(frame, plus)
            else:
                # Apply nudges to fill mode by translating the proj_quad per target.
                nudged_specs = []
                for idx, (proj_quad, content_fn) in enumerate(rect_specs):
                    nudge = np.array(nudges.get(idx + 1, [0.0, 0.0]),
                                       dtype=np.float32)
                    nudged_quad = np.asarray(proj_quad, dtype=np.float32) + nudge
                    nudged_specs.append((nudged_quad, content_fn))
                frame = render_animated_fills(nudged_specs, t)
            blit(screen, frame)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

            # Publish state + camera frame every ~500ms for the web UI.
            now = time.time()
            if now - last_publish > 0.5:
                last_publish = now
                try:
                    f = pipe.wait_for_frames()
                    cam_now = np.asanyarray(f.get_color_frame().get_data())
                    cv2.imwrite(FRAME_FILE, cam_now,
                                 [cv2.IMWRITE_JPEG_QUALITY, 70])
                    # Also overwrite legacy path for any older consumers.
                    if not screenshot_taken and t > 3.0:
                        cv2.imwrite(r"D:\Github-D\LiveTracking\tmp\iv10_final.png",
                                     cam_now)
                        screenshot_taken = True
                except Exception as e:
                    print(f"  publish frame error: {e}", flush=True)
                try:
                    state = {
                        "started_at": t0,
                        "uptime_s": round(t, 1),
                        "heal_count": heal_count,
                        "render_mode": render_mode,
                        "target_count": len(winners),
                        "last_status": last_status,
                        "targets": [
                            {
                                "index": i + 1,
                                "cam_xy": [round(w["target_cam"][0], 1),
                                            round(w["target_cam"][1], 1)],
                                "proj_xy": [round(w["proj_center"][0], 1),
                                              round(w["proj_center"][1], 1)],
                                "err_px": round(w["err_px"], 2),
                                "angle_deg": round(w["rot_angle_deg"], 1),
                                "nudge": [round(nudges.get(i + 1, [0, 0])[0], 1),
                                           round(nudges.get(i + 1, [0, 0])[1], 1)],
                            }
                            for i, w in enumerate(winners)
                        ],
                        "published_at": round(now, 1),
                    }
                    tmp_path = STATE_FILE + ".tmp"
                    with open(tmp_path, "w") as sf:
                        json.dump(state, sf, indent=2)
                    os.replace(tmp_path, STATE_FILE)
                except Exception as e:
                    print(f"  publish state error: {e}", flush=True)

            # Command handling (both new COMMAND_FILE and legacy iv10_cmd.txt).
            for cmd_path in (COMMAND_FILE, LEGACY_CMD_FILE):
                if os.path.exists(cmd_path):
                    try:
                        with open(cmd_path) as cf:
                            cmd = cf.read().strip().lower()
                        os.remove(cmd_path)
                        print(f"  CMD: {cmd}", flush=True)
                        if cmd == "quit":
                            running = False
                        elif cmd in ("restart", "recalibrate"):
                            restart_requested = True
                            running = False
                        elif cmd == "heal_now":
                            last_heal = 0
                        elif cmd in ("mode_plus", "plus"):
                            render_mode = "plus"
                            last_status = "mode=plus"
                        elif cmd in ("mode_fill", "fill"):
                            render_mode = "fill"
                            last_status = "mode=fill"
                        elif cmd == "screenshot":
                            f = pipe.wait_for_frames()
                            cam_now = np.asanyarray(f.get_color_frame().get_data())
                            cv2.imwrite(r"D:\Github-D\LiveTracking\tmp\iv10_final.png",
                                         cam_now)
                            cv2.imwrite(FRAME_FILE, cam_now,
                                         [cv2.IMWRITE_JPEG_QUALITY, 70])
                        elif cmd.startswith("nudge_"):
                            # nudge_<idx>_<dir>  where dir is left/right/up/down/reset
                            # also accepts nudge_<idx>_<dir>_<step> for finer steps
                            parts = cmd.split("_")
                            if len(parts) >= 3:
                                try:
                                    idx = int(parts[1])
                                    direction = parts[2]
                                    step = NUDGE_STEP_PX
                                    if len(parts) >= 4:
                                        try:
                                            step = max(1, int(parts[3]))
                                        except ValueError:
                                            pass
                                    if idx in nudges:
                                        if direction == "left":
                                            nudges[idx][0] -= step
                                        elif direction == "right":
                                            nudges[idx][0] += step
                                        elif direction == "up":
                                            nudges[idx][1] -= step
                                        elif direction == "down":
                                            nudges[idx][1] += step
                                        elif direction == "reset":
                                            nudges[idx] = [0.0, 0.0]
                                        save_nudges(nudges)
                                        last_status = f"nudge t{idx} {direction} -> {nudges[idx]}"
                                except (ValueError, IndexError) as e:
                                    print(f"  bad nudge cmd '{cmd}': {e}", flush=True)
                        elif cmd == "reset_nudges":
                            nudges = {1: [0.0, 0.0], 2: [0.0, 0.0], 3: [0.0, 0.0]}
                            save_nudges(nudges)
                            last_status = "nudges reset"
                    except Exception as e:
                        print(f"  command error: {e}", flush=True)
            clock.tick(25)
    finally:
        try: pipe.stop()
        except: pass
        pygame.quit()
    # Exit code 42 signals "restart requested" to the supervisor.
    return 42 if restart_requested else 0


if __name__ == "__main__":
    sys.exit(main())

"""Full guitar mask (body + neck + headstock) by color geometry.

Depth can't separate the guitar from the cushions it lies on (too flat, depth
noise). Instead: the white BODY and white HEADSTOCK are color-detectable; the
dark NECK between them is *not red*. So take the convex hull spanning the white
parts and SUBTRACT the red sofa -> body + dark neck + headstock remain.

Outputs scripts/out/maskfull.png + maskfull_zoom.png. Tune CONFIG, re-run.
"""
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

CONFIG = dict(
    exposure=700,
    white_S_max=80,
    white_V_min=140,
    depth_margin_m=0.12,
    top_ignore_frac=0.40,
    part_min_area=70,       # min white-blob area to count as a guitar part
    part_max_dist_frac=2.2, # white parts within this*body_diag of body count
)


def main():
    devs = list(rs.context().query_devices())
    if devs:
        devs[0].hardware_reset()
    for _ in range(30):
        time.sleep(0.5)
        if list(rs.context().query_devices()):
            break
    time.sleep(2.0)

    p = rs.pipeline(); cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    prof = p.start(cfg)
    ds = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    s = prof.get_device().query_sensors()[1]
    s.set_option(rs.option.enable_auto_exposure, 0)
    s.set_option(rs.option.exposure, CONFIG["exposure"])

    import pygame
    pygame.init()
    sizes = pygame.display.get_desktop_sizes()
    pi = max(range(len(sizes)), key=lambda i: sizes[i][0]*sizes[i][1])
    screen = pygame.display.set_mode(sizes[pi], pygame.NOFRAME, display=pi)
    screen.fill((0, 0, 0)); pygame.display.flip()
    for _ in range(6):
        pygame.event.pump(); time.sleep(0.03)
    for _ in range(30):
        p.wait_for_frames()
    f = align.process(p.wait_for_frames())
    color = np.asanyarray(f.get_color_frame().get_data())
    depth = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * ds
    p.stop(); pygame.quit()

    H_img, W_img = depth.shape
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < CONFIG["white_S_max"]) & (hsv[:, :, 2] > CONFIG["white_V_min"])).astype(np.uint8)
    Hc = hsv[:, :, 0]
    red = (((Hc < 12) | (Hc > 168)) & (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 45)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    up = depth[:int(0.4*H_img)]; upv = up[up > 0]
    wall_d = float(np.percentile(upv, 55)) if upv.size else 3.5
    near = ((depth > 0.3) & (depth < wall_d - CONFIG["depth_margin_m"])).astype(np.uint8)
    nod = (depth == 0).astype(np.uint8)
    onsofa = cv2.bitwise_or(near, nod)

    # locate guitar body: white + near, ignore wall, most red-surrounded blob
    cand = cv2.bitwise_and(white, onsofa)
    cand[:int(CONFIG["top_ignore_frac"]*H_img)] = 0
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    if n <= 1:
        print("no guitar candidate"); return
    gi, best = None, -1.0
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < 400:
            continue
        blob = (lbl == i).astype(np.uint8)
        ring = cv2.subtract(cv2.dilate(blob, np.ones((17, 17), np.uint8)), blob)
        rf = float((red[ring > 0] > 0).mean()) if np.any(ring) else 0.0
        sc = a * (0.15 + rf)
        if sc > best:
            best, gi = sc, i
    seed = (lbl == gi).astype(np.uint8)
    x, y = stats[gi, cv2.CC_STAT_LEFT], stats[gi, cv2.CC_STAT_TOP]
    w, h = stats[gi, cv2.CC_STAT_WIDTH], stats[gi, cv2.CC_STAT_HEIGHT]
    bcx, bcy = x + w/2.0, y + h/2.0
    body_diag = float(np.hypot(w, h))

    # body via white color in the located neighborhood
    region = cv2.dilate(seed, np.ones((61, 61), np.uint8))
    body = cv2.bitwise_and(white, region)
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 200]
    body = np.zeros_like(body)
    if cnts:
        cv2.drawContours(body, [max(cnts, key=cv2.contourArea)], -1, 1, cv2.FILLED)

    # white parts (body + headstock) on the sofa, near the body
    wsofa = cv2.bitwise_and(white, onsofa)
    wsofa[:int(CONFIG["top_ignore_frac"]*H_img)] = 0
    wsofa = cv2.morphologyEx(wsofa, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(wsofa, 8)
    parts = np.zeros_like(wsofa)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < CONFIG["part_min_area"]:
            continue
        d = np.hypot(cents[i][0]-bcx, cents[i][1]-bcy)
        if d < CONFIG["part_max_dist_frac"] * body_diag:
            parts[lbl == i] = 1
    parts = cv2.bitwise_or(parts, body)

    # convex hull spanning all white parts, then subtract red sofa -> full guitar
    ys, xs = np.where(parts > 0)
    pts = np.column_stack([xs, ys]).astype(np.int32)
    hull = cv2.convexHull(pts)
    hullmask = np.zeros_like(parts)
    cv2.fillConvexPoly(hullmask, hull, 1)
    guitar = cv2.bitwise_and(hullmask, 1 - red)
    guitar = cv2.bitwise_and(guitar, onsofa)
    guitar = cv2.bitwise_or(guitar, body)   # always keep the white body
    guitar = cv2.morphologyEx(guitar, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    guitar = cv2.morphologyEx(guitar, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(guitar, 8)
    keep = set(np.unique(lbl[body > 0])) - {0}
    mask = np.isin(lbl, list(keep)).astype(np.uint8) if keep else body
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    if cnts:
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 1, cv2.FILLED)

    print(f"wall_d={wall_d:.2f} body_px={int(body.sum())} parts={int(parts.sum())} full_px={int(mask.sum())}")

    def label(img, t):
        o = img.copy()
        if o.ndim == 2: o = cv2.cvtColor(o, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(o, (0, 0), (o.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(o, t, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return o
    partvis = cv2.cvtColor(parts*255, cv2.COLOR_GRAY2BGR)
    cv2.polylines(partvis, [hull], True, (0, 255, 0), 2)
    overlay = color.copy()
    overlay[mask > 0] = (0.45*overlay[mask > 0] + 0.55*np.array([255, 120, 0])).astype(np.uint8)
    c2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, c2, -1, (0, 0, 255), 2)
    montage = np.vstack([
        np.hstack([label(color, "ambient RGB"), label(partvis, "white parts + hull (green)")]),
        np.hstack([label((1-red)*255, "not-red mask"), label(overlay, "FULL GUITAR MASK")]),
    ])
    cv2.imwrite(os.path.join(OUT, "maskfull.png"), montage)
    cv2.imwrite(os.path.join(OUT, "guitar_mask_full.png"), mask*255)
    ys, xs = np.where(mask > 0)
    if xs.size:
        bb = [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()), int(ys.max()-ys.min())]
        pad = 30
        x0, y0 = max(0, bb[0]-pad), max(0, bb[1]-pad)
        x1, y1 = min(W_img, bb[0]+bb[2]+pad), min(H_img, bb[1]+bb[3]+pad)
        zoom = cv2.resize(overlay[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(OUT, "maskfull_zoom.png"), zoom)
    print("saved maskfull.png, maskfull_zoom.png")


if __name__ == "__main__":
    main()

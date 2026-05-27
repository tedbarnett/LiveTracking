"""Accurate guitar mask: locked-exposure color + depth detection, GrabCut refine.

Uses the detection recipe proven during dot-calibration (locked exposure ~700 so
the bright right side doesn't blow out; white + near-depth + ignore the wall; take
the largest blob) to LOCATE the guitar, then GrabCut for a clean boundary.

Outputs scripts/out/mask.png (montage), guitar_mask.png, mask_overlay.png,
mask_zoom.png. Tune CONFIG, re-run, inspect.
"""
import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

CONFIG = dict(
    exposure=700,           # locked (auto-exposure blows out the window -> false blobs)
    white_S_max=80,
    white_V_min=140,
    depth_margin_m=0.12,    # guitar nearer than wall by at least this
    top_ignore_frac=0.40,   # ignore wall region (top of frame)
    grabcut_iters=6,
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
    white = ((hsv[:, :, 1] < CONFIG["white_S_max"]) &
             (hsv[:, :, 2] > CONFIG["white_V_min"])).astype(np.uint8)

    up = depth[:int(0.4*H_img)]; upv = up[up > 0]
    wall_d = float(np.percentile(upv, 55)) if upv.size else 3.5
    near = ((depth > 0.3) & (depth < wall_d - CONFIG["depth_margin_m"])).astype(np.uint8)
    nod = (depth == 0).astype(np.uint8)

    # red sofa fabric (for the "surrounded by red" discriminator)
    Hc = hsv[:, :, 0]
    red = (((Hc < 12) | (Hc > 168)) & (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 45)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # candidate white + near blobs; pick the one most ringed by red sofa (the
    # guitar). The right-side window/table blobs are similar size but not red-ringed.
    cand = cv2.bitwise_and(white, cv2.bitwise_or(near, nod))
    cand[:int(CONFIG["top_ignore_frac"]*H_img)] = 0
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    if n <= 1:
        print("no guitar candidate"); return
    gi, best, best_rf = None, -1.0, 0.0
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < 400:
            continue
        blob = (lbl == i).astype(np.uint8)
        ring = cv2.subtract(cv2.dilate(blob, np.ones((17, 17), np.uint8)), blob)
        rf = float((red[ring > 0] > 0).mean()) if np.any(ring) else 0.0
        sc = a * (0.15 + rf)            # big AND red-surrounded
        if sc > best:
            best, gi, best_rf = sc, i, rf
    if gi is None:
        print("no guitar candidate"); return
    seed = (lbl == gi).astype(np.uint8)
    x, y = stats[gi, cv2.CC_STAT_LEFT], stats[gi, cv2.CC_STAT_TOP]
    w, h = stats[gi, cv2.CC_STAT_WIDTH], stats[gi, cv2.CC_STAT_HEIGHT]
    print(f"wall_d={wall_d:.2f}  guitar seed bbox=({x},{y},{w},{h}) "
          f"area={int(stats[gi, cv2.CC_STAT_AREA])} red_surround={best_rf:.2f}")

    # full guitar body = white pixels in the located neighborhood (the cushion
    # around it is NOT white, so color gives a clean boundary here). The seed only
    # covered the depth-clean lower body; dilate it to reach the full white body.
    region = cv2.dilate(seed, np.ones((61, 61), np.uint8))
    mask = cv2.bitwise_and(white, region)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # keep the largest external contour, filled (closes soundhole / logo / strap gaps)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 200]
    mask = np.zeros_like(mask)
    if cnts:
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 1, cv2.FILLED)

    area = int(mask.sum())
    ys, xs = np.where(mask > 0)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()), int(ys.max()-ys.min())] if area else None
    print(f"mask_area={area}  mask_bbox={bbox}")

    def label(img, t):
        o = img.copy()
        if o.ndim == 2: o = cv2.cvtColor(o, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(o, (0, 0), (o.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(o, t, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return o
    overlay = color.copy()
    if area:
        overlay[mask > 0] = (0.45*overlay[mask > 0] + 0.55*np.array([255, 120, 0])).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 0, 255), 2)
    montage = np.vstack([
        np.hstack([label(color, "ambient RGB (exp 700)"), label(white*255, "white (color) mask")]),
        np.hstack([label(near*255, "near (depth) mask"), label(overlay, "GUITAR MASK (refined)")]),
    ])
    cv2.imwrite(os.path.join(OUT, "mask.png"), montage)
    cv2.imwrite(os.path.join(OUT, "guitar_mask.png"), mask*255)
    cv2.imwrite(os.path.join(OUT, "mask_overlay.png"), overlay)
    if bbox:
        pad = 35
        x0, y0 = max(0, bbox[0]-pad), max(0, bbox[1]-pad)
        x1, y1 = min(W_img, bbox[0]+bbox[2]+pad), min(H_img, bbox[1]+bbox[3]+pad)
        zoom = cv2.resize(overlay[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(OUT, "mask_zoom.png"), zoom)
    json.dump({"area": area, "bbox": bbox, "wall_d": round(wall_d, 3)},
              open(os.path.join(OUT, "mask_meta.json"), "w"), indent=2)
    print("saved mask.png, mask_zoom.png, guitar_mask.png")


if __name__ == "__main__":
    main()

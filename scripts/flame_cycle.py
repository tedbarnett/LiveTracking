"""Closed-loop test cycle for 'blue flame on the white guitar'.

One run:
  1. reset camera, start color+depth (aligned), lock exposure
  2. fullscreen white-flood on the JMGO -> projection quad -> H (camera -> projector)
  3. detect the white guitar (bright + low-saturation, in front of the wall)
  4. project a blue flame warped onto the guitar
  5. capture the RGB + depth RESULT and evaluate against the goal
  6. save a montage (scripts/out/cycle.png) + print metrics

Tune CONFIG, re-run, inspect cycle.png. That's the optimization loop.
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

CONFIG = dict(
    exposure=1200,        # locked exposure (this lit room needs ~1200, not 150)
    white_S_max=70,       # white guitar: low saturation
    white_V_min=150,      # white guitar: bright
    depth_margin_m=0.15,  # guitar must be at least this much in front of the wall
    flame_scale_w=1.0,    # flame width  vs guitar box
    flame_scale_h=1.6,    # flame height vs guitar box (flames rise above)
    flame_intensity=1.0,
    settle_frames=25,
)


def order_quad(pts):
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def make_blue_flame(w, h, t=0.7, intensity=1.0):
    """Procedural blue flame, BGR uint8. Hot (white-blue) core at the base."""
    w, h = max(2, int(w)), max(2, int(h))
    v = np.linspace(0.0, 1.0, h)[:, None]          # 0 top .. 1 bottom
    x = np.linspace(0.0, 1.0, w)[None, :]
    cx = 0.5 + 0.05 * np.sin(t * 3 + v * 7)        # wavering centerline
    width = 0.05 + 0.12 * (1 - v)                  # wider at the base
    horiz = np.exp(-((x - cx) ** 2) / width)
    flicker = 0.65 + 0.35 * np.sin(t * 5 + x * 18 + v * 9)
    i = np.clip(v ** 0.8 * horiz * flicker * intensity, 0, 1)
    b = np.clip(i * 1.5, 0, 1)
    g = np.clip(i * 1.25 - 0.15, 0, 1)
    r = np.clip(i * 1.15 - 0.55, 0, 1)
    return (np.stack([b, g, r], -1) * 255).astype(np.uint8)


def main():
    # ---------- camera ----------
    devs = list(rs.context().query_devices())
    if devs:
        devs[0].hardware_reset()
    for _ in range(30):
        time.sleep(0.5)
        if list(rs.context().query_devices()):
            break
    time.sleep(2.0)

    p = rs.pipeline()
    c = rs.config()
    c.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    c.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    prof = p.start(c)
    depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    try:
        csensor = prof.get_device().query_sensors()[1]
        csensor.set_option(rs.option.enable_auto_exposure, 0)
        csensor.set_option(rs.option.exposure, CONFIG["exposure"])
    except Exception as e:
        print("exposure lock failed:", e)

    import pygame
    pygame.init()
    sizes = pygame.display.get_desktop_sizes()
    proj_idx = max(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1])
    screen = pygame.display.set_mode(sizes[proj_idx], pygame.NOFRAME, display=proj_idx)
    PW, PH = screen.get_size()
    print(f"projector {PW}x{PH} (display {proj_idx}); camera 848x480")

    def show_rgb(rgb):
        screen.fill(rgb); pygame.display.flip()
        for _ in range(5):
            pygame.event.pump(); time.sleep(0.02)

    def show_canvas(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        surf = pygame.image.frombuffer(rgb.tobytes(), (bgr.shape[1], bgr.shape[0]), "RGB")
        screen.blit(surf, (0, 0)); pygame.display.flip()
        for _ in range(5):
            pygame.event.pump(); time.sleep(0.02)

    def grab():
        for _ in range(CONFIG["settle_frames"]):
            p.wait_for_frames()
        f = align.process(p.wait_for_frames())
        col = np.asanyarray(f.get_color_frame().get_data())
        dep = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * depth_scale
        return col, dep

    # ---------- calibrate ----------
    show_rgb((0, 0, 0)); time.sleep(1.0); cam_black, depth_black = grab()
    show_rgb((255, 255, 255)); time.sleep(1.0); cam_white, _ = grab()
    gb = cv2.cvtColor(cam_black, cv2.COLOR_BGR2GRAY).astype(np.int16)
    gw = cv2.cvtColor(cam_white, cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff = np.clip(gw - gb, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(big, 0.02 * cv2.arcLength(big, True), True)
    cam_quad = order_quad(approx.reshape(-1, 2) if len(approx) == 4
                          else cv2.boxPoints(cv2.minAreaRect(big)))
    proj_quad = np.array([[0, 0], [PW - 1, 0], [PW - 1, PH - 1], [0, PH - 1]], np.float32)
    H = cv2.getPerspectiveTransform(cam_quad, proj_quad)
    proj_mask = np.zeros(diff.shape, np.uint8)
    cv2.fillPoly(proj_mask, [cam_quad.astype(np.int32)], 255)

    # ---------- detect guitar ----------
    show_rgb((0, 0, 0)); time.sleep(0.8); cam, depth = grab()
    hsv = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < CONFIG["white_S_max"]) &
             (hsv[:, :, 2] > CONFIG["white_V_min"])).astype(np.uint8) * 255
    wall_vals = depth_black[(proj_mask > 0) & (depth_black > 0)]
    wall_d = float(np.median(wall_vals)) if wall_vals.size else 0.0
    near = ((depth > 0.3) & (depth < wall_d - CONFIG["depth_margin_m"])).astype(np.uint8) * 255
    nodepth = (depth == 0).astype(np.uint8) * 255
    guitar = cv2.bitwise_and(cv2.bitwise_and(white, proj_mask),
                             cv2.bitwise_or(near, nodepth))
    guitar = cv2.morphologyEx(guitar, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    guitar = cv2.morphologyEx(guitar, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    gcnts, _ = cv2.findContours(guitar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gcnts = [g for g in gcnts if cv2.contourArea(g) > 400]

    metrics = dict(wall_d=round(wall_d, 3), proj_rect=cv2.boundingRect(big),
                   guitar_found=bool(gcnts))
    projected = np.zeros((PH, PW, 3), np.uint8)
    guitar_box_cam = None

    if gcnts:
        gbig = max(gcnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(gbig)
        (cx, cy), (rw, rh), ang = rect
        guitar_box_cam = cv2.boxPoints(rect)
        metrics["guitar_center_cam"] = [round(cx, 1), round(cy, 1)]
        metrics["guitar_size_cam"] = [round(rw, 1), round(rh, 1)]
        metrics["guitar_area"] = int(cv2.contourArea(gbig))

        # flame box in camera coords (scaled around guitar center), then -> projector
        sw, sh = CONFIG["flame_scale_w"], CONFIG["flame_scale_h"]
        big_rect = ((cx, cy), (rw * sw, rh * sh), ang)
        flame_box_cam = cv2.boxPoints(big_rect).astype(np.float32)
        flame_box_proj = cv2.perspectiveTransform(flame_box_cam.reshape(1, -1, 2), H).reshape(-1, 2)
        # orient flame texture so its base sits at the bottom (max-y) edge in camera
        fb = order_quad(flame_box_cam)  # TL,TR,BR,BL in camera
        fb_proj = cv2.perspectiveTransform(fb.reshape(1, -1, 2), H).reshape(-1, 2).astype(np.float32)
        tex = make_blue_flame(256, 384, intensity=CONFIG["flame_intensity"])
        tex_src = np.array([[0, 0], [tex.shape[1]-1, 0],
                            [tex.shape[1]-1, tex.shape[0]-1], [0, tex.shape[0]-1]], np.float32)
        Mtex = cv2.getPerspectiveTransform(tex_src, fb_proj)
        projected = cv2.warpPerspective(tex, Mtex, (PW, PH))

    # ---------- project flame + capture result ----------
    show_canvas(projected); time.sleep(1.0)
    cam_res, depth_res = grab()
    show_rgb((0, 0, 0))
    p.stop(); pygame.quit()

    # ---------- evaluate ----------
    if guitar_box_cam is not None:
        gm = np.zeros(guitar.shape, np.uint8)
        cv2.fillPoly(gm, [guitar_box_cam.astype(np.int32)], 255)
        sel = gm > 0
        base_bright = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)[sel].mean()
        res_bright = cv2.cvtColor(cam_res, cv2.COLOR_BGR2GRAY)[sel].mean()
        b = cam_res[:, :, 0][sel].astype(np.int16)
        r = cam_res[:, :, 2][sel].astype(np.int16)
        metrics["guitar_brightness_gain"] = round(float(res_bright - base_bright), 1)
        metrics["guitar_blueness_B_minus_R"] = round(float((b - r).mean()), 1)
        lit = (cv2.cvtColor(cam_res, cv2.COLOR_BGR2GRAY).astype(np.int16)
               - cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY).astype(np.int16)) > 20
        metrics["flame_on_guitar_coverage"] = round(float(lit[sel].mean()), 3)
        spill = lit & ~sel & (proj_mask > 0)
        metrics["flame_spill_ratio"] = round(float(spill.sum() / max(1, lit.sum())), 3)

    # ---------- montage ----------
    def label(img, txt):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(out, txt, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out

    res_vis = cam_res.copy()
    cv2.polylines(res_vis, [cam_quad.astype(np.int32)], True, (0, 255, 0), 1)
    if guitar_box_cam is not None:
        cv2.drawContours(res_vis, [guitar_box_cam.astype(np.int32)], 0, (255, 120, 0), 2)
    dnorm = np.clip((depth_res - 0.3) / (3.7), 0, 1)
    dvis = cv2.applyColorMap((dnorm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    dvis[depth_res == 0] = 0
    proj_small = cv2.resize(projected, (848, 480))
    det_vis = cv2.cvtColor(guitar, cv2.COLOR_GRAY2BGR)
    top = np.hstack([label(res_vis, "RESULT: flame projected (cam RGB)"),
                     label(dvis, "depth")])
    bot = np.hstack([label(proj_small, "what we projected"),
                     label(det_vis, "guitar mask")])
    montage = np.vstack([top, bot])
    cv2.imwrite(os.path.join(OUT, "cycle.png"), montage)
    cv2.imwrite(os.path.join(OUT, "cycle_result.png"), cam_res)
    with open(os.path.join(OUT, "cycle_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("METRICS:", json.dumps(metrics))
    print("saved cycle.png to", OUT)


if __name__ == "__main__":
    main()

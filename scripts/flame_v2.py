"""Project a blue flame onto the guitar using the saved dot-calibration.
Loads scripts/out/calib.json (H, guitar position). Projects flame, captures the
result through the camera, builds a montage + metrics. Tune CONFIG, re-run.

Run calibrate_dots.py first (or whenever the guitar/scene moves)."""
import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

OUT = os.path.join(os.path.dirname(__file__), "out")

CONFIG = dict(
    exposure=700,
    flame_scale_w=1.8,    # flame width  vs detected guitar size (proj space)
    flame_scale_h=2.6,    # flame height vs detected guitar size (rises above)
    rise_frac=0.72,       # fraction of flame height above the guitar center
    intensity=1.15,
    animate_secs=3.0,     # show animated flame this long; capture mid-way
)


def make_blue_flame(w, h, t=0.7, intensity=1.0):
    w, h = max(2, int(w)), max(2, int(h))
    v = np.linspace(0.0, 1.0, h)[:, None]          # 0 top .. 1 bottom (hot base)
    x = np.linspace(0.0, 1.0, w)[None, :]
    cx = 0.5 + 0.06 * np.sin(t * 3.0 + v * 7)
    width = 0.04 + 0.13 * (1 - v)
    horiz = np.exp(-((x - cx) ** 2) / width)
    flicker = 0.6 + 0.4 * np.sin(t * 6 + x * 16 + v * 9)
    i = np.clip((v ** 0.75) * horiz * flicker * intensity, 0, 1)
    b = np.clip(i * 1.55, 0, 1)
    g = np.clip(i * 1.25 - 0.12, 0, 1)
    r = np.clip(i * 1.15 - 0.55, 0, 1)
    return (np.stack([b, g, r], -1) * 255).astype(np.uint8)


def main():
    cal = json.load(open(os.path.join(OUT, "calib.json")))
    PW, PH = cal["PW"], cal["PH"]
    H = np.array(cal["H"], np.float32)
    g = cal["guitar_cam"]
    gcx, gcy = g["center"]
    gbox_cam = np.array(g["box"], np.float32)

    # guitar size in projector space
    gbox_proj = cv2.perspectiveTransform(gbox_cam.reshape(1, -1, 2), H).reshape(-1, 2)
    Cp = cv2.perspectiveTransform(np.array([[[gcx, gcy]]], np.float32), H)[0, 0]
    gw_p = gbox_proj[:, 0].max() - gbox_proj[:, 0].min()
    gh_p = gbox_proj[:, 1].max() - gbox_proj[:, 1].min()
    Wf = max(gw_p, gh_p) * CONFIG["flame_scale_w"]
    Hf = max(gw_p, gh_p) * CONFIG["flame_scale_h"]
    rf = CONFIG["rise_frac"]
    # axis-aligned flame quad in projector space; base below center, top above
    flame_quad = np.array([
        [Cp[0] - Wf/2, Cp[1] - Hf*rf],          # TL
        [Cp[0] + Wf/2, Cp[1] - Hf*rf],          # TR
        [Cp[0] + Wf/2, Cp[1] + Hf*(1-rf)],      # BR
        [Cp[0] - Wf/2, Cp[1] + Hf*(1-rf)],      # BL
    ], np.float32)
    print(f"guitar proj center=({Cp[0]:.0f},{Cp[1]:.0f}) size~({gw_p:.0f}x{gh_p:.0f}) "
          f"flame {Wf:.0f}x{Hf:.0f}")

    # ---- camera + projector ----
    devs = list(rs.context().query_devices())
    if devs:
        devs[0].hardware_reset()
    for _ in range(30):
        time.sleep(0.5)
        if list(rs.context().query_devices()):
            break
    time.sleep(2.0)
    p = rs.pipeline(); c = rs.config()
    c.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    prof = p.start(c)
    s = prof.get_device().query_sensors()[1]
    s.set_option(rs.option.enable_auto_exposure, 0)
    s.set_option(rs.option.exposure, CONFIG["exposure"])

    import pygame
    pygame.init()
    sizes = pygame.display.get_desktop_sizes()
    pi = max(range(len(sizes)), key=lambda i: sizes[i][0]*sizes[i][1])
    screen = pygame.display.set_mode(sizes[pi], pygame.NOFRAME, display=pi)

    def grab():
        for _ in range(15):
            p.wait_for_frames()
        return np.asanyarray(p.wait_for_frames().get_color_frame().get_data())

    # baseline (black) for evaluation
    screen.fill((0, 0, 0)); pygame.display.flip()
    for _ in range(6):
        pygame.event.pump(); time.sleep(0.03)
    time.sleep(0.5)
    cam_base = grab()

    tex_src = None
    cam_res = cam_base
    projected = np.zeros((PH, PW, 3), np.uint8)
    t0 = time.time()
    captured = False
    while time.time() - t0 < CONFIG["animate_secs"]:
        t = time.time() - t0
        tex = make_blue_flame(220, 360, t=t, intensity=CONFIG["intensity"])
        if tex_src is None:
            tex_src = np.array([[0, 0], [tex.shape[1]-1, 0],
                                [tex.shape[1]-1, tex.shape[0]-1], [0, tex.shape[0]-1]], np.float32)
        M = cv2.getPerspectiveTransform(tex_src, flame_quad)
        projected = cv2.warpPerspective(tex, M, (PW, PH))
        rgb = cv2.cvtColor(projected, cv2.COLOR_BGR2RGB)
        surf = pygame.image.frombuffer(rgb.tobytes(), (PW, PH), "RGB")
        screen.blit(surf, (0, 0)); pygame.display.flip()
        for _ in range(2):
            pygame.event.pump()
        if not captured and t > CONFIG["animate_secs"] * 0.5:
            time.sleep(0.2)
            cam_res = grab()
            captured = True
        time.sleep(0.03)

    screen.fill((0, 0, 0)); pygame.display.flip()
    p.stop(); pygame.quit()

    # ---- evaluate over the guitar region ----
    gm = np.zeros(cam_base.shape[:2], np.uint8)
    cv2.fillPoly(gm, [gbox_cam.astype(np.int32)], 255)
    sel = gm > 0
    metrics = {}
    if sel.sum() > 0:
        b = cam_res[:, :, 0][sel].astype(np.int16)
        r = cam_res[:, :, 2][sel].astype(np.int16)
        gain = (cv2.cvtColor(cam_res, cv2.COLOR_BGR2GRAY)[sel].astype(np.int16)
                - cv2.cvtColor(cam_base, cv2.COLOR_BGR2GRAY)[sel].astype(np.int16))
        metrics["guitar_blueness_B_minus_R"] = round(float((b - r).mean()), 1)
        metrics["guitar_brightness_gain"] = round(float(gain.mean()), 1)
        lit = (cv2.cvtColor(cam_res, cv2.COLOR_BGR2GRAY).astype(np.int16)
               - cv2.cvtColor(cam_base, cv2.COLOR_BGR2GRAY).astype(np.int16)) > 18
        metrics["coverage_on_guitar"] = round(float(lit[sel].mean()), 3)
        metrics["spill_off_guitar"] = round(float((lit & ~sel).sum() / max(1, lit.sum())), 3)
    print("METRICS:", json.dumps(metrics))

    def label(img, t):
        o = img.copy()
        cv2.rectangle(o, (0, 0), (o.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(o, t, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return o

    res_vis = cam_res.copy()
    cv2.drawContours(res_vis, [gbox_cam.astype(np.int32)], 0, (0, 140, 255), 2)
    proj_small = cv2.resize(projected, (848, 480))
    montage = np.vstack([
        np.hstack([label(res_vis, "RESULT: blue flame on guitar (cam)"),
                   label(cam_base, "baseline (projector black)")]),
        np.hstack([label(proj_small, "what we projected (flame canvas)"),
                   label(np.zeros_like(proj_small), "")]),
    ])
    cv2.imwrite(os.path.join(OUT, "flame.png"), montage)
    cv2.imwrite(os.path.join(OUT, "flame_result.png"), cam_res)
    print("saved flame.png, flame_result.png")


if __name__ == "__main__":
    main()

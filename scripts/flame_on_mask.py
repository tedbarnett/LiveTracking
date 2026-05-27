"""Blue flame conformed to the guitar BODY MASK.

Self-contained: dot-calibrate (camera->projector H) -> detect guitar body mask
-> warp mask into projector space -> fill it with animated blue fire -> project,
capture the result, score. Keep the scene static / stay out of frame during the
dot flashes.

Outputs scripts/out/flamemask.png (montage) + flamemask_result.png.
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
    exposure=700,
    dot_r=70,
    white_S_max=80, white_V_min=140, depth_margin_m=0.12, top_ignore_frac=0.40,
    animate_secs=4.0,
    fire_intensity=1.15,
)
DOT_FRACS = [(fx, fy) for fy in (0.2, 0.5, 0.8) for fx in (0.2, 0.5, 0.8)]


def blue_fire(w, h, t, intensity=1.0):
    """Animated blue fire field (BGR uint8): hot white-blue at the base, blue up top."""
    w, h = max(2, int(w)), max(2, int(h))
    yy = np.linspace(1.0, 0.0, h)[:, None]      # 1 bottom .. 0 top
    xx = np.linspace(0.0, 1.0, w)[None, :]
    # turbulent intensity from a few moving sine octaves
    n = (0.5 + 0.5*np.sin(xx*9 + t*3.0)) * (0.5 + 0.5*np.sin(yy*7 - t*4.0))
    n += 0.6*(0.5 + 0.5*np.sin(xx*19 - t*5.0)) * (0.5 + 0.5*np.sin(yy*15 + t*6.0))
    n /= 1.6
    i = np.clip((yy**0.7) * (0.55 + 0.75*n) * intensity, 0, 1)
    b = np.clip(i*1.5, 0, 1)
    g = np.clip(i*1.25 - 0.12, 0, 1)
    r = np.clip(i*1.15 - 0.55, 0, 1)
    return (np.stack([b, g, r], -1) * 255).astype(np.uint8)


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
    PW, PH = screen.get_size()
    print(f"projector {PW}x{PH}")

    def dot(xy=None):
        screen.fill((0, 0, 0))
        if xy is not None:
            pygame.draw.circle(screen, (255, 255, 255), (int(xy[0]), int(xy[1])), CONFIG["dot_r"])
        pygame.display.flip()
        for _ in range(6):
            pygame.event.pump(); time.sleep(0.03)

    def grab():
        for _ in range(18):
            p.wait_for_frames()
        f = align.process(p.wait_for_frames())
        return (np.asanyarray(f.get_color_frame().get_data()),
                np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32)*ds)

    # ---- dot calibration ----
    dot(None); time.sleep(0.7)
    cam_black, depth = grab()
    gblk = cv2.cvtColor(cam_black, cv2.COLOR_BGR2GRAY).astype(np.int16)
    campts, projpts = [], []
    diffs = []
    ch, cw = cam_black.shape[:2]
    for fx, fy in DOT_FRACS:
        pxy = (fx*PW, fy*PH)
        dot(pxy); time.sleep(0.2)
        cd, _ = grab()
        d = cv2.GaussianBlur(np.clip(cv2.cvtColor(cd, cv2.COLOR_BGR2GRAY).astype(np.int16)-gblk, 0, 255).astype(np.uint8), (0, 0), 5)
        _, mx, _, loc = cv2.minMaxLoc(d)
        diffs.append(int(mx))
        bxx, byy = loc
        on_border = bxx < 8 or byy < 8 or bxx > cw-8 or byy > ch-8
        if mx > 30 and not on_border:     # strong diff, not a frame-edge artifact
            campts.append([bxx, byy]); projpts.append([pxy[0], pxy[1]])
    dot(None)
    print(f"dot maxdiffs: {diffs}  kept {len(campts)}")
    if len(campts) < 4:
        print(f"calibration FAILED: only {len(campts)} dots detected"); p.stop(); pygame.quit(); return
    H, _ = cv2.findHomography(np.array(campts, np.float32), np.array(projpts, np.float32), cv2.RANSAC, 5.0)
    print(f"calibration: {len(campts)}/9 dots")
    print(f"H matrix:\n{H}")
    # Sanity: where do the 4 camera-frame corners map to in projector space?
    ch_d, cw_d = cam_black.shape[:2]
    corners_cam = np.array([[[0, 0]], [[cw_d - 1, 0]],
                             [[cw_d - 1, ch_d - 1]], [[0, ch_d - 1]]], dtype=np.float32)
    corners_proj = cv2.perspectiveTransform(corners_cam, H).reshape(-1, 2)
    print(f"cam frame corners map to projector:\n{corners_proj}")
    cv2.imwrite(os.path.join(OUT, "fom_baseline.png"), cam_black)

    # Map the 4 projector corners back into camera space so we can restrict
    # candidate-blob analysis to the region the projector can actually cover.
    # Anything outside this quad cannot be lit by the projector and must not
    # be considered as the "guitar" - otherwise off-axis white objects (e.g.
    # a storage box outside the projection field) hijack the area-based score.
    H_inv = np.linalg.inv(H)
    proj_corners = np.array([[[0, 0]], [[PW - 1, 0]],
                              [[PW - 1, PH - 1]], [[0, PH - 1]]], dtype=np.float32)
    proj_in_cam = cv2.perspectiveTransform(proj_corners, H_inv).reshape(-1, 2).astype(np.float32)
    print(f"projector quad in camera space:\n{proj_in_cam}")
    proj_quad_mask = np.zeros((ch_d, cw_d), dtype=np.uint8)
    cv2.fillConvexPoly(proj_quad_mask, proj_in_cam.astype(np.int32), 1)
    cv2.imwrite(os.path.join(OUT, "fom_proj_quad_in_cam.png"), proj_quad_mask * 255)

    # ---- guitar body mask (proven recipe) ----
    hsv = cv2.cvtColor(cam_black, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < CONFIG["white_S_max"]) & (hsv[:, :, 2] > CONFIG["white_V_min"])).astype(np.uint8)
    Hc = hsv[:, :, 0]
    red = (((Hc < 12) | (Hc > 168)) & (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 45)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    H_img, W_img = depth.shape
    up = depth[:int(0.4*H_img)]; upv = up[up > 0]
    wall_d = float(np.percentile(upv, 55)) if upv.size else 3.5
    near = ((depth > 0.3) & (depth < wall_d - CONFIG["depth_margin_m"])).astype(np.uint8)
    nod = (depth == 0).astype(np.uint8)
    cand = cv2.bitwise_and(white, cv2.bitwise_or(near, nod))
    cand[:int(CONFIG["top_ignore_frac"]*H_img)] = 0
    # Restrict candidates to the projector's reachable footprint.
    cand = cv2.bitwise_and(cand, proj_quad_mask)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    if n <= 1:
        print("no guitar"); p.stop(); pygame.quit(); return
    gi, best = None, -1.0
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < 400:
            continue
        blob = (lbl == i).astype(np.uint8)
        ring = cv2.subtract(cv2.dilate(blob, np.ones((17, 17), np.uint8)), blob)
        rf = float((red[ring > 0] > 0).mean()) if np.any(ring) else 0.0
        sc = a*(0.15+rf)
        if sc > best:
            best, gi = sc, i
    seed = (lbl == gi).astype(np.uint8)
    region = cv2.dilate(seed, np.ones((61, 61), np.uint8))
    # Also clip the dilation region to the projector footprint so the final
    # body contour can never bleed off the reachable area.
    region = cv2.bitwise_and(region, proj_quad_mask)
    body = cv2.morphologyEx(cv2.bitwise_and(white, region), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 200]
    body = np.zeros_like(body)
    if cnts:
        cv2.drawContours(body, [max(cnts, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    cam_mask = (body > 0).astype(np.uint8)
    print(f"cam_mask: {int(cam_mask.sum())} px, bbox in cam: ", end="")
    yc, xc = np.where(cam_mask > 0)
    if xc.size:
        print(f"({xc.min()},{yc.min()})-({xc.max()},{yc.max()}) shape={cam_mask.shape}")
    else:
        print("EMPTY")
    cv2.imwrite(os.path.join(OUT, "fom_cam_mask.png"), cam_mask * 255)
    # Annotated baseline showing where the mask landed in the camera frame
    overlay = cam_black.copy()
    overlay[cam_mask > 0] = (0, 255, 0)
    blended = cv2.addWeighted(cam_black, 0.5, overlay, 0.5, 0)
    cv2.imwrite(os.path.join(OUT, "fom_cam_mask_overlay.png"), blended)

    # ---- warp mask to projector space ----
    proj_mask = cv2.warpPerspective(cam_mask*255, H, (PW, PH), flags=cv2.INTER_NEAREST)
    proj_mask = (proj_mask > 127).astype(np.uint8)
    ys, xs = np.where(proj_mask > 0)
    cv2.imwrite(os.path.join(OUT, "fom_proj_mask.png"), proj_mask * 255)
    if xs.size == 0:
        print("mask did not map into projector (saved diagnostics to scripts/out/fom_*.png)")
        p.stop(); pygame.quit(); return
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max(), ys.max()
    pm3 = cv2.merge([proj_mask, proj_mask, proj_mask])
    print(f"proj mask bbox=({bx0},{by0},{bx1-bx0},{by1-by0})")

    # ---- animate blue fire clipped to the mask ----
    cam_res = cam_black
    t0 = time.time(); captured = False
    canvas = np.zeros((PH, PW, 3), np.uint8)
    while time.time()-t0 < CONFIG["animate_secs"]:
        t = time.time()-t0
        canvas[:] = 0
        fire = blue_fire(bx1-bx0+1, by1-by0+1, t, CONFIG["fire_intensity"])
        canvas[by0:by1+1, bx0:bx1+1] = fire
        frame = canvas * pm3            # clip fire to the guitar mask (pm3 is 0/1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.image.frombuffer(rgb.tobytes(), (PW, PH), "RGB")
        screen.blit(surf, (0, 0)); pygame.display.flip()
        for _ in range(2):
            pygame.event.pump()
        if not captured and t > CONFIG["animate_secs"]*0.5:
            time.sleep(0.2); cam_res, _ = grab(); captured = True
        time.sleep(0.03)
    screen.fill((0, 0, 0)); pygame.display.flip()
    p.stop(); pygame.quit()

    # ---- evaluate over the camera body mask ----
    sel = cam_mask > 0
    metrics = {}
    if sel.sum():
        b = cam_res[:, :, 0][sel].astype(np.int16); r = cam_res[:, :, 2][sel].astype(np.int16)
        gain = (cv2.cvtColor(cam_res, cv2.COLOR_BGR2GRAY)[sel].astype(np.int16)
                - cv2.cvtColor(cam_black, cv2.COLOR_BGR2GRAY)[sel].astype(np.int16))
        lit = (cv2.cvtColor(cam_res, cv2.COLOR_BGR2GRAY).astype(np.int16)
               - cv2.cvtColor(cam_black, cv2.COLOR_BGR2GRAY).astype(np.int16)) > 18
        metrics = dict(blueness_B_minus_R=round(float((b-r).mean()), 1),
                       brightness_gain=round(float(gain.mean()), 1),
                       coverage_on_body=round(float(lit[sel].mean()), 3),
                       spill_off_body=round(float((lit & ~sel).sum()/max(1, lit.sum())), 3))
    print("METRICS:", json.dumps(metrics))

    def label(img, tx):
        o = img.copy()
        cv2.rectangle(o, (0, 0), (o.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(o, tx, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return o
    res_vis = cam_res.copy()
    cv2.drawContours(res_vis, cv2.findContours(cam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 0, 255), 1)
    pm_small = cv2.resize((proj_mask*255), (848, 480))
    montage = np.vstack([
        np.hstack([label(res_vis, "RESULT: blue fire on body mask"), label(cam_black, "baseline")]),
        np.hstack([label(cv2.cvtColor(cam_mask*255, cv2.COLOR_GRAY2BGR), "camera body mask"),
                   label(cv2.cvtColor(pm_small, cv2.COLOR_GRAY2BGR), "mask warped to projector")]),
    ])
    cv2.imwrite(os.path.join(OUT, "flamemask.png"), montage)
    cv2.imwrite(os.path.join(OUT, "flamemask_result.png"), cam_res)
    print("saved flamemask.png, flamemask_result.png")


if __name__ == "__main__":
    main()

"""v2 calibration: image-registration by coordinate descent.

Idea (Ted's): project the camera's view back onto the scene and adjust the
camera->projector homography until the projection lines up with reality.

Concretely, per iteration:
  - warp the camera EDGE map into projector space with the current homography
  - project it; the camera sees those edges land somewhere on the real scene
  - score = overlap between where the projected edges landed and the real edges
  - coordinate-descend the 4 projector-corner positions (in camera px) to maximize it

Coarse init from a white-flood bounding box, then refine. Saves calib.json
(compatible with flame_v2.py) + a verification montage.

Keep the scene static and stay out of the camera's view while this runs (~1-2 min).
"""
import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
EXP = 700
SETTLE = 6


def order_quad(pts):
    pts = np.array(pts, np.float32).reshape(-1, 2)
    s = pts.sum(1); d = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


class Rig:
    def __init__(self):
        devs = list(rs.context().query_devices())
        if devs:
            devs[0].hardware_reset()
        for _ in range(30):
            time.sleep(0.5)
            if list(rs.context().query_devices()):
                break
        time.sleep(2.0)
        self.p = rs.pipeline(); cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        prof = self.p.start(cfg)
        self.ds = prof.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        s = prof.get_device().query_sensors()[1]
        s.set_option(rs.option.enable_auto_exposure, 0)
        s.set_option(rs.option.exposure, EXP)
        import pygame
        self.pygame = pygame
        pygame.init()
        sizes = pygame.display.get_desktop_sizes()
        pi = max(range(len(sizes)), key=lambda i: sizes[i][0]*sizes[i][1])
        self.screen = pygame.display.set_mode(sizes[pi], pygame.NOFRAME, display=pi)
        self.PW, self.PH = self.screen.get_size()

    def project(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        surf = self.pygame.image.frombuffer(rgb.tobytes(), (bgr.shape[1], bgr.shape[0]), "RGB")
        self.screen.blit(surf, (0, 0)); self.pygame.display.flip()
        for _ in range(4):
            self.pygame.event.pump(); time.sleep(0.02)

    def black(self):
        self.screen.fill((0, 0, 0)); self.pygame.display.flip()
        for _ in range(4):
            self.pygame.event.pump(); time.sleep(0.02)

    def grab(self):
        for _ in range(SETTLE):
            self.p.wait_for_frames()
        f = self.align.process(self.p.wait_for_frames())
        return (np.asanyarray(f.get_color_frame().get_data()),
                np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * self.ds)

    def close(self):
        self.p.stop(); self.pygame.quit()


def main():
    rig = Rig()
    PW, PH = rig.PW, rig.PH
    proj_corners = np.array([[0, 0], [PW-1, 0], [PW-1, PH-1], [0, PH-1]], np.float32)
    print(f"projector {PW}x{PH}")

    # baseline (real scene)
    rig.black(); time.sleep(0.6)
    cam_base, depth = rig.grab()
    S = cv2.cvtColor(cam_base, cv2.COLOR_BGR2GRAY)
    # real edges (for scoring) + a blurred version giving a convergence basin
    E_real = cv2.Canny(cv2.GaussianBlur(S, (3, 3), 0), 40, 120)
    E_blur = cv2.GaussianBlur(E_real.astype(np.float32), (0, 0), 6)
    E_blur /= (E_blur.max() + 1e-6)
    # the edges we PROJECT (bright thin lines on black)
    E_proj_src = cv2.cvtColor(cv2.dilate(E_real, np.ones((3, 3), np.uint8)), cv2.COLOR_GRAY2BGR)

    # coarse init: white-flood for a rough CENTER only (it under-counts size on
    # dark surfaces), then start from a generous rectangle and let descent shrink it.
    rig.screen.fill((255, 255, 255)); rig.pygame.display.flip()
    for _ in range(4):
        rig.pygame.event.pump(); time.sleep(0.02)
    time.sleep(0.6)
    cam_w, _ = rig.grab()
    rig.black()
    d = np.clip(cv2.cvtColor(cam_w, cv2.COLOR_BGR2GRAY).astype(np.int16) - S.astype(np.int16), 0, 255).astype(np.uint8)
    _, m = cv2.threshold(d, 15, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H_img, W_img = S.shape
    if cnts:
        bx, by, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        cx0, cy0 = bx + bw / 2.0, by + bh / 2.0
    else:
        cx0, cy0 = W_img / 2.0, H_img / 2.0
    # generous starting rectangle (half-sizes); descent will fit it
    params = [cx0, cy0, 0.28 * W_img, 0.26 * H_img]   # cx, cy, hw, hh
    print(f"init center=({cx0:.0f},{cy0:.0f}) rect half=({params[2]:.0f},{params[3]:.0f})")

    def score(qc):
        H_p2c = cv2.getPerspectiveTransform(proj_corners, qc.astype(np.float32))
        H_c2p = np.linalg.inv(H_p2c)
        proj_img = cv2.warpPerspective(E_proj_src, H_c2p, (PW, PH))
        rig.project(proj_img)
        lit, _ = rig.grab()
        P = np.clip(cv2.cvtColor(lit, cv2.COLOR_BGR2GRAY).astype(np.float32) - S.astype(np.float32), 0, None)
        P /= (P.max() + 1e-6)
        return float((P * E_blur).sum())

    def rect_to_q(cx, cy, hw, hh):
        return np.array([[cx-hw, cy-hh], [cx+hw, cy-hh],
                         [cx+hw, cy+hh], [cx-hw, cy+hh]], np.float32)

    # ---- Stage 1: coarse rectangle (center + size), big steps ----
    best = score(rect_to_q(*params))
    history = [best]
    print(f"init score = {best:.1f}")
    for step in (48, 24, 12, 6):
        improved, passes = True, 0
        while improved and passes < 4:
            improved, passes = False, passes + 1
            for i in range(4):
                for sgn in (+1, -1):
                    cand = list(params); cand[i] += sgn * step
                    if cand[2] < 25 or cand[3] < 25:
                        continue
                    sc = score(rect_to_q(*cand))
                    if sc > best + 1e-6:
                        params, best, improved = cand, sc, True
        history.append(best)
        print(f"  [rect] step={step:2d} score={best:.1f} params={[round(v,1) for v in params]}")
    q = rect_to_q(*params)

    # ---- Stage 2: refine the 4 corners (projective), small steps ----
    for step in (16, 8, 4, 2):
        improved, passes = True, 0
        while improved and passes < 3:
            improved, passes = False, passes + 1
            for i in range(4):
                for ax in (0, 1):
                    for sgn in (+1, -1):
                        cand = q.copy(); cand[i, ax] += sgn * step
                        sc = score(cand)
                        if sc > best + 1e-6:
                            q, best, improved = cand, sc, True
        history.append(best)
        print(f"  [corner] step={step:2d} score={best:.1f}")
    print(f"final score = {best:.1f}  (init {history[0]:.1f})")

    H_p2c = cv2.getPerspectiveTransform(proj_corners, q.astype(np.float32))
    H_c2p = np.linalg.inv(H_p2c)

    # ---- verification: project the real scene warped; should overlay reality ----
    proj_scene = cv2.warpPerspective(cam_base, H_c2p, (PW, PH))
    rig.project(proj_scene); time.sleep(0.6)
    cam_v, _ = rig.grab()
    rig.black()

    # ---- detect guitar (white + near) and map to projector ----
    hsv = cv2.cvtColor(cam_base, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 140)).astype(np.uint8) * 255
    up = depth[:int(0.4*depth.shape[0])]; upv = up[up > 0]
    wall_d = float(np.percentile(upv, 60)) if upv.size else 3.5
    near = ((depth > 0.3) & (depth < wall_d - 0.15)).astype(np.uint8) * 255
    nod = (depth == 0).astype(np.uint8) * 255
    guitar = cv2.bitwise_and(white, cv2.bitwise_or(near, nod))
    guitar[:int(0.40*guitar.shape[0])] = 0
    guitar = cv2.morphologyEx(guitar, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    gc, _ = cv2.findContours(guitar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gc = [x for x in gc if cv2.contourArea(x) > 500]
    out = {"PW": PW, "PH": PH, "H": H_c2p.tolist(), "wall_d": round(wall_d, 3),
           "cam_corners": q.tolist(), "init_score": round(history[0], 1),
           "final_score": round(best, 1)}
    gvis = cam_base.copy()
    if gc:
        rect = cv2.minAreaRect(max(gc, key=cv2.contourArea))
        box = cv2.boxPoints(rect)
        out["guitar_cam"] = {"center": list(rect[0]), "size": list(rect[1]),
                             "angle": rect[2], "box": box.tolist(),
                             "area": cv2.contourArea(max(gc, key=cv2.contourArea))}
        cv2.drawContours(gvis, [box.astype(np.int32)], 0, (0, 140, 255), 2)

    rig.close()

    # montage
    def label(img, t):
        o = img.copy()
        cv2.rectangle(o, (0, 0), (o.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(o, t, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return o
    # overlay projected-scene-as-seen vs real edges
    Pv = np.clip(cv2.cvtColor(cam_v, cv2.COLOR_BGR2GRAY).astype(np.int16) - S.astype(np.int16), 0, 255).astype(np.uint8)
    ov = cam_base.copy()
    ov[E_real > 0] = (0, 255, 0)             # real edges green
    ov[Pv > 40] = (0, 0, 255)                # where projection landed, red
    montage = np.vstack([
        np.hstack([label(cam_base, "real scene (baseline)"), label(cam_v, "projected scene warped back (cam)")]),
        np.hstack([label(ov, "green=real edges  red=projected light (aligned=overlap)"), label(gvis, "guitar mask via this calibration")]),
    ])
    cv2.imwrite(os.path.join(OUT, "reg_verify.png"), montage)
    with open(os.path.join(OUT, "calib.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("METRICS:", json.dumps({k: out[k] for k in ("init_score", "final_score", "wall_d")}))
    print("saved reg_verify.png, calib.json")


if __name__ == "__main__":
    main()

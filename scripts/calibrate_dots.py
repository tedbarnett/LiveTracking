"""Robust camera<->projector calibration via projected dots, + verification.

  1. project bright dots at known projector positions; find each in the camera (diff).
     -> accurate correspondences across the WHOLE projected area (works on dark sofa too).
  2. solve homography H (camera px -> projector px).
  3. detect the white guitar (white + near; no projection-rect gating).
  4. VERIFY: project a marker at the guitar's mapped projector position; confirm via
     the camera that it lands on the guitar.

Saves dots/guitar/verify overlays + calib.json. Tune and re-run.
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
DOT_R = 70                      # projector-px radius of each calibration dot
DOT_FRACS = [(fx, fy) for fy in (0.2, 0.5, 0.8) for fx in (0.2, 0.5, 0.8)]
GUITAR_S_MAX = 80
GUITAR_V_MIN = 140
GUITAR_TOP_FRAC = 0.40          # ignore wall: search guitar in lower part of frame


def main():
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
    c.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    prof = p.start(c)
    ds = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    s = prof.get_device().query_sensors()[1]
    s.set_option(rs.option.enable_auto_exposure, 0)
    s.set_option(rs.option.exposure, EXP)

    import pygame
    pygame.init()
    sizes = pygame.display.get_desktop_sizes()
    pi = max(range(len(sizes)), key=lambda i: sizes[i][0]*sizes[i][1])
    screen = pygame.display.set_mode(sizes[pi], pygame.NOFRAME, display=pi)
    PW, PH = screen.get_size()
    print(f"projector {PW}x{PH}")

    def show_dot(proj_xy=None):
        screen.fill((0, 0, 0))
        if proj_xy is not None:
            pygame.draw.circle(screen, (255, 255, 255),
                               (int(proj_xy[0]), int(proj_xy[1])), DOT_R)
        pygame.display.flip()
        for _ in range(6):
            pygame.event.pump(); time.sleep(0.03)

    def grab():
        for _ in range(20):
            p.wait_for_frames()
        f = align.process(p.wait_for_frames())
        return (np.asanyarray(f.get_color_frame().get_data()),
                np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32)*ds)

    # ---- baseline ----
    show_dot(None); time.sleep(0.8)
    cam_black, depth = grab()
    gblk = cv2.cvtColor(cam_black, cv2.COLOR_BGR2GRAY).astype(np.int16)

    # ---- project dots, find each in camera ----
    cam_pts, proj_pts = [], []
    dotvis = cam_black.copy()
    for i, (fx, fy) in enumerate(DOT_FRACS):
        proj_xy = (fx * PW, fy * PH)
        show_dot(proj_xy); time.sleep(0.25)
        cam_dot, _ = grab()
        d = np.clip(cv2.cvtColor(cam_dot, cv2.COLOR_BGR2GRAY).astype(np.int16) - gblk, 0, 255).astype(np.uint8)
        d = cv2.GaussianBlur(d, (0, 0), 5)
        _, mx, _, loc = cv2.minMaxLoc(d)
        if mx > 25:
            cam_pts.append([loc[0], loc[1]]); proj_pts.append([proj_xy[0], proj_xy[1]])
            cv2.circle(dotvis, loc, 6, (0, 255, 0), -1)
            cv2.putText(dotvis, str(i), (loc[0]+6, loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(f"dot {i} proj=({proj_xy[0]:.0f},{proj_xy[1]:.0f}) maxdiff={mx} cam={loc if mx>25 else 'MISS'}")
    show_dot(None)
    print(f"got {len(cam_pts)}/{len(DOT_FRACS)} correspondences")

    H = None
    if len(cam_pts) >= 4:
        H, _ = cv2.findHomography(np.array(cam_pts, np.float32),
                                  np.array(proj_pts, np.float32), cv2.RANSAC, 5.0)

    # ---- detect guitar (white + near, lower region) ----
    hsv = cv2.cvtColor(cam_black, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < GUITAR_S_MAX) & (hsv[:, :, 2] > GUITAR_V_MIN)).astype(np.uint8)*255
    wall_d = float(np.median(depth[(depth > 0)]))  # rough; refine below
    # use a robust wall estimate = far mode in upper area
    up = depth[:int(0.4*depth.shape[0]), :]
    upv = up[up > 0]
    wall_d = float(np.percentile(upv, 60)) if upv.size else wall_d
    near = ((depth > 0.3) & (depth < wall_d - 0.15)).astype(np.uint8)*255
    nod = (depth == 0).astype(np.uint8)*255
    guitar = cv2.bitwise_and(white, cv2.bitwise_or(near, nod))
    guitar[:int(GUITAR_TOP_FRAC*guitar.shape[0]), :] = 0
    guitar = cv2.morphologyEx(guitar, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    guitar = cv2.morphologyEx(guitar, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    gc, _ = cv2.findContours(guitar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gc = [x for x in gc if cv2.contourArea(x) > 500]

    gvis = cam_black.copy()
    result = {"PW": PW, "PH": PH, "wall_d": round(wall_d, 3),
              "n_corr": len(cam_pts), "H": H.tolist() if H is not None else None}
    guitar_cam = None
    if gc:
        gb = max(gc, key=cv2.contourArea)
        rect = cv2.minAreaRect(gb)
        guitar_cam = rect
        box = cv2.boxPoints(rect)
        cv2.drawContours(gvis, [box.astype(np.int32)], 0, (0, 140, 255), 2)
        cv2.circle(gvis, (int(rect[0][0]), int(rect[0][1])), 5, (0, 0, 255), -1)
        result["guitar_cam"] = {"center": list(rect[0]), "size": list(rect[1]), "angle": rect[2],
                                "box": box.tolist(), "area": cv2.contourArea(gb)}
        print(f"guitar cam center={rect[0]} size={rect[1]} angle={rect[2]:.0f}")
    else:
        print("guitar NOT found")

    # ---- verify: project marker at guitar's mapped projector position ----
    if H is not None and guitar_cam is not None:
        gcx, gcy = guitar_cam[0]
        proj_pt = cv2.perspectiveTransform(np.array([[[gcx, gcy]]], np.float32), H)[0, 0]
        show_dot((proj_pt[0], proj_pt[1])); time.sleep(0.8)
        cam_v, _ = grab()
        show_dot(None)
        dv = np.clip(cv2.cvtColor(cam_v, cv2.COLOR_BGR2GRAY).astype(np.int16) - gblk, 0, 255).astype(np.uint8)
        dv = cv2.GaussianBlur(dv, (0, 0), 5)
        _, mxv, _, locv = cv2.minMaxLoc(dv)
        err = float(np.hypot(locv[0]-gcx, locv[1]-gcy))
        result["verify_proj_pt"] = [float(proj_pt[0]), float(proj_pt[1])]
        result["verify_marker_cam"] = [locv[0], locv[1]]
        result["verify_error_px"] = round(err, 1)
        print(f"VERIFY: marker landed at cam {locv}, guitar at ({gcx:.0f},{gcy:.0f}), error={err:.1f}px (maxdiff={mxv})")
        vv = cam_v.copy()
        cv2.circle(vv, (int(gcx), int(gcy)), 8, (0, 140, 255), 2)   # guitar
        cv2.circle(vv, locv, 8, (0, 0, 255), 2)                     # where marker landed
        cv2.line(vv, (int(gcx), int(gcy)), locv, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(OUT, "verify.png"), vv)

    p.stop(); pygame.quit()
    cv2.imwrite(os.path.join(OUT, "dots.png"), dotvis)
    cv2.imwrite(os.path.join(OUT, "guitar.png"), gvis)
    with open(os.path.join(OUT, "calib.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("saved dots.png, guitar.png, verify.png, calib.json")


if __name__ == "__main__":
    main()

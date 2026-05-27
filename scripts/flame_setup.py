"""Stage 1+2 for 'blue flame on the white guitar':
  - fullscreen white-flood on the JMGO -> projection quad in camera coords -> H (cam->proj)
  - detect the white guitar (bright + low-saturation, in front of the wall, inside the
    projection area)
  - save annotated camera snapshot + calibration json for the flame stage.
Run with the venv python. Saves to scripts/out/.
"""
import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def order_quad(pts):
    """Return TL, TR, BR, BL."""
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


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
    csensor.set_option(rs.option.exposure, 150)
except Exception as e:
    print("exposure lock failed:", e)

# ---------- projector (fullscreen on the JMGO) ----------
import pygame
pygame.init()
sizes = pygame.display.get_desktop_sizes()
proj_idx = max(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1])  # biggest = 4K JMGO
screen = pygame.display.set_mode(sizes[proj_idx], pygame.FULLSCREEN, display=proj_idx)
PW, PH = screen.get_size()
print(f"projector {PW}x{PH} on display {proj_idx}; camera 848x480")


def show(rgb):
    screen.fill(rgb)
    pygame.display.flip()
    for _ in range(5):
        pygame.event.pump()
        time.sleep(0.02)


def grab():
    for _ in range(25):
        p.wait_for_frames()
    f = align.process(p.wait_for_frames())
    color = np.asanyarray(f.get_color_frame().get_data())
    depth = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * depth_scale
    return color, depth


# ---------- calibrate: black vs white ----------
show((0, 0, 0)); time.sleep(1.0); cam_black, depth_black = grab()
show((255, 255, 255)); time.sleep(1.0); cam_white, _ = grab()

gb = cv2.cvtColor(cam_black, cv2.COLOR_BGR2GRAY).astype(np.int16)
gw = cv2.cvtColor(cam_white, cv2.COLOR_BGR2GRAY).astype(np.int16)
diff = np.clip(gw - gb, 0, 255).astype(np.uint8)
_, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
big = max(cnts, key=cv2.contourArea)
peri = cv2.arcLength(big, True)
approx = cv2.approxPolyDP(big, 0.02 * peri, True)
if len(approx) == 4:
    cam_quad = order_quad(approx.reshape(-1, 2))
else:
    cam_quad = order_quad(cv2.boxPoints(cv2.minAreaRect(big)))
proj_quad = np.array([[0, 0], [PW - 1, 0], [PW - 1, PH - 1], [0, PH - 1]], dtype=np.float32)
H_cam2proj = cv2.getPerspectiveTransform(cam_quad, proj_quad)
x, y, w, h = cv2.boundingRect(big)
print(f"projection rect in camera: x={x} y={y} w={w} h={h}")

proj_mask = np.zeros(diff.shape, np.uint8)
cv2.fillPoly(proj_mask, [cam_quad.astype(np.int32)], 255)

# ---------- detect white guitar (projector black) ----------
show((0, 0, 0)); time.sleep(0.8); cam, depth = grab()
hsv = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
S, V = hsv[:, :, 1], hsv[:, :, 2]
white = ((S < 70) & (V > 150)).astype(np.uint8) * 255

# wall distance from the black-projection depth, inside the projection area
wall_vals = depth_black[(proj_mask > 0) & (depth_black > 0)]
wall_d = float(np.median(wall_vals)) if wall_vals.size else 0.0
print(f"wall distance ~{wall_d:.2f} m")
near = ((depth > 0.3) & (depth < wall_d - 0.15)).astype(np.uint8) * 255

guitar = cv2.bitwise_and(white, proj_mask)
# keep white pixels that are either in front of the wall, or have no depth (glossy holes)
nodepth = (depth == 0).astype(np.uint8) * 255
front = cv2.bitwise_or(near, nodepth)
guitar = cv2.bitwise_and(guitar, front)
guitar = cv2.morphologyEx(guitar, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
guitar = cv2.morphologyEx(guitar, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))

gcnts, _ = cv2.findContours(guitar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
gcnts = [c for c in gcnts if cv2.contourArea(c) > 400]
result = {"PW": PW, "PH": PH, "H_cam2proj": H_cam2proj.tolist(),
          "cam_quad": cam_quad.tolist(), "wall_d": wall_d}

vis = cam.copy()
cv2.polylines(vis, [cam_quad.astype(np.int32)], True, (0, 255, 0), 2)
if gcnts:
    gbig = max(gcnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(gbig)
    box = cv2.boxPoints(rect)
    cx, cy = rect[0]
    cv2.drawContours(vis, [box.astype(np.int32)], 0, (255, 100, 0), 2)
    cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    result["guitar_rect"] = {"center": [float(cx), float(cy)],
                             "size": [float(rect[1][0]), float(rect[1][1])],
                             "angle": float(rect[2]),
                             "area": float(cv2.contourArea(gbig)),
                             "box": box.tolist()}
    print(f"guitar detected: center=({cx:.0f},{cy:.0f}) size={rect[1][0]:.0f}x{rect[1][1]:.0f} "
          f"angle={rect[2]:.0f} area={cv2.contourArea(gbig):.0f}")
else:
    print("NO guitar candidate found")

show((0, 0, 0))
p.stop()
pygame.quit()

cv2.imwrite(os.path.join(OUT, "flame_detect.png"), vis)
cv2.imwrite(os.path.join(OUT, "flame_whitemask.png"), guitar)
with open(os.path.join(OUT, "calib.json"), "w") as f:
    json.dump(result, f, indent=2)
print("saved flame_detect.png, flame_whitemask.png, calib.json to", OUT)

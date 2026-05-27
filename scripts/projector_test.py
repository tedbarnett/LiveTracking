"""Prove the camera<->projector loop on the laptop:
  1. Open a fullscreen pygame window on the JMGO (chosen by pygame display index).
  2. Project BLACK, capture camera; project WHITE, capture camera.
  3. Diff the two -> the projection rectangle in CAMERA coordinates.
This validates projector output AND locates where it lands in the camera view.
"""
import os
import time

import cv2
import numpy as np
import pygame
import pyrealsense2 as rs

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# ---------- camera: reset, then start color ----------
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
prof = p.start(c)
# lock exposure so the diff isn't fought by auto-exposure (README standing rule)
try:
    csensor = prof.get_device().query_sensors()[1]
    csensor.set_option(rs.option.enable_auto_exposure, 0)
    csensor.set_option(rs.option.exposure, 150)
except Exception as e:
    print("exposure lock failed:", e)

# ---------- projector: fullscreen on the JMGO ----------
pygame.init()
sizes = pygame.display.get_desktop_sizes()
print("pygame displays:", sizes)
# pick the 1280x720 display (the JMGO); fallback to last index
proj_idx = next((i for i, s in enumerate(sizes) if tuple(s) == (1280, 720)),
                len(sizes) - 1)
print("using display index", proj_idx)
screen = pygame.display.set_mode((1280, 720), pygame.NOFRAME, display=proj_idx)


def show(color_rgb):
    screen.fill(color_rgb)
    pygame.display.flip()
    for _ in range(5):
        pygame.event.pump()
        time.sleep(0.02)


def grab():
    for _ in range(20):
        p.wait_for_frames()
    f = p.wait_for_frames()
    return np.asanyarray(f.get_color_frame().get_data())


show((0, 0, 0))
time.sleep(1.0)
cam_black = grab()
show((255, 255, 255))
time.sleep(1.0)
cam_white = grab()

p.stop()
pygame.quit()

# ---------- diff -> projection rectangle ----------
gb = cv2.cvtColor(cam_black, cv2.COLOR_BGR2GRAY).astype(np.int16)
gw = cv2.cvtColor(cam_white, cv2.COLOR_BGR2GRAY).astype(np.int16)
diff = np.clip(gw - gb, 0, 255).astype(np.uint8)
print(f"diff mean {diff.mean():.1f}, max {int(diff.max())}, "
      f">30 px {int((diff > 30).sum())}")

_, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
vis = cam_white.copy()
if cnts:
    big = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(big)
    area = cv2.contourArea(big)
    print(f"projection rect in camera coords: x={x} y={y} w={w} h={h} area={int(area)}")
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
else:
    print("NO projection detected in camera view -- window may be on the wrong display")

cv2.imwrite(os.path.join(OUT, "proj_black.png"), cam_black)
cv2.imwrite(os.path.join(OUT, "proj_white.png"), cam_white)
cv2.imwrite(os.path.join(OUT, "proj_diff.png"),
            cv2.applyColorMap(diff, cv2.COLORMAP_JET))
cv2.imwrite(os.path.join(OUT, "proj_rect.png"), vis)
print("saved to", OUT)

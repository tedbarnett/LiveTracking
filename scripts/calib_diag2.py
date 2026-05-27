"""Locked-exposure calibration sweep (static scene).
For each exposure: project black, then white, diff -> projection rectangle.
Saves a diff heatmap + rect overlay per exposure so we can pick the best."""
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
EXPOSURES = [120, 300, 600, 1200]

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
csensor = prof.get_device().query_sensors()[1]

import pygame
pygame.init()
sizes = pygame.display.get_desktop_sizes()
proj_idx = max(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1])
screen = pygame.display.set_mode(sizes[proj_idx], pygame.NOFRAME, display=proj_idx)
PW, PH = screen.get_size()
print(f"projector {PW}x{PH}")


def show(rgb):
    screen.fill(rgb); pygame.display.flip()
    for _ in range(8):
        pygame.event.pump(); time.sleep(0.03)


def grab():
    for _ in range(25):
        p.wait_for_frames()
    return np.asanyarray(p.wait_for_frames().get_color_frame().get_data())


csensor.set_option(rs.option.enable_auto_exposure, 0)
best = None
for exp in EXPOSURES:
    csensor.set_option(rs.option.exposure, exp)
    time.sleep(0.3)
    show((0, 0, 0)); time.sleep(0.8); blk = grab()
    show((255, 255, 255)); time.sleep(0.8); wht = grab()
    diff = np.clip(cv2.cvtColor(wht, cv2.COLOR_BGR2GRAY).astype(np.int16)
                   - cv2.cvtColor(blk, cv2.COLOR_BGR2GRAY).astype(np.int16), 0, 255).astype(np.uint8)
    _, m = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = bbox = 0
    vis = wht.copy()
    if cnts:
        big = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(big)
        bbox = cv2.boundingRect(big)
        cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
    print(f"exp={exp:5d}  blkMean={blk.mean():5.1f} whtMean={wht.mean():5.1f} "
          f"diffMax={int(diff.max()):3d} >30={int((diff>30).sum()):6d}  rectArea={int(area):6d} bbox={bbox}")
    cv2.imwrite(os.path.join(OUT, f"calib_e{exp}_white.png"), vis)
    cv2.imwrite(os.path.join(OUT, f"calib_e{exp}_diff.png"), cv2.applyColorMap(diff, cv2.COLORMAP_JET))
    if area > (best[0] if best else 0):
        best = (area, exp)

show((0, 0, 0))
p.stop(); pygame.quit()
print("best exposure by projection area:", best)

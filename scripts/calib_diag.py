"""Diagnostic: what does the camera see when the projector shows black vs white?
Captures with auto-exposure (to actually see the lit room) and saves frames + diff."""
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

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
# AUTO exposure: we want to clearly see the scene for this diagnostic
try:
    prof.get_device().query_sensors()[1].set_option(rs.option.enable_auto_exposure, 1)
except Exception as e:
    print("ae set failed", e)

import pygame
pygame.init()
sizes = pygame.display.get_desktop_sizes()
proj_idx = max(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1])
screen = pygame.display.set_mode(sizes[proj_idx], pygame.NOFRAME, display=proj_idx)
PW, PH = screen.get_size()
print(f"projector {PW}x{PH} on display {proj_idx}")


def show(rgb):
    screen.fill(rgb); pygame.display.flip()
    for _ in range(10):
        pygame.event.pump(); time.sleep(0.03)


def grab():
    for _ in range(40):  # let auto-exposure settle
        p.wait_for_frames()
    return np.asanyarray(p.wait_for_frames().get_color_frame().get_data())


show((0, 0, 0)); time.sleep(1.5); blk = grab()
show((255, 255, 255)); time.sleep(1.5); wht = grab()
show((0, 0, 0))
p.stop(); pygame.quit()

diff = np.clip(cv2.cvtColor(wht, cv2.COLOR_BGR2GRAY).astype(np.int16)
               - cv2.cvtColor(blk, cv2.COLOR_BGR2GRAY).astype(np.int16), 0, 255).astype(np.uint8)
print(f"black mean {blk.mean():.1f}  white mean {wht.mean():.1f}  diff mean {diff.mean():.1f} max {int(diff.max())}")
for th in (15, 30, 50, 80):
    print(f"  diff > {th}: {int((diff > th).sum())} px")
cv2.imwrite(os.path.join(OUT, "diag_black.png"), blk)
cv2.imwrite(os.path.join(OUT, "diag_white.png"), wht)
cv2.imwrite(os.path.join(OUT, "diag_diff.png"), cv2.applyColorMap(diff, cv2.COLORMAP_JET))
print("saved diag_black/white/diff.png")

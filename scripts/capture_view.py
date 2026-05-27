"""Grab one aligned RGB + depth frame from the D455 and save visualizations.
Hardware-resets first (this D455 needs it to stream color+depth together)."""
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# --- reset to clear the stuck dual-stream state ---
devs = list(rs.context().query_devices())
if devs:
    devs[0].hardware_reset()
for _ in range(30):
    time.sleep(0.5)
    if list(rs.context().query_devices()):
        break
time.sleep(2.0)

# --- stream color + depth, align depth to color ---
p = rs.pipeline()
c = rs.config()
c.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
c.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
prof = p.start(c)
depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

# let auto-exposure settle
for _ in range(30):
    p.wait_for_frames()

f = align.process(p.wait_for_frames())
color = np.asanyarray(f.get_color_frame().get_data())
depth_raw = np.asanyarray(f.get_depth_frame().get_data())
p.stop()

depth_m = depth_raw.astype(np.float32) * depth_scale

# colorize depth: clamp 0.3-4.0 m, JET colormap, no-data -> black
valid = depth_m > 0
near, far = 0.3, 4.0
norm = np.clip((depth_m - near) / (far - near), 0, 1)
depth_vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
depth_vis[~valid] = (0, 0, 0)

cv2.imwrite(os.path.join(OUT, "rs_color.png"), color)
cv2.imwrite(os.path.join(OUT, "rs_depth.png"), depth_vis)

# side-by-side
combo = np.hstack([color, depth_vis])
cv2.imwrite(os.path.join(OUT, "rs_combo.png"), combo)

vals = depth_m[valid]
print("color:", color.shape, "mean", round(float(color.mean()), 1))
print("depth valid px:", int(valid.sum()), f"({100*valid.mean():.0f}%)")
if vals.size:
    print(f"depth range: {vals.min():.2f}-{vals.max():.2f} m, median {np.median(vals):.2f} m")
print("saved:", OUT)

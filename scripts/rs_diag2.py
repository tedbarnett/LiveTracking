"""Categorize what the pipeline delivers with both streams enabled:
are framesets arriving but unmatched (sync/metadata issue), or not at all (USB)?
Also reports timestamp domains, which is the usual culprit."""
import time

import pyrealsense2 as rs

p = rs.pipeline()
c = rs.config()
c.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
c.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
prof = p.start(c)
time.sleep(1.0)

both = color_only = depth_only = empty = 0
dom_color = dom_depth = None
for _ in range(150):
    fs = p.poll_for_frames()
    if not fs:
        empty += 1
    else:
        cf = fs.get_color_frame()
        df = fs.get_depth_frame()
        if cf and dom_color is None:
            dom_color = cf.get_frame_timestamp_domain()
        if df and dom_depth is None:
            dom_depth = df.get_frame_timestamp_domain()
        if cf and df:
            both += 1
        elif cf:
            color_only += 1
        elif df:
            depth_only += 1
    time.sleep(0.02)

p.stop()
print(f"framesets over ~3s of polling:")
print(f"  both color+depth : {both}")
print(f"  color only       : {color_only}")
print(f"  depth only       : {depth_only}")
print(f"  empty poll       : {empty}")
print(f"  color ts domain  : {dom_color}")
print(f"  depth ts domain  : {dom_depth}")

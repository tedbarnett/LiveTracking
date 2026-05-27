"""Hardware-reset the D455, wait for re-enumeration, then retry dual stream."""
import time

import numpy as np
import pyrealsense2 as rs

ctx = rs.context()
devs = list(ctx.query_devices())
print("before reset: devices =", len(devs))
if devs:
    devs[0].hardware_reset()
    print("hardware_reset() sent; waiting for re-enumeration...")

# wait up to 15s for the device to come back
for i in range(30):
    time.sleep(0.5)
    devs = list(rs.context().query_devices())
    if devs:
        print(f"re-enumerated after ~{(i+1)*0.5:.1f}s")
        break
else:
    print("device did not re-enumerate")
    raise SystemExit(1)

time.sleep(2.0)  # settle

p = rs.pipeline()
c = rs.config()
c.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
c.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
p.start(c)
try:
    for _ in range(15):
        p.wait_for_frames(10000)
    f = p.wait_for_frames(10000)
    col = np.asanyarray(f.get_color_frame().get_data())
    dep = np.asanyarray(f.get_depth_frame().get_data())
    print(f"DUAL STREAM OK after reset: color {col.shape} depth {dep.shape} "
          f"depth_nonzero {int((dep > 0).sum())}")
finally:
    p.stop()

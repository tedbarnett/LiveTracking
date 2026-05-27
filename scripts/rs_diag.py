"""RealSense dual-stream diagnostic. Finds a color+depth config that delivers
coherent framesets on this machine. Run with the venv python."""
import sys
import time

import numpy as np
import pyrealsense2 as rs


def try_config(cw, ch, cfps, dw, dh, dfps, warmup=10, timeout_ms=10000):
    p = rs.pipeline()
    c = rs.config()
    c.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, cfps)
    c.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
    label = f"color {cw}x{ch}@{cfps} + depth {dw}x{dh}@{dfps}"
    try:
        p.start(c)
    except Exception as e:
        print(f"  FAIL start  {label}: {e}")
        return False
    try:
        for _ in range(warmup):
            p.wait_for_frames(timeout_ms)
        f = p.wait_for_frames(timeout_ms)
        col = np.asanyarray(f.get_color_frame().get_data())
        dep = np.asanyarray(f.get_depth_frame().get_data())
        print(f"  OK          {label}: color {col.shape} depth {dep.shape} "
              f"depth_nonzero {int((dep > 0).sum())}")
        return True
    except Exception as e:
        print(f"  FAIL frames {label}: {e}")
        return False
    finally:
        try:
            p.stop()
        except Exception:
            pass


CONFIGS = [
    (848, 480, 30, 848, 480, 30),   # app default
    (848, 480, 15, 848, 480, 15),
    (640, 480, 30, 640, 480, 30),
    (640, 480, 15, 640, 480, 15),
    (1280, 720, 30, 848, 480, 30),  # color native, depth standard
]

if __name__ == "__main__":
    print("RealSense dual-stream config sweep:")
    for cfg in CONFIGS:
        try_config(*cfg)
        time.sleep(1.0)  # let the device fully release between attempts

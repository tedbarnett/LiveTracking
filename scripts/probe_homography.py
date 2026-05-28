"""Live test of the camera->projector mapping for a few selected camera points.

Reads H from the saved calibration, picks a handful of camera-space points
(centers of the bodhran, guitar, pillow, couch arm), warps each through H to
projector coordinates, then drives the projector daemon to flash a bright
cross at each warped point in sequence. The user (or a camera capture)
verifies whether the cross actually lands on the named object.

Bypasses perception entirely so we test ONLY the calibration H.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import zmq

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from livetracking.paths import HOMOGRAPHY_FILE, CALIB_META_FILE  # noqa
from livetracking.perception.footprint import load_homography


# Camera-pixel locations of physical objects, manually noted from the
# vision overlays in scripts/out/probe_obj_*.jpg. These are in 848x480 camera
# coordinates.
TARGETS = [
    ("bodhran_drum", 156, 324),     # round white drum, center of couch
    ("toy_guitar",   301, 300),     # small white solidbody on couch
    ("couch_left",   100, 350),     # left arm of couch
    ("couch_right",  500, 350),     # right end of couch
    ("frame_center", 350, 130),     # center wall map
]


def main():
    H, meta = load_homography()
    PW = int(meta["proj_w"])
    PH = int(meta["proj_h"])
    H_inv = np.linalg.inv(H)
    print(f"[probe] H loaded; projector {PW}x{PH}")

    # Connect to projector ZMQ PULL endpoint
    from livetracking.paths import ZMQ_PROJECTOR_PULL
    ctx = zmq.Context.instance()
    push = ctx.socket(zmq.PUSH)
    push.connect(ZMQ_PROJECTOR_PULL)
    print(f"[probe] connected to projector at {ZMQ_PROJECTOR_PULL}")

    # We need to inject a custom mask into the projector. The projector
    # daemon supports `highlight` messages with a `mask_path` field; write a
    # bright square mask centered at each warped point.
    import cv2
    masks_dir = os.path.join(os.path.dirname(HERE), "runtime", "masks")
    os.makedirs(masks_dir, exist_ok=True)
    out_dir = os.path.join(os.path.dirname(HERE), "scripts", "out")
    os.makedirs(out_dir, exist_ok=True)

    import urllib.request
    BASE = os.environ.get("LIVETRACKING_BASE", "http://localhost:5070")

    def grab_jpeg(timeout=4.0):
        """Pull one JPEG from the MJPEG stream so we can see where the
        square landed in the camera. Requires perception to be running."""
        try:
            with urllib.request.urlopen(BASE + "/stream.mjpg", timeout=timeout) as r:
                buf = b""
                t0 = time.time()
                while time.time() - t0 < timeout:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    buf += chunk
                    soi = buf.find(b"\xff\xd8\xff")
                    if soi < 0:
                        continue
                    eoi = buf.find(b"\xff\xd9", soi + 3)
                    if eoi >= 0:
                        return buf[soi:eoi + 2]
        except Exception as e:
            print(f"  jpeg grab failed: {e!r}")
        return None

    # Don't pause perception — the paused loop sends `clear` every 200ms which
    # wipes our highlights. Instead we'll keep re-sending the highlight at 10Hz
    # for a couple of seconds per target so it wins the race against any
    # competing messages.

    for name, cx_c, cy_c in TARGETS:
        # Warp camera point through H to projector coords
        pt_cam = np.array([[[float(cx_c), float(cy_c)]]], dtype=np.float32)
        pt_proj = cv2.perspectiveTransform(pt_cam, H).reshape(2)
        px, py = float(pt_proj[0]), float(pt_proj[1])
        print(f"[probe] {name}: cam=({cx_c},{cy_c}) -> proj=({px:.0f},{py:.0f})")

        if not (0 <= px < PW and 0 <= py < PH):
            print(f"  -> outside projector frame, skipping")
            continue

        # Build a large white square mask centered at (px, py)
        SIZE = 300  # projector px on each side
        mask = np.zeros((PH, PW), dtype=np.uint8)
        x0 = int(round(px - SIZE / 2)); y0 = int(round(py - SIZE / 2))
        x1 = min(PW, x0 + SIZE); y1 = min(PH, y0 + SIZE)
        x0 = max(0, x0); y0 = max(0, y0)
        mask[y0:y1, x0:x1] = 255
        mask_path = os.path.join(masks_dir, f"probe_{name}.png")
        cv2.imwrite(mask_path, mask)

        # Spam highlight at 20 Hz for 2 s so it wins the race against
        # perception's per-frame projector updates / pause clears.
        msg = {
            "type": "highlight",
            "id": 99,
            "color": [255, 255, 255],
            "mask_path": mask_path,
            "proj_centroid": [px, py],
        }
        t_end = time.time() + 2.0
        while time.time() < t_end:
            push.send_json(msg)
            time.sleep(0.05)
        jpg = grab_jpeg()
        if jpg:
            out_path = os.path.join(out_dir, f"H_probe_{name}.jpg")
            with open(out_path, "wb") as f:
                f.write(jpg)
            print(f"  -> camera saw it; wrote {out_path}")
        # Brief pause without highlighting so PIPE drains between targets
        push.send_json({"type": "clear"})
        time.sleep(0.4)

    # Clear and resume
    push.send_json({"type": "clear"})
    try:
        urllib.request.urlopen(urllib.request.Request(BASE + "/run", method="POST"),
                                timeout=4).read()
    except Exception:
        pass
    print("[probe] done.")


if __name__ == "__main__":
    main()

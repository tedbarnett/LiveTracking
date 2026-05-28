"""Parallax tuning probe.

For each tracked object, send `test_point` via perception ctrl with
parallax=True and the object's depth_m. Capture an MJPEG frame from
flame_web for visual inspection. Optionally compare against a baseline
(raw_H, no parallax) for the same object.

Usage:
  python scripts/probe_parallax.py            # parallax ON (current cfg)
  python scripts/probe_parallax.py --raw      # raw H, no parallax (baseline)
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import zmq

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "runtime", "probe_parallax")
os.makedirs(OUT, exist_ok=True)


def send_ctrl(cmd, **kw):
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.RCVTIMEO, 3000)
    s.setsockopt(zmq.SNDTIMEO, 3000)
    s.connect("tcp://127.0.0.1:5573")
    s.send_json({"cmd": cmd, **kw})
    rep = s.recv_json()
    s.close()
    return rep


def get_objects():
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"objects")
    sub.setsockopt(zmq.RCVTIMEO, 4000)
    sub.connect("tcp://127.0.0.1:5571")
    raw = sub.recv_multipart()  # [topic, json]
    sub.close()
    return json.loads(raw[1].decode())


def grab_frame(path):
    # Single-shot JPEG from flame_web.
    req = urllib.request.urlopen("http://127.0.0.1:5070/snapshot.jpg",
                                 timeout=4)
    data = req.read()
    req.close()
    if not data or not data.startswith(b"\xff\xd8"):
        return False
    with open(path, "wb") as f:
        f.write(data)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true",
                    help="bypass parallax (use raw H)")
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--size", type=int, default=300)
    args = ap.parse_args()

    print("[probe] fetching object list from perception...")
    payload = get_objects()
    objs = [o for o in payload.get("objects", []) if not o.get("hidden")]
    print(f"[probe] {len(objs)} objects:")
    for o in objs:
        print(f"   id={o['id']:>2} {o['name'][:18]:>18} "
              f"cam=({o['centroid_cam'][0]:.0f},{o['centroid_cam'][1]:.0f}) "
              f"depth={o['depth_m']:.2f}m")

    # Clear first.
    send_ctrl("test_clear")
    time.sleep(0.3)

    use_parallax = not args.raw
    tag = "parallax" if use_parallax else "raw"
    results = []
    for o in objs:
        cx, cy = o["centroid_cam"]
        depth = o["depth_m"]
        name = o["name"][:18].replace(" ", "_")
        print(f"\n[probe] -- {name} id={o['id']} cam=({cx:.0f},{cy:.0f}) "
              f"d={depth:.2f}m  ({tag}) --")
        rep = send_ctrl("test_point",
                        cam_x=cx, cam_y=cy,
                        size_px=args.size, hold_s=args.hold,
                        parallax=use_parallax, depth_m=depth)
        print(f"   ctrl rep: {rep}")
        # Give the projector ~600 ms to render, then grab a frame.
        time.sleep(0.6)
        path = os.path.join(OUT, f"{tag}_{o['id']:02d}_{name}.jpg")
        if grab_frame(path):
            print(f"   saved {path}")
            results.append({"id": o["id"], "name": o["name"],
                            "cam": [cx, cy], "depth": depth,
                            "proj": rep.get("proj_xy"),
                            "method": rep.get("method"), "path": path})
        else:
            print("   [WARN] frame grab failed")
        # Let the hold expire before the next test.
        time.sleep(max(0, args.hold - 0.6))

    send_ctrl("test_clear")
    summary = os.path.join(OUT, f"{tag}_summary.json")
    with open(summary, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[probe] summary -> {summary}")


if __name__ == "__main__":
    sys.exit(main())

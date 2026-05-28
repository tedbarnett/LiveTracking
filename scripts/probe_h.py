"""H QA via the perception 'test_point' ctrl. Captures one MJPEG frame
per target via curl (more robust than urllib).
"""
import os, time, subprocess, json
import zmq

TARGETS = [
    ("A_wall_topleft",  150, 130),
    ("B_wall_center",   400, 140),
    ("C_wall_right",    560, 145),
    ("D_couch_top",     400, 270),
    ("E_bodhran",       156, 324),
    ("F_toy_guitar",    301, 300),
    ("G_couch_left",    100, 360),
    ("H_couch_right",   440, 360),
]

OUT = r"C:\Users\timew\Github\LiveTracking\scripts\out"
os.makedirs(OUT, exist_ok=True)

ctx = zmq.Context.instance()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.RCVTIMEO, 5000); sock.setsockopt(zmq.SNDTIMEO, 5000)
sock.connect("tcp://127.0.0.1:5573")

for name, cx, cy in TARGETS:
    sock.send_json({"cmd": "test_point", "cam_x": cx, "cam_y": cy,
                    "size_px": 350, "hold_s": 4.0})
    r = sock.recv_json()
    print(f"{name}: cam=({cx},{cy}) proj_xy={r.get('proj_xy')} in_frame={r.get('in_frame')}")
    # Hold a moment so the projector + camera converge.
    time.sleep(1.8)
    out_path = os.path.join(OUT, f"H3_{name}.jpg")
    # Use curl to grab one MJPEG frame: --max-time short, output binary,
    # then extract first JPEG with Python.
    raw_path = os.path.join(OUT, f"H3_{name}.raw")
    subprocess.run(
        ["curl", "-s", "-m", "3", "-o", raw_path, "http://localhost:5070/stream.mjpg"],
        check=False,
    )
    with open(raw_path, "rb") as f:
        buf = f.read()
    soi = buf.find(b"\xff\xd8\xff")
    eoi = buf.find(b"\xff\xd9", soi + 3) if soi >= 0 else -1
    if soi >= 0 and eoi >= 0:
        with open(out_path, "wb") as f:
            f.write(buf[soi:eoi + 2])
        print(f"  wrote {out_path} ({eoi+2-soi} bytes)")
    else:
        print(f"  no JPEG in {len(buf)} bytes raw")
    try:
        os.remove(raw_path)
    except OSError:
        pass
    time.sleep(0.3)

sock.send_json({"cmd": "test_clear"}); sock.recv_json()
print("done")

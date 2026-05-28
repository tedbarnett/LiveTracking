"""Drive the perception/projector loop from outside.

For each currently-tracked object: clear the projector, pin the object so it
lights up alone, grab one camera JPEG from the MJPEG stream, save it to
scripts/out/probe_obj_<id>.jpg. Then unpin and clear.

Used to vision-verify that the projected wash for object N lands on the
actual physical object N, with the camera as ground-truth witness.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
BASE = os.environ.get("LIVETRACKING_BASE", "http://localhost:5070")


def grab_one_jpeg(url: str, timeout: float = 5.0) -> bytes:
    """Pull one JPEG out of an MJPEG multipart stream."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
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
        raise TimeoutError("no JPEG found in MJPEG stream within timeout")


def post(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        BASE + path, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=5) as r:
        return json.loads(r.read())


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    # SSE objects: read one event
    with urllib.request.urlopen(BASE + "/objects", timeout=5) as r:
        # read until first data line
        payload = None
        for raw_line in r:
            line = raw_line.decode().strip()
            if line.startswith("data:"):
                payload = json.loads(line[len("data:"):].strip())
                break
    if not payload:
        print("[probe] no SSE payload received")
        return 2

    objects = payload.get("objects", [])
    print(f"[probe] {len(objects)} tracked objects")
    summary = []
    # Make sure we're not paused
    post("/run")
    post("/clear")
    time.sleep(0.3)

    for obj in objects:
        oid = obj["id"]
        name = obj.get("name", "?")
        print(f"[probe] pinning id={oid} ({name})")
        r = post("/pin", {"id": oid})
        if not r.get("ok"):
            print(f"  pin failed: {r}")
            continue
        # Let the projector paint + camera see it through the room.
        time.sleep(1.2)
        try:
            jpg = grab_one_jpeg(BASE + "/stream.mjpg", timeout=4.0)
        except Exception as e:
            print(f"  jpeg grab failed: {e!r}")
            post("/unpin")
            continue
        out_path = os.path.join(OUT, f"probe_obj_{oid:02d}_{name.replace(' ', '_')[:20]}.jpg")
        with open(out_path, "wb") as f:
            f.write(jpg)
        print(f"  wrote {out_path} ({len(jpg)} bytes)")
        summary.append({
            "id": oid, "name": name,
            "centroid_cam": obj.get("centroid_cam"),
            "centroid_proj": obj.get("centroid_proj"),
            "bbox_cam": obj.get("bbox_cam"),
            "depth_m": obj.get("depth_m"),
            "image": out_path,
        })
        post("/unpin")
        time.sleep(0.4)

    post("/clear")
    with open(os.path.join(OUT, "probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[probe] done. summary: {OUT}\\probe_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

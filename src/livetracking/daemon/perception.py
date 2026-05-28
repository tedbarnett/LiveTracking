"""Perception daemon — runs the full pipeline in a loop and publishes results.

Three outputs over ZMQ on localhost:
  - PUB on $LIVETRACKING_ZMQ_OBJECTS (default tcp://127.0.0.1:5571)
    publishes a JSON object list every frame:
        {"t": float, "frame_idx": int, "objects": [
            {"id":1, "name":"sofa", "color":[R,G,B], "centroid_cam":[x,y],
             "bbox_cam":[x,y,w,h], "depth_m":2.5, "score":0.71}, ...
        ]}
  - PUB topic "frame" carries the latest JPEG-encoded annotated RGB frame
    so flame_web.py can MJPEG-stream it.
  - REP on the same socket handles "rename" / "highlight" requests routed
    in by the web app via a separate REQ socket — see daemon/projector.py.

Persistence: object names are saved to runtime/object_names.json by the
tracker itself.

Holds the D455 exclusively for the lifetime of the process. NSSM-install as
`LiveTrackingPerception`, AUTO_START.
"""
from __future__ import annotations

import json
import signal
import sys
import threading
import time
from typing import List, Optional

import cv2
import numpy as np
import zmq

from livetracking.paths import (
    RUNTIME_DIR,
    ZMQ_OBJECTS_PUB,
    ZMQ_PROJECTOR_PULL,
    describe,
)
from livetracking.perception.capture import RealSenseCapture
from livetracking.perception.footprint import (
    footprint_outline_in_camera,
    load_homography,
)
from livetracking.perception.pipeline import Pipeline, PipelineConfig
from livetracking.perception.types import DetectedObject


JPEG_QUALITY = 78


def _render_annotated(
    color: np.ndarray,
    footprint: np.ndarray,
    fp_corners: np.ndarray,
    objects: List[DetectedObject],
) -> np.ndarray:
    out = color.copy()
    outside = (footprint == 0)
    out[outside] = (out[outside] * 0.45).astype(np.uint8)
    cv2.polylines(out, [fp_corners.astype(np.int32)], True, (0, 255, 255), 2)
    for o in objects:
        col = o.color_rgb
        contours, _ = cv2.findContours(o.cam_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, col, 3)
        cx, cy = int(o.centroid_cam[0]), int(o.centroid_cam[1])
        cv2.circle(out, (cx, cy), 22, (0, 0, 0), -1)
        cv2.circle(out, (cx, cy), 22, col, 2)
        cv2.putText(out, str(o.object_id), (cx - 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def _objects_to_payload(objects: List[DetectedObject], timings: dict) -> dict:
    return {
        "t": time.time(),
        "timings_ms": timings,
        "objects": [
            {
                "id": o.object_id,
                "name": o.name,
                "color": list(o.color_rgb),
                "centroid_cam": [round(o.centroid_cam[0], 1),
                                 round(o.centroid_cam[1], 1)],
                "centroid_proj": (
                    [round(o.centroid_proj[0], 1), round(o.centroid_proj[1], 1)]
                    if o.centroid_proj else None
                ),
                "bbox_cam": list(o.bbox_cam),
                "depth_m": round(o.median_depth_m, 3),
                "score": round(o.label_score, 3),
                "has_proj_mask": o.proj_mask is not None,
            }
            for o in objects
        ],
    }


class PerceptionDaemon:
    def __init__(self):
        print(f"[perception] {describe()}")
        H, meta = load_homography()
        PW, PH = int(meta["proj_w"]), int(meta["proj_h"])
        self.cap = RealSenseCapture()
        cw, ch = self.cap.size()
        self.pipeline = Pipeline(
            H, cw, ch, PipelineConfig(proj_w=PW, proj_h=PH)
        )
        self.fp_corners = footprint_outline_in_camera(H, PW, PH, cw, ch)

        ctx = zmq.Context.instance()
        self.objects_pub = ctx.socket(zmq.PUB)
        self.objects_pub.bind(ZMQ_OBJECTS_PUB)
        # PUSH/PULL into the projector daemon — passthrough only when a
        # /highlight comes in from the web app via REQ.
        self.proj_push = ctx.socket(zmq.PUSH)
        self.proj_push.connect(ZMQ_PROJECTOR_PULL)

        # REP socket so flame_web.py can issue control commands.
        # Served from a dedicated thread so the slow perception loop never
        # holds it back — hover -> highlight is < 100 ms.
        self.ctrl = ctx.socket(zmq.REP)
        self.ctrl.bind("tcp://127.0.0.1:5573")
        self._ctrl_lock = threading.Lock()

        self.running = True
        self.paused = False
        self._pinned_id: Optional[int] = None
        self.frame_idx = 0
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)
        self._ctrl_thread = threading.Thread(target=self._ctrl_loop, daemon=True)
        self._ctrl_thread.start()

    def _ctrl_loop(self):
        while self.running:
            try:
                msg = self.ctrl.recv_json(flags=0)
            except Exception:
                break
            try:
                with self._ctrl_lock:
                    reply = self._handle_ctrl_message(msg)
            except Exception as e:
                reply = {"ok": False, "reason": repr(e)}
            try:
                self.ctrl.send_json(reply)
            except Exception:
                break

    def _stop(self, *_):
        self.running = False

    def _handle_ctrl_message(self, msg: dict):
        """Web-app -> daemon control: rename, highlight, clear, snapshot."""
        cmd = msg.get("cmd")
        if cmd == "rename":
            ok = self.pipeline.tracker.rename(int(msg["id"]), str(msg["name"]))
            return {"ok": ok}
        if cmd == "highlight":
            obj_id = int(msg["id"])
            with self.pipeline.tracker_lock:
                tracked = self.pipeline.tracker.active()
            target = next((o for o in tracked if o.object_id == obj_id), None)
            if target is None or target.proj_mask is None:
                self.proj_push.send_json({"type": "clear"})
                return {"ok": False, "reason": "object not found or no proj_mask"}
            self.proj_push.send_json({
                "type": "highlight",
                "id": obj_id,
                "color": list(target.color_rgb),
                "proj_centroid": (
                    list(target.centroid_proj) if target.centroid_proj else None
                ),
                # mask bytes are sent as a separate side-channel via masks/ dir
                "mask_path": _save_mask_png(target),
            })
            return {"ok": True}
        if cmd == "clear":
            # Don't clear if something is pinned.
            if self._pinned_id is not None:
                return {"ok": True, "ignored": "pinned"}
            self.proj_push.send_json({"type": "clear"})
            return {"ok": True}
        if cmd == "cycle_color":
            new = self.pipeline.tracker.cycle_color(int(msg["id"]))
            if new is None:
                return {"ok": False, "reason": "no such object"}
            return {"ok": True, "color": list(new)}
        if cmd == "pin":
            # Like highlight but doesn't auto-clear on mouseleave.
            obj_id = int(msg["id"])
            with self.pipeline.tracker_lock:
                tracked = self.pipeline.tracker.active()
            target = next((o for o in tracked if o.object_id == obj_id), None)
            if target is None or target.proj_mask is None:
                return {"ok": False, "reason": "no such object / no proj_mask"}
            self.proj_push.send_json({
                "type": "highlight",
                "id": obj_id,
                "color": list(target.color_rgb),
                "proj_centroid": (list(target.centroid_proj)
                                  if target.centroid_proj else None),
                "mask_path": _save_mask_png(target),
                "pinned": True,
            })
            self._pinned_id = obj_id
            return {"ok": True}
        if cmd == "unpin":
            self._pinned_id = None
            self.proj_push.send_json({"type": "clear"})
            return {"ok": True}
        if cmd == "intensity":
            try:
                v = float(msg.get("value", 0.78))
            except Exception:
                return {"ok": False, "reason": "bad intensity"}
            v = max(0.0, min(1.0, v))
            self.proj_push.send_json({"type": "set_intensity", "value": v})
            return {"ok": True, "value": v}
        if cmd == "highlight_all":
            with self.pipeline.tracker_lock:
                tracked = self.pipeline.tracker.active()
            self.proj_push.send_json({
                "type": "highlight_all",
                "objects": [
                    {
                        "id": o.object_id,
                        "color": list(o.color_rgb),
                        "proj_centroid": (list(o.centroid_proj)
                                          if o.centroid_proj else None),
                        "mask_path": _save_mask_png(o),
                    }
                    for o in tracked if o.proj_mask is not None
                ],
            })
            return {"ok": True, "count": sum(1 for o in tracked if o.proj_mask is not None)}
        if cmd == "pause":
            self.paused = True
            self.proj_push.send_json({"type": "clear"})
            return {"ok": True, "paused": True}
        if cmd == "run":
            self.paused = False
            return {"ok": True, "paused": False}
        if cmd == "state":
            return {"ok": True, "paused": self.paused}
        if cmd == "list":
            with self.pipeline.tracker_lock:
                act = self.pipeline.tracker.active()
            return {"ok": True, "objects": _objects_to_payload(
                act, self.pipeline.last_timings_ms
            )["objects"]}
        return {"ok": False, "reason": f"unknown cmd {cmd!r}"}

    def run(self):
        print("[perception] entering main loop")
        last_print = 0.0
        while self.running:
            frame = self.cap.read()
            if self.paused:
                # Tell the projector to stay black and publish an empty
                # "paused" payload so the UI knows.
                self.proj_push.send_json({"type": "clear"})
                payload = {
                    "t": time.time(),
                    "paused": True,
                    "timings_ms": {"total_ms": 0, "stage1_ms": 0, "dino_ms": 0,
                                   "sam_ms": 0, "merge_ms": 0,
                                   "n_dino_raw": 0, "n_dino_kept": 0,
                                   "n_objects": 0},
                    "objects": [],
                }
                self.objects_pub.send_multipart([
                    b"objects", json.dumps(payload).encode("utf-8"),
                ])
                # Paint a dim "PAUSED" frame so the MJPEG isn't stale.
                dim = (frame.color * 0.35).astype(np.uint8)
                cv2.putText(dim, "PAUSED", (24, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            2.5, (255, 255, 255), 4)
                ok, jpeg = cv2.imencode(".jpg", dim,
                                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    self.objects_pub.send_multipart([b"frame", jpeg.tobytes()])
                # Idle the loop — don't burn the GPU while paused.
                time.sleep(0.2)
                self.frame_idx += 1
                continue

            # No need to hold _ctrl_lock here — Pipeline owns its own
            # tracker_lock; ctrl ops contend only on tracker, not on the GPU.
            objects = self.pipeline.step_auto(frame.color, frame.depth_m)
            payload = _objects_to_payload(objects, self.pipeline.last_timings_ms)
            payload["paused"] = False
            self.objects_pub.send_multipart([
                b"objects", json.dumps(payload).encode("utf-8"),
            ])
            annotated = _render_annotated(
                frame.color, self.pipeline.footprint, self.fp_corners, objects
            )
            ok, jpeg = cv2.imencode(".jpg", annotated,
                                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                self.objects_pub.send_multipart([b"frame", jpeg.tobytes()])

            self.frame_idx += 1
            now = time.time()
            if now - last_print > 2.0:
                t = self.pipeline.last_timings_ms
                kind = "FAST" if t.get("fast") else "FULL"
                print(f"[perception] frame {self.frame_idx} [{kind}] "
                      f"total={t['total_ms']:.0f}ms "
                      f"(s1={t['stage1_ms']:.0f} dino={t['dino_ms']:.0f} "
                      f"sam={t['sam_ms']:.0f}) "
                      f"dino={t['n_dino_raw']}->{t['n_dino_kept']} "
                      f"objects={t['n_objects']}")
                last_print = now

        self.pipeline.stop_async()
        self.cap.close()
        print("[perception] exited")


def _save_mask_png(obj: DetectedObject) -> str:
    """Write the projector-space mask as a PNG to runtime/masks/ for the
    projector daemon to pick up. Filename is keyed by object id."""
    import os
    path = os.path.join(RUNTIME_DIR, "masks", f"obj_{obj.object_id:03d}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if obj.proj_mask is not None:
        cv2.imwrite(path, obj.proj_mask)
    return path


def main() -> int:
    PerceptionDaemon().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

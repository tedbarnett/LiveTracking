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
import os
import signal
import sys
import threading
import time
from typing import List, Optional

import cv2
import numpy as np
import zmq

from livetracking.config.env import parse_bool, parse_float, parse_str
from livetracking.paths import (
    RUNTIME_DIR,
    ZMQ_CTRL_ENDPOINT,
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
from livetracking.perception.recognize import create_recognizer, read_active_detector, VALID_DETECTORS
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
        # color_rgb is RGB (matches the web UI swatch + pygame projector),
        # but `out` is an OpenCV BGR frame — swap or the preview outline
        # shows the wrong hue (blue objects ring red, amber rings cyan)
        # and the user thinks the projector ignores the color box.
        col = tuple(int(c) for c in o.color_rgb[::-1])
        # Smooth the mask before contour extraction so frame-to-frame
        # pixel-step noise on the SAM boundary doesn't drive jitter.
        # o.cam_mask is uint8 with values {0,1}; normalize to {0,255}
        # first so the Gaussian + threshold actually has signal.
        m = (o.cam_mask > 0).astype(np.uint8) * 255
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=2.0)
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        smoothed = []
        for c in contours:
            if len(c) < 4:
                smoothed.append(c)
                continue
            eps = 0.004 * cv2.arcLength(c, True)
            smoothed.append(cv2.approxPolyDP(c, eps, True))
        cv2.drawContours(out, smoothed, -1, col, 2, cv2.LINE_AA)
        cx, cy = int(o.centroid_cam[0]), int(o.centroid_cam[1])
        # Smaller preview number badge (~half prior size) to reduce
        # visual clutter in the web preview.
        cv2.circle(out, (cx, cy), 12, (0, 0, 0), -1)
        cv2.circle(out, (cx, cy), 12, col, 1)
        cv2.putText(out, str(o.object_id), (cx - 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
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
        # Per-object parallax compensation knobs (env overrides for live tuning
        # without code edits). All parsed via livetracking.config.env so a
        # garbage value in the environment (e.g. LIVETRACKING_PARALLAX_K=1200x
        # from a fat-fingered shell command) warns + keeps default instead of
        # crashing the daemon at startup.
        cfg = PipelineConfig(proj_w=PW, proj_h=PH)
        cfg.parallax_compensate = parse_bool(
            "LIVETRACKING_PARALLAX_COMPENSATE", cfg.parallax_compensate)
        cfg.parallax_sign = parse_float(
            "LIVETRACKING_PARALLAX_SIGN", cfg.parallax_sign,
            min_value=-1.0, max_value=1.0)
        cfg.parallax_scale = parse_float(
            "LIVETRACKING_PARALLAX_SCALE", cfg.parallax_scale,
            min_value=0.0, max_value=10.0)
        cfg.parallax_k_px_m = parse_float(
            "LIVETRACKING_PARALLAX_K", cfg.parallax_k_px_m,
            min_value=0.0, max_value=10000.0)
        print(f"[perception] parallax: compensate={cfg.parallax_compensate} "
              f"sign={cfg.parallax_sign} scale={cfg.parallax_scale} "
              f"k_px_m={cfg.parallax_k_px_m}")

        # Detector backend: persisted in runtime/active_detector.json.
        # An env override is honored so we can spin up an alternate detector
        # without touching the file (mostly useful from a smoke-test shell).
        # parse_str with choices=VALID_DETECTORS rejects typos cleanly.
        env_detector = parse_str("LIVETRACKING_DETECTOR", "",
                                 choices=list(VALID_DETECTORS))
        detector_name = env_detector or read_active_detector()
        print(f"[perception] detector backend: {detector_name}")
        self.detector_name = detector_name
        recognizer = create_recognizer(detector_name)
        self.pipeline = Pipeline(
            H, cw, ch, cfg, recognizer=recognizer,
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
        self.ctrl.bind(ZMQ_CTRL_ENDPOINT)
        self._ctrl_lock = threading.Lock()

        self.running = True
        self.paused = False
        self._pinned_id: Optional[int] = None
        # Track what we last told the projector to draw, so a live config
        # tune (e.g. /mask edge softness) can re-push with the updated
        # mask without waiting for the next user hover. None = nothing
        # shown. {"kind":"single","id":N} or {"kind":"all"}.
        self._last_highlight: Optional[dict] = None
        # When > current time, suppress perception's own projector messages
        # so a `test_point` highlight from /test_light stays on screen.
        self._test_hold_until: float = 0.0
        self.frame_idx = 0
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)
        self._ctrl_thread = threading.Thread(target=self._ctrl_loop, daemon=True)
        self._ctrl_thread.start()

    def _ctrl_loop(self):
        while self.running:
            try:
                msg = self.ctrl.recv_json(flags=0)
            except Exception as e:
                # A dead ctrl socket means every UI hover/POST times out
                # while the daemon looks healthy. Be loud about it.
                print(f"[perception] FATAL: ctrl socket recv failed: {e!r} "
                      f"— control plane is DOWN, restart the daemon",
                      flush=True)
                break
            try:
                with self._ctrl_lock:
                    reply = self._handle_ctrl_message(msg)
            except Exception as e:
                reply = {"ok": False, "reason": repr(e)}
            try:
                self.ctrl.send_json(reply)
            except Exception as e:
                print(f"[perception] FATAL: ctrl socket send failed: {e!r} "
                      f"— control plane is DOWN, restart the daemon",
                      flush=True)
                break

    def _stop(self, *_):
        self.running = False

    def _push_highlight(self, obj_id: int, pinned: bool = False) -> bool:
        """Look up an object, warp its current cam_mask through the
        pipeline (so live cfg changes like mask_smooth_px take effect),
        and push the highlight to the projector. Returns True on success.
        """
        with self.pipeline.tracker_lock:
            tracked = self.pipeline.tracker.visible()
        target = next((o for o in tracked if o.object_id == obj_id), None)
        if target is None or target.cam_mask is None:
            return False
        # Re-warp from raw cam_mask using the LIVE cfg so a slider tweak
        # is reflected immediately. Falls back to the cached proj_mask
        # if re-warp fails (shouldn't, but safety net).
        try:
            new_proj, new_centroid = (
                self.pipeline._warp_with_parallax_image_first(
                    target.cam_mask, target.median_depth_m
                )
            )
            if new_proj is not None:
                target.proj_mask = new_proj
                target.centroid_proj = new_centroid
        except Exception as e:  # noqa: BLE001
            print(f"[perception] _push_highlight rewarp failed: {e!r}")
        if target.proj_mask is None:
            return False
        payload = {
            "type": "highlight",
            "id": obj_id,
            "name": target.name,
            "color": list(target.color_rgb),
            "proj_centroid": (
                list(target.centroid_proj) if target.centroid_proj else None
            ),
            "mask_path": _save_mask_png(target),
        }
        if pinned:
            payload["pinned"] = True
        self.proj_push.send_json(payload)
        return True

    def _push_highlight_all(self) -> int:
        """Like _push_highlight but for the 'illuminate everything'
        broadcast. Returns the count actually pushed."""
        with self.pipeline.tracker_lock:
            tracked = self.pipeline.tracker.visible()
        # Re-warp every visible object from its raw cam_mask so live
        # cfg changes apply on the rebroadcast.
        objects: List[dict] = []
        for o in tracked:
            if o.cam_mask is None:
                continue
            try:
                new_proj, new_centroid = (
                    self.pipeline._warp_with_parallax_image_first(
                        o.cam_mask, o.median_depth_m
                    )
                )
                if new_proj is not None:
                    o.proj_mask = new_proj
                    o.centroid_proj = new_centroid
            except Exception:
                pass
            if o.proj_mask is None:
                continue
            objects.append({
                "id": o.object_id,
                "name": o.name,
                "color": list(o.color_rgb),
                "proj_centroid": (
                    list(o.centroid_proj) if o.centroid_proj else None
                ),
                "mask_path": _save_mask_png(o),
            })
        self.proj_push.send_json({
            "type": "highlight_all", "objects": objects,
        })
        return len(objects)

    def _refresh_active_highlight(self) -> None:
        """Re-emit whatever is currently shown on the projector. Called
        after a live cfg tune so the user sees the change without
        having to re-hover."""
        last = self._last_highlight
        if not last:
            return
        try:
            if last.get("kind") == "single":
                self._push_highlight(
                    int(last["id"]),
                    pinned=bool(last.get("pinned", False)),
                )
            elif last.get("kind") == "all":
                self._push_highlight_all()
        except Exception as e:  # noqa: BLE001
            print(f"[perception] refresh failed: {e!r}")

    def _handle_ctrl_message(self, msg: dict):
        """Web-app -> daemon control: rename, highlight, clear, snapshot."""
        cmd = msg.get("cmd")
        if cmd == "rename":
            ok = self.pipeline.tracker.rename(int(msg["id"]), str(msg["name"]))
            return {"ok": ok}
        if cmd == "hide":
            ok = self.pipeline.tracker.hide(int(msg["id"]))
            # Also clear the projection if we just hid the currently lit object.
            self.proj_push.send_json({"type": "clear"})
            return {"ok": ok}
        if cmd == "unhide":
            ok = self.pipeline.tracker.unhide(int(msg["id"]))
            return {"ok": ok}
        if cmd == "unhide_all":
            n = self.pipeline.tracker.unhide_all()
            return {"ok": True, "count": n}
        if cmd == "hidden_list":
            return {"ok": True, "ids": self.pipeline.tracker.hidden_ids()}
        if cmd == "highlight":
            obj_id = int(msg["id"])
            ok = self._push_highlight(obj_id)
            if not ok:
                self.proj_push.send_json({"type": "clear"})
                self._last_highlight = None
                return {"ok": False, "reason": "object not found or no proj_mask"}
            self._last_highlight = {"kind": "single", "id": obj_id}
            return {"ok": True}
        if cmd == "clear":
            # Don't clear if something is pinned.
            if self._pinned_id is not None:
                return {"ok": True, "ignored": "pinned"}
            self.proj_push.send_json({"type": "clear"})
            self._last_highlight = None
            return {"ok": True}
        if cmd == "cycle_color":
            new = self.pipeline.tracker.cycle_color(int(msg["id"]))
            if new is None:
                return {"ok": False, "reason": "no such object"}
            # Re-emit the active highlight so the projected wash changes
            # color immediately — the projector paints the last pushed
            # payload, which otherwise still carries the old color.
            self._refresh_active_highlight()
            return {"ok": True, "color": list(new)}
        if cmd == "pin":
            # Like highlight but doesn't auto-clear on mouseleave.
            obj_id = int(msg["id"])
            ok = self._push_highlight(obj_id, pinned=True)
            if not ok:
                return {"ok": False, "reason": "no such object / no proj_mask"}
            self._pinned_id = obj_id
            self._last_highlight = {"kind": "single", "id": obj_id, "pinned": True}
            return {"ok": True}
        if cmd == "unpin":
            self._pinned_id = None
            self.proj_push.send_json({"type": "clear"})
            self._last_highlight = None
            return {"ok": True}
        if cmd == "test_point":
            # Project a fixed-size white square at the projector coordinates
            # for camera (cam_x, cam_y). If parallax=True and depth_m is
            # provided (or readable from the last frame), routes through
            # Pipeline._warp_with_parallax so the wash lands on the *object*
            # at that depth, not the wall behind it. Used by the
            # calibration QA probe.
            try:
                cx_c = float(msg["cam_x"])
                cy_c = float(msg["cam_y"])
                size_px = int(msg.get("size_px", 300))
                hold_s = float(msg.get("hold_s", 2.0))
                use_parallax = bool(msg.get("parallax", False))
                depth_m_val = msg.get("depth_m", None)
                if depth_m_val is not None:
                    depth_m_val = float(depth_m_val)
            except Exception:
                return {"ok": False, "reason": "bad test_point payload"}
            import cv2 as _cv2
            PW = self.pipeline.cfg.proj_w
            PH = self.pipeline.cfg.proj_h
            CW = self.pipeline.cam_w
            CH = self.pipeline.cam_h
            px = py = 0.0
            method = "raw_H"
            if use_parallax and depth_m_val and depth_m_val > 0.1:
                # Build a small cam-mask disc, run through parallax-aware warp.
                disc = np.zeros((CH, CW), dtype=np.uint8)
                _cv2.circle(disc, (int(round(cx_c)), int(round(cy_c))),
                            max(4, size_px // 16), 255, -1)
                # Synthesize a depth array that says "depth_m at the disc".
                fake_depth = np.zeros((CH, CW), dtype=np.float32)
                fake_depth[disc > 0] = depth_m_val
                # The compat shim ignores `plane` — the pipeline uses its
                # own calibrated self.wall_plane. (The old last_stage1_debug
                # plane lookup was dead code from the depth-first era.)
                pm, pc, _med = self.pipeline._warp_with_parallax(
                    disc, fake_depth, None,
                )
                if pm is not None and pc is not None:
                    # Re-render as a centered square at the parallax-shifted
                    # projector centroid (so the visual is consistent).
                    px, py = float(pc[0]), float(pc[1])
                    method = "parallax"
                else:
                    # parallax couldn't compute (e.g. mask warped offscreen);
                    # fall through to raw H so something still projects.
                    pt = np.array([[[cx_c, cy_c]]], dtype=np.float32)
                    pp = _cv2.perspectiveTransform(pt, self.pipeline.H).reshape(2)
                    px, py = float(pp[0]), float(pp[1])
                    method = "raw_H_fallback"
            else:
                pt = np.array([[[cx_c, cy_c]]], dtype=np.float32)
                pp = _cv2.perspectiveTransform(pt, self.pipeline.H).reshape(2)
                px, py = float(pp[0]), float(pp[1])
            mask = np.zeros((PH, PW), dtype=np.uint8)
            x0 = max(0, int(round(px - size_px / 2)))
            y0 = max(0, int(round(py - size_px / 2)))
            x1 = min(PW, x0 + size_px)
            y1 = min(PH, y0 + size_px)
            mask[y0:y1, x0:x1] = 255
            mask_path = os.path.join(RUNTIME_DIR, "masks", "test_point.png")
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            _cv2.imwrite(mask_path, mask)
            self.proj_push.send_json({
                "type": "highlight",
                "id": 99,
                "color": [255, 255, 255],
                "mask_path": mask_path,
                "proj_centroid": [px, py],
            })
            # Hold-lock: suppress perception's own projector messages for
            # hold_s seconds.
            self._test_hold_until = time.time() + hold_s
            return {
                "ok": True,
                "proj_xy": [px, py],
                "in_frame": (0 <= px < PW and 0 <= py < PH),
                "method": method,
                "parallax_sign": self.pipeline.cfg.parallax_sign,
                "parallax_scale": self.pipeline.cfg.parallax_scale,
            }
        if cmd == "test_clear":
            self._test_hold_until = 0.0
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
        if cmd == "white_light":
            v = bool(msg.get("value", False))
            self.proj_push.send_json({"type": "set_white_light", "value": v})
            return {"ok": True, "value": v}
        if cmd == "highlight_all":
            n = self._push_highlight_all()
            self._last_highlight = {"kind": "all"}
            return {"ok": True, "count": n}
        if cmd == "pause":
            self.paused = True
            self.proj_push.send_json({"type": "clear"})
            return {"ok": True, "paused": True}
        if cmd == "run":
            self.paused = False
            return {"ok": True, "paused": False}
        if cmd == "state":
            return {"ok": True, "paused": self.paused,
                    "detector": getattr(self, "detector_name", "dino")}
        if cmd == "detector_info":
            return {"ok": True, "detector": getattr(self, "detector_name", "dino")}
        if cmd == "list":
            with self.pipeline.tracker_lock:
                act = self.pipeline.tracker.visible()
            return {"ok": True, "objects": _objects_to_payload(
                act, self.pipeline.last_timings_ms
            )["objects"]}
        if cmd == "parallax_get":
            cfg = self.pipeline.cfg
            return {"ok": True,
                    "compensate": cfg.parallax_compensate,
                    "sign": cfg.parallax_sign,
                    "scale": cfg.parallax_scale,
                    "k_px_m": cfg.parallax_k_px_m}
        if cmd == "mask_get":
            cfg = self.pipeline.cfg
            return {"ok": True,
                    "smooth_px": int(cfg.mask_smooth_px)}
        if cmd == "mask_tune":
            cfg = self.pipeline.cfg
            changed = {}
            if "smooth_px" in msg:
                v = int(msg["smooth_px"])
                cfg.mask_smooth_px = max(0, min(25, v))
                changed["smooth_px"] = cfg.mask_smooth_px
            print(f"[perception] mask_tune applied: {changed}")
            # Re-push the currently-shown highlight so the user sees the
            # softness change immediately, without having to mouse-off
            # and mouse-back-on the object row.
            if changed:
                self._refresh_active_highlight()
            return {"ok": True, "changed": changed,
                    "current": {"smooth_px": int(cfg.mask_smooth_px)}}
        if cmd == "parallax_tune":
            # Live-mutate the pipeline config from a remote HTTP request.
            # Pipeline reads cfg on every frame, so changes take effect on
            # the next frame — no restart needed. Bounds match the
            # daemon-boot env-parse clamps so we can't push a value that
            # would be rejected on next restart.
            cfg = self.pipeline.cfg
            changed = {}
            if "compensate" in msg:
                cfg.parallax_compensate = bool(msg["compensate"])
                changed["compensate"] = cfg.parallax_compensate
            if "sign" in msg:
                v = float(msg["sign"])
                cfg.parallax_sign = max(-1.0, min(1.0, v))
                changed["sign"] = cfg.parallax_sign
            if "scale" in msg:
                v = float(msg["scale"])
                cfg.parallax_scale = max(0.0, min(10.0, v))
                changed["scale"] = cfg.parallax_scale
            if "k_px_m" in msg:
                v = float(msg["k_px_m"])
                cfg.parallax_k_px_m = max(0.0, min(10000.0, v))
                changed["k_px_m"] = cfg.parallax_k_px_m
            print(f"[perception] parallax_tune applied: {changed}")
            if changed:
                self._refresh_active_highlight()
            return {"ok": True, "changed": changed,
                    "current": {"compensate": cfg.parallax_compensate,
                                "sign": cfg.parallax_sign,
                                "scale": cfg.parallax_scale,
                                "k_px_m": cfg.parallax_k_px_m}}
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
            objects_all = self.pipeline.step_auto(frame.color, frame.depth_m)
            # User-hidden tracks stay in the matcher but never reach the UI
            # or the projector.
            objects = [o for o in objects_all if not o.hidden]
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
                      f"sam={t['sam_ms']:.0f} stageD={t.get('stageD_ms',0):.0f}"
                      f" [clip={t.get('d_clip_ms',0):.0f}"
                      f" dband={t.get('d_depthband_ms',0):.0f}"
                      f" morph={t.get('d_morph_ms',0):.0f}"
                      f" cc={t.get('d_cc_ms',0):.0f}"
                      f" warp={t.get('d_warp_ms',0):.0f}])"
                      f" dino={t['n_dino_raw']}->{t['n_dino_kept']} "
                      f"objects={t['n_objects']}")
                last_print = now

        self.pipeline.stop_async()
        self.cap.close()
        print("[perception] exited")


def _save_mask_png(obj: DetectedObject) -> str:
    """Write the projector-space mask as a PNG to runtime/masks/ for the
    projector daemon to pick up. Filename is keyed by object id. Written
    via tmp+rename so the projector never reads a torn file."""
    import os
    path = os.path.join(RUNTIME_DIR, "masks", f"obj_{obj.object_id:03d}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if obj.proj_mask is not None:
        tmp = path + ".tmp.png"
        if cv2.imwrite(tmp, obj.proj_mask):
            os.replace(tmp, path)
    return path


def main() -> int:
    PerceptionDaemon().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
from typing import List, Optional, Tuple

import cv2
import numpy as np
import zmq

from livetracking.config.env import parse_bool, parse_float, parse_int, parse_str
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


def _repush_decision(
    fast_track: bool,
    cur_seq: int,
    last_repush_seq: int,
    has_active_highlight: bool,
    now: float,
    test_hold_until: float,
) -> tuple[bool, int]:
    """Pure decision for the Step-1 fast-follow re-push (no I/O, no locks).

    Returns ``(should_push, new_last_repush_seq)``.

    Logic, kept in one testable place:
      - fast_track off            -> never push, seq unchanged.
      - seq has not advanced      -> never push, seq unchanged (no redundant
                                     PNG rewrites between SAM passes — this is
                                     what protects the projector mask cache).
      - seq advanced              -> consume the new seq (advance the marker
                                     even when we ultimately skip, so we don't
                                     re-evaluate the same pass), then push iff
                                     a highlight is active AND we're not inside
                                     a /test_light hold window.
    """
    if not fast_track:
        return False, last_repush_seq
    if cur_seq == last_repush_seq:
        return False, last_repush_seq
    new_seq = cur_seq
    if not has_active_highlight:
        return False, new_seq
    if now < test_hold_until:
        return False, new_seq
    return True, new_seq


def _cam_mask_iou(a, b) -> float:
    """IoU of two camera-space masks (any nonzero = foreground). Returns 1.0
    for two empty masks (nothing vs nothing = unchanged), 0.0 when exactly
    one is empty. Cheap: operates on the 848x480 SAM masks, not the 8.3 MP
    projector buffer."""
    import numpy as _np
    ab = a > 0
    bb = b > 0
    inter = int(_np.logical_and(ab, bb).sum())
    union = int(_np.logical_or(ab, bb).sum())
    if union == 0:
        return 1.0
    return inter / union


def _highlight_mask_stable(
    prev_centroid_cam,
    new_centroid_cam,
    prev_cam_mask,
    new_cam_mask,
    move_px: float,
    iou_thresh: float,
) -> bool:
    """Hysteresis gate for the select-all re-push (pure, no I/O).

    Returns True when the freshly-detected mask is close enough to what is
    already shown that re-pushing would only inject SAM's run-to-run wobble
    (so we should HOLD the current mask). Returns False when the object
    genuinely moved or changed shape and the projector should update.

    Stable iff BOTH:
      * camera-space centroid moved <= ``move_px``, AND
      * IoU(new, shown) >= ``iou_thresh``.

    Any missing input (first show, no prior mask) => not stable (update),
    so a newly-highlighted object always paints immediately.
    """
    if prev_centroid_cam is None or new_centroid_cam is None:
        return False
    if prev_cam_mask is None or new_cam_mask is None:
        return False
    dx = float(new_centroid_cam[0]) - float(prev_centroid_cam[0])
    dy = float(new_centroid_cam[1]) - float(prev_centroid_cam[1])
    moved = (dx * dx + dy * dy) ** 0.5
    if moved > move_px:
        return False
    if _cam_mask_iou(new_cam_mask, prev_cam_mask) < iou_thresh:
        return False
    return True


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
                "effect": getattr(o, "effect", "color"),
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

        # Fast object-following. When a highlighted object moves, the
        # projector wash only re-lands on it when perception re-pushes the
        # highlight. Step 1 (this flag, default ON): re-push the active
        # highlight every time a heavy DINO+SAM pass produces fresh
        # positions, so the wash follows within one SAM pass (~2.5 Hz)
        # instead of staying frozen until an unrelated UI poke. Set
        # LIVETRACKING_FAST_TRACK=0 to restore the old push-on-UI-event-only
        # behavior if the re-push ever misbehaves at the rig.
        self.fast_track = parse_bool("LIVETRACKING_FAST_TRACK", True)
        print(f"[perception] fast_track (auto re-push on SAM pass): "
              f"{self.fast_track}")

        # Step 2: inter-pass fusion tracking (depth-blob + CSRT). Default OFF
        # — Step 1 (re-push on SAM pass) is the safe floor. When ON, the main
        # loop runs a fast per-frame tracker for highlighted objects and sends
        # the projector live position offsets so the wash follows in real time
        # between SAM passes. Requires fast_track on to do anything useful
        # (the offsets ride alongside the Step-1 re-push).
        self.fast_track_fusion = parse_bool(
            "LIVETRACKING_FAST_TRACK_FUSION", False)
        print(f"[perception] fast_track_fusion (inter-pass depth+CSRT): "
              f"{self.fast_track_fusion}")

        # Select-all / multi-highlight stabilization. The "illuminate
        # everything" path re-warps every object from its fresh SAM mask on
        # every DINO+SAM pass (~2.5 Hz) and rewrites the projector PNG. SAM
        # boundaries wobble a few px run-to-run even on a static object, so
        # the wash visibly JERKS every ~2 s. With this on, an object's mask
        # is only re-pushed when it actually moved (centroid shift in camera
        # px) or changed shape (IoU drop vs the last shown mask); otherwise
        # the displayed mask is held, killing the jitter (and skipping the
        # 8.3 MP warp + PNG write for stable objects). Set
        # LIVETRACKING_HIGHLIGHT_HYSTERESIS=0 to restore re-warp-every-pass.
        self.highlight_hysteresis = parse_bool(
            "LIVETRACKING_HIGHLIGHT_HYSTERESIS", True)
        # Re-push when the camera-space centroid moves more than this many
        # pixels since the last shown mask (real motion should update).
        self.highlight_move_px = parse_float(
            "LIVETRACKING_HIGHLIGHT_MOVE_PX", 6.0, min_value=0.0,
            max_value=200.0)
        # Re-push when IoU(new_cam_mask, shown_cam_mask) drops below this
        # (shape changed for real). SAM run-to-run wobble on a static object
        # sits well above ~0.9; genuine reshape falls below.
        self.highlight_iou = parse_float(
            "LIVETRACKING_HIGHLIGHT_IOU", 0.90, min_value=0.0, max_value=1.0)
        print(f"[perception] highlight_hysteresis: "
              f"{self.highlight_hysteresis} "
              f"(move>{self.highlight_move_px}px or IoU<{self.highlight_iou})")
        # Per-object record of what is CURRENTLY shown on the projector for
        # the select-all path: {oid: {"cam_mask", "centroid_cam",
        # "proj_mask", "centroid_proj", "mask_path"}}. Used by the hysteresis
        # gate to decide whether a fresh SAM pass should actually re-push.
        self._highlight_shown: dict = {}

        # DINO detection knobs (env overrides for boot-time tuning; the web
        # UI's Detection panel live-mutates the same fields via dino_tune).
        cfg.dino_box_thresh = parse_float(
            "LIVETRACKING_DINO_BOX_THRESH", cfg.dino_box_thresh,
            min_value=0.01, max_value=0.95)
        cfg.dino_text_thresh = parse_float(
            "LIVETRACKING_DINO_TEXT_THRESH", cfg.dino_text_thresh,
            min_value=0.01, max_value=0.95)
        cfg.min_dino_score = parse_float(
            "LIVETRACKING_DINO_MIN_SCORE", cfg.min_dino_score,
            min_value=0.01, max_value=0.95)
        cfg.min_obj_area_px = parse_int(
            "LIVETRACKING_MIN_OBJ_AREA_PX", cfg.min_obj_area_px,
            min_value=50, max_value=50000)
        env_prompt = parse_str("LIVETRACKING_DINO_PROMPT", "")
        if env_prompt.strip():
            cfg.dino_prompt = env_prompt.strip()
        print(f"[perception] dino: box={cfg.dino_box_thresh} "
              f"text={cfg.dino_text_thresh} min_score={cfg.min_dino_score} "
              f"min_area={cfg.min_obj_area_px}")

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
        # Last pipeline.recognize_seq we re-pushed the highlight for. When
        # fast_track is on, the main loop re-pushes the active highlight
        # whenever this falls behind the pipeline counter (i.e. a new SAM
        # pass landed fresh positions), so the wash follows a moving object.
        self._last_repush_seq: int = 0
        # Step-2 fusion tracker (created lazily only if the flag is on, so the
        # import cost / cv2 tracker allocation is skipped when unused).
        self._fast: Optional[object] = None
        self._fast_seed_seq: int = 0   # recognize_seq we last reseeded at
        # Telemetry for the live tuning pass: per-object follow stats.
        self._fast_stats: dict = {}
        # ---- Lock mode (interactive "light the thing I'm holding/playing") --
        # When an object is LOCKED, the slow DINO+SAM tracker is forbidden from
        # touching its wash: we freeze the mask captured at lock time and drive
        # position purely from the fast tracker's depth-band gate + CSRT, which
        # reject the player's body (it sits at a different, nearer depth). This
        # severs all three ways the slow tracker poisons a held object: id
        # reassignment, reseed from an arm-merged mask, and Step-1 re-push of a
        # poisoned mask shape. None = not locked.
        self._locked_id: Optional[int] = None
        # Anchor captured at lock time (cam centroid + depth) — the offset the
        # projector applies is (live_proj - anchor_proj), so the frozen mask
        # slides to follow the object without ever re-warping.
        self._lock_anchor_cam: Optional[Tuple[float, float]] = None
        self._lock_depth: float = 0.0
        # Seed state: 'pending' = lock requested, fast tracker not yet seeded
        # (the ctrl thread has no live frame; the main loop seeds on the next
        # frame when the object is visible). 'seeded' = running. None = unlocked.
        self._lock_state: Optional[str] = None
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
            "effect": getattr(target, "effect", "color"),
            "proj_centroid": (
                list(target.centroid_proj) if target.centroid_proj else None
            ),
            "mask_path": _save_mask_png(target),
        }
        if pinned:
            payload["pinned"] = True
        self.proj_push.send_json(payload)
        return True

    def _push_highlight_all(self, ids: Optional[List[int]] = None) -> int:
        """Like _push_highlight but for the 'illuminate everything'
        broadcast. When `ids` is given, only that subset of visible
        objects is pushed (checkbox multi-select from the web UI).
        Returns the count actually pushed.

        Hysteresis: when self.highlight_hysteresis is on, an object whose
        fresh SAM mask hasn't moved/reshaped beyond threshold reuses the
        mask already on screen instead of re-warping. This kills the ~2 s
        jerk that SAM's run-to-run boundary wobble would otherwise inject on
        static objects (and skips the 8.3 MP warp + PNG write for them)."""
        with self.pipeline.tracker_lock:
            tracked = self.pipeline.tracker.visible()
        idset = {int(i) for i in ids} if ids is not None else None
        objects: List[dict] = []
        live_ids = set()
        for o in tracked:
            if idset is not None and o.object_id not in idset:
                continue
            if o.cam_mask is None:
                continue
            live_ids.add(o.object_id)
            prev = self._highlight_shown.get(o.object_id)
            # Decide whether the on-screen mask still represents this object.
            stable = False
            if self.highlight_hysteresis and prev is not None:
                stable = _highlight_mask_stable(
                    prev.get("centroid_cam"),
                    tuple(o.centroid_cam) if o.centroid_cam is not None
                    else None,
                    prev.get("cam_mask"),
                    o.cam_mask,
                    self.highlight_move_px,
                    self.highlight_iou,
                )
            if stable:
                # Hold: reuse the mask already on the projector. Refresh only
                # the live metadata (color/effect/name can change without a
                # reshape) — the heavy mask PNG and warp are untouched.
                o.proj_mask = prev.get("proj_mask")
                o.centroid_proj = prev.get("centroid_proj")
                objects.append({
                    "id": o.object_id,
                    "name": o.name,
                    "color": list(o.color_rgb),
                    "effect": getattr(o, "effect", "color"),
                    "proj_centroid": (
                        list(prev["centroid_proj"])
                        if prev.get("centroid_proj") else None
                    ),
                    "mask_path": prev.get("mask_path"),
                })
                continue
            # Update: re-warp from the fresh cam_mask (live cfg changes like
            # mask softness also apply here) and rewrite the projector PNG.
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
            mask_path = _save_mask_png(o)
            objects.append({
                "id": o.object_id,
                "name": o.name,
                "color": list(o.color_rgb),
                "effect": getattr(o, "effect", "color"),
                "proj_centroid": (
                    list(o.centroid_proj) if o.centroid_proj else None
                ),
                "mask_path": mask_path,
            })
            # Record what is now shown for the next pass's gate. Copy the
            # cam_mask so a later in-place tracker update can't mutate our
            # reference out from under the comparison.
            self._highlight_shown[o.object_id] = {
                "centroid_cam": (
                    tuple(o.centroid_cam)
                    if o.centroid_cam is not None else None
                ),
                "cam_mask": o.cam_mask.copy(),
                "proj_mask": o.proj_mask,
                "centroid_proj": o.centroid_proj,
                "mask_path": mask_path,
            }
        # Drop records for objects no longer in this highlight set so the
        # dict can't grow without bound across many select/deselect cycles.
        for dead in [k for k in self._highlight_shown if k not in live_ids]:
            del self._highlight_shown[dead]
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
            elif last.get("kind") == "set":
                self._push_highlight_all(ids=list(last.get("ids", [])))
        except Exception as e:  # noqa: BLE001
            print(f"[perception] refresh failed: {e!r}")

    def _maybe_repush_active_highlight(self) -> None:
        """Step-1 fast-follow: when a heavy DINO+SAM pass has landed fresh
        object positions (pipeline.recognize_seq advanced), re-emit the
        active highlight so the projector wash re-lands on the object's new
        location. Without this the wash only moves when an unrelated UI event
        (color change, object appear/disappear) happens to re-push.

        Gated on:
          - fast_track flag (off => legacy push-on-UI-event-only behavior),
          - an active highlight existing,
          - the test-point hold lock (don't stomp a /test_light square),
          - the recognize_seq actually advancing (no redundant PNG rewrites
            between SAM passes — protects the projector's mask-decode cache).

        Runs under _ctrl_lock so it can't interleave with a ctrl-thread push
        mid-payload. Tracker reads inside _push_highlight* take tracker_lock
        separately, so this never blocks the GPU loop.
        """
        should, new_seq = _repush_decision(
            fast_track=self.fast_track,
            cur_seq=self.pipeline.recognize_seq,
            last_repush_seq=self._last_repush_seq,
            has_active_highlight=bool(self._last_highlight),
            now=time.time(),
            test_hold_until=self._test_hold_until,
        )
        self._last_repush_seq = new_seq
        if not should:
            return
        with self._ctrl_lock:
            self._refresh_active_highlight()

    def _highlighted_ids(self, visible) -> list:
        """Resolve the currently-shown highlight to a concrete id list.
        ``visible`` is the tracker's visible() list (passed in so we don't
        re-lock). Returns [] when nothing is highlighted."""
        last = self._last_highlight
        if not last:
            return []
        kind = last.get("kind")
        if kind == "single":
            return [int(last["id"])]
        if kind == "set":
            return [int(i) for i in last.get("ids", [])]
        if kind == "all":
            return [o.object_id for o in visible]
        return []

    def _lock_follow_step(self, color, depth_m) -> None:
        """Lock-mode follow: drive the LOCKED object's wash purely from the
        fast tracker (depth-band gate + CSRT), with ZERO input from the slow
        DINO+SAM tracker after the initial clean acquire.

        This is the interactive "light the guitar I'm playing" path. The slow
        tracker is the thing the player's arm/body poisons (id reassignment,
        arm-merged masks, poisoned re-push). Lock severs all of it:

          * Seed ONCE from the mask captured at lock time (a clean moment),
            never reseed from the slow tracker again.
          * Position comes only from the fast tracker, whose depth band rejects
            the player's body (it sits ~1 m closer than the guitar) and whose
            CSRT holds the guitar's appearance through partial occlusion.
          * The projector keeps the frozen lock-time mask and just slides it by
            (live_proj - anchor_proj); no re-warp, no PNG rewrite.
        """
        oid = self._locked_id
        if oid is None:
            return
        if self._fast is None:
            from livetracking.perception.fasttrack import FastTracker
            self._fast = FastTracker()

        # First frame after a lock request: seed from the object's current
        # clean mask. The ctrl thread captured the anchor but had no live color
        # frame, so the CSRT seed (which needs the frame) happens here.
        if self._lock_state == "pending":
            with self.pipeline.tracker_lock:
                visible = self.pipeline.tracker.visible()
                target = next(
                    (o for o in visible if o.object_id == oid), None)
            if target is None or target.cam_mask is None:
                # Object not visible this frame (e.g. briefly occluded during
                # the acquire). Keep waiting — the frozen mask still shows.
                return
            self._fast.retain_only([oid])
            self._fast.reseed(oid, target.cam_mask, target.bbox_cam,
                              target.median_depth_m, color=color)
            self._lock_anchor_cam = tuple(target.centroid_cam)
            self._lock_depth = float(target.median_depth_m)
            self._lock_state = "seeded"
            return

        # Running: estimate live camera centroid from depth+CSRT only.
        est = self._fast.update(oid, color, depth_m)
        if est is None or self._lock_anchor_cam is None:
            return
        live_proj = self.pipeline.cam_to_proj_point(
            (est.cx, est.cy), self._lock_depth)
        anchor_proj = self.pipeline.cam_to_proj_point(
            self._lock_anchor_cam, self._lock_depth)
        if live_proj is None or anchor_proj is None:
            return
        ox = live_proj[0] - anchor_proj[0]
        oy = live_proj[1] - anchor_proj[1]
        self._fast_stats[oid] = {
            "moved_cam_px": round(est.moved_px, 1),
            "offset_proj_px": [round(ox, 1), round(oy, 1)],
            "conf": round(est.confidence, 2),
            "source": est.source,
            "locked": True,
        }
        self.proj_push.send_json({
            "type": "set_offsets",
            "offsets": {str(oid): [round(ox, 1), round(oy, 1)]},
        })

    def _fast_track_step(self, color, depth_m) -> None:
        """Step-2 inter-pass fusion: between SAM passes, estimate each
        highlighted object's live camera centroid and send the projector a
        per-object projector-pixel offset so the wash follows in real time.

        Reseeds the fusion tracker from fresh SAM masks whenever a heavy pass
        lands (recognize_seq advances) — that's the drift-bounding ground
        truth. Between passes, only the cheap depth-blob + CSRT estimators run
        and we emit a `set_offsets` message (no mask data, no PNG rewrite).
        """
        from livetracking.perception.fasttrack import FastTracker
        if self._fast is None:
            self._fast = FastTracker()

        with self.pipeline.tracker_lock:
            visible = self.pipeline.tracker.visible()
            ids = self._highlighted_ids(visible)
            by_id = {o.object_id: o for o in visible}
            seq = self.pipeline.recognize_seq

        if not ids:
            # Nothing highlighted — drop all tracks and clear any offsets once.
            if self._fast.active_ids():
                self._fast.retain_only([])
                self.proj_push.send_json({"type": "set_offsets",
                                          "offsets": {}})
            return

        # Reseed on a fresh SAM pass (positions just refreshed = ground truth).
        if seq != self._fast_seed_seq:
            self._fast_seed_seq = seq
            self._fast.retain_only(ids)
            for oid in ids:
                obj = by_id.get(oid)
                if obj is None or obj.cam_mask is None:
                    continue
                self._fast.reseed(
                    oid, obj.cam_mask, obj.bbox_cam,
                    obj.median_depth_m, color=color,
                )

        # Split the highlighted ids into those whose track is still live and
        # those whose track has died (churned away / disappeared since the last
        # SAM pass). A dead highlighted track MUST have its offset cleared this
        # frame: set_offsets fully replaces the projector's offset table, so if
        # we silently skip a dead id (the old behavior) the projector keeps
        # applying that object's last offset to its still-cached mask and the
        # wash flies off across the room. Prune dead ids from the fast tracker
        # and telemetry, and make sure we still emit set_offsets below so the
        # stale offset is dropped (wash falls back to its last anchored
        # position instead of running away).
        live_ids = [oid for oid in ids if oid in by_id]
        dead_ids = [oid for oid in ids if oid not in by_id]
        if dead_ids:
            self._fast.retain_only(live_ids)
            for oid in dead_ids:
                self._fast_stats.pop(oid, None)

        # Per-frame estimate -> projector-pixel offset for each live object.
        offsets = {}
        for oid in live_ids:
            obj = by_id[oid]
            est = self._fast.update(oid, color, depth_m)
            if est is None:
                continue
            # Map the live camera centroid to projector space, and the object's
            # SAM-anchored camera centroid too; the offset is the delta. Using
            # the delta (not absolute) means the cached mask — already warped
            # and positioned at the anchor — just slides by how far the object
            # moved, which is exactly right for a rigid translation.
            live_proj = self.pipeline.cam_to_proj_point(
                (est.cx, est.cy), obj.median_depth_m)
            anchor_proj = self.pipeline.cam_to_proj_point(
                obj.centroid_cam, obj.median_depth_m)
            if live_proj is None or anchor_proj is None:
                continue
            ox = live_proj[0] - anchor_proj[0]
            oy = live_proj[1] - anchor_proj[1]
            offsets[str(oid)] = [round(ox, 1), round(oy, 1)]
            self._fast_stats[oid] = {
                "moved_cam_px": round(est.moved_px, 1),
                "offset_proj_px": [round(ox, 1), round(oy, 1)],
                "conf": round(est.confidence, 2),
                "source": est.source,
            }
        # Emit when we have live offsets OR a track just died (so the
        # projector replaces its offset table and drops the dead object's
        # stale offset — sending an empty dict is the clear).
        if offsets or dead_ids:
            self.proj_push.send_json({"type": "set_offsets",
                                      "offsets": offsets})

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
        if cmd == "set_color":
            rgb = msg.get("rgb") or msg.get("color")
            if not (isinstance(rgb, (list, tuple)) and len(rgb) == 3):
                return {"ok": False, "reason": "bad rgb"}
            new = self.pipeline.tracker.set_color(int(msg["id"]), tuple(rgb))
            if new is None:
                return {"ok": False, "reason": "no such object"}
            self._refresh_active_highlight()
            return {"ok": True, "color": list(new)}
        if cmd == "cycle_effect":
            new = self.pipeline.tracker.cycle_effect(int(msg["id"]))
            if new is None:
                return {"ok": False, "reason": "no such object"}
            # Re-emit so the projector swaps to the new effect immediately.
            self._refresh_active_highlight()
            return {"ok": True, "effect": new}
        if cmd == "set_effect":
            new = self.pipeline.tracker.set_effect(
                int(msg["id"]), str(msg.get("effect", "color"))
            )
            if new is None:
                return {"ok": False, "reason": "no such object"}
            self._refresh_active_highlight()
            return {"ok": True, "effect": new}
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
        if cmd == "lock":
            # Interactive "light the thing I'm holding/playing": acquire the
            # object cleanly NOW, then hand it to the fast tracker and forbid
            # the slow DINO+SAM tracker from ever touching its wash again. The
            # depth-band gate + CSRT reject the player's body (different depth),
            # so reaching in / playing can't steal the highlight.
            obj_id = int(msg["id"])
            with self.pipeline.tracker_lock:
                tracked = self.pipeline.tracker.visible()
            target = next((o for o in tracked if o.object_id == obj_id), None)
            if target is None or target.cam_mask is None:
                return {"ok": False, "reason": "no such object / no cam_mask"}
            # Push the clean highlight one last time (frozen mask the projector
            # will keep and just slide via set_offsets from here on).
            ok = self._push_highlight(obj_id)
            if not ok:
                return {"ok": False, "reason": "highlight push failed"}
            self._locked_id = obj_id
            self._last_highlight = {"kind": "single", "id": obj_id,
                                    "pinned": True, "locked": True}
            self._pinned_id = obj_id  # locked implies pinned (no auto-clear)
            self._lock_anchor_cam = tuple(target.centroid_cam)
            self._lock_depth = float(target.median_depth_m)
            # Seed happens in the main loop where the live color frame exists.
            self._lock_state = "pending"
            if not (self.fast_track and self.fast_track_fusion):
                return {"ok": True, "warn": (
                    "locked, but fast_track + fast_track_fusion must both be "
                    "ON for the wash to follow without the slow tracker")}
            return {"ok": True, "locked_id": obj_id}
        if cmd == "unlock":
            self._locked_id = None
            self._lock_state = None
            self._lock_anchor_cam = None
            self._lock_depth = 0.0
            self._pinned_id = None
            if self._fast is not None:
                try:
                    self._fast.retain_only([])
                except Exception:
                    pass
            self.proj_push.send_json({"type": "set_offsets", "offsets": {}})
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
        if cmd == "highlight_set":
            # Checkbox multi-select: project exactly this set of objects.
            # Empty list == clear (unless something is pinned).
            try:
                ids = [int(i) for i in (msg.get("ids") or [])]
            except Exception:
                return {"ok": False, "reason": "bad 'ids' payload"}
            if not ids:
                if self._pinned_id is not None:
                    return {"ok": True, "count": 0, "ignored": "pinned"}
                self.proj_push.send_json({"type": "clear"})
                self._last_highlight = None
                return {"ok": True, "count": 0}
            n = self._push_highlight_all(ids=ids)
            self._last_highlight = {"kind": "set", "ids": ids}
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
                    "detector": getattr(self, "detector_name", "dino"),
                    "highlight": self._last_highlight,
                    "pinned_id": self._pinned_id,
                    "locked_id": self._locked_id,
                    "lock_state": self._lock_state}
        if cmd == "detector_info":
            return {"ok": True, "detector": getattr(self, "detector_name", "dino")}
        if cmd == "fast_stats":
            # Step-2 telemetry for the live tuning pass: per-object follow
            # state (how far it moved, the projector offset applied, fusion
            # source, confidence) plus the flag states.
            return {"ok": True,
                    "fast_track": self.fast_track,
                    "fast_track_fusion": self.fast_track_fusion,
                    "locked_id": self._locked_id,
                    "lock_state": self._lock_state,
                    "objects": {str(k): v for k, v in self._fast_stats.items()}}
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
        if cmd == "dino_get":
            cfg = self.pipeline.cfg
            return {"ok": True,
                    "box_thresh": cfg.dino_box_thresh,
                    "text_thresh": cfg.dino_text_thresh,
                    "min_score": cfg.min_dino_score,
                    "min_area_px": int(cfg.min_obj_area_px),
                    "prompt": cfg.dino_prompt}
        if cmd == "dino_tune":
            # Live-mutate DINO detection knobs. Pipeline reads cfg on every
            # recognize pass, so changes apply on the next DINO frame (~1 s)
            # without a restart. Bounds match the boot-time env clamps.
            cfg = self.pipeline.cfg
            changed = {}
            if "box_thresh" in msg:
                v = float(msg["box_thresh"])
                cfg.dino_box_thresh = max(0.01, min(0.95, v))
                changed["box_thresh"] = cfg.dino_box_thresh
            if "text_thresh" in msg:
                v = float(msg["text_thresh"])
                cfg.dino_text_thresh = max(0.01, min(0.95, v))
                changed["text_thresh"] = cfg.dino_text_thresh
            if "min_score" in msg:
                v = float(msg["min_score"])
                cfg.min_dino_score = max(0.01, min(0.95, v))
                changed["min_score"] = cfg.min_dino_score
            if "min_area_px" in msg:
                v = int(msg["min_area_px"])
                cfg.min_obj_area_px = max(50, min(50000, v))
                changed["min_area_px"] = cfg.min_obj_area_px
            if "prompt" in msg:
                p = str(msg["prompt"]).strip()
                if p:
                    # DINO wants 'a. b. c.' — make sure it ends with a dot,
                    # otherwise the last class silently scores near zero.
                    if not p.endswith("."):
                        p += "."
                    cfg.dino_prompt = p
                    changed["prompt"] = cfg.dino_prompt
            print(f"[perception] dino_tune applied: "
                  f"{ {k: (v[:60] + '…' if isinstance(v, str) and len(v) > 60 else v) for k, v in changed.items()} }")
            return {"ok": True, "changed": changed,
                    "current": {"box_thresh": cfg.dino_box_thresh,
                                "text_thresh": cfg.dino_text_thresh,
                                "min_score": cfg.min_dino_score,
                                "min_area_px": int(cfg.min_obj_area_px),
                                "prompt": cfg.dino_prompt}}
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
            # Fast-follow: if a heavy pass just landed fresh positions,
            # re-push the active highlight so the wash tracks the object.
            #
            # LOCK MODE takes over completely when an object is locked: the
            # slow tracker is barred from touching the locked wash (no re-push,
            # no fusion reseed), and a dedicated depth+CSRT follow drives it so
            # the player's body can't steal it. Otherwise run the normal
            # Step-1 re-push (+ Step-2 fusion) highlight flow.
            if self._locked_id is not None:
                try:
                    self._lock_follow_step(frame.color, frame.depth_m)
                except Exception as e:  # noqa: BLE001
                    print(f"[perception] lock_follow_step error: {e!r}")
            else:
                self._maybe_repush_active_highlight()
                # Step-2: between passes, slide the cached wash to the object's
                # live position via cheap depth+CSRT tracking (flag-gated).
                if self.fast_track and self.fast_track_fusion:
                    try:
                        self._fast_track_step(frame.color, frame.depth_m)
                    except Exception as e:  # noqa: BLE001
                        print(f"[perception] fast_track_step error: {e!r}")
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

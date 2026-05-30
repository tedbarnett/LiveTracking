"""Image-first perception pipeline.

For each (color, depth) frame:

  1. Run Grounding DINO on the color frame -> per-object bboxes.
  2. Filter detections whose bbox center is OUTSIDE the projector footprint.
  3. Run SAM2 at each surviving bbox center -> pixel-accurate cam_mask.
  4. AND with the footprint, drop tiny masks.
  5. Median depth per mask. If wall_plane is loaded, gate against it
     (drop masks whose median depth is beyond the wall -- looking through
     the doorway etc.).
  6. Warp each cam_mask with parallax compensation into projector pixels.
  7. Hand FreshDetections to ObjectTracker for stable IDs.

Async mode: a background thread runs the heavy DINO+SAM pass on the freshest
frame; the main loop returns immediately with the current tracker state.
This decouples perception's output rate from DINO's ~80 ms cost.

This replaces the older depth-first (Stage-1 blob) pipeline. Depth is now
used only for:
  * Median depth per object (for parallax + UI).
  * Optional wall-plane gating (drop "objects" sitting beyond the wall).

The footprint mask remains depth-free: it's a homography-derived camera-space
region representing where the projector can reach.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .footprint import footprint_mask_in_camera
from .recognize import DEFAULT_DINO_PROMPT, Recognizer
from .tracker import FreshDetection, ObjectTracker
from .types import DetectedObject


@dataclass
class PipelineConfig:
    # D455 default intrinsics @ 848x480 aligned-to-color stream.
    intrinsics: Tuple[float, float, float, float] = (615.0, 615.0, 424.0, 240.0)
    proj_w: int = 3840
    proj_h: int = 2160

    # DINO open-vocab detection.
    dino_prompt: str = DEFAULT_DINO_PROMPT
    dino_box_thresh: float = 0.30
    dino_text_thresh: float = 0.25
    min_dino_score: float = 0.30

    # Object viability after SAM.
    min_obj_area_px: int = 800

    # Per-object parallax compensation. The wall-plane homography aims at the
    # wall behind every object; for objects sitting in front of the wall we
    # shift their camera-space mask before warping so the projector hits the
    # object itself, not its shadow on the wall.
    #
    # Geometry: camera and projector are separated by a horizontal baseline
    # B (projector RIGHT of camera in the current rig). For a point on the
    # wall, H is exact. For a point at depth z_obj < z_wall the projector
    # pixel that hits the WALL along the camera ray sits to the RIGHT of
    # the projector pixel that would hit the OBJECT, by approximately
    #     Δx_proj  ≈  f_proj * B * (1/z_obj − 1/z_wall)
    # so we shift the warped projector mask LEFT (negative x) by that
    # amount. We lump (f_proj * B) into a single tunable constant
    # `parallax_k_px_m` (pixels · meters); default 1200 ≈ f_proj ~3000 px *
    # B ~0.40 m (effective; tune live with LIVETRACKING_PARALLAX_K).
    # `parallax_sign` sets the baseline direction: +1 for projector right
    # of camera (default, shifts wash RIGHTWARD in projector pixels for
    # near objects -- pulls wash from the wall-shadow back onto the
    # object), -1 for projector left of camera.
    # `parallax_scale` is a final multiplier for live tuning.
    parallax_compensate: bool = True
    parallax_sign: float = 1.0
    parallax_scale: float = 1.0
    parallax_k_px_m: float = 1200.0

    # Wall-plane depth gating. Drop SAM masks whose median depth is more
    # than this many meters DEEPER than the calibrated wall plane at the
    # mask centroid. Catches detections seen through a doorway, in a
    # mirror, etc. Disabled when wall_plane is not loaded.
    wall_gate_m: float = 0.40

    # Tracker.
    iou_match_threshold: float = 0.25
    stale_after_s: float = 60.0

    # Async DINO+SAM. Main loop returns immediately with the cached
    # tracker state; a worker thread updates it whenever DINO+SAM
    # finishes on the freshest submitted frame.
    async_recognize: bool = True


class Pipeline:
    """Image-first pipeline: DINO -> SAM -> depth-assist -> tracker."""

    def __init__(
        self,
        H: np.ndarray,
        cam_w: int,
        cam_h: int,
        config: Optional[PipelineConfig] = None,
        recognizer: Optional[Recognizer] = None,
    ):
        self.H = H
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.cfg = config or PipelineConfig()
        self.footprint = footprint_mask_in_camera(
            H, self.cfg.proj_w, self.cfg.proj_h, cam_w, cam_h
        )
        self.recognizer = recognizer or Recognizer()

        # Load calibrated wall plane (a, b, c, d) so aX+bY+cZ+d=0 for
        # points on the back wall in camera 3D space. Saved by
        # scripts/calibrate_homography.py.
        self.wall_plane: Optional[List[float]] = None
        try:
            wp_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))),
                "runtime", "calibration", "wall_plane.npy",
            )
            if os.path.exists(wp_path):
                wp = np.load(wp_path)
                if wp.size == 4:
                    self.wall_plane = wp.tolist()
                    print(f"[pipeline] loaded calibrated wall plane: "
                          f"{[round(x, 4) for x in self.wall_plane]}")
        except Exception as _e:  # noqa: BLE001
            print(f"[pipeline] wall_plane.npy load failed: {_e}")

        self.tracker = ObjectTracker(
            iou_match_threshold=self.cfg.iou_match_threshold,
            stale_after_s=self.cfg.stale_after_s,
        )
        self.tracker_lock = threading.Lock()

        # Telemetry compatibility with the perception daemon.
        self.last_stage1_debug: dict = {}  # legacy key; kept for ctrl handlers
        self.last_timings_ms: dict = {
            "total_ms": 0.0, "dino_ms": 0.0, "sam_ms": 0.0,
            "stage1_ms": 0.0, "fast": False,
        }

        # Async machinery.
        self._async_box: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._async_cv = threading.Condition()
        self._async_thread: Optional[threading.Thread] = None
        self._async_running = False

        self._frame_idx = 0

    # ---- async lifecycle -----------------------------------------------
    def start_async(self):
        if self._async_running:
            return
        self._async_running = True
        self._async_thread = threading.Thread(
            target=self._async_loop, daemon=True, name="recognize"
        )
        self._async_thread.start()

    def stop_async(self):
        self._async_running = False
        with self._async_cv:
            self._async_cv.notify_all()
        if self._async_thread:
            self._async_thread.join(timeout=2.0)

    def _async_loop(self):
        """Pop the freshest submitted frame, run DINO+SAM, update tracker."""
        while self._async_running:
            with self._async_cv:
                while self._async_running and self._async_box is None:
                    self._async_cv.wait(timeout=0.5)
                if not self._async_running:
                    return
                box = self._async_box
                self._async_box = None
            color, depth_m = box
            try:
                fresh, timings = self._recognize_one(color, depth_m)
            except Exception as e:  # noqa: BLE001
                print(f"[pipeline] async recognize crashed: {e}")
                continue
            with self.tracker_lock:
                self.tracker.update(fresh)
                self.last_timings_ms = timings

    # ---- public step entry --------------------------------------------
    def step_auto(self, color: np.ndarray, depth_m: np.ndarray
                  ) -> List[DetectedObject]:
        """Main-loop entry point. Always returns immediately with the
        current tracker state; submits this frame to the worker."""
        self._frame_idx += 1
        if self.cfg.async_recognize:
            if not self._async_running:
                self.start_async()
            # Replace any pending frame -- worker always uses the newest.
            with self._async_cv:
                self._async_box = (color, depth_m)
                self._async_cv.notify()
            # Bootstrap: if tracker is empty, do one sync pass so we have
            # objects to display from frame ~1 instead of after the first
            # ~100 ms DINO finishes asynchronously.
            with self.tracker_lock:
                if not self.tracker.active():
                    fresh, timings = self._recognize_one(color, depth_m)
                    self.tracker.update(fresh)
                    self.last_timings_ms = timings
                return self.tracker.visible()

        # Sync mode.
        fresh, timings = self._recognize_one(color, depth_m)
        with self.tracker_lock:
            self.tracker.update(fresh)
            self.last_timings_ms = timings
            return self.tracker.visible()

    # ---- the heavy pass -----------------------------------------------
    def _recognize_one(self, color: np.ndarray, depth_m: np.ndarray
                       ) -> Tuple[List[FreshDetection], dict]:
        """Image-first pipeline: DINO -> footprint filter -> SAM -> mask
        gates -> parallax warp -> FreshDetection list."""
        t0 = time.perf_counter()
        timings = {"total_ms": 0.0, "dino_ms": 0.0, "sam_ms": 0.0,
                   "stage1_ms": 0.0, "fast": False,
                   "n_dino_raw": 0, "n_dino_kept": 0, "n_objects": 0,
                   "dino_n": 0, "kept_n": 0, "sam_n": 0}

        # Stage A: DINO.
        t = time.perf_counter()
        dets = self.recognizer.label_image(
            color, prompt=self.cfg.dino_prompt,
            box_threshold=self.cfg.dino_box_thresh,
            text_threshold=self.cfg.dino_text_thresh,
        )
        timings["dino_ms"] = (time.perf_counter() - t) * 1000.0
        timings["dino_n"] = len(dets)
        timings["n_dino_raw"] = len(dets)

        # Stage B: drop low-confidence + outside-footprint.
        kept: List[dict] = []
        for d in dets:
            if d["score"] < self.cfg.min_dino_score:
                continue
            x, y, w, h = d["bbox"]
            cx_b, cy_b = int(x + w / 2), int(y + h / 2)
            if not (0 <= cx_b < self.cam_w and 0 <= cy_b < self.cam_h):
                continue
            if self.footprint[cy_b, cx_b] == 0:
                continue
            kept.append(d)
        kept = self._dedupe_dino(kept)
        timings["kept_n"] = len(kept)
        timings["n_dino_kept"] = len(kept)

        if not kept:
            timings["total_ms"] = (time.perf_counter() - t0) * 1000.0
            return [], timings

        # Stage C: SAM. Point-prompt each kept bbox at its center.
        centers = [(d["bbox"][0] + d["bbox"][2] / 2,
                    d["bbox"][1] + d["bbox"][3] / 2)
                   for d in kept]
        t = time.perf_counter()
        sam_out = self.recognizer.segment_with_points(color, centers)
        timings["sam_ms"] = (time.perf_counter() - t) * 1000.0
        timings["sam_n"] = len(sam_out)

        # Stage D: each (det, sam_mask) -> FreshDetection if it survives
        # area + wall-depth gates + mask-quality cleanup.
        fresh: List[FreshDetection] = []
        # Track which (label, bbox) won the same-label NMS round.
        _winners: List[Tuple[str, Tuple[int, int, int, int], int]] = []
        for det, (cam_mask, _sam_score) in zip(kept, sam_out):
            _lbl = det.get("label", "")
            _bx, _by, _bw, _bh = det["bbox"]
            _trace = (f"  lbl='{_lbl}' score={det['score']:.2f} "
                      f"dino_bbox=[{int(_bx)},{int(_by)},"
                      f"{int(_bw)}x{int(_bh)}] sam_raw={int((cam_mask>0).sum())}")
            def _log(msg):
                try:
                    p = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(
                            os.path.dirname(os.path.abspath(__file__))))),
                        "runtime", "logs", "detection_trace.log")
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "a", encoding="utf-8") as f:
                        f.write(msg + "\n")
                except Exception:
                    pass
            # Clip to footprint AND to a small expansion of the DINO bbox.
            # SAM is generous and routinely leaks beyond DINO's box --
            # but DINO's box is the *signal* about what was actually
            # detected. Anything outside bbox*1.5 around its center is
            # almost always a leak onto a neighbor (the drum mask
            # creeping onto the cushion behind it).
            bx, by, bw_, bh_ = det["bbox"]
            cxb = bx + bw_ * 0.5
            cyb = by + bh_ * 0.5
            ew = bw_ * 0.75   # half-extent (1.5x bbox total)
            eh = bh_ * 0.75
            x0 = max(0, int(cxb - ew))
            y0 = max(0, int(cyb - eh))
            x1 = min(self.cam_w, int(cxb + ew))
            y1 = min(self.cam_h, int(cyb + eh))
            bbox_mask = np.zeros_like(self.footprint)
            bbox_mask[y0:y1, x0:x1] = 255
            cam_mask = cv2.bitwise_and(cam_mask, self.footprint)
            cam_mask = cv2.bitwise_and(cam_mask, bbox_mask)
            area = int((cam_mask > 0).sum())
            if area < self.cfg.min_obj_area_px:
                print(_trace + f" -> DROP bbox_clip area={area}", flush=True)
                _log(_trace + f" -> DROP bbox_clip area={area}")
                continue

            # --- Mask-quality cleanup -----------------------------------
            # Two failure modes the photos showed:
            #   (a) wall-object label leaks onto closer foreground (Poster
            #       1609 grabbed half the couch). Pixel depths span a huge
            #       range; median is misleading.
            #   (b) right-object label boundary-leaks onto a near neighbor
            #       (bodhran mask creeping onto the cushion behind it).
            #       Centroid drifts -> H warps wrong -> projection misses.
            # Fix: depth-band the mask. Keep only pixels within 0.25 m of
            # the **mode/median** depth, then morphologically clean it.
            zp_all = depth_m[(cam_mask > 0) & (depth_m > 0)]
            if zp_all.size < 50:
                # Too little depth signal to trust; fall back to raw mask.
                med_z = float(np.median(zp_all)) if zp_all.size else 0.0
            else:
                med_z0 = float(np.median(zp_all))
                # Depth-band: drop mask pixels too far from median depth.
                z_band = 0.25  # meters
                depth_keep = ((cam_mask > 0)
                              & (depth_m > med_z0 - z_band)
                              & (depth_m < med_z0 + z_band))
                cleaned = depth_keep.astype(np.uint8) * 255
                # Morph open to break leak tendrils, then close to seal
                # small holes punched by depth noise.
                k3 = np.ones((3, 3), np.uint8)
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k3,
                                           iterations=2)
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k3,
                                           iterations=1)
                # Keep only the largest connected component -- a
                # boundary-leak that survived open/close is usually a
                # separate blob clinging to the real object.
                n_lbl, lbl_img, stats, _ = cv2.connectedComponentsWithStats(
                    cleaned, connectivity=8)
                if n_lbl > 1:
                    # Skip background label 0.
                    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    cleaned = ((lbl_img == largest).astype(np.uint8) * 255)
                area2 = int((cleaned > 0).sum())
                # If cleanup ate too much of the mask the SAM mask was
                # garbage -- drop the whole detection.
                if area2 < max(self.cfg.min_obj_area_px, area // 4):
                    _log(_trace + f" -> DROP depth_clean pre={area} post={area2} med_z={med_z0:.2f}")
                    continue
                cam_mask = cleaned
                zp_clean = depth_m[(cam_mask > 0) & (depth_m > 0)]
                med_z = (float(np.median(zp_clean)) if zp_clean.size
                         else med_z0)
                area = area2

            # --- Sanity gates -------------------------------------------
            # Frame-fraction cap: anything covering > 25% of the frame is
            # almost certainly a leak across multiple objects.
            if area > 0.25 * self.cam_w * self.cam_h:
                _log(_trace + f" -> DROP frame_frac area={area}")
                continue
            # Aspect-ratio cap for "poster|map|frame|painting|sign" labels:
            # phantom seam strips between two real posters are very tall +
            # narrow.
            ys_chk, xs_chk = np.where(cam_mask > 0)
            if ys_chk.size:
                bw = int(xs_chk.max() - xs_chk.min() + 1)
                bh = int(ys_chk.max() - ys_chk.min() + 1)
                aspect = max(bw, bh) / max(1, min(bw, bh))
                dlbl = det.get("label", "").lower()
                if aspect > 4.0 and any(w in dlbl for w in
                                        ("poster", "map", "frame",
                                         "painting", "sign", "picture")):
                    _log(_trace + f" -> DROP aspect={aspect:.1f}")
                    continue
                det_bbox_xywh = (int(xs_chk.min()), int(ys_chk.min()),
                                 bw, bh)
            else:
                _log(_trace + " -> DROP empty_post_clean")
                continue

            # Same-label NMS on the post-cleanup bbox: among detections
            # sharing a label word, drop later (lower-score) bboxes that
            # overlap a kept one with IoU > 0.30.
            dlbl_words = set(det.get("label", "").lower().split())
            dropped_by_nms = False
            for (wlbl, wbbox, _) in _winners:
                if not (dlbl_words & set(wlbl.split())):
                    continue
                x, y, w, h = det_bbox_xywh
                kx, ky, kw, kh = wbbox
                ix0 = max(x, kx); iy0 = max(y, ky)
                ix1 = min(x + w, kx + kw); iy1 = min(y + h, ky + kh)
                iw = max(0, ix1 - ix0); ih2 = max(0, iy1 - iy0)
                inter = iw * ih2
                union = w * h + kw * kh - inter
                if union > 0 and inter / union > 0.30:
                    dropped_by_nms = True
                    break
            if dropped_by_nms:
                _log(_trace + " -> DROP same_label_nms")
                continue
            _winners.append((det.get("label", "").lower(),
                             det_bbox_xywh, area))
            _log(_trace + f" -> KEEP area={area} z={med_z:.2f}")

            # Wall-depth gate: drop masks that sit BEHIND the wall plane.
            # (Objects in a doorway / mirror / through-window detections.)
            if self.wall_plane is not None and med_z > 0.1:
                ys_m, xs_m = np.where(cam_mask > 0)
                cx_m = float(xs_m.mean()); cy_m = float(ys_m.mean())
                z_wall = self._wall_z_at(cx_m, cy_m)
                if z_wall > 0.1 and med_z > z_wall + self.cfg.wall_gate_m:
                    # Past the wall -> ignore.
                    continue

            # Warp through H with parallax shift.
            proj_mask, proj_centroid = self._warp_with_parallax_image_first(
                cam_mask, med_z
            )
            if proj_mask is None:
                continue

            fresh.append(FreshDetection(
                cam_mask=cam_mask,
                label=det["label"] or "object",
                label_score=float(det["score"]),
                median_depth_m=med_z,
                proj_mask=proj_mask,
                proj_centroid=proj_centroid,
            ))

        timings["total_ms"] = (time.perf_counter() - t0) * 1000.0
        timings["n_objects"] = len(fresh)
        return fresh, timings

    # ---- geometry helpers ---------------------------------------------
    def _wall_z_at(self, u: float, v: float) -> float:
        """Predicted wall-plane depth at camera pixel (u, v)."""
        if self.wall_plane is None:
            return 0.0
        a, b, c, d = self.wall_plane
        fx, fy, cx, cy = self.cfg.intrinsics
        denom = a * (u - cx) / fx + b * (v - cy) / fy + c
        if abs(denom) < 1e-6:
            return 0.0
        return -d / denom

    def _warp_with_parallax_image_first(
        self, cam_mask: np.ndarray, med_z: float,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]]]:
        """Parallax-shift cam_mask then warp into projector pixels."""
        fx, fy, cx, cy = self.cfg.intrinsics
        ys_m, xs_m = np.where(cam_mask > 0)
        if ys_m.size == 0:
            return None, None

        # INTER_LINEAR + re-threshold gives smooth, anti-aliased
        # projector-mask edges instead of the pixel staircases that
        # INTER_NEAREST produces on diagonals.
        proj_mask = cv2.warpPerspective(
            cam_mask, self.H,
            (self.cfg.proj_w, self.cfg.proj_h),
            flags=cv2.INTER_LINEAR,
        )
        _, proj_mask = cv2.threshold(proj_mask, 127, 255, cv2.THRESH_BINARY)

        # Constant baseline parallax shift in PROJECTOR pixels. Applied
        # AFTER warpPerspective because the parallax error is a property
        # of the camera-projector baseline in the projector's view, not a
        # radial scaling around the camera's principal point (the old
        # cam-space (cx_m - cx) * (z_wall-z_obj)/z_wall formula zeroed
        # out for centered objects like the bodhran -- the bug we just
        # fixed).
        if (self.cfg.parallax_compensate
                and self.wall_plane is not None
                and med_z > 0.1):
            cx_m = float(xs_m.mean()); cy_m = float(ys_m.mean())
            z_wall = self._wall_z_at(cx_m, cy_m)
            if z_wall > 0.1 and med_z < z_wall - 0.05:
                disparity = (1.0 / med_z) - (1.0 / z_wall)  # 1/m, positive
                shift_x = (self.cfg.parallax_sign
                           * self.cfg.parallax_scale
                           * self.cfg.parallax_k_px_m
                           * disparity)
                if abs(shift_x) > 0.5:
                    M = np.array([[1.0, 0.0, shift_x],
                                  [0.0, 1.0, 0.0]], dtype=np.float32)
                    proj_mask = cv2.warpAffine(
                        proj_mask, M,
                        (self.cfg.proj_w, self.cfg.proj_h),
                        flags=cv2.INTER_LINEAR, borderValue=0,
                    )
                    _, proj_mask = cv2.threshold(
                        proj_mask, 127, 255, cv2.THRESH_BINARY)
        if not proj_mask.any():
            return None, None
        py, px = np.where(proj_mask > 0)
        return proj_mask, (float(px.mean()), float(py.mean()))

    # ---- back-compat _warp_with_parallax used by perception ctrl ------
    def _warp_with_parallax(
        self, cam_mask: np.ndarray, depth_m: np.ndarray, plane
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]], float]:
        """Compat shim for perception.test_point ctrl handler. The new
        path doesn't need the `plane` arg (we use self.wall_plane), but
        we keep the signature so existing callers still work."""
        zp = depth_m[(cam_mask > 0) & (depth_m > 0)]
        med_z = float(np.median(zp)) if zp.size else 0.0
        proj_mask, proj_centroid = self._warp_with_parallax_image_first(
            cam_mask, med_z
        )
        return proj_mask, proj_centroid, med_z

    # ---- misc ----------------------------------------------------------
    def _dedupe_dino(self, dets: List[dict]) -> List[dict]:
        """Drop redundant DINO bboxes only when SAME label and high IoU.
        Image-first: a guitar bbox sitting INSIDE a couch bbox is a real
        separate object, not a duplicate -- don't drop it just because
        the bigger box contains it."""
        if not dets:
            return []
        dets = sorted(dets, key=lambda d: -d["score"])
        keep: List[dict] = []
        for d in dets:
            x, y, w, h = d["bbox"]
            ok = True
            for k in keep:
                # Only consider as a duplicate if labels share a word
                # (DINO's prompt-token labels often glue several into one
                # string like 'guitar acoustic guitar electric guitar').
                d_words = set(d["label"].lower().split())
                k_words = set(k["label"].lower().split())
                if not (d_words & k_words):
                    continue
                kx, ky, kw, kh = k["bbox"]
                ix0 = max(x, kx); iy0 = max(y, ky)
                ix1 = min(x + w, kx + kw); iy1 = min(y + h, ky + kh)
                iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
                inter = iw * ih
                union = w * h + kw * kh - inter
                if union > 0 and inter / union > 0.5:
                    ok = False
                    break
            if ok:
                keep.append(d)
        return keep

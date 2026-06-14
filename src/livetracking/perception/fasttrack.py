"""Inter-pass fast object tracking (Step 2 of fast object-following).

The heavy DINO+SAM pass runs ~2.5x/sec. Step 1 makes the projector wash
re-land on an object whenever that pass produces fresh positions, so a moving
object's wash follows at ~2.5 Hz. This module closes the remaining gap:
between SAM passes it cheaply estimates where each *highlighted* object moved,
at camera-frame rate (~30-60 fps), so the wash follows in real time.

Per the design agreed with Ted, each object is tracked by FUSING two
estimators whose failure modes are opposite:

  * Depth-blob (POSITION ANCHOR, drift-free): in a window around the object's
    last position, keep pixels within +/- a band of the object's known depth
    and take the largest blob's centroid. Cheap (~2-3 ms). Robust precisely
    because the rig's objects sit in front of the wall at a separable depth --
    great on plain/shiny surfaces (the bodhran) where appearance trackers
    choke. Weak when two objects share a depth, or depth drops out on a shiny
    frame.

  * CSRT (IDENTITY + BRIDGE): an OpenCV appearance tracker, re-seeded from the
    fresh SAM mask each heavy pass. Locks onto identity, bridges depth-dropout
    frames, and disambiguates same-depth neighbours. Drifts over time and is
    weak on plain surfaces -- which is exactly where depth-blob is strong.

Fusion (decided per frame from quality signals):
  1. Agree (centroids within agree_px)      -> use depth (no drift). HIGH conf.
  2. Depth clean, CSRT drifted               -> snap CSRT onto depth. HIGH.
  3. Depth ambiguous/dropped, CSRT ok        -> trust CSRT (bridge). MED.
  4. Both low confidence                     -> FREEZE at last good position.
                                                The wash never flies off.

The heavy pass is the ground truth: ``reseed`` re-anchors BOTH estimators
every SAM pass, bounding drift for both.

Pure cv2 + numpy -- no camera, no torch, no pygame -- so it unit-tests with
synthetic frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ---- CSRT factory (OpenCV moved trackers between namespaces by version) --

def _make_csrt():
    """Return a fresh CSRT tracker across OpenCV API variants, or None if
    this build has no CSRT (depth-blob-only fallback still works)."""
    # cv2 >= 4.5.1 main namespace
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT") and hasattr(cv2.TrackerCSRT, "create"):
        return cv2.TrackerCSRT.create()
    # legacy namespace (contrib)
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
        return legacy.TrackerCSRT_create()
    return None


@dataclass
class FastEstimate:
    """One fast-frame position estimate for an object."""
    cx: float
    cy: float
    confidence: float          # 0..1
    source: str                # "depth" | "csrt" | "fused" | "frozen"
    moved_px: float = 0.0      # distance from the last SAM-anchored centroid


@dataclass
class _ObjState:
    ref_depth_m: float
    anchor_xy: Tuple[float, float]      # centroid at last SAM reseed
    last_xy: Tuple[float, float]        # last good centroid (any source)
    bbox_wh: Tuple[int, int]            # object size in camera px (w, h)
    csrt: object = None                 # CSRT tracker or None
    csrt_ok: bool = True
    last_conf: float = 1.0
    misses: int = 0
    frozen: bool = False


def _fuse_estimates(
    depth_xy: Optional[Tuple[float, float]],
    depth_conf: float,
    csrt_xy: Optional[Tuple[float, float]],
    csrt_conf: float,
    last_xy: Tuple[float, float],
    agree_px: float,
) -> Tuple[float, float, float, str]:
    """Pure fusion decision. Returns (cx, cy, confidence, source).

    Encodes the 4-case table from the module docstring. Kept pure (no cv2,
    no object state) so every branch is unit-testable in isolation.

    depth_conf / csrt_conf are 0..1 quality signals; an estimate with conf
    <= 0 is treated as "unavailable". ``last_xy`` is the freeze fallback.
    """
    depth_ok = depth_xy is not None and depth_conf > 0.0
    csrt_ok = csrt_xy is not None and csrt_conf > 0.0

    # Case 4 (both unavailable): freeze.
    if not depth_ok and not csrt_ok:
        return last_xy[0], last_xy[1], 0.0, "frozen"

    # Depth alone.
    if depth_ok and not csrt_ok:
        return depth_xy[0], depth_xy[1], depth_conf, "depth"

    # CSRT alone (Case 3: depth dropped/ambiguous, appearance bridges).
    if csrt_ok and not depth_ok:
        return csrt_xy[0], csrt_xy[1], csrt_conf * 0.8, "csrt"

    # Both available -> compare.
    dx = depth_xy[0] - csrt_xy[0]
    dy = depth_xy[1] - csrt_xy[1]
    dist = (dx * dx + dy * dy) ** 0.5
    if dist <= agree_px:
        # Case 1: agree -> trust depth (drift-free), boosted confidence.
        conf = min(1.0, 0.5 * (depth_conf + csrt_conf) + 0.25)
        return depth_xy[0], depth_xy[1], conf, "fused"
    # Case 2: disagree. Depth is the absolute anchor when it's confident;
    # otherwise the trackers genuinely conflict -> low confidence, but still
    # prefer depth (it doesn't drift) and let the caller's freeze-on-low-conf
    # hysteresis decide whether to hold.
    if depth_conf >= csrt_conf:
        return depth_xy[0], depth_xy[1], depth_conf * 0.6, "depth"
    return csrt_xy[0], csrt_xy[1], csrt_conf * 0.6, "csrt"


class FastTracker:
    def __init__(
        self,
        depth_band_m: float = 0.25,
        search_scale: float = 1.6,
        agree_px: float = 40.0,
        min_blob_frac: float = 0.15,
        freeze_after_misses: int = 3,
    ):
        """
        depth_band_m:    keep pixels within +/- this of the object's ref depth.
        search_scale:    search window = bbox * this, centered on last pos.
        agree_px:        depth vs CSRT centroids within this => "agree".
        min_blob_frac:   depth blob must be >= this fraction of the object's
                         seed area to count as a confident detection.
        freeze_after_misses: consecutive low-conf frames before we report a
                         frozen (held) position with zero confidence.
        """
        self.depth_band_m = depth_band_m
        self.search_scale = search_scale
        self.agree_px = agree_px
        self.min_blob_frac = min_blob_frac
        self.freeze_after_misses = freeze_after_misses
        self._objs: Dict[int, _ObjState] = {}
        self._seed_area: Dict[int, int] = {}

    # ---- lifecycle ----------------------------------------------------

    def active_ids(self):
        return list(self._objs.keys())

    def drop(self, object_id: int) -> None:
        self._objs.pop(object_id, None)
        self._seed_area.pop(object_id, None)

    def retain_only(self, ids) -> None:
        """Drop any tracked object not in ``ids`` (called when the highlight
        selection changes so we don't track stale objects)."""
        keep = set(int(i) for i in ids)
        for oid in list(self._objs.keys()):
            if oid not in keep:
                self.drop(oid)

    def reseed(
        self,
        object_id: int,
        cam_mask: np.ndarray,
        bbox_cam: Tuple[int, int, int, int],
        depth_m_value: float,
        color: Optional[np.ndarray] = None,
    ) -> None:
        """Re-anchor an object from a fresh SAM result. Resets the depth
        reference, the centroid anchor, and (if a color frame is given)
        re-seeds the CSRT box. Called once per heavy pass per highlighted
        object -- this is the ground truth that bounds drift."""
        ys, xs = np.where(cam_mask > 0)
        if ys.size == 0:
            return
        cx = float(xs.mean())
        cy = float(ys.mean())
        x, y, w, h = bbox_cam
        if w <= 0 or h <= 0:
            w = int(xs.max() - xs.min() + 1)
            h = int(ys.max() - ys.min() + 1)
            x = int(xs.min())
            y = int(ys.min())
        area = int((cam_mask > 0).sum())
        self._seed_area[object_id] = max(1, area)

        csrt = None
        if color is not None:
            csrt = _make_csrt()
            if csrt is not None:
                try:
                    csrt.init(color, (int(x), int(y), int(w), int(h)))
                except Exception:
                    csrt = None

        self._objs[object_id] = _ObjState(
            ref_depth_m=float(depth_m_value),
            anchor_xy=(cx, cy),
            last_xy=(cx, cy),
            bbox_wh=(int(w), int(h)),
            csrt=csrt,
            csrt_ok=csrt is not None,
            last_conf=1.0,
            misses=0,
            frozen=False,
        )

    # ---- per-fast-frame update ----------------------------------------

    def _depth_blob_estimate(
        self, st: _ObjState, object_id: int, depth_m: np.ndarray,
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """Largest near-depth blob centroid inside the search window."""
        if st.ref_depth_m <= 0.1:
            return None, 0.0
        H, W = depth_m.shape
        lx, ly = st.last_xy
        bw, bh = st.bbox_wh
        half_w = max(8, int(bw * self.search_scale * 0.5))
        half_h = max(8, int(bh * self.search_scale * 0.5))
        x0 = max(0, int(lx - half_w)); x1 = min(W, int(lx + half_w))
        y0 = max(0, int(ly - half_h)); y1 = min(H, int(ly + half_h))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None, 0.0
        sub = depth_m[y0:y1, x0:x1]
        band = (np.abs(sub - st.ref_depth_m) < self.depth_band_m) & (sub > 0.1)
        mask = band.astype(np.uint8) * 255
        if mask.sum() == 0:
            return None, 0.0
        n_lab, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        if n_lab <= 1:
            return None, 0.0
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[largest, cv2.CC_STAT_AREA])
        seed_area = self._seed_area.get(object_id, area)
        # Confidence scales with how much of the expected area we recovered,
        # capped at 1.0. A tiny sliver in the window is low confidence.
        conf = float(min(1.0, area / max(1.0, seed_area * self.min_blob_frac)))
        cxl, cyl = centroids[largest]
        return (float(cxl + x0), float(cyl + y0)), conf

    def _csrt_estimate(
        self, st: _ObjState, color: Optional[np.ndarray],
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        if st.csrt is None or color is None or not st.csrt_ok:
            return None, 0.0
        try:
            ok, box = st.csrt.update(color)
        except Exception:
            st.csrt_ok = False
            return None, 0.0
        if not ok:
            st.csrt_ok = False
            return None, 0.0
        x, y, w, h = box
        return (float(x + w / 2.0), float(y + h / 2.0)), 0.7

    def update(
        self,
        object_id: int,
        color: Optional[np.ndarray],
        depth_m: np.ndarray,
    ) -> Optional[FastEstimate]:
        """Estimate the object's current camera centroid. Returns None if the
        object isn't tracked. On low confidence, returns a frozen estimate at
        the last good position (the wash holds instead of chasing garbage)."""
        st = self._objs.get(object_id)
        if st is None:
            return None

        depth_xy, depth_conf = self._depth_blob_estimate(st, object_id, depth_m)
        csrt_xy, csrt_conf = self._csrt_estimate(st, color)

        cx, cy, conf, source = _fuse_estimates(
            depth_xy, depth_conf, csrt_xy, csrt_conf,
            st.last_xy, self.agree_px,
        )

        # Freeze hysteresis: a single low-conf frame holds the last position
        # but keeps trying; sustained low conf reports frozen with 0 conf.
        if conf <= 0.0 or source == "frozen":
            st.misses += 1
            st.frozen = st.misses >= self.freeze_after_misses
            cx, cy = st.last_xy
            out_conf = 0.0 if st.frozen else st.last_conf * 0.5
            return FastEstimate(cx, cy, out_conf, "frozen",
                                self._moved(st, cx, cy))

        st.misses = 0
        st.frozen = False
        st.last_xy = (cx, cy)
        st.last_conf = conf
        return FastEstimate(cx, cy, conf, source, self._moved(st, cx, cy))

    @staticmethod
    def _moved(st: _ObjState, cx: float, cy: float) -> float:
        ax, ay = st.anchor_xy
        return ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5

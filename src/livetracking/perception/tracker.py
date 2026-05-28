"""Object tracker — stable ids and palette colors across frames.

Stage 1 + Stage 2 produce a fresh list of (mask, label) per frame. The tracker
matches each fresh detection to an existing tracked object by mask IoU, so
``DetectedObject.object_id`` stays stable as the scene jitters or the user
moves.

Within a session:
- ids are assigned 1, 2, 3, ... and never reused.
- each id gets a stable palette color (cycled by ``id % len(PALETTE)``).
- names default to the DINO label and persist to ``runtime/object_names.json``
  so user-edited names survive perception_daemon restarts.
- an object whose mask drops out for > ``stale_after_s`` is retired.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from livetracking.paths import HIDDEN_OBJECTS_FILE, OBJECT_NAMES_FILE
from .types import DetectedObject


# 12-color palette — distinct under projector light, all sufficiently saturated.
PALETTE: List[Tuple[int, int, int]] = [
    (255, 80, 80),    # red
    (80, 255, 80),    # green
    (80, 80, 255),    # blue
    (255, 200, 0),    # amber
    (255, 80, 255),   # magenta
    (80, 255, 255),   # cyan
    (255, 130, 0),    # orange
    (160, 80, 255),   # purple
    (0, 255, 160),    # teal
    (255, 255, 80),   # yellow
    (255, 80, 160),   # pink
    (80, 200, 255),   # sky
]


@dataclass
class _Track:
    obj: DetectedObject
    age_frames: int = 0
    misses: int = 0


@dataclass
class _Candidate:
    """A would-be-new track that isn't promoted to a real id until it survives
    ``promote_after_frames`` consecutive frames. Suppresses single-frame
    DINO/SAM hallucinations from churning the id space."""
    last_cam_mask: np.ndarray
    last_centroid: Tuple[float, float]
    last_label: str
    last_label_score: float
    last_depth_m: float
    last_proj_mask: Optional[np.ndarray]
    last_proj_centroid: Optional[Tuple[float, float]]
    consecutive_hits: int = 1
    first_seen_t: float = 0.0
    last_hit_t: float = 0.0


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    ab = (a > 0)
    bb = (b > 0)
    inter = int(np.logical_and(ab, bb).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(ab, bb).sum())
    return inter / union if union > 0 else 0.0


def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


@dataclass
class FreshDetection:
    """What perception delivers per frame, pre-tracking."""
    cam_mask: np.ndarray
    label: str
    label_score: float
    median_depth_m: float
    proj_mask: Optional[np.ndarray] = None
    proj_centroid: Optional[Tuple[float, float]] = None


class ObjectTracker:
    def __init__(
        self,
        iou_match_threshold: float = 0.3,
        stale_after_s: float = 2.0,
        names_path: str = OBJECT_NAMES_FILE,
        hidden_path: str = HIDDEN_OBJECTS_FILE,
        promote_after_frames: int = 3,
        candidate_match_iou: float = 0.2,
        candidate_match_dist_px: float = 50.0,
    ):
        self.iou_match_threshold = iou_match_threshold
        self.stale_after_s = stale_after_s
        self.names_path = names_path
        self.hidden_path = hidden_path
        self.promote_after_frames = promote_after_frames
        self.candidate_match_iou = candidate_match_iou
        self.candidate_match_dist_px = candidate_match_dist_px
        self._tracks: Dict[int, _Track] = {}
        self._candidates: List[_Candidate] = []
        self._next_id: int = 1
        self._names: Dict[str, str] = self._load_names()
        self._hidden_ids: set = self._load_hidden()

    # ---- name persistence ----
    # New schema (v2): JSON dict with two top-level keys.
    #   "by_id":          legacy {id_str: name} mapping kept for back-compat.
    #   "by_fingerprint": [{name, label, cam_xy, depth_m, last_seen}, ...]
    #                     allows renames to survive perception restarts: when
    #                     a new track is promoted we look up its fingerprint
    #                     (label class + camera centroid within FP_XY_TOL_PX
    #                     + depth within FP_DEPTH_TOL_M) and inherit the name.
    # Old schema (v1) ({id_str: name}) is still read and treated as by_id.
    _FP_XY_TOL_PX = 60.0
    _FP_DEPTH_TOL_M = 0.40

    def _load_names(self) -> Dict[str, str]:
        """Load the legacy id -> name map (v1) or by_id sub-dict (v2)."""
        self._fp_renames: List[dict] = []
        if not os.path.exists(self.names_path):
            return {}
        try:
            with open(self.names_path) as f:
                data = json.load(f)
            # v2?
            if isinstance(data, dict) and (
                "by_id" in data or "by_fingerprint" in data
            ):
                self._fp_renames = list(data.get("by_fingerprint") or [])
                return dict(data.get("by_id") or {})
            # v1 (flat id -> name dict)
            if isinstance(data, dict):
                return dict(data)
        except Exception:
            pass
        return {}

    def _save_names(self) -> None:
        """Save both id-keyed and fingerprint-keyed renames (v2 schema)."""
        os.makedirs(os.path.dirname(self.names_path), exist_ok=True)
        payload = {
            "by_id": self._names,
            "by_fingerprint": self._fp_renames,
        }
        with open(self.names_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _lookup_fingerprint_name(
        self, label: str, cam_xy: Tuple[float, float], depth_m: float,
    ) -> Optional[str]:
        """Return a previously-saved user name whose fingerprint matches."""
        best: Optional[dict] = None
        best_score: float = -1.0
        cx, cy = cam_xy
        for fp in self._fp_renames:
            # Require same label class (so a couch rename doesn't poach a
            # pillow at the same coords).
            if (fp.get("label") or "").lower() != (label or "").lower():
                continue
            fcx, fcy = fp.get("cam_xy") or (0, 0)
            dx, dy = cx - float(fcx), cy - float(fcy)
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > self._FP_XY_TOL_PX:
                continue
            fd = float(fp.get("depth_m") or 0.0)
            if depth_m > 0.1 and fd > 0.1 \
                    and abs(depth_m - fd) > self._FP_DEPTH_TOL_M:
                continue
            # Pick the closest match.
            score = -dist
            if score > best_score:
                best_score = score
                best = fp
        return best.get("name") if best else None

    # ---- hidden persistence ----
    def _load_hidden(self) -> set:
        if not os.path.exists(self.hidden_path):
            return set()
        try:
            with open(self.hidden_path) as f:
                return set(int(x) for x in json.load(f))
        except Exception:
            return set()

    def _save_hidden(self) -> None:
        os.makedirs(os.path.dirname(self.hidden_path), exist_ok=True)
        with open(self.hidden_path, "w") as f:
            json.dump(sorted(self._hidden_ids), f, indent=2)

    def hide(self, object_id: int) -> bool:
        """Mark a track hidden — it stays in the matcher (so Stage-1 blobs at
        its location don't keep promoting fresh duplicates) but is excluded
        from active()/visible() and from the projector wash."""
        tr = self._tracks.get(object_id)
        if tr is None:
            return False
        tr.obj.hidden = True
        self._hidden_ids.add(object_id)
        self._save_hidden()
        return True

    def unhide(self, object_id: int) -> bool:
        tr = self._tracks.get(object_id)
        if tr is not None:
            tr.obj.hidden = False
        self._hidden_ids.discard(object_id)
        self._save_hidden()
        return tr is not None

    def hidden_ids(self) -> List[int]:
        return sorted(self._hidden_ids)

    def rename(self, object_id: int, new_name: str) -> bool:
        track = self._tracks.get(object_id)
        if track is None:
            return False
        track.obj.name = new_name
        self._names[str(object_id)] = new_name
        # Also store a fingerprint so the rename survives a perception
        # restart (where object_id will be reassigned).
        obj = track.obj
        # The fingerprint label is the DINO class (object.name once was
        # the DINO label before rename); we use the CURRENT cam centroid
        # + depth and *the label DINO assigned*, captured here at rename
        # time from obj.bbox_cam-derived locality. We don't store
        # obj.name -- that's the new user-supplied name.
        fp = {
            "name": new_name,
            "label": obj.dino_label or obj.name,
            "cam_xy": [float(obj.centroid_cam[0]),
                       float(obj.centroid_cam[1])],
            "depth_m": float(obj.median_depth_m),
            "last_seen": time.time(),
        }
        # Replace any prior fingerprint at this location for the same label.
        self._fp_renames = [
            f for f in self._fp_renames
            if not (
                (f.get("label") or "").lower() == (fp["label"] or "").lower()
                and abs((f.get("cam_xy") or [0, 0])[0] - fp["cam_xy"][0])
                    < self._FP_XY_TOL_PX
                and abs((f.get("cam_xy") or [0, 0])[1] - fp["cam_xy"][1])
                    < self._FP_XY_TOL_PX
            )
        ]
        self._fp_renames.append(fp)
        self._save_names()
        return True

    def cycle_color(self, object_id: int) -> Optional[Tuple[int, int, int]]:
        """Advance the object's color to the next palette entry. Returns the
        new color, or None if no such object."""
        track = self._tracks.get(object_id)
        if track is None:
            return None
        cur = tuple(track.obj.color_rgb)
        try:
            idx = PALETTE.index(cur)
        except ValueError:
            idx = -1
        new = PALETTE[(idx + 1) % len(PALETTE)]
        track.obj.color_rgb = new
        return new

    # ---- per-frame update ----
    def update(self, fresh: List[FreshDetection]) -> List[DetectedObject]:
        now = time.time()
        # Greedy matching: prefer mask IoU, fall back to centroid proximity.
        # The match is INTENTIONALLY loose because the heavy DINO+SAM thread
        # only runs every ~2s, and the same physical object's SAM mask wobbles
        # noticeably between passes — strict IoU would spawn a new candidate
        # every cycle and inflate ids.
        unmatched_track_ids = set(self._tracks.keys())
        matched_fresh = set()
        pairs: List[Tuple[int, int, float]] = []  # (fresh_idx, track_id, score)
        for fi, fd in enumerate(fresh):
            f_cent = _mask_centroid(fd.cam_mask)
            f_bbox = _mask_bbox(fd.cam_mask)
            for tid, tr in self._tracks.items():
                iou = _mask_iou(fd.cam_mask, tr.obj.cam_mask)
                t_cent = tr.obj.centroid_cam
                dx, dy = f_cent[0] - t_cent[0], f_cent[1] - t_cent[1]
                dist = (dx * dx + dy * dy) ** 0.5
                tx, ty, tw, th = tr.obj.bbox_cam
                fx_, fy_, fw, fh = f_bbox
                # Track-bbox-contains-fresh-centroid (or vice versa) is a
                # strong signal even when masks shifted shape between passes.
                fresh_cent_in_track = (tx <= f_cent[0] <= tx + tw
                                       and ty <= f_cent[1] <= ty + th)
                track_cent_in_fresh = (fx_ <= t_cent[0] <= fx_ + fw
                                       and fy_ <= t_cent[1] <= fy_ + fh)
                if iou >= 0.15:
                    score = 2.0 + iou
                elif fresh_cent_in_track or track_cent_in_fresh:
                    score = 1.0 + iou
                elif dist < 60.0:
                    score = 0.6 - dist / 200.0
                else:
                    continue
                pairs.append((fi, tid, score))
        pairs.sort(key=lambda x: -x[2])
        used_tracks = set()
        for fi, tid, iou in pairs:
            if fi in matched_fresh or tid in used_tracks:
                continue
            matched_fresh.add(fi)
            used_tracks.add(tid)
            unmatched_track_ids.discard(tid)
            # update existing track in place
            tr = self._tracks[tid]
            fd = fresh[fi]
            tr.obj.cam_mask = fd.cam_mask
            tr.obj.proj_mask = fd.proj_mask
            tr.obj.centroid_cam = _mask_centroid(fd.cam_mask)
            tr.obj.centroid_proj = fd.proj_centroid
            tr.obj.bbox_cam = _mask_bbox(fd.cam_mask)
            tr.obj.median_depth_m = fd.median_depth_m
            tr.obj.label_score = max(tr.obj.label_score, fd.label_score)
            # Always track the latest DINO label (fingerprint key), even
            # if the user renamed the display name.
            if fd.label:
                tr.obj.dino_label = fd.label
            # Only overwrite label/name with DINO's pick if user hasn't renamed.
            user_named = str(tid) in self._names
            if not user_named and fd.label and fd.label_score > tr.obj.label_score - 0.05:
                tr.obj.name = fd.label
            tr.obj.last_seen_t = now
            tr.age_frames += 1
            tr.misses = 0

        # Create new tracks ONLY for fresh detections that have survived
        # ``promote_after_frames`` consecutive frames as a candidate.
        for fi, fd in enumerate(fresh):
            if fi in matched_fresh:
                continue
            f_cent = _mask_centroid(fd.cam_mask)
            # Match this unmatched fresh to an existing candidate?
            best_cand_idx: Optional[int] = None
            best_cand_score = -1.0
            for ci, cand in enumerate(self._candidates):
                iou = _mask_iou(fd.cam_mask, cand.last_cam_mask)
                dx, dy = f_cent[0] - cand.last_centroid[0], f_cent[1] - cand.last_centroid[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if iou >= self.candidate_match_iou:
                    score = 1.0 + iou
                elif dist < self.candidate_match_dist_px:
                    score = 0.5 - dist / (2 * self.candidate_match_dist_px)
                else:
                    continue
                if score > best_cand_score:
                    best_cand_score = score
                    best_cand_idx = ci
            if best_cand_idx is not None:
                cand = self._candidates[best_cand_idx]
                cand.last_cam_mask = fd.cam_mask
                cand.last_centroid = f_cent
                cand.last_label = fd.label
                cand.last_label_score = fd.label_score
                cand.last_depth_m = fd.median_depth_m
                cand.last_proj_mask = fd.proj_mask
                cand.last_proj_centroid = fd.proj_centroid
                cand.consecutive_hits += 1
                cand.last_hit_t = now
                if cand.consecutive_hits >= self.promote_after_frames:
                    new_id = self._next_id
                    self._next_id += 1
                    color = PALETTE[(new_id - 1) % len(PALETTE)]
                    # Prefer a fingerprint-matched name (persists across
                    # perception restarts where ids get reassigned).
                    fp_name = self._lookup_fingerprint_name(
                        cand.last_label or "",
                        cand.last_centroid,
                        cand.last_depth_m,
                    )
                    saved_name = fp_name or self._names.get(str(new_id))
                    obj = DetectedObject(
                        object_id=new_id,
                        name=saved_name or cand.last_label or f"object {new_id}",
                        color_rgb=color,
                        cam_mask=cand.last_cam_mask,
                        proj_mask=cand.last_proj_mask,
                        centroid_cam=cand.last_centroid,
                        centroid_proj=cand.last_proj_centroid,
                        bbox_cam=_mask_bbox(cand.last_cam_mask),
                        median_depth_m=cand.last_depth_m,
                        last_seen_t=now,
                        label_score=cand.last_label_score,
                        dino_label=cand.last_label or "",
                    )
                    # If we picked up the name from a fingerprint, also
                    # register it under the new id so subsequent renames
                    # / lookups still work the legacy way.
                    if fp_name:
                        self._names[str(new_id)] = fp_name
                    self._tracks[new_id] = _Track(obj=obj, age_frames=cand.consecutive_hits, misses=0)
                    if new_id in self._hidden_ids:
                        obj.hidden = True
                    self._candidates.pop(best_cand_idx)
            else:
                self._candidates.append(_Candidate(
                    last_cam_mask=fd.cam_mask,
                    last_centroid=f_cent,
                    last_label=fd.label,
                    last_label_score=fd.label_score,
                    last_depth_m=fd.median_depth_m,
                    last_proj_mask=fd.proj_mask,
                    last_proj_centroid=fd.proj_centroid,
                    consecutive_hits=1,
                    first_seen_t=now,
                    last_hit_t=now,
                ))

        # Reap candidates that haven't been re-hit recently (1.5 s without a
        # matching detection means it was a flicker, not a real object).
        kept_candidates: List[_Candidate] = []
        for cand in self._candidates:
            if now - cand.last_hit_t > 1.5:
                continue
            kept_candidates.append(cand)
        self._candidates = kept_candidates

        # Retire stale tracks.
        for tid in list(self._tracks.keys()):
            tr = self._tracks[tid]
            if tid in unmatched_track_ids:
                tr.misses += 1
            if now - tr.obj.last_seen_t > self.stale_after_s:
                del self._tracks[tid]

        # Post-merge: collapse pairs of LIVE tracks that point at the same
        # physical object. This happens when DINO skips one of two
        # overlapping detections for a frame, the original track times
        # out, a duplicate track gets promoted at the same position, and
        # the original then re-matches on a later frame. We merge by
        # keeping the LOWER id (older / user-renamed) and discarding the
        # higher one. Trigger: mask IoU >= 0.5 or centroid-inside-bbox
        # both ways. Skips hidden tracks (the user explicitly wants
        # those kept as distinct entries even if they overlap).
        tid_list = [tid for tid, tr in self._tracks.items()
                    if not tr.obj.hidden]
        tid_list.sort()  # so we keep lower ids
        merged_away: set = set()
        for i, a_id in enumerate(tid_list):
            if a_id in merged_away:
                continue
            a = self._tracks[a_id]
            ax, ay, aw, ah = a.obj.bbox_cam
            a_cent = a.obj.centroid_cam
            for b_id in tid_list[i + 1:]:
                if b_id in merged_away:
                    continue
                b = self._tracks[b_id]
                iou = _mask_iou(a.obj.cam_mask, b.obj.cam_mask)
                bx, by, bw, bh = b.obj.bbox_cam
                b_cent = b.obj.centroid_cam
                a_in_b = (bx <= a_cent[0] <= bx + bw
                          and by <= a_cent[1] <= by + bh)
                b_in_a = (ax <= b_cent[0] <= ax + aw
                          and ay <= b_cent[1] <= ay + ah)
                if iou >= 0.5 or (a_in_b and b_in_a):
                    # Merge b -> a: keep a's id/name/color, but if b has
                    # the higher score, adopt b's mask (it's likely the
                    # fresher detection).
                    if b.obj.label_score > a.obj.label_score:
                        a.obj.cam_mask = b.obj.cam_mask
                        a.obj.proj_mask = b.obj.proj_mask
                        a.obj.centroid_cam = b.obj.centroid_cam
                        a.obj.centroid_proj = b.obj.centroid_proj
                        a.obj.bbox_cam = b.obj.bbox_cam
                        a.obj.median_depth_m = b.obj.median_depth_m
                        a.obj.label_score = b.obj.label_score
                    a.obj.last_seen_t = max(a.obj.last_seen_t,
                                            b.obj.last_seen_t)
                    merged_away.add(b_id)
                    del self._tracks[b_id]

        return [tr.obj for tr in self._tracks.values()]

    def active(self) -> List[DetectedObject]:
        """All live tracks, including hidden. Used internally for matching."""
        return [tr.obj for tr in self._tracks.values()]

    def visible(self) -> List[DetectedObject]:
        """Live tracks excluding user-hidden ones. Use for UI/projector."""
        return [tr.obj for tr in self._tracks.values() if not tr.obj.hidden]

    def refresh_track(
        self,
        object_id: int,
        cam_mask: np.ndarray,
        proj_mask: Optional[np.ndarray],
        proj_centroid: Optional[Tuple[float, float]],
        median_depth_m: float,
    ) -> bool:
        """Used by Pipeline.fast_step() — updates geometry-only fields of an
        existing track without rerunning DINO/SAM. Label, color, name stay put.
        """
        tr = self._tracks.get(object_id)
        if tr is None:
            return False
        tr.obj.cam_mask = cam_mask
        tr.obj.proj_mask = proj_mask
        tr.obj.centroid_cam = _mask_centroid(cam_mask)
        tr.obj.centroid_proj = proj_centroid
        tr.obj.bbox_cam = _mask_bbox(cam_mask)
        tr.obj.median_depth_m = median_depth_m
        tr.obj.last_seen_t = time.time()
        tr.age_frames += 1
        tr.misses = 0
        return True

    def promote_provisional(
        self,
        cam_mask: np.ndarray,
        proj_mask: Optional[np.ndarray],
        proj_centroid: Optional[Tuple[float, float]],
        median_depth_m: float,
    ) -> int:
        """Create a new track immediately, no DINO label yet.

        Used by Pipeline.fast_step() when a Stage-1 blob has been stable for
        enough fast frames that we're confident it's a real object. The async
        recognizer thread will attach a proper label on its next pass.

        Returns the new object id.
        """
        now = time.time()
        new_id = self._next_id
        self._next_id += 1
        color = PALETTE[(new_id - 1) % len(PALETTE)]
        saved_name = self._names.get(str(new_id))
        obj = DetectedObject(
            object_id=new_id,
            name=saved_name or f"object {new_id}",
            color_rgb=color,
            cam_mask=cam_mask,
            proj_mask=proj_mask,
            centroid_cam=_mask_centroid(cam_mask),
            centroid_proj=proj_centroid,
            bbox_cam=_mask_bbox(cam_mask),
            median_depth_m=median_depth_m,
            last_seen_t=now,
            label_score=0.0,  # no DINO confidence yet
        )
        self._tracks[new_id] = _Track(obj=obj, age_frames=1, misses=0)
        if new_id in self._hidden_ids:
            obj.hidden = True
        return new_id

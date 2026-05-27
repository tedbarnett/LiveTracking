"""Persistent learned-offset memory keyed by camera position.

When a user nudges a target's projector position (via the web UI), we
record the (cam_xy, projector_dx_dy) tuple to disk. On future convergences
at any cam_xy near a stored entry, the recorded offset is auto-applied
before the user even sees the misalignment.

This is the system's "memory" of parallax / hardware quirks per spatial
region of the camera view.

Storage: JSONL file, one entry per line:
  {"cam_xy": [x, y], "offset": [dx, dy], "timestamp": <unix>}

Lookup: nearest entry by Euclidean cam-distance. If nearest is within
LOOKUP_RADIUS_PX, return its offset. Otherwise no offset.

This module has NO opencv / pygame / RealSense deps so it's unit-testable.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import List, Optional, Sequence, Tuple

LOOKUP_RADIUS_PX = 50.0  # max distance for a stored offset to apply


class NudgeMemory:
    """In-memory representation of the persistent learned-offset log."""

    def __init__(self, path: str):
        self.path = path
        self.entries: List[dict] = []
        self.load()

    def load(self) -> None:
        self.entries = []
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if (isinstance(rec.get("cam_xy"), list) and
                                isinstance(rec.get("offset"), list) and
                                len(rec["cam_xy"]) == 2 and
                                len(rec["offset"]) == 2):
                            self.entries.append({
                                "cam_xy": [float(rec["cam_xy"][0]),
                                             float(rec["cam_xy"][1])],
                                "offset": [float(rec["offset"][0]),
                                             float(rec["offset"][1])],
                                "timestamp": float(rec.get("timestamp", 0)),
                            })
                    except (json.JSONDecodeError, KeyError, TypeError,
                            ValueError):
                        continue
        except OSError:
            pass

    def append(self, cam_xy: Sequence[float], offset: Sequence[float]) -> None:
        """Record a new learned offset. Persists immediately."""
        rec = {
            "cam_xy": [float(cam_xy[0]), float(cam_xy[1])],
            "offset": [float(offset[0]), float(offset[1])],
            "timestamp": time.time(),
        }
        self.entries.append(rec)
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def lookup(
        self,
        cam_xy: Sequence[float],
        radius_px: float = LOOKUP_RADIUS_PX,
    ) -> Optional[List[float]]:
        """Return offset (dx, dy) of the nearest stored entry within radius.

        If multiple entries are inside radius, the most recent one wins
        (so a fresh nudge overrides older stale ones for the same region).
        Returns None if no entries within radius.
        """
        if not self.entries:
            return None
        best = None
        best_dist = float("inf")
        for e in self.entries:
            dx = cam_xy[0] - e["cam_xy"][0]
            dy = cam_xy[1] - e["cam_xy"][1]
            d = math.hypot(dx, dy)
            if d > radius_px:
                continue
            # Within radius: take the most recent (largest timestamp), and
            # within that, the closest.
            if best is None or e["timestamp"] > best["timestamp"] or (
                    e["timestamp"] == best["timestamp"] and d < best_dist):
                best = e
                best_dist = d
        if best is None:
            return None
        return [float(best["offset"][0]), float(best["offset"][1])]

    def clear(self) -> None:
        self.entries = []
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except OSError:
            pass


# Self-tests live in ``tests/test_nudge_memory.py`` (pytest-runnable).

# Self-tests for this module live in tests/ (pytest-runnable).

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


# ----- self-tests -----

if __name__ == "__main__":
    import tempfile, shutil

    tmp = tempfile.mkdtemp(prefix="nudge_mem_test_")
    path = os.path.join(tmp, "test.jsonl")

    try:
        # Test 1: empty
        m = NudgeMemory(path)
        assert m.lookup((100, 100)) is None
        print("Test 1 (empty lookup): PASS")

        # Test 2: append + lookup at same point
        m.append((100, 100), (5, -3))
        result = m.lookup((100, 100))
        assert result == [5.0, -3.0], f"got {result}"
        print(f"Test 2 (append + exact lookup): PASS ({result})")

        # Test 3: lookup within radius
        result = m.lookup((110, 105))  # ~11 px from (100, 100), within 50
        assert result == [5.0, -3.0], f"got {result}"
        print(f"Test 3 (lookup within radius): PASS ({result})")

        # Test 4: lookup outside radius
        result = m.lookup((200, 200))
        assert result is None, f"expected None, got {result}"
        print("Test 4 (lookup outside radius): PASS (None)")

        # Test 5: persistence - reload from disk
        m2 = NudgeMemory(path)
        assert len(m2.entries) == 1
        assert m2.lookup((100, 100)) == [5.0, -3.0]
        print("Test 5 (persistence reload): PASS")

        # Test 6: multiple entries in same region - most recent wins
        time.sleep(0.01)
        m2.append((105, 105), (10, -10))  # 7 px from first, newer
        result = m2.lookup((100, 100))
        assert result == [10.0, -10.0], f"expected newer (10,-10), got {result}"
        print(f"Test 6 (newest wins in region): PASS ({result})")

        # Test 7: entry far from any stored
        result = m2.lookup((500, 500))
        assert result is None
        print("Test 7 (far lookup None): PASS")

        # Test 8: return type is list of float
        result = m2.lookup((100, 100))
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)
        print("Test 8 (return type): PASS")

        # Test 9: clear
        m2.clear()
        assert m2.lookup((100, 100)) is None
        assert not os.path.exists(path)
        print("Test 9 (clear): PASS")

        # Test 10: real-world T1 scenario
        m3 = NudgeMemory(path)
        # Ted nudged T1 (-5, 0) earlier and (135, -280) later when post-it moved
        m3.append((488, 200), (-5, 0))      # T1 original position
        time.sleep(0.01)
        m3.append((483, 266), (135, -280))  # T1 new position after move
        # Future T1 detection at original position - should get (-5, 0)
        r1 = m3.lookup((488, 200))
        assert r1 == [-5.0, 0.0], f"orig T1 expected (-5,0), got {r1}"
        # Future T1 detection at moved position - should get (135, -280)
        r2 = m3.lookup((483, 266))
        assert r2 == [135.0, -280.0], f"moved T1 expected (135,-280), got {r2}"
        # New T2 at unmarked position - should get None
        r3 = m3.lookup((600, 400))
        assert r3 is None
        print(f"Test 10 (real-world T1): PASS (orig={r1} moved={r2} new=None)")

        print("\nAll 10 self-tests passed.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

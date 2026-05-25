"""Simple persistent ID tracker for segmented LiveTracking objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .depth_segmenter import DepthObject


@dataclass
class TrackedObject:
    """A detected object with a persistent ID across frames."""

    track_id: int
    detection: DepthObject
    age_frames: int = 1
    stable_frames: int = 1
    lost_frames: int = 0
    selected: bool = False

    @property
    def centroid(self) -> tuple[float, float]:
        return self.detection.centroid

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.detection.bbox


class ObjectTracker:
    """Track segmented objects by nearest centroid.

    This is intentionally lightweight for the first MVP. It gives stable IDs for
    modest scenes with a few separated props inside the projector rectangle.
    """

    def __init__(self, max_centroid_jump_px: float = 90.0, max_lost_frames: int = 10) -> None:
        self.max_centroid_jump_px = max_centroid_jump_px
        self.max_lost_frames = max_lost_frames
        self._next_track_id = 1
        self.tracks: list[TrackedObject] = []
        self.selected_track_id: Optional[int] = None

    def update(self, detections: list[DepthObject]) -> list[TrackedObject]:
        remaining = list(detections)

        for track in self.tracks:
            best_index = None
            best_distance = self.max_centroid_jump_px
            for index, detection in enumerate(remaining):
                distance = self._distance(track.centroid, detection.centroid)
                if distance < best_distance:
                    best_distance = distance
                    best_index = index

            if best_index is None:
                track.lost_frames += 1
                track.stable_frames = 0
                continue

            track.detection = remaining.pop(best_index)
            track.age_frames += 1
            track.stable_frames += 1
            track.lost_frames = 0

        self.tracks = [track for track in self.tracks if track.lost_frames <= self.max_lost_frames]

        for detection in remaining:
            self.tracks.append(TrackedObject(track_id=self._next_track_id, detection=detection))
            self._next_track_id += 1

        self._sync_selection()
        return self.visible_tracks()

    def visible_tracks(self) -> list[TrackedObject]:
        return [track for track in self.tracks if track.lost_frames == 0]

    def cycle_selection(self, direction: int = 1) -> Optional[TrackedObject]:
        visible = self.visible_tracks()
        if not visible:
            self.selected_track_id = None
            self._sync_selection()
            return None

        ids = [track.track_id for track in visible]
        if self.selected_track_id not in ids:
            self.selected_track_id = ids[0]
        else:
            index = ids.index(self.selected_track_id)
            self.selected_track_id = ids[(index + direction) % len(ids)]
        self._sync_selection()
        return self.selected_track()

    def selected_track(self) -> Optional[TrackedObject]:
        for track in self.tracks:
            if track.track_id == self.selected_track_id and track.lost_frames == 0:
                return track
        return None

    def clear_selection(self) -> None:
        self.selected_track_id = None
        self._sync_selection()

    def _sync_selection(self) -> None:
        live_ids = {track.track_id for track in self.visible_tracks()}
        if self.selected_track_id not in live_ids:
            self.selected_track_id = None
        for track in self.tracks:
            track.selected = track.track_id == self.selected_track_id

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

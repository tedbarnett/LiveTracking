"""Fast depth-first object segmentation.

This module is intentionally independent of projector output. It detects all candidate
foreground objects inside the calibrated projector region in one pass using RealSense
depth, then returns contours/bounding boxes for the UI/effects layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthObject:
    """A single detected foreground object in camera pixel coordinates."""

    label: int
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area_px: int
    mean_depth_mm: float
    mask: np.ndarray


class DepthSegmenter:
    """Segment foreground objects using a captured background depth model.

    Expected inputs are 2D uint16 or float depth frames in millimeters. The optional
    projector ROI mask should be a uint8/bool mask in camera coordinates where nonzero
    pixels are inside the calibrated projector rectangle.
    """

    def __init__(
        self,
        min_depth_delta_mm: int = 60,
        min_depth_mm: int = 250,
        max_depth_mm: int = 4500,
        min_object_area_px: int = 500,
        max_object_area_px: int = 500_000,
        morphology_kernel_px: int = 5,
        blur_kernel_px: int = 5,
    ) -> None:
        self.min_depth_delta_mm = min_depth_delta_mm
        self.min_depth_mm = min_depth_mm
        self.max_depth_mm = max_depth_mm
        self.min_object_area_px = min_object_area_px
        self.max_object_area_px = max_object_area_px
        self.morphology_kernel_px = morphology_kernel_px
        self.blur_kernel_px = blur_kernel_px
        self.background_depth_mm: Optional[np.ndarray] = None
        self.projector_roi_mask: Optional[np.ndarray] = None

    def set_background(self, depth_mm: np.ndarray) -> None:
        """Store a background/wall depth frame for later subtraction."""
        self.background_depth_mm = self._clean_depth(depth_mm)

    def set_projector_roi(self, roi_mask: Optional[np.ndarray]) -> None:
        """Set mask of camera pixels inside the calibrated projector rectangle."""
        if roi_mask is None:
            self.projector_roi_mask = None
            return
        self.projector_roi_mask = roi_mask.astype(bool)

    def segment(self, depth_mm: np.ndarray) -> list[DepthObject]:
        """Return detected foreground objects for the current depth frame."""
        depth = self._clean_depth(depth_mm)
        foreground = self._foreground_mask(depth)
        foreground = self._postprocess_mask(foreground)
        return self._objects_from_mask(foreground, depth)

    def _clean_depth(self, depth_mm: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_mm)
        if depth.ndim != 2:
            raise ValueError(f"Expected 2D depth frame, got shape {depth.shape}")
        depth = depth.astype(np.float32, copy=False)
        invalid = ~np.isfinite(depth) | (depth <= 0)
        depth = depth.copy()
        depth[invalid] = 0
        return depth

    def _foreground_mask(self, depth: np.ndarray) -> np.ndarray:
        valid_depth = (depth >= self.min_depth_mm) & (depth <= self.max_depth_mm)

        if self.background_depth_mm is not None:
            bg = self.background_depth_mm
            if bg.shape != depth.shape:
                raise ValueError(
                    f"Background shape {bg.shape} does not match depth shape {depth.shape}"
                )
            valid_bg = bg > 0
            # Foreground objects are closer to the camera than the stored background.
            foreground = valid_depth & valid_bg & ((bg - depth) >= self.min_depth_delta_mm)
        else:
            # Without a background, keep all valid depth inside the ROI. This is useful
            # for diagnostics but less selective than background subtraction.
            foreground = valid_depth

        if self.projector_roi_mask is not None:
            if self.projector_roi_mask.shape != depth.shape:
                raise ValueError(
                    "Projector ROI mask shape "
                    f"{self.projector_roi_mask.shape} does not match depth shape {depth.shape}"
                )
            foreground &= self.projector_roi_mask

        return foreground.astype(np.uint8) * 255

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.blur_kernel_px and self.blur_kernel_px > 1:
            k = self._odd(self.blur_kernel_px)
            mask = cv2.medianBlur(mask, k)

        if self.morphology_kernel_px and self.morphology_kernel_px > 1:
            k = self._odd(self.morphology_kernel_px)
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask

    def _objects_from_mask(self, mask: np.ndarray, depth: np.ndarray) -> list[DepthObject]:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        objects: list[DepthObject] = []

        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            area = int(area)
            if area < self.min_object_area_px or area > self.max_object_area_px:
                continue

            component_mask = (labels == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            depth_values = depth[labels == label]
            valid_depth_values = depth_values[depth_values > 0]
            mean_depth = float(np.mean(valid_depth_values)) if valid_depth_values.size else 0.0
            cx, cy = centroids[label]

            objects.append(
                DepthObject(
                    label=label,
                    contour=contour,
                    bbox=(int(x), int(y), int(w), int(h)),
                    centroid=(float(cx), float(cy)),
                    area_px=area,
                    mean_depth_mm=mean_depth,
                    mask=component_mask,
                )
            )

        objects.sort(key=lambda obj: obj.area_px, reverse=True)
        return objects

    @staticmethod
    def _odd(value: int) -> int:
        value = max(1, int(value))
        return value if value % 2 == 1 else value + 1

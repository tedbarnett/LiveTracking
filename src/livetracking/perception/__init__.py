"""Perception package — detect objects inside the projector footprint."""
from .types import Blob, DetectedObject
from .footprint import footprint_mask_in_camera, load_homography, save_homography

__all__ = [
    "Blob",
    "DetectedObject",
    "footprint_mask_in_camera",
    "load_homography",
    "save_homography",
]

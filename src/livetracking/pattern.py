"""Cobblestone pattern generator.

8x6 hidden grid of irregular polygon stones. Each stone has a deterministic
position-encoded color used for structured-light decoding (calibration mode)
or a warm earth-tone palette (idle mode). Stones get a subtle drop-shadow
so they read as carved-in, not painted-on.
"""
from __future__ import annotations

import colorsys
import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

GRID_COLS = 8
GRID_ROWS = 6


# Warm earth-tone palette for idle (BGR triples in 0..255).
# Chosen to read as "old cobblestone street at golden hour".
IDLE_PALETTE_BGR: List[Tuple[int, int, int]] = [
    ( 38,  52,  76),   # deep umber
    ( 56,  78, 105),   # warm brown
    ( 74,  98, 128),   # terracotta-shadow
    ( 92, 120, 150),   # ochre
    (110, 140, 170),   # sand
    ( 68,  90, 118),   # russet
    ( 84, 110, 140),   # clay
    (100, 130, 160),   # straw
]


@dataclass
class Stone:
    grid_i: int          # column 0..GRID_COLS-1
    grid_j: int          # row    0..GRID_ROWS-1
    polygon_unit: np.ndarray  # (N,2) float32 in [-0.5, 0.5] cell-local coords
    palette_idx: int     # which idle color


@dataclass
class CobblestoneLayout:
    width: int
    height: int
    stones: List[Stone] = field(default_factory=list)


def _make_irregular_polygon(rng: np.random.Generator, n_sides: int = 7,
                            radius: float = 0.46, jitter: float = 0.10) -> np.ndarray:
    """A roughly-round irregular polygon centered at origin, inscribed in cell."""
    angles = np.linspace(0, 2 * math.pi, n_sides, endpoint=False)
    angles = angles + rng.uniform(-0.15, 0.15, size=n_sides)
    radii = radius + rng.uniform(-jitter, jitter, size=n_sides)
    radii = radii.clip(radius - jitter, radius + jitter)
    xs = np.cos(angles) * radii
    ys = np.sin(angles) * radii
    return np.stack([xs, ys], axis=-1).astype(np.float32)


def build_layout(width: int, height: int, seed: int = 1664) -> CobblestoneLayout:
    """Create a deterministic cobblestone layout for the given canvas size."""
    rng = np.random.default_rng(seed)
    layout = CobblestoneLayout(width=width, height=height)
    for j in range(GRID_ROWS):
        for i in range(GRID_COLS):
            n_sides = int(rng.integers(6, 10))
            poly = _make_irregular_polygon(rng, n_sides=n_sides,
                                           radius=rng.uniform(0.40, 0.50),
                                           jitter=rng.uniform(0.06, 0.14))
            palette_idx = (i * 3 + j * 5 + int(rng.integers(0, len(IDLE_PALETTE_BGR)))) % len(IDLE_PALETTE_BGR)
            layout.stones.append(Stone(grid_i=i, grid_j=j, polygon_unit=poly,
                                       palette_idx=palette_idx))
    return layout


def _cell_center(width: int, height: int, i: int, j: int) -> Tuple[float, float]:
    cw = width / GRID_COLS
    ch = height / GRID_ROWS
    return (i + 0.5) * cw, (j + 0.5) * ch


def _cell_size(width: int, height: int) -> Tuple[float, float]:
    return width / GRID_COLS, height / GRID_ROWS


def _encoded_color_bgr(i: int, j: int) -> Tuple[int, int, int]:
    """Vivid color uniquely encoding the (i, j) cell. HSV-rotated for distinguishability."""
    hue = ((i * 47 + j * 79) % 256) / 256.0
    sat = 0.85 + ((i + j) % 3) * 0.05
    val = 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return int(b * 255), int(g * 255), int(r * 255)


def _polygon_pixels(stone: Stone, width: int, height: int,
                    breathe: float = 0.0) -> np.ndarray:
    cw, ch = _cell_size(width, height)
    cx, cy = _cell_center(width, height, stone.grid_i, stone.grid_j)
    scale = 1.0 + breathe
    pts = stone.polygon_unit.copy()
    pts[:, 0] = cx + pts[:, 0] * cw * scale
    pts[:, 1] = cy + pts[:, 1] * ch * scale
    return pts.astype(np.int32)


def render(layout: CobblestoneLayout, mode: str, t: float,
           dim: float = 1.0) -> np.ndarray:
    """Render the cobblestone pattern as a BGR uint8 image.

    mode: "idle" (warm low-contrast, slow animation)
        | "calibration" (high-contrast position-encoded colors)
        | "selection_bg" (dim warm palette, used as a backdrop)
    dim: scales final brightness in 0..1 (selection_bg uses ~0.25).
    """
    w, h = layout.width, layout.height
    # Mortar background (cool slate). Subtle vignette so edges read softer.
    img = np.full((h, w, 3), (28, 30, 36), dtype=np.uint8)

    # Shadow pass (offset, blurred). Cheap drop shadow.
    shadow = np.zeros((h, w), dtype=np.uint8)
    shadow_offset = (3, 4)
    for stone in layout.stones:
        pts = _polygon_pixels(stone, w, h, breathe=0.0)
        pts_shadow = pts + np.array(shadow_offset, dtype=np.int32)
        cv2.fillPoly(shadow, [pts_shadow], 255)
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=4.0)
    shadow_rgb = (shadow.astype(np.float32) * 0.35)[..., None]
    img = (img.astype(np.float32) - shadow_rgb).clip(0, 255).astype(np.uint8)

    # Stone fills.
    pulse = 0.5 + 0.5 * math.sin(t * 0.6)
    for stone in layout.stones:
        if mode == "calibration":
            base = _encoded_color_bgr(stone.grid_i, stone.grid_j)
            # Encoded colors should stay vivid; tiny breathing keeps decoder happy.
            breathe = 0.0
        elif mode == "selection_bg":
            base = IDLE_PALETTE_BGR[stone.palette_idx]
            base = tuple(int(c * 0.45) for c in base)
            breathe = 0.0
        else:  # idle
            base = IDLE_PALETTE_BGR[stone.palette_idx]
            # Per-stone phase so the field "breathes" rather than blinks in unison.
            phase = (stone.grid_i * 0.7 + stone.grid_j * 1.1)
            local = 0.85 + 0.15 * math.sin(t * 0.5 + phase)
            base = tuple(int(c * local) for c in base)
            breathe = 0.02 * math.sin(t * 0.4 + phase)

        pts = _polygon_pixels(stone, w, h, breathe=breathe)
        cv2.fillPoly(img, [pts], base)
        # Subtle highlight rim (a brighter outline) so stones look 3D.
        cv2.polylines(img, [pts], isClosed=True,
                      color=tuple(min(255, int(c * 1.25)) for c in base),
                      thickness=1, lineType=cv2.LINE_AA)

    if mode == "idle":
        # Slow global pulse for the idle state — very gentle.
        gain = 0.85 + 0.10 * pulse
        img = (img.astype(np.float32) * gain).clip(0, 255).astype(np.uint8)

    if dim != 1.0:
        img = (img.astype(np.float32) * dim).clip(0, 255).astype(np.uint8)
    return img


def render_idle(layout: CobblestoneLayout, t: float) -> np.ndarray:
    return render(layout, "idle", t)


def render_calibration(layout: CobblestoneLayout, t: float) -> np.ndarray:
    return render(layout, "calibration", t)


def render_selection_bg(layout: CobblestoneLayout, t: float) -> np.ndarray:
    return render(layout, "selection_bg", t, dim=1.0)


if __name__ == "__main__":
    # Quick local smoke test: dump three frames.
    L = build_layout(1280, 720)
    for name, mode in [("idle", "idle"), ("calib", "calibration"), ("sel", "selection_bg")]:
        out = render(L, mode, time.time())
        cv2.imwrite(f"_pattern_{name}.png", out)
        print(f"wrote _pattern_{name}.png  shape={out.shape}")

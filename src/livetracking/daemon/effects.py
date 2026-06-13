"""Animated projection effects — procedural textures stenciled through an
object's mask by the projector daemon.

Design constraints (read before editing):

* A projector ADDS light; it cannot project black. So every effect here is
  authored on a black background: dark texels vanish on the wall, bright
  texels light up the object. "Fire on black" is the ideal case — the dark
  gaps between flames simply don't illuminate, so the guitar shows real
  flame shapes rather than a glowing rectangle.

* The projector render loop runs at 60 Hz over a 4K surface. Generating a
  full-frame noise field per object per frame would blow the budget, so
  every effect renders ONLY at the object's projector-space bbox size
  (typically a few hundred px), then the projector composites it through
  the mask alpha. Keep per-frame work O(bbox), never O(screen).

* Effects are pure functions of (width, height, t_seconds). No persistent
  per-object state lives here — the projector owns the clock and the mask.
  That keeps effects hot-swappable and testable offline (see
  scripts/preview_effects.py).

Public API:
    render_effect(name, w, h, t) -> np.ndarray (h, w, 3) uint8 RGB on black
    EFFECTS: tuple of available effect names (excluding the flat "color").
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - cv2 always present in the daemon venv
    cv2 = None


# --- gradient lookup tables -------------------------------------------------

def _build_fire_lut() -> np.ndarray:
    """256-entry black->red->orange->yellow->white ramp (RGB uint8).

    Indexed by heat 0..255. Low heat stays near-black so it disappears on
    the wall; high heat goes white-hot."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    # Control points: (heat, (r, g, b)).
    stops = [
        (0.00, (0, 0, 0)),
        (0.18, (40, 0, 0)),
        (0.38, (160, 16, 0)),
        (0.58, (240, 70, 0)),
        (0.78, (255, 160, 20)),
        (0.92, (255, 230, 110)),
        (1.00, (255, 255, 220)),
    ]
    for i in range(256):
        t = i / 255.0
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t0 <= t <= t1:
                u = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                lut[i] = [int(c0[k] + u * (c1[k] - c0[k])) for k in range(3)]
                break
    return lut


def _build_cloud_lut() -> np.ndarray:
    """Cool blue-white ramp for drifting cloud/smoke light.

    Authored dim so it reads as soft drifting light/shadow rather than a
    solid white wash (projectors can't subtract light, so a 'cloud' is
    really gentle moving highlights)."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    stops = [
        (0.00, (0, 0, 0)),
        (0.45, (10, 14, 24)),
        (0.70, (60, 80, 120)),
        (0.88, (140, 165, 205)),
        (1.00, (220, 232, 255)),
    ]
    for i in range(256):
        t = i / 255.0
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t0 <= t <= t1:
                u = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                lut[i] = [int(c0[k] + u * (c1[k] - c0[k])) for k in range(3)]
                break
    return lut


_FIRE_LUT = _build_fire_lut()
_CLOUD_LUT = _build_cloud_lut()


# --- tileable value-noise field (precomputed once) --------------------------

def _make_tileable_noise(h: int, w: int, seed: int) -> np.ndarray:
    """Multi-octave value noise that wraps vertically and horizontally.

    Built by summing a few octaves of low-res random grids upsampled with
    bilinear interpolation. Vertically tileable so we can scroll it forever
    by rolling rows. Returns float32 in [0, 1]."""
    rng = np.random.default_rng(seed)
    field = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    amp_sum = 0.0
    for octave in (4, 8, 16, 32):
        # Low-res grid with one extra wrap row/col so resize tiles cleanly.
        g = rng.random((octave + 1, octave + 1)).astype(np.float32)
        g[-1, :] = g[0, :]
        g[:, -1] = g[:, 0]
        if cv2 is not None:
            up = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
        else:  # pragma: no cover
            up = np.kron(g[:-1, :-1], np.ones((h // octave + 1, w // octave + 1),
                                              dtype=np.float32))[:h, :w]
        field += amp * up
        amp_sum += amp
        amp *= 0.5
    field /= max(amp_sum, 1e-6)
    field -= field.min()
    field /= max(field.max(), 1e-6)
    return field


# A tall noise canvas we scroll through. 512 rows gives a long loop before
# the pattern visibly repeats; 256 cols is plenty since we resize per object.
_NOISE_H, _NOISE_W = 512, 256

# Cap the internal render resolution (long side, px). Effects are soft noise,
# so generating at <=360px and bilinear-upscaling to the object's bbox is
# visually indistinguishable but keeps cost flat regardless of how close the
# object is to the camera/projector. Tuned so even a full-height object holds
# 60 Hz with several effects active.
_RENDER_CAP_PX = 360
_FLAME_NOISE = _make_tileable_noise(_NOISE_H, _NOISE_W, seed=1664)
_CLOUD_NOISE = _make_tileable_noise(_NOISE_H, _NOISE_W, seed=4661)


def _resize(src: np.ndarray, w: int, h: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
    # Nearest-ish fallback.
    ys = (np.linspace(0, src.shape[0] - 1, h)).astype(int)
    xs = (np.linspace(0, src.shape[1] - 1, w)).astype(int)
    return src[ys][:, xs]


def _flame(w: int, h: int, t: float) -> np.ndarray:
    """Upward-licking fire. Hotter at the bottom, flickering, scrolling up."""
    if w <= 0 or h <= 0:
        return np.zeros((max(h, 1), max(w, 1), 3), dtype=np.uint8)
    # Scroll the noise upward over time (rows roll); flames rise.
    scroll = int((t * 220.0) % _NOISE_H)
    base = np.roll(_FLAME_NOISE, -scroll, axis=0)
    # Add a second, faster, horizontally-shifted layer for turbulence.
    scroll2 = int((t * 360.0) % _NOISE_H)
    shift2 = int((t * 90.0) % _NOISE_W)
    layer2 = np.roll(np.roll(_FLAME_NOISE, -scroll2, axis=0), shift2, axis=1)
    noise = _resize(base, w, h) * 0.65 + _resize(layer2, w, h) * 0.35

    # Vertical heat gradient: 0.0 at the TOP row -> 1.0 at the BOTTOM row.
    # (row 0 is the top in image coords, so the bottom of the silhouette is
    # the white-hot base and the flame tips at the top fade to black and
    # vanish on the wall.) A raised floor keeps the base genuinely hot
    # rather than smoldering.
    grad = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    # Bias the gradient up so the lower half is strongly lit, then clamp.
    grad = np.clip(grad * 1.35 + 0.15, 0.0, 1.0)
    # Gentle global flicker so the whole flame breathes.
    flicker = 0.88 + 0.12 * np.sin(t * 9.0)
    # Combine: gradient dominates (drives the hot base), noise carves the
    # licking flame shapes on top of it.
    heat = ((0.55 * grad + 0.45 * noise) * grad * flicker)
    # Light contrast lift — drop only the deep shadows toward black so the
    # flame keeps its shape, but let the base ramp all the way to white-hot.
    heat = np.clip((heat - 0.08) / 0.92, 0.0, 1.0) ** 0.85
    idx = (heat * 255.0).astype(np.uint8)
    return _FIRE_LUT[idx]


def _cloud(w: int, h: int, t: float) -> np.ndarray:
    """Slow drifting cloud/smoke — gentle moving highlights, no hard edges."""
    if w <= 0 or h <= 0:
        return np.zeros((max(h, 1), max(w, 1), 3), dtype=np.uint8)
    # Slow diagonal drift.
    sy = int((t * 26.0) % _NOISE_H)
    sx = int((t * 40.0) % _NOISE_W)
    base = np.roll(np.roll(_CLOUD_NOISE, -sy, axis=0), sx, axis=1)
    sy2 = int((t * 14.0) % _NOISE_H)
    layer2 = np.roll(_CLOUD_NOISE, -sy2, axis=1 if False else 0)
    noise = _resize(base, w, h) * 0.6 + _resize(layer2, w, h) * 0.4
    # Soft S-curve so it reads as billows, not a flat fog.
    soft = np.clip((noise - 0.30) / 0.70, 0.0, 1.0)
    soft = soft * soft * (3 - 2 * soft)  # smoothstep
    idx = (soft * 255.0).astype(np.uint8)
    return _CLOUD_LUT[idx]


_RENDERERS = {
    "flame": _flame,
    "cloud": _cloud,
}

# Effect names selectable in the UI (the flat single-color mode is "color"
# and is handled directly by the projector, not here).
EFFECTS = tuple(_RENDERERS.keys())


def is_effect(name: str) -> bool:
    return name in _RENDERERS


def render_effect(name: str, w: int, h: int, t: float) -> np.ndarray:
    """Render `name` at (w, h) for animation time `t` seconds.

    Returns an (h, w, 3) uint8 RGB image on a black background. Unknown
    names render as black (caller should fall back to flat color).

    Perf: procedural noise is soft, so we render at a capped internal
    resolution (long side <= _RENDER_CAP_PX) and upscale the finished RGB
    once with a single bilinear resize. This decouples per-frame cost from
    object size — a close-up guitar that fills 600x1040 px on the wall would
    otherwise cost ~40 ms/frame (noise is generated per output pixel);
    capped + upscaled it stays in the few-ms range and holds 60 Hz even
    with several objects lit at once."""
    fn = _RENDERERS.get(name)
    if fn is None:
        return np.zeros((max(h, 1), max(w, 1), 3), dtype=np.uint8)
    w, h = int(w), int(h)
    if w <= 0 or h <= 0:
        return np.zeros((max(h, 1), max(w, 1), 3), dtype=np.uint8)
    long_side = max(w, h)
    if long_side > _RENDER_CAP_PX:
        scale = _RENDER_CAP_PX / float(long_side)
        rw, rh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        small = fn(rw, rh, float(t))
        return _resize(small, w, h)
    return fn(w, h, float(t))

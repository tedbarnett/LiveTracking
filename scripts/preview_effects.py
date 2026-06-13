"""Offline preview for daemon/effects.py — renders flame & cloud effects
stenciled through a guitar-ish mask to PNG strips, so the animated textures
can be eyeballed (and vision-verified) without the projector running.

Usage:
    .venv/Scripts/python.exe scripts/preview_effects.py [out_dir]

Writes <out>/effect_<name>.png : a horizontal strip of frames over time,
each composited on black exactly as the projector will (mask = alpha).
"""
from __future__ import annotations

import os
import sys

import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover
    print("cv2 required:", e)
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from livetracking.daemon import effects  # noqa: E402


def _guitar_mask(w: int, h: int) -> np.ndarray:
    """A rough guitar silhouette as a {0,255} mask for realistic stenciling."""
    m = np.zeros((h, w), dtype=np.uint8)
    # Body: two stacked ellipses (lower bout bigger).
    cv2.ellipse(m, (w // 2, int(h * 0.70)), (int(w * 0.26), int(h * 0.22)),
                0, 0, 360, 255, -1)
    cv2.ellipse(m, (w // 2, int(h * 0.52)), (int(w * 0.19), int(h * 0.16)),
                0, 0, 360, 255, -1)
    # Neck.
    cv2.rectangle(m, (int(w * 0.46), int(h * 0.10)),
                  (int(w * 0.54), int(h * 0.55)), 255, -1)
    # Headstock.
    cv2.rectangle(m, (int(w * 0.43), int(h * 0.05)),
                  (int(w * 0.57), int(h * 0.13)), 255, -1)
    return m


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "runtime/effect_preview"
    os.makedirs(out, exist_ok=True)

    W, H = 220, 380          # one object's bbox, guitar-ish aspect
    n_frames = 6
    dt = 0.12                # seconds between sampled frames
    mask = _guitar_mask(W, H)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]

    for name in effects.EFFECTS:
        cells = []
        for i in range(n_frames):
            t = i * dt
            tex = effects.render_effect(name, W, H, t)          # RGB on black
            comp = (tex.astype(np.float32) * alpha).astype(np.uint8)
            # Convert RGB->BGR for cv2.imwrite.
            cells.append(comp[:, :, ::-1])
            # 4px black gutter between frames.
            cells.append(np.zeros((H, 4, 3), dtype=np.uint8))
        strip = np.concatenate(cells[:-1], axis=1)
        path = os.path.join(out, f"effect_{name}.png")
        cv2.imwrite(path, strip)
        print(f"wrote {path}  ({strip.shape[1]}x{strip.shape[0]}, "
              f"{n_frames} frames @ dt={dt}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

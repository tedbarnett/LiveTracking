"""Manual two-plane parallax calibration via projected-image alignment.

Why: a single homography H maps the camera plane to ONE projector-plane
target depth (the back wall, in our case). Objects nearer than the wall
suffer parallax error because the camera's ray to the object hits the
wall at a different point than the projector's ray, so the wall-target H
puts the projection on the wall-shadow instead of the object.

The clean fix is a SECOND homography at a known nearer depth. At runtime
we interpolate per-object based on its median depth z_obj:

    alpha = (1/z_obj - 1/z_wall) / (1/z_near - 1/z_wall)
    H(z)  = (1 - alpha) * H_wall + alpha * H_near

(Inverse-depth lerp because parallax disparity is linear in 1/z.)

This script captures both homographies via manual operator alignment:

  Pass 1 (WALL):
    - Project the live RealSense color frame, full-screen on the
      projector, modulated through the *current* H_wall (so it starts
      already roughly aligned). The operator sees the projected camera
      image and the real wall side by side.
    - Operator nudges translate / scale / rotate of the projected image
      until features in the projected image LAND ON the matching real
      features (poster edges, etc.).
    - On Enter, we apply the nudge as a similarity transform S on top of
      the existing H: H_wall <- S * H. Saved to runtime/calibration/.

  Pass 2 (NEAR):
    - Same UI. Operator stands a flat target (bodhran on a stand) at
      typical instrument depth, faces it toward the camera.
    - Operator aligns the projected image to the bodhran.
    - On Enter, we save H_near and also the median depth at the camera-
      space centroid of the bodhran (read from the RealSense depth
      stream) as z_near.

Keyboard during alignment:
  Arrow keys  : translate the projected image by 5 px (Shift = 50 px)
  + / -       : scale up / down by 1% (Shift = 5%)
  [ / ]       : rotate -/+ 0.3 deg (Shift = 3 deg)
  R           : reset nudge to identity
  N           : skip to next pass (or quit on last pass)
  Enter / Space : commit current pass and continue
  Esc         : abort without writing anything
  H           : toggle the operator help overlay (default ON)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from livetracking.paths import (  # noqa: E402
    CALIB_DIR, DISPLAY_INDEX, describe,
)
from livetracking.perception.capture import RealSenseCapture  # noqa: E402
from livetracking.perception.footprint import load_homography  # noqa: E402


# ---- output paths --------------------------------------------------------
H_WALL_FILE = os.path.join(CALIB_DIR, "H_wall.npy")
H_NEAR_FILE = os.path.join(CALIB_DIR, "H_near.npy")
DEPTHS_FILE = os.path.join(CALIB_DIR, "parallax_depths.json")


# ---- pygame plumbing -----------------------------------------------------
def pick_projector_display(pygame):
    pygame.display.init()
    sizes = pygame.display.get_desktop_sizes()
    if not sizes:
        raise RuntimeError("pygame found 0 displays.")
    if DISPLAY_INDEX is not None and 0 <= DISPLAY_INDEX < len(sizes):
        idx = DISPLAY_INDEX
    else:
        idx = max(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1])
    return idx, sizes[idx]


# ---- alignment state ----------------------------------------------------
@dataclass
class Nudge:
    """Similarity transform applied in projector pixel space (post-H).

    tx, ty in projector pixels, scale unitless (1.0 = no change),
    angle in degrees.
    """
    tx: float = 0.0
    ty: float = 0.0
    scale: float = 1.0
    angle: float = 0.0
    pivot: tuple[float, float] = (0.0, 0.0)  # set to projector center

    def reset(self):
        self.tx = 0.0
        self.ty = 0.0
        self.scale = 1.0
        self.angle = 0.0

    def to_matrix(self) -> np.ndarray:
        """Return the 3x3 matrix S such that p' = S @ [p, 1].T."""
        cx, cy = self.pivot
        # Rotate+scale about pivot, then translate.
        a = math.radians(self.angle)
        c, s = math.cos(a), math.sin(a)
        k = self.scale
        # Translate pivot to origin -> scale+rotate -> translate back -> tx,ty.
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
        RS = np.array([[k * c, -k * s, 0],
                       [k * s,  k * c, 0],
                       [0,         0,  1]], dtype=np.float64)
        T2 = np.array([[1, 0, cx + self.tx],
                       [0, 1, cy + self.ty],
                       [0, 0,           1]], dtype=np.float64)
        return T2 @ RS @ T1


def composed_H(H: np.ndarray, nudge: Nudge) -> np.ndarray:
    """Return S(nudge) @ H."""
    return nudge.to_matrix() @ H


# ---- depth median at mask centroid --------------------------------------
def median_depth_in_centerbox(depth_m: np.ndarray, half: int = 60) -> float:
    """Return median depth in a 2*half x 2*half window at the camera center
    (where the operator was instructed to hold the NEAR target).
    Ignores zero/invalid pixels.
    """
    h, w = depth_m.shape[:2]
    cx, cy = w // 2, h // 2
    win = depth_m[max(0, cy - half):cy + half,
                  max(0, cx - half):cx + half]
    vals = win[win > 0]
    if vals.size < 50:
        return 0.0
    return float(np.median(vals))


# ---- single-pass interactive alignment ----------------------------------
def run_alignment_pass(
    pygame, screen, font, font_small,
    cap: RealSenseCapture,
    PW: int, PH: int,
    starting_H: np.ndarray,
    pass_label: str,
    instructions: list[str],
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Returns (final_H, last_depth_m) on commit, None on skip/quit."""
    nudge = Nudge(pivot=(PW / 2.0, PH / 2.0))
    show_help = True
    last_depth_m = np.zeros((480, 848), dtype=np.float32)

    clock = pygame.time.Clock()
    while True:
        # Pump events.
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return None
            if ev.type != pygame.KEYDOWN:
                continue
            mods = pygame.key.get_mods()
            shift = bool(mods & pygame.KMOD_SHIFT)
            step = 50 if shift else 5
            sstep = 0.05 if shift else 0.01
            rstep = 3.0 if shift else 0.3
            if ev.key == pygame.K_ESCAPE:
                return None
            if ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                final_H = composed_H(starting_H, nudge)
                return final_H, last_depth_m
            if ev.key == pygame.K_n:
                return None
            if ev.key == pygame.K_h:
                show_help = not show_help
            if ev.key == pygame.K_r:
                nudge.reset()
            if ev.key == pygame.K_LEFT:
                nudge.tx -= step
            if ev.key == pygame.K_RIGHT:
                nudge.tx += step
            if ev.key == pygame.K_UP:
                nudge.ty -= step
            if ev.key == pygame.K_DOWN:
                nudge.ty += step
            if ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
                nudge.scale *= (1.0 + sstep)
            if ev.key == pygame.K_MINUS:
                nudge.scale *= (1.0 - sstep)
            if ev.key == pygame.K_LEFTBRACKET:
                nudge.angle -= rstep
            if ev.key == pygame.K_RIGHTBRACKET:
                nudge.angle += rstep

        # Grab a fresh camera frame.
        frame = cap.read()
        last_depth_m = frame.depth_m

        # Warp camera color through composed H -> projector.
        Hc = composed_H(starting_H, nudge)
        proj_img = cv2.warpPerspective(
            frame.color, Hc, (PW, PH),
            flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
        )
        # pygame expects RGB.
        proj_rgb = cv2.cvtColor(proj_img, cv2.COLOR_BGR2RGB)
        # Surface from array (swap axes to get (W, H, 3) for pygame).
        surf = pygame.surfarray.make_surface(np.transpose(proj_rgb, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        if show_help:
            _draw_overlay(
                screen, font, font_small,
                pass_label, instructions, nudge,
                center_depth=median_depth_in_centerbox(frame.depth_m),
            )

        pygame.display.flip()
        clock.tick(20)


def _draw_overlay(screen, font, font_small, pass_label, instructions,
                  nudge: Nudge, center_depth: float):
    import pygame  # local import: this is only called from inside main()
    pad = 30
    lines = [
        f"PARALLAX CALIBRATION — {pass_label}",
        "",
    ]
    lines.extend(instructions)
    lines.extend([
        "",
        "Arrows: translate (Shift = x10)   + / - : scale (Shift = x5)",
        "[ / ]: rotate (Shift = x10)       R: reset nudge",
        "Enter/Space: COMMIT this pass     N: skip/next     Esc: ABORT",
        "H: toggle help",
        "",
        (f"nudge: tx={nudge.tx:+.0f}px ty={nudge.ty:+.0f}px "
         f"scale={nudge.scale:.3f} angle={nudge.angle:+.2f}deg"),
        f"depth at camera center: {center_depth:.2f} m",
    ])
    # Translucent backing for readability.
    h = len(lines) * 42 + pad * 2
    bg = pygame.Surface((screen.get_width() - 2 * pad, h))
    bg.set_alpha(170)
    bg.fill((0, 0, 0))
    screen.blit(bg, (pad, pad))
    y = pad + 10
    for i, line in enumerate(lines):
        f = font if i == 0 else font_small
        col = (255, 220, 0) if i == 0 else (255, 255, 255)
        surf = f.render(line, True, col)
        screen.blit(surf, (pad + 16, y))
        y += 42


# ---- main ----------------------------------------------------------------
def main() -> int:
    print(f"[parallax_calib] {describe()}")
    os.makedirs(CALIB_DIR, exist_ok=True)

    # Load the existing wall-target homography as the starting point. Without
    # it we have no idea what scale/orientation to start at; the operator
    # would have to hunt blindly in a 4K canvas.
    H_start, meta = load_homography()
    PW = int(meta.get("proj_w", 3840))
    PH = int(meta.get("proj_h", 2160))
    print(f"[parallax_calib] starting H loaded, proj={PW}x{PH}")

    # Open the RealSense BEFORE pygame so we fail fast if the device is busy.
    cap = RealSenseCapture()
    cw, ch = cap.size()
    print(f"[parallax_calib] camera={cw}x{ch}")

    # Pygame fullscreen on projector display.
    import pygame
    pygame.init()
    idx, (PW2, PH2) = pick_projector_display(pygame)
    if (PW2, PH2) != (PW, PH):
        # Trust pygame's reported size; warpPerspective writes to PW2 x PH2.
        print(f"[parallax_calib] WARN: meta says {PW}x{PH}, pygame says "
              f"{PW2}x{PH2}; using pygame value")
        PW, PH = PW2, PH2
    screen = pygame.display.set_mode((PW, PH), pygame.NOFRAME, display=idx)
    pygame.mouse.set_visible(False)
    font = pygame.font.SysFont(None, 56)
    font_small = pygame.font.SysFont(None, 36)

    try:
        # ---- PASS 1: WALL ----
        wall_instr = [
            "Goal: align the projected CAMERA image to the BACK WALL.",
            "The projected colors should land on the matching real features",
            "(poster edges, map borders, etc) on the wall behind the scene.",
        ]
        res1 = run_alignment_pass(
            pygame, screen, font, font_small, cap, PW, PH,
            H_start, "PASS 1 / 2  —  WALL", wall_instr,
        )
        if res1 is None:
            print("[parallax_calib] WALL pass aborted; nothing saved.")
            return 1
        H_wall, _depth_wall = res1
        np.save(H_WALL_FILE, H_wall)
        print(f"[parallax_calib] saved {H_WALL_FILE}")

        # ---- PASS 2: NEAR ----
        near_instr = [
            "Goal: align the projected CAMERA image to a FLAT TARGET",
            "held at INSTRUMENT depth (e.g. the BODHRAN on a stand,",
            "or a poster held up at ~1.5-2.5 m). Centre the target in",
            "the camera view so the depth readout below reflects it.",
        ]
        res2 = run_alignment_pass(
            pygame, screen, font, font_small, cap, PW, PH,
            H_wall, "PASS 2 / 2  —  NEAR  (bodhran)", near_instr,
        )
        if res2 is None:
            print("[parallax_calib] NEAR pass aborted; only H_wall saved.")
            return 2
        H_near, depth_near_arr = res2
        np.save(H_NEAR_FILE, H_near)
        z_near = median_depth_in_centerbox(depth_near_arr)
        # Also pull z_wall: prefer the existing wall_plane.npy if present,
        # else fall back to a frame-average over a back-of-room patch.
        z_wall = _estimate_z_wall()
        if z_near <= 0.0 or z_near >= 10.0:
            print(f"[parallax_calib] WARN: implausible z_near={z_near:.2f} m; "
                  "using fallback 2.0 m")
            z_near = 2.0
        with open(DEPTHS_FILE, "w") as f:
            json.dump({"z_near_m": z_near, "z_wall_m": z_wall,
                       "captured_at": time.time()},
                      f, indent=2)
        print(f"[parallax_calib] saved {H_NEAR_FILE}")
        print(f"[parallax_calib] saved {DEPTHS_FILE} "
              f"(z_near={z_near:.2f} m, z_wall={z_wall:.2f} m)")
        return 0
    finally:
        try:
            pygame.quit()
        except Exception:
            pass
        try:
            cap.close()
        except Exception:
            pass


def _estimate_z_wall() -> float:
    """Best-effort wall depth: load the calibrated wall plane if present and
    sample it at the projector-footprint center; else 3.5 m."""
    wp_path = os.path.join(CALIB_DIR, "wall_plane.npy")
    if os.path.exists(wp_path):
        try:
            wp = np.load(wp_path)
            if wp.size == 4:
                # Plane: aX + bY + cZ + d = 0. At image center (u,v) we want
                # z such that a*x_ray*z + b*y_ray*z + c*z + d = 0.
                # Use D455 intrinsics from PipelineConfig defaults.
                fx, fy, cx, cy = 615.0, 615.0, 424.0, 240.0
                u, v = cx, cy
                a, b, c, d = wp.tolist()
                denom = a * (u - cx) / fx + b * (v - cy) / fy + c
                if abs(denom) > 1e-6:
                    z = -d / denom
                    if 0.5 < z < 10.0:
                        return float(z)
        except Exception:
            pass
    return 3.5


if __name__ == "__main__":
    sys.exit(main())

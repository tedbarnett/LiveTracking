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
from livetracking.perception.depth_probe import (  # noqa: E402
    depth_at_point,
    median_depth_in_centerbox,
    nearest_depth_in_centerbox,
)
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


# ---- depth probes are imported from livetracking.perception.depth_probe -
# (median_depth_in_centerbox, nearest_depth_in_centerbox) -- see imports above.


# ---- single-pass interactive alignment ----------------------------------
def run_alignment_pass(
    pygame, screen, font, font_small,
    cap: RealSenseCapture,
    PW: int, PH: int,
    starting_H: np.ndarray,
    pass_label: str,
    instructions: list[str],
    allow_click_probe: bool = False,
) -> Optional[tuple[np.ndarray, np.ndarray, Optional[tuple[int, int]]]]:
    """Returns (final_H, last_depth_m, probed_xy_cam) on commit, None on skip/quit.

    When ``allow_click_probe`` is True (NEAR pass), a left-click on the
    projector screen drops a depth probe at the corresponding CAMERA pixel
    — found by inverse-mapping the click through the current composed H.
    Useful when the NEAR target is off-center (off-axis bodhran on couch).

    ``probed_xy_cam`` is (x, y) in camera coords or None if no click was
    placed. The caller decides whether to use it (NEAR pass does;
    WALL pass ignores it).
    """
    nudge = Nudge(pivot=(PW / 2.0, PH / 2.0))
    show_help = True
    last_depth_m = np.zeros((480, 848), dtype=np.float32)
    probed_xy_cam: Optional[tuple[int, int]] = None
    probed_xy_proj: Optional[tuple[int, int]] = None
    # Brightness multiplier applied to the projected camera image so the
    # operator can SEE the real bodhran/wall through the projection.
    # 0.7 default (-30%). B/V keys adjust live.
    brightness = float(os.environ.get("LIVETRACKING_CALIB_BRIGHTNESS", "0.7"))
    # Make the mouse cursor visible during the NEAR pass so the operator
    # knows where they're clicking. Hidden by main() for the WALL pass.
    if allow_click_probe:
        pygame.mouse.set_visible(True)

    clock = pygame.time.Clock()
    while True:
        # Pump events.
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return None
            if (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                    and allow_click_probe):
                # Inverse-map click from projector coords -> camera coords.
                px, py = ev.pos
                Hc = composed_H(starting_H, nudge)
                try:
                    Hc_inv = np.linalg.inv(Hc)
                except np.linalg.LinAlgError:
                    continue
                vec = Hc_inv @ np.array([px, py, 1.0])
                if abs(vec[2]) < 1e-9:
                    continue
                cx = int(vec[0] / vec[2])
                cy = int(vec[1] / vec[2])
                # Reject clicks that land outside the camera frame.
                if 0 <= cx < last_depth_m.shape[1] and 0 <= cy < last_depth_m.shape[0]:
                    probed_xy_cam = (cx, cy)
                    probed_xy_proj = (px, py)
                continue
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
                return final_H, last_depth_m, probed_xy_cam
            if ev.key == pygame.K_n:
                return None
            if ev.key == pygame.K_h:
                show_help = not show_help
            if ev.key == pygame.K_r:
                nudge.reset()
            if ev.key == pygame.K_c and allow_click_probe:
                # Clear the click probe; fall back to centerbox sampling.
                probed_xy_cam = None
                probed_xy_proj = None
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
            if ev.key == pygame.K_b:
                brightness = min(1.0, brightness + 0.05)
            if ev.key == pygame.K_v:
                brightness = max(0.05, brightness - 0.05)

        # Grab a fresh camera frame.
        frame = cap.read()
        last_depth_m = frame.depth_m

        # Warp camera color through composed H -> projector.
        Hc = composed_H(starting_H, nudge)
        proj_img = cv2.warpPerspective(
            frame.color, Hc, (PW, PH),
            flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
        )
        # Apply brightness multiplier so the real scene is visible behind
        # the projection during alignment.
        if brightness < 0.999:
            proj_img = cv2.convertScaleAbs(proj_img, alpha=brightness, beta=0)
        # pygame expects RGB.
        proj_rgb = cv2.cvtColor(proj_img, cv2.COLOR_BGR2RGB)
        # Surface from array (swap axes to get (W, H, 3) for pygame).
        surf = pygame.surfarray.make_surface(np.transpose(proj_rgb, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        # If a click probe is placed, draw a crosshair at the projector-
        # space location so the operator can confirm it followed the
        # target after a nudge.
        if probed_xy_proj is not None:
            pygame.draw.circle(screen, (0, 255, 0), probed_xy_proj, 20, 3)
            pygame.draw.line(screen, (0, 255, 0),
                              (probed_xy_proj[0] - 30, probed_xy_proj[1]),
                              (probed_xy_proj[0] + 30, probed_xy_proj[1]), 2)
            pygame.draw.line(screen, (0, 255, 0),
                              (probed_xy_proj[0], probed_xy_proj[1] - 30),
                              (probed_xy_proj[0], probed_xy_proj[1] + 30), 2)

        if show_help:
            # Probed depth: centerbox by default; click-probed if available.
            if probed_xy_cam is not None:
                probed_depth = depth_at_point(frame.depth_m, *probed_xy_cam,
                                               half=20)
            else:
                probed_depth = median_depth_in_centerbox(frame.depth_m)
            _draw_overlay(
                screen, font, font_small,
                pass_label, instructions, nudge,
                center_depth=probed_depth,
                brightness=brightness,
                probed_xy_cam=probed_xy_cam,
                allow_click_probe=allow_click_probe,
            )

        pygame.display.flip()
        clock.tick(20)


# ---- depth probes are imported from livetracking.perception.depth_probe -
# (median_depth_in_centerbox, nearest_depth_in_centerbox, depth_at_point)


def _draw_overlay(screen, font, font_small, pass_label, instructions,
                  nudge: Nudge, center_depth: float, brightness: float = 1.0,
                  probed_xy_cam: Optional[tuple[int, int]] = None,
                  allow_click_probe: bool = False):
    import pygame  # local import: this is only called from inside main()
    pad = 30
    lines = [
        f"PARALLAX CALIBRATION — {pass_label}",
        "",
    ]
    lines.extend(instructions)
    extras = [
        "",
        "Arrows: translate (Shift = x10)   + / - : scale (Shift = x5)",
        "[ / ]: rotate (Shift = x10)       R: reset nudge",
        "B / V: brightness up / down       H: toggle help",
    ]
    if allow_click_probe:
        extras.append(
            "CLICK on the NEAR target to set probe   C: clear probe"
        )
    extras.append(
        "Enter/Space: COMMIT this pass     N: skip/next     Esc: ABORT"
    )
    if probed_xy_cam is not None:
        probe_str = (f"PROBED at cam ({probed_xy_cam[0]}, "
                     f"{probed_xy_cam[1]})   depth: {center_depth:.2f} m")
    else:
        probe_str = (f"depth at camera center: {center_depth:.2f} m    "
                     f"brightness: {brightness:.0%}")
    extras.extend([
        "",
        (f"nudge: tx={nudge.tx:+.0f}px ty={nudge.ty:+.0f}px "
         f"scale={nudge.scale:.3f} angle={nudge.angle:+.2f}deg"),
        probe_str,
    ])
    lines.extend(extras)
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


def _interstitial_prompt(pygame, screen, font, font_small,
                          title: str, lines: list[str], cap) -> bool:
    """Big banner between passes. Shows live depth so the operator can verify
    they've actually aimed the camera at the NEAR target before pressing Enter.
    Returns True on Enter, False on Esc/quit."""
    clock = pygame.time.Clock()
    PW, PH = screen.get_width(), screen.get_height()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return False
                if ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return True
        frame = cap.read()
        depth_median = median_depth_in_centerbox(frame.depth_m)
        depth_near = nearest_depth_in_centerbox(frame.depth_m, pct=10.0)
        # Dim background.
        screen.fill((10, 10, 20))
        # Title.
        title_surf = font.render(title, True, (255, 220, 0))
        screen.blit(title_surf, ((PW - title_surf.get_width()) // 2, 120))
        # Body lines.
        y = 240
        for ln in lines:
            s = font_small.render(ln, True, (255, 255, 255))
            screen.blit(s, ((PW - s.get_width()) // 2, y))
            y += 56
        # Live depth reads so operator sees they've aimed the camera right.
        # The "nearest" probe is what we'll actually save as z_near, so it
        # drives the traffic light. The median is shown so the operator can
        # diagnose a noisy/empty foreground (median >> nearest = foreground
        # is small in the box; near both = uniform target).
        dcol = (120, 255, 120) if 0.5 < depth_near < 2.5 else (255, 120, 120)
        dsurf = font.render(
            f"camera-center depth — nearest: {depth_near:.2f} m   "
            f"median: {depth_median:.2f} m",
            True, dcol,
        )
        screen.blit(dsurf, ((PW - dsurf.get_width()) // 2, y + 40))
        pygame.display.flip()
        clock.tick(15)


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
        H_wall, _depth_wall, _probe_wall = res1
        np.save(H_WALL_FILE, H_wall)
        print(f"[parallax_calib] saved {H_WALL_FILE}")

        # ---- INTERSTITIAL BANNER between passes ----
        if not _interstitial_prompt(
            pygame, screen, font, font_small,
            "PASS 1 / 2 COMPLETE  —  WALL saved",
            [
                "Next: NEAR pass.",
                "1. Place the BODHRAN on the couch (or held at ~1-2 m).",
                "2. CLICK directly on the bodhran in the projected image",
                "   to set a depth probe at that exact pixel (best for",
                "   off-center targets). Or center the camera on it and",
                "   skip the click (centerbox fallback).",
                "3. Depth readout should drop to ~1-2 m, NOT ~3 m.",
                "",
                "Press ENTER when ready to start PASS 2.   Esc to abort.",
            ],
            cap,
        ):
            print("[parallax_calib] aborted between passes; only H_wall saved.")
            return 2

        # ---- PASS 2: NEAR ----
        near_instr = [
            "Goal: align the projected CAMERA image to a FLAT TARGET",
            "at INSTRUMENT depth (BODHRAN on couch, or any object",
            "1-2 m from camera). LEFT-CLICK on the target in the",
            "projected image to set the depth probe — works even when",
            "the target is OFF-CENTER. C clears the probe.",
        ]
        res2 = run_alignment_pass(
            pygame, screen, font, font_small, cap, PW, PH,
            H_wall, "PASS 2 / 2  —  NEAR  (bodhran)", near_instr,
            allow_click_probe=True,
        )
        if res2 is None:
            print("[parallax_calib] NEAR pass aborted; only H_wall saved.")
            return 2
        H_near, depth_near_arr, probe_xy = res2
        np.save(H_NEAR_FILE, H_near)
        # z_near sampling: click-probe wins (operator explicitly aimed),
        # else fall back to nearest-decile centerbox (resilient to small
        # foreground targets but assumes camera is aimed roughly right).
        if probe_xy is not None:
            z_near = depth_at_point(depth_near_arr, *probe_xy, half=20)
            print(f"[parallax_calib] z_near from click probe at "
                  f"cam({probe_xy[0]}, {probe_xy[1]}): {z_near:.2f} m")
        else:
            z_near = nearest_depth_in_centerbox(depth_near_arr, pct=10.0)
            z_near_median = median_depth_in_centerbox(depth_near_arr)
            print(f"[parallax_calib] z_near centerbox: nearest-decile="
                  f"{z_near:.2f} m, median={z_near_median:.2f} m")
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

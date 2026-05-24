"""Main window, keyboard handler, mode state machine.

Modes:
  IDLE        — low-contrast cobblestone idle animation
  CALIBRATE   — 3-second high-contrast capture, then back to idle
  SELECT      — segmented foreground objects shown with colored halos (1..9)
  EFFECT      — fire/glow/colorshift painted on the selected object

Hotkeys: see HELP_LINE below.
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Disable pygame's startup banner before importing.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402
import moderngl  # noqa: E402

from .capture import FrameSource, Frame
from .pattern import build_layout, render as render_pattern
from .calibrate import run_calibration, DEFAULT_HOMOGRAPHY_PATH
from .segment import segment as segment_depth, Segment
from .effects import (
    EffectRenderer,
    EFFECT_NONE, EFFECT_FIRE, EFFECT_GLOW, EFFECT_COLORSHIFT,
    CYCLE_ORDER, EFFECT_NAMES,
)


MODE_IDLE = "idle"
MODE_CALIBRATE = "calibrate"
MODE_SELECT = "select"
MODE_EFFECT = "effect"

# Visually distinct halo colors for objects 1..9 (RGBA).
HALO_COLORS_RGB = [
    (255,  90,  90),
    ( 90, 220, 120),
    ( 90, 160, 255),
    (255, 200,  80),
    (220, 100, 255),
    ( 80, 235, 235),
    (255, 140,  60),
    (170, 110, 255),
    ( 80, 255, 170),
]

HELP_LINE = "C=calibrate  S=select  1-9=pick  E=effect  R=reseg  Esc=idle  Q=quit"


@dataclass
class AppOptions:
    width: int = 1280
    height: int = 720
    capture_width: int = 848
    capture_height: int = 480
    capture_fps: int = 30
    test_mode: bool = False
    test_frames: int = 30
    headless_camera: bool = False
    fullscreen: bool = False


class App:
    def __init__(self, opts: AppOptions):
        self.opts = opts
        self.width = opts.width
        self.height = opts.height
        self.mode = MODE_IDLE
        self.effect = EFFECT_NONE
        self.effect_cycle_idx = 0
        self.calibration_start: Optional[float] = None
        self.last_frame: Optional[Frame] = None
        self.segments: List[Segment] = []
        self.selected_index: Optional[int] = None  # 1..9
        self.frame_count = 0
        self.fps_times: deque = deque(maxlen=60)
        self.last_render_ms = 0.0
        self.errors: List[str] = []
        self.running = True

    # ------------------------------------------------------------------ setup
    def setup(self):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if self.opts.fullscreen:
            flags |= pygame.FULLSCREEN
        self.screen = pygame.display.set_mode((self.width, self.height), flags)
        pygame.display.set_caption("LiveTracking — Phase 0")

        self.ctx = moderngl.create_context()
        self.fx = EffectRenderer(self.ctx, self.width, self.height)

        # Pattern layout is generated at display resolution.
        self.layout = build_layout(self.width, self.height)

        # Frame source.
        prefer_rs = not self.opts.headless_camera
        self.source = FrameSource(
            prefer_realsense=prefer_rs,
            width=self.opts.capture_width,
            height=self.opts.capture_height,
            fps=self.opts.capture_fps,
        )
        print(f"[ui] capture source: {self.source.source}")

        self.font = pygame.font.SysFont("Consolas", 18)
        self.font_big = pygame.font.SysFont("Consolas", 28, bold=True)

        # Reusable RGBA buffers for selection and overlay layers.
        self._sel_rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self._overlay_rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        self.start_time = time.perf_counter()
        self.last_capture_time = self.start_time

    # ------------------------------------------------------------------ input
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type != pygame.KEYDOWN:
                continue
            key = event.key
            if key == pygame.K_q:
                self.running = False
            elif key == pygame.K_ESCAPE:
                self._enter_idle()
            elif key == pygame.K_c:
                self._enter_calibrate()
            elif key == pygame.K_s:
                self._enter_select()
            elif key == pygame.K_r:
                self._run_segmentation()
            elif key == pygame.K_e:
                if self.mode == MODE_EFFECT:
                    self.effect_cycle_idx = (self.effect_cycle_idx + 1) % len(CYCLE_ORDER)
                    self.effect = CYCLE_ORDER[self.effect_cycle_idx]
            elif pygame.K_1 <= key <= pygame.K_9:
                idx = key - pygame.K_0
                self._select_object(idx)

    def _enter_idle(self):
        self.mode = MODE_IDLE
        self.effect = EFFECT_NONE
        self.calibration_start = None

    def _enter_calibrate(self):
        self.mode = MODE_CALIBRATE
        self.calibration_start = time.perf_counter()
        self._calib_frames = 0

    def _enter_select(self):
        self._run_segmentation()
        self.mode = MODE_SELECT
        self.effect = EFFECT_NONE

    def _run_segmentation(self):
        if self.last_frame is None:
            self.segments = []
            return
        try:
            self.segments = segment_depth(self.last_frame.depth_m)
        except Exception as e:
            self.errors.append(f"segment: {e}")
            self.segments = []
        if self.selected_index is not None and self.selected_index > len(self.segments):
            self.selected_index = None
        print(f"[ui] segmented {len(self.segments)} objects")

    def _select_object(self, n: int):
        if not self.segments:
            return
        if n < 1 or n > len(self.segments):
            return
        self.selected_index = n
        self.effect_cycle_idx = 0
        self.effect = CYCLE_ORDER[self.effect_cycle_idx]
        self.mode = MODE_EFFECT

    # ------------------------------------------------------------------ update
    def step_capture(self):
        try:
            self.last_frame = self.source.read()
            self.last_capture_time = time.perf_counter()
        except Exception as e:
            self.errors.append(f"capture: {e}")

    def step_calibration(self):
        if self.mode != MODE_CALIBRATE or self.calibration_start is None:
            return
        elapsed = time.perf_counter() - self.calibration_start
        self._calib_frames += 1
        if elapsed >= 3.0:
            # Drive the calibrate module with our already-captured frame source.
            try:
                result = run_calibration(
                    capture_fn=self.source.read,
                    save_path=Path(DEFAULT_HOMOGRAPHY_PATH),
                    duration_s=0.05,  # we already burned 3s projecting; this is bookkeeping
                )
                print(f"[ui] calibration done: placeholder={result.placeholder}, "
                      f"frames_during_window={self._calib_frames}")
            except Exception as e:
                self.errors.append(f"calibrate: {e}")
            self._enter_idle()

    # ----------------------------------------------------------------- render
    def build_pattern(self, t: float) -> np.ndarray:
        if self.mode == MODE_CALIBRATE:
            return render_pattern(self.layout, "calibration", t)
        if self.mode in (MODE_SELECT, MODE_EFFECT):
            return render_pattern(self.layout, "selection_bg", t)
        return render_pattern(self.layout, "idle", t)

    def build_selection_layer(self) -> Optional[np.ndarray]:
        if self.mode != MODE_SELECT or not self.segments:
            self._sel_rgba.fill(0)
            return self._sel_rgba

        self._sel_rgba.fill(0)
        sx = self.width / self.last_frame.color.shape[1]
        sy = self.height / self.last_frame.color.shape[0]

        for seg in self.segments:
            color = HALO_COLORS_RGB[(seg.index - 1) % len(HALO_COLORS_RGB)]
            mask_u8 = seg.mask.astype(np.uint8) * 255
            mask_disp = cv2.resize(
                mask_u8, (self.width, self.height), interpolation=cv2.INTER_LINEAR
            )
            halo = cv2.GaussianBlur(mask_disp, (0, 0), sigmaX=18.0)
            alpha = (halo.astype(np.uint16) * 200 // 255).clip(0, 220).astype(np.uint8)
            # Painters' overlay: max-blend each channel.
            np.maximum(self._sel_rgba[..., 0], (alpha * color[0] // 255).astype(np.uint8),
                       out=self._sel_rgba[..., 0])
            np.maximum(self._sel_rgba[..., 1], (alpha * color[1] // 255).astype(np.uint8),
                       out=self._sel_rgba[..., 1])
            np.maximum(self._sel_rgba[..., 2], (alpha * color[2] // 255).astype(np.uint8),
                       out=self._sel_rgba[..., 2])
            np.maximum(self._sel_rgba[..., 3], alpha, out=self._sel_rgba[..., 3])

        # Numeric labels — easier on CPU via pygame surface blits, then read back.
        label_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for seg in self.segments:
            cx, cy = seg.centroid_xy
            cx_d = int(cx * sx)
            cy_d = int(cy * sy)
            color = HALO_COLORS_RGB[(seg.index - 1) % len(HALO_COLORS_RGB)]
            badge_r = 18
            pygame.draw.circle(label_surf, (*color, 230), (cx_d, cy_d), badge_r)
            pygame.draw.circle(label_surf, (0, 0, 0, 255), (cx_d, cy_d), badge_r, 2)
            text = self.font_big.render(str(seg.index), True, (10, 10, 10))
            label_surf.blit(text, (cx_d - text.get_width() // 2,
                                   cy_d - text.get_height() // 2))
        label_arr = pygame.surfarray.array3d(label_surf).swapaxes(0, 1)
        label_a = pygame.surfarray.array_alpha(label_surf).swapaxes(0, 1)
        mask = label_a > 0
        self._sel_rgba[mask, 0] = label_arr[mask, 0]
        self._sel_rgba[mask, 1] = label_arr[mask, 1]
        self._sel_rgba[mask, 2] = label_arr[mask, 2]
        self._sel_rgba[mask, 3] = np.maximum(self._sel_rgba[mask, 3], label_a[mask])
        return self._sel_rgba

    def build_effect_mask(self) -> Optional[np.ndarray]:
        if self.mode != MODE_EFFECT:
            return None
        if self.selected_index is None or not self.segments:
            return None
        idx = self.selected_index - 1
        if idx >= len(self.segments):
            return None
        seg = self.segments[idx]
        return (seg.mask.astype(np.uint8) * 255)

    def build_overlay(self, t: float) -> np.ndarray:
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        fps = self._compute_fps()
        lat = self.last_render_ms
        cap_src = self.source.source if hasattr(self, "source") else "?"
        if self.mode == MODE_EFFECT and self.selected_index is not None:
            sel_str = f"#{self.selected_index} (effect: {EFFECT_NAMES[self.effect]})"
        elif self.mode == MODE_SELECT:
            sel_str = f"{len(self.segments)} candidates — press 1..{len(self.segments)}"
        else:
            sel_str = "-"
        lines = [
            f"FPS: {fps:5.1f}    render: {lat:5.1f} ms",
            f"mode: {self.mode}",
            f"objects: {len(self.segments)}   selected: {sel_str}",
            f"capture: {cap_src}",
            HELP_LINE,
        ]
        if self.mode == MODE_CALIBRATE and self.calibration_start is not None:
            remaining = max(0.0, 3.0 - (time.perf_counter() - self.calibration_start))
            lines.insert(0, f"CALIBRATING — {remaining:.1f}s remaining")
        pad = 8
        line_h = 22
        box_h = pad * 2 + line_h * len(lines)
        pygame.draw.rect(surf, (0, 0, 0, 170), pygame.Rect(0, 0, 620, box_h))
        for i, line in enumerate(lines):
            color = (255, 220, 120) if i == 0 and self.mode == MODE_CALIBRATE else (240, 240, 240)
            text = self.font.render(line, True, color)
            surf.blit(text, (pad, pad + i * line_h))

        if self.errors:
            err_text = self.font.render(f"errors: {len(self.errors)} — last: {self.errors[-1][:60]}",
                                        True, (255, 120, 120))
            surf.blit(err_text, (pad, self.height - 28))

        rgba = pygame.surfarray.array3d(surf).swapaxes(0, 1)
        alpha = pygame.surfarray.array_alpha(surf).swapaxes(0, 1)
        out = np.dstack([rgba, alpha]).astype(np.uint8)
        return out

    def _compute_fps(self) -> float:
        if len(self.fps_times) < 2:
            return 0.0
        dt = self.fps_times[-1] - self.fps_times[0]
        if dt <= 0:
            return 0.0
        return (len(self.fps_times) - 1) / dt

    # ------------------------------------------------------------------- loop
    def frame(self):
        frame_start = time.perf_counter()
        self.handle_events()
        if not self.running:
            return
        self.step_capture()
        self.step_calibration()

        t = time.perf_counter() - self.start_time

        pattern = self.build_pattern(t)
        self.fx.upload_pattern_bgr(pattern)

        sel = self.build_selection_layer()
        self.fx.upload_selection_rgba(sel)

        mask = self.build_effect_mask()
        self.fx.upload_mask(mask)

        overlay = self.build_overlay(t)
        self.fx.upload_overlay_rgba(overlay)

        self.fx.render(t=t, effect=self.effect)
        pygame.display.flip()

        now = time.perf_counter()
        self.fps_times.append(now)
        self.last_render_ms = (now - frame_start) * 1000.0
        self.frame_count += 1

    def run(self):
        try:
            while self.running:
                self.frame()
                if self.opts.test_mode and self.frame_count >= self.opts.test_frames:
                    print(f"[ui] test mode — ran {self.frame_count} frames; exiting.")
                    self.running = False
        finally:
            self.shutdown()

    def shutdown(self):
        try:
            self.source.close()
        except Exception as e:
            self.errors.append(f"capture.close: {e}")
        try:
            pygame.quit()
        except Exception:
            pass

    # ----------------------------------------------------------- test summary
    def print_test_summary(self):
        fps = self._compute_fps()
        print("-" * 60)
        print(f"frames rendered : {self.frame_count}")
        print(f"avg FPS          : {fps:.2f}")
        print(f"last render ms   : {self.last_render_ms:.2f}")
        print(f"capture source   : {self.source.source}")
        print(f"errors           : {len(self.errors)}")
        for e in self.errors:
            print(f"  - {e}")
        print("-" * 60)

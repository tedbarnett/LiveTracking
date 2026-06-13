"""Projector daemon — owns the JMGO fullscreen pygame surface.

Subscribes (PULL) to highlight/clear commands from the perception daemon.
On `highlight`, paints the object's projector-space mask in its assigned
color, and draws the object number large at the mask centroid. On `clear`,
goes to black.

Idle-default: black. Hover-time response: < 1 frame at 60 Hz.

NSSM-install as `LiveTrackingProjector`, AUTO_START.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
import zmq

from livetracking.daemon import effects
from livetracking.paths import DISPLAY_INDEX, ZMQ_PROJECTOR_PULL, describe


class ProjectorDaemon:
    def __init__(self):
        print(f"[projector] {describe()}")
        import pygame
        pygame.init()
        pygame.display.init()
        sizes = pygame.display.get_desktop_sizes()
        if not sizes:
            raise RuntimeError("pygame found 0 displays")
        if DISPLAY_INDEX is not None and 0 <= DISPLAY_INDEX < len(sizes):
            idx = DISPLAY_INDEX
        else:
            idx = max(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1])
        self.PW, self.PH = sizes[idx]
        print(f"[projector] using display {idx}: {self.PW}x{self.PH}")
        self.screen = pygame.display.set_mode(
            (self.PW, self.PH), pygame.NOFRAME, display=idx
        )
        self.pygame = pygame
        # Number font 115 (-20%); label font ~0.5x for caption hierarchy.
        self.font_big = pygame.font.SysFont(None, 115)
        self.font_lbl = pygame.font.SysFont(None, 58)

        # Banner font: ~3x the number font for full-room legibility.
        self.font_banner = pygame.font.SysFont(None, 340)

        self.state_lock = threading.Lock()
        self.current = None        # single highlight
        self.current_many = None   # highlight_all payload
        self.intensity = 0.78      # alpha multiplier (0..1)
        self.white_light = False   # if True, paint full white over everything
        # When non-empty, the render loop paints a full-screen banner on top
        # of (or in place of) everything else. Used by flame_web/perception
        # to signal "Rebuilding…" during a detector switch or recalibrate
        # cycle while the model is reloading. Cleared by a set_busy with an
        # empty/None text, or by clear_busy.
        self.busy_text: Optional[str] = None

        # Monotonic clock for animated effects (flame/cloud). All effect
        # phases derive from now()-t0 so every object animates in lockstep
        # and the loop stays stateless per object.
        self._anim_t0 = time.perf_counter()

        # Per-object mask cache. Decoding a 4K mask PNG (~16 ms) and scanning
        # it for its bbox (~24 ms) every frame, for every painted object, is
        # what made multi-object animations crawl (Select-all + 2 animated
        # objects = ~80 ms/frame). The mask only changes when perception
        # rewrites the PNG (~every 2 s on a DINO refresh), so we cache the
        # decoded mask + bbox keyed by (path, mtime, size) and only re-read
        # when the file actually changes. Keyed by mask_path.
        # value: {"key": (mtime, size), "mask": ndarray, "bbox": (x0,y0,x1,y1)}
        self._mask_cache: dict = {}

        ctx = zmq.Context.instance()
        self.pull = ctx.socket(zmq.PULL)
        self.pull.bind(ZMQ_PROJECTOR_PULL)

        self.running = True
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)
        self.zmq_thread = threading.Thread(target=self._zmq_loop, daemon=True)

    def _stop(self, *_):
        self.running = False

    def _zmq_loop(self):
        while self.running:
            try:
                msg = self.pull.recv_json(flags=0)
            except Exception:
                break
            with self.state_lock:
                t = msg.get("type")
                if t == "clear":
                    self.current = None
                    self.current_many = None
                elif t == "highlight":
                    self.current = msg
                    self.current_many = None
                elif t == "highlight_all":
                    self.current = None
                    self.current_many = msg
                elif t == "set_intensity":
                    self.intensity = float(msg.get("value", 0.78))
                elif t == "set_white_light":
                    self.white_light = bool(msg.get("value", False))
                elif t == "set_busy":
                    txt = msg.get("text")
                    self.busy_text = (
                        str(txt).strip() if txt else None
                    ) or None
                elif t == "clear_busy":
                    self.busy_text = None
            time.sleep(0)

    def _load_mask(self, mask_path: str):
        """Return (mask, bbox) for mask_path, decoding from disk only when the
        file has changed since last read. bbox is (x0, y0, x1, y1) or None for
        an empty mask. Returns (None, None) if the file is missing/unreadable.

        This is the hot-path optimization: without caching, every frame
        re-decoded the 4K PNG (~16 ms) and re-scanned it for its bbox
        (~24 ms) per object, so animating multiple selected objects dropped
        to single-digit fps. The mask only changes on a perception refresh,
        so we key the cache on (mtime, size) and reuse otherwise."""
        if not mask_path:
            return None, None
        try:
            st = os.stat(mask_path)
        except OSError:
            return None, None
        key = (st.st_mtime, st.st_size)
        ent = self._mask_cache.get(mask_path)
        if ent is not None and ent["key"] == key:
            return ent["mask"], ent["bbox"]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, None
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            bbox = None
        else:
            bbox = (int(xs.min()), int(ys.min()),
                    int(xs.max()) + 1, int(ys.max()) + 1)
        self._mask_cache[mask_path] = {"key": key, "mask": mask, "bbox": bbox}
        # Bound the cache so a long session with many transient ids doesn't
        # grow without limit. Drop the oldest entries past a soft cap.
        if len(self._mask_cache) > 64:
            for k in list(self._mask_cache.keys())[:16]:
                self._mask_cache.pop(k, None)
        return mask, bbox

    def _blit_flat(self, mask: np.ndarray, color, intensity: float):
        """Original flat-color path: tint the whole silhouette one color.
        The mask grayscale is the alpha (anti-aliased soft edges preserved)."""
        tint = np.zeros((self.PH, self.PW, 4), dtype=np.uint8)
        tint[..., 0] = color[0]
        tint[..., 1] = color[1]
        tint[..., 2] = color[2]
        tint[..., 3] = (
            mask.astype(np.float32) * intensity
        ).clip(0, 255).astype(np.uint8)
        surf = self.pygame.image.frombuffer(
            tint.tobytes(), (self.PW, self.PH), "RGBA"
        )
        self.screen.blit(surf, (0, 0))

    def _blit_effect(self, mask: np.ndarray, bbox, effect: str,
                     intensity: float):
        """Animated-texture path (flame/cloud/water).

        Renders the procedural effect ONLY at the mask's bounding box (not
        the full 4K surface) for speed, stencils it through the mask alpha,
        and blits it at the bbox offset. The effect texture is RGB on black;
        the mask grayscale drives alpha so soft warp edges stay soft and the
        dark parts of the flame fall to zero alpha (vanish on the wall).

        `bbox` (x0, y0, x1, y1) is precomputed by _load_mask and cached, so
        the hot loop no longer scans the full 4K mask every frame. The RGBA
        (incl. the effect's own luminance alpha) is composited at capped
        resolution inside render_effect_rgba; here we only fold in the object
        mask + user intensity on the alpha channel."""
        if bbox is None:
            return
        x0, y0, x1, y1 = bbox
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            return
        t = time.perf_counter() - self._anim_t0
        rgba = effects.render_effect_rgba(effect, bw, bh, t)  # (bh, bw, 4)
        # Fold the object mask + intensity into alpha. Both in 0..255 uint8;
        # use a uint16 multiply then >>8 (≈ /255) — far cheaper than building
        # two float32 million-pixel arrays every frame.
        sub_mask = mask[y0:y1, x0:x1]
        a = rgba[..., 3].astype(np.uint16)
        a *= sub_mask.astype(np.uint16)
        a >>= 8
        if intensity < 0.999:
            a = (a * int(intensity * 256)) >> 8
        rgba[..., 3] = a.astype(np.uint8)
        surf = self.pygame.image.frombuffer(
            np.ascontiguousarray(rgba).tobytes(), (bw, bh), "RGBA"
        )
        self.screen.blit(surf, (x0, y0))

    def _paint_one(self, cur: dict):
        mask_path = cur.get("mask_path")
        color = tuple(cur.get("color", (255, 200, 0)))
        effect = cur.get("effect", "color")
        mask, bbox = self._load_mask(mask_path)
        if mask is not None and mask.shape == (self.PH, self.PW):
            intensity = max(0.0, min(1.0, self.intensity))
            if effects.is_effect(effect):
                self._blit_effect(mask, bbox, effect, intensity)
            else:
                self._blit_flat(mask, color, intensity)
        pc = cur.get("proj_centroid")
        if pc is not None:
            cx_p, cy_p = int(pc[0]), int(pc[1])
            num_str = "#" + str(cur.get("id", "?"))
            name_str = str(cur.get("name", "")).strip()

            # Number, centered on the object centroid, with a small black
            # drop shadow for legibility against bright projector colors.
            txt = self.font_big.render(num_str, True, (255, 255, 255))
            tw, th = txt.get_size()
            shadow = self.font_big.render(num_str, True, (0, 0, 0))
            self.screen.blit(shadow, (cx_p - tw // 2 + 3,
                                      cy_p - th // 2 + 3))
            self.screen.blit(txt, (cx_p - tw // 2, cy_p - th // 2))

            # Name (same font size), directly under the number.
            if name_str:
                lbl = self.font_lbl.render(name_str, True, (255, 255, 255))
                lw, lh = lbl.get_size()
                lbl_shadow = self.font_lbl.render(name_str, True, (0, 0, 0))
                ly = cy_p + th // 2 + 8  # small gap below number
                self.screen.blit(lbl_shadow,
                                 (cx_p - lw // 2 + 3, ly + 3))
                self.screen.blit(lbl, (cx_p - lw // 2, ly))

    def _render(self):
        with self.state_lock:
            cur = self.current
            many = self.current_many
            white = self.white_light
        if white:
            self.screen.fill((255, 255, 255))
            self.pygame.display.flip()
            for _ in self.pygame.event.get():
                pass
            return
        self.screen.fill((0, 0, 0))
        if many is not None:
            for obj in many.get("objects", []):
                self._paint_one(obj)
        elif cur is not None:
            self._paint_one(cur)
        self.pygame.display.flip()
        for _ in self.pygame.event.get():
            pass

    def run(self):
        self.zmq_thread.start()
        print("[projector] entering render loop")
        try:
            while self.running:
                self._render()
                time.sleep(1 / 60)
        finally:
            self.pygame.quit()
            print("[projector] exited")


def main() -> int:
    ProjectorDaemon().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

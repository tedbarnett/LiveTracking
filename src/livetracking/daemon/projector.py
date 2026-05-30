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

        self.state_lock = threading.Lock()
        self.current = None        # single highlight
        self.current_many = None   # highlight_all payload
        self.intensity = 0.78      # alpha multiplier (0..1)
        self.white_light = False   # if True, paint full white over everything

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
            time.sleep(0)

    def _paint_one(self, cur: dict):
        mask_path = cur.get("mask_path")
        color = tuple(cur.get("color", (255, 200, 0)))
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape == (self.PH, self.PW):
            alpha_val = int(255 * max(0.0, min(1.0, self.intensity)))
            tint = np.zeros((self.PH, self.PW, 4), dtype=np.uint8)
            tint[..., 0] = color[0]
            tint[..., 1] = color[1]
            tint[..., 2] = color[2]
            tint[..., 3] = (mask > 0).astype(np.uint8) * alpha_val
            surf = self.pygame.image.frombuffer(
                tint.tobytes(), (self.PW, self.PH), "RGBA"
            )
            self.screen.blit(surf, (0, 0))
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

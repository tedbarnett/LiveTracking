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
        self.font_big = pygame.font.SysFont(None, 720)
        self.font_lbl = pygame.font.SysFont(None, 160)

        self.state_lock = threading.Lock()
        self.current = None  # dict | None: {id, color, proj_centroid, mask_path}

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
                if msg.get("type") == "clear":
                    self.current = None
                elif msg.get("type") == "highlight":
                    self.current = msg
            time.sleep(0)

    def _render(self):
        with self.state_lock:
            cur = self.current
        self.screen.fill((0, 0, 0))
        if cur is not None:
            mask_path = cur.get("mask_path")
            color = tuple(cur.get("color", (255, 200, 0)))
            mask = None
            if mask_path and os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape == (self.PH, self.PW):
                # Build a 4-channel RGBA surface with the mask as alpha
                tint = np.zeros((self.PH, self.PW, 4), dtype=np.uint8)
                tint[..., 0] = color[0]
                tint[..., 1] = color[1]
                tint[..., 2] = color[2]
                tint[..., 3] = (mask > 0).astype(np.uint8) * 200
                surf = self.pygame.image.frombuffer(
                    tint.tobytes(), (self.PW, self.PH), "RGBA"
                )
                self.screen.blit(surf, (0, 0))
            # number at centroid
            pc = cur.get("proj_centroid")
            if pc is not None:
                num_str = str(cur.get("id", "?"))
                txt = self.font_big.render(num_str, True, (255, 255, 255))
                tw, th = txt.get_size()
                shadow = self.font_big.render(num_str, True, (0, 0, 0))
                self.screen.blit(shadow, (int(pc[0]) - tw // 2 + 6,
                                          int(pc[1]) - th // 2 + 6))
                self.screen.blit(txt, (int(pc[0]) - tw // 2,
                                       int(pc[1]) - th // 2))
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

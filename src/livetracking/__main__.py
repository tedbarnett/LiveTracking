"""Entry point: `python -m livetracking [--test] [--no-camera] [--fullscreen]`."""
from __future__ import annotations

import argparse
import sys

from .ui import App, AppOptions


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="livetracking",
        description="LiveTracking Phase 0 prototype — cobblestone calibration + "
                    "depth-cluster segmentation + effect compositor.",
    )
    p.add_argument("--width", type=int, default=1280, help="display window width (default 1280)")
    p.add_argument("--height", type=int, default=720, help="display window height (default 720)")
    p.add_argument("--capture-width", type=int, default=848,
                   help="camera capture width (default 848)")
    p.add_argument("--capture-height", type=int, default=480,
                   help="camera capture height (default 480)")
    p.add_argument("--capture-fps", type=int, default=30, help="camera fps (default 30)")
    p.add_argument("--no-camera", action="store_true",
                   help="skip all real cameras; use synthetic frames only")
    p.add_argument("--no-realsense", action="store_true",
                   help="skip the RealSense; allow webcam, else synthetic")
    p.add_argument("--fullscreen", action="store_true", help="fullscreen window")
    p.add_argument("--test", action="store_true",
                   help="run a headless 30-frame test and exit with FPS report")
    p.add_argument("--test-frames", type=int, default=30,
                   help="number of frames in --test mode (default 30)")
    p.add_argument("--headless", action="store_true",
                   help="run without a window (moderngl standalone context)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # --test implies headless + no camera (deterministic synthetic frames).
    headless = args.headless or args.test
    headless_camera = args.no_camera or args.test
    opts = AppOptions(
        width=args.width,
        height=args.height,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        capture_fps=args.capture_fps,
        test_mode=args.test,
        test_frames=args.test_frames,
        headless_camera=headless_camera,
        no_realsense=args.no_realsense,
        fullscreen=args.fullscreen,
        headless=headless,
    )

    app = App(opts)
    try:
        app.setup()
        app.run()
    except KeyboardInterrupt:
        print("[main] interrupted")
    except Exception as e:
        print(f"[main] fatal: {e}", file=sys.stderr)
        if args.test:
            app.errors.append(f"fatal: {e}")
        else:
            raise

    if args.test:
        app.print_test_summary()
        # Exit code reflects whether we caught any errors.
        return 1 if app.errors else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

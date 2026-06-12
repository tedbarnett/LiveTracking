"""Orchestrate a full re-calibration cycle.

Runs in the user's desktop session (as the LiveTrackingCalibrate scheduled
task) so it can grab the JMGO on display 1 and the RealSense camera. The
sequence:

  1. Stop perception (frees the D455).
  2. Stop projector (frees pygame display 1).
  3. Run calibrate_homography.py.
  4. Restart projector.
  5. Restart perception (it re-reads H.npy on startup).

Status is written to runtime/calibration_status.json so the Flask UI
(running in Session 0) can poll it without needing to reach into the user
session.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from livetracking.paths import RUNTIME_DIR  # noqa: E402

STATUS_FILE = os.path.join(RUNTIME_DIR, "calibration_status.json")
LOG_FILE = os.path.join(RUNTIME_DIR, "service-logs", "calibrate.log")
VENV_PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
# Gray-code structured light is the default: it needs no flat wall patches
# (the room's guitars/AC/couch broke ArUco marker detection). Set
# LIVETRACKING_CALIB_METHOD=aruco to fall back to the old marker approach.
_METHOD = os.environ.get("LIVETRACKING_CALIB_METHOD", "graycode").lower()
CALIB_SCRIPT = os.path.join(
    HERE,
    "calibrate_homography.py" if _METHOD == "aruco"
    else "calibrate_graycode.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(phase: str, ok: bool | None = None, detail: str = "") -> None:
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    payload = {
        "phase": phase,
        "ok": ok,
        "detail": detail,
        "t": _now(),
        "pid": os.getpid(),
    }
    tmp = STATUS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, STATUS_FILE)


def _schtasks(args: list[str]) -> tuple[int, str]:
    r = subprocess.run(
        ["schtasks"] + args,
        capture_output=True, text=True, shell=False,
    )
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode, out.strip()


def stop_task(name: str) -> None:
    _schtasks(["/end", "/tn", name])  # idempotent: ignores errors


def start_task(name: str) -> None:
    _schtasks(["/run", "/tn", name])


def task_running(name: str) -> bool:
    rc, out = _schtasks(["/query", "/tn", name, "/fo", "LIST", "/v"])
    return "Running" in out


def wait_for_stop(name: str, timeout_s: float = 8.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not task_running(name):
            return
        time.sleep(0.3)


def main() -> int:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log = open(LOG_FILE, "a", buffering=1)
    log.write(f"\n=== recalibrate run @ {_now()} ===\n")

    try:
        write_status("stopping_perception")
        log.write("[orch] stopping LiveTrackingPerception\n")
        stop_task("LiveTrackingPerception")
        wait_for_stop("LiveTrackingPerception")

        write_status("stopping_projector")
        log.write("[orch] stopping LiveTrackingProjector\n")
        stop_task("LiveTrackingProjector")
        wait_for_stop("LiveTrackingProjector")
        # Extra settle time for pygame/RealSense USB teardown.
        time.sleep(1.5)

        write_status("calibrating")
        log.write(f"[orch] running {CALIB_SCRIPT}\n")
        env = os.environ.copy()
        # Force project venv (rule from memory: clobber inherited PYTHONPATH).
        env["PYTHONPATH"] = ""
        env["VIRTUAL_ENV"] = os.path.join(REPO, ".venv")
        r = subprocess.run(
            [VENV_PY, CALIB_SCRIPT],
            cwd=REPO, env=env,
            stdout=log, stderr=subprocess.STDOUT,
            timeout=240,
        )
        calib_ok = (r.returncode == 0)
        log.write(f"[orch] calibration exit={r.returncode}\n")

        write_status("restarting_projector")
        log.write("[orch] restarting LiveTrackingProjector\n")
        start_task("LiveTrackingProjector")
        time.sleep(1.5)

        write_status("restarting_perception")
        log.write("[orch] restarting LiveTrackingPerception\n")
        start_task("LiveTrackingPerception")

        if calib_ok:
            write_status("done", ok=True, detail="calibration succeeded")
            log.write("[orch] DONE ok\n")
            return 0
        else:
            write_status("done", ok=False,
                         detail=f"calibrate_homography.py exit={r.returncode}")
            log.write("[orch] DONE failed\n")
            return r.returncode

    except Exception as e:
        log.write(f"[orch] EXCEPTION: {e!r}\n")
        write_status("done", ok=False, detail=f"exception: {e!r}")
        # Best-effort restart so the system isn't left in a broken state.
        start_task("LiveTrackingProjector")
        start_task("LiveTrackingPerception")
        return 99
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())

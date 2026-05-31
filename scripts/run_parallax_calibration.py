"""Orchestrate a parallax calibration cycle.

Same pattern as run_calibration.py:
  1. Stop perception (frees the D455).
  2. Stop projector (frees pygame display 1).
  3. Run scripts/calibrate_parallax.py (interactive UI on the projector).
  4. Restart projector + perception.

Status is written to runtime/parallax_calibration_status.json so the Flask
UI can poll progress.
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

STATUS_FILE = os.path.join(RUNTIME_DIR, "parallax_calibration_status.json")
LOG_FILE = os.path.join(RUNTIME_DIR, "service-logs", "parallax_calibrate.log")
VENV_PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
CALIB_SCRIPT = os.path.join(HERE, "calibrate_parallax.py")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(phase: str, ok: bool | None = None, detail: str = "") -> None:
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    payload = {"phase": phase, "ok": ok, "detail": detail,
               "t": _now(), "pid": os.getpid()}
    tmp = STATUS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, STATUS_FILE)


def _schtasks(args: list[str]) -> tuple[int, str]:
    r = subprocess.run(["schtasks"] + args,
                       capture_output=True, text=True, shell=False)
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode, out.strip()


def stop_task(name: str) -> None:
    _schtasks(["/end", "/tn", name])


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
    log.write(f"\n=== parallax calibrate run @ {_now()} ===\n")

    try:
        write_status("stopping_perception")
        log.write("[orch] stopping LiveTrackingPerception\n")
        stop_task("LiveTrackingPerception")
        wait_for_stop("LiveTrackingPerception")

        write_status("stopping_projector")
        log.write("[orch] stopping LiveTrackingProjector\n")
        stop_task("LiveTrackingProjector")
        wait_for_stop("LiveTrackingProjector")
        time.sleep(1.5)

        write_status("calibrating")
        log.write(f"[orch] running {CALIB_SCRIPT}\n")
        env = os.environ.copy()
        env["PYTHONPATH"] = ""
        env["VIRTUAL_ENV"] = os.path.join(REPO, ".venv")
        r = subprocess.run(
            [VENV_PY, CALIB_SCRIPT],
            cwd=REPO, env=env,
            stdout=log, stderr=subprocess.STDOUT,
            # No timeout — this is an interactive UI; operator may take
            # several minutes to align by eye. If they walk away the
            # daemon stays paused, which is fine.
        )
        calib_ok = (r.returncode == 0)
        log.write(f"[orch] calibration exit={r.returncode}\n")

        write_status("restarting_projector")
        start_task("LiveTrackingProjector")
        time.sleep(1.5)

        write_status("restarting_perception")
        start_task("LiveTrackingPerception")

        if calib_ok:
            write_status("done", ok=True, detail="parallax calibration succeeded")
            return 0
        else:
            write_status("done", ok=False,
                         detail=f"calibrate_parallax.py exit={r.returncode}")
            return r.returncode

    except Exception as e:
        log.write(f"[orch] EXCEPTION: {e!r}\n")
        write_status("done", ok=False, detail=f"exception: {e!r}")
        start_task("LiveTrackingProjector")
        start_task("LiveTrackingPerception")
        return 99
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())

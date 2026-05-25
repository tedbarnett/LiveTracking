"""Daemon supervisor: keeps projection_daemon.py running.

If the daemon exits with code 42 (restart requested via web UI), relaunch it.
Any other non-zero exit means real failure - log and relaunch after a wait.
Exit code 0 = clean shutdown, supervisor exits too.
"""
import os
import subprocess
import sys
import time

DAEMON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "projection_daemon.py")


def main():
    while True:
        print(f"[supervisor] launching {DAEMON}", flush=True)
        rc = subprocess.call([sys.executable, DAEMON])
        print(f"[supervisor] daemon exited with code {rc}", flush=True)
        if rc == 42:
            print("[supervisor] restart requested - relaunching in 2s", flush=True)
            time.sleep(2.0)
            continue
        if rc == 0:
            print("[supervisor] clean exit - shutting down", flush=True)
            return 0
        print(f"[supervisor] crash (rc={rc}) - relaunching in 5s", flush=True)
        time.sleep(5.0)


if __name__ == "__main__":
    sys.exit(main())

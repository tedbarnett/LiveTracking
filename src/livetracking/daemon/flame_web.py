"""Web UI at livetracking.barnettlabs.tech — Flask on :5070.

- GET  /              -> single page with live MJPEG + numbered object list
- GET  /stream.mjpg   -> multipart JPEG from the perception daemon PUB socket
- GET  /objects       -> SSE: object-list updates as perception sees them
- POST /rename        -> {id, name}     (goes to perception daemon over REQ)
- POST /highlight     -> {id}           (perception forwards to projector)
- POST /clear         -> {}             (perception forwards to projector)

Cloudflare tunnel `livetracking-laptop` (NSSM service `Cloudflared`) already
routes livetracking.barnettlabs.tech -> http://localhost:5070, so this is
visible publicly the moment the daemon is up.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from typing import Optional

import zmq
from flask import Flask, Response, jsonify, render_template, request

from livetracking.paths import RUNTIME_DIR, WEB_UI_PORT, ZMQ_OBJECTS_PUB


CTRL_ENDPOINT = "tcp://127.0.0.1:5573"


# ---- shared latest state ---------------------------------------------------
class LatestState:
    def __init__(self):
        self._lock = threading.Lock()
        self._objects_payload: Optional[dict] = None
        self._jpeg: Optional[bytes] = None
        self._subscribers: list[queue.Queue] = []

    def update_objects(self, payload: dict):
        with self._lock:
            self._objects_payload = payload
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                try:
                    self._subscribers.remove(q)
                except ValueError:
                    pass

    def update_frame(self, jpeg: bytes):
        with self._lock:
            self._jpeg = jpeg

    def get_objects(self) -> Optional[dict]:
        with self._lock:
            return self._objects_payload

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=8)
        with self._lock:
            self._subscribers.append(q)
            if self._objects_payload is not None:
                try:
                    q.put_nowait(self._objects_payload)
                except queue.Full:
                    pass
        return q


STATE = LatestState()


# ---- ZMQ subscriber thread -------------------------------------------------
def _zmq_subscriber():
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(ZMQ_OBJECTS_PUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVHWM, 4)
    while True:
        try:
            topic, body = sub.recv_multipart()
        except Exception as e:
            print(f"[web] zmq recv err: {e}")
            time.sleep(0.5)
            continue
        if topic == b"objects":
            try:
                payload = json.loads(body.decode("utf-8"))
                STATE.update_objects(payload)
            except Exception as e:
                print(f"[web] bad objects payload: {e}")
        elif topic == b"frame":
            STATE.update_frame(body)


def _send_ctrl(msg: dict, timeout_ms: int = 3000) -> dict:
    """One-shot REQ to the perception daemon's control socket."""
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVTIMEO, timeout_ms)
    s.setsockopt(zmq.SNDTIMEO, timeout_ms)
    try:
        s.connect(CTRL_ENDPOINT)
        s.send_json(msg)
        return s.recv_json()
    except zmq.Again:
        return {"ok": False, "reason": "perception daemon timeout"}
    except Exception as e:
        return {"ok": False, "reason": repr(e)}
    finally:
        s.close()


# ---- Flask app -------------------------------------------------------------
def create_app() -> Flask:
    here = os.path.dirname(os.path.abspath(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(here, "templates"),
        static_folder=os.path.join(here, "static"),
    )

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/snapshot.jpg")
    def snapshot_jpg():
        """One-shot JPEG of the latest annotated frame — finite response."""
        jpeg = STATE.get_jpeg()
        if jpeg is None:
            return Response(b"", status=503)
        return Response(jpeg, mimetype="image/jpeg")

    @app.route("/stream.mjpg")
    def stream_mjpg():
        def gen():
            last_id = id(None)
            while True:
                jpeg = STATE.get_jpeg()
                if jpeg is not None and id(jpeg) != last_id:
                    last_id = id(jpeg)
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg)).encode()
                           + b"\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(1 / 20)  # 20 fps to the browser
        return Response(
            gen(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/objects")
    def objects_sse():
        def gen():
            q = STATE.subscribe()
            # send initial
            payload = STATE.get_objects()
            if payload is not None:
                yield f"event: update\ndata: {json.dumps(payload)}\n\n"
            while True:
                try:
                    payload = q.get(timeout=15)
                    yield f"event: update\ndata: {json.dumps(payload)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        return Response(gen(), mimetype="text/event-stream")

    @app.route("/objects.json")
    def objects_json():
        return jsonify(STATE.get_objects() or {"objects": []})

    @app.route("/rename", methods=["POST"])
    def rename():
        data = request.get_json(silent=True) or {}
        return jsonify(_send_ctrl({
            "cmd": "rename",
            "id": int(data["id"]),
            "name": str(data["name"]),
        }))

    @app.route("/highlight", methods=["POST"])
    def highlight():
        data = request.get_json(silent=True) or {}
        return jsonify(_send_ctrl({"cmd": "highlight", "id": int(data["id"])}))

    @app.route("/clear", methods=["POST"])
    def clear():
        return jsonify(_send_ctrl({"cmd": "clear"}))

    @app.route("/pause", methods=["POST"])
    def pause():
        return jsonify(_send_ctrl({"cmd": "pause"}))

    @app.route("/run", methods=["POST"])
    def runcmd():
        return jsonify(_send_ctrl({"cmd": "run"}))

    @app.route("/state")
    def state():
        return jsonify(_send_ctrl({"cmd": "state"}))

    @app.route("/highlight_all", methods=["POST"])
    def highlight_all():
        return jsonify(_send_ctrl({"cmd": "highlight_all"}))

    @app.route("/cycle_color", methods=["POST"])
    def cycle_color():
        data = request.get_json(silent=True) or {}
        return jsonify(_send_ctrl({"cmd": "cycle_color", "id": int(data["id"])}))

    @app.route("/pin", methods=["POST"])
    def pin():
        data = request.get_json(silent=True) or {}
        return jsonify(_send_ctrl({"cmd": "pin", "id": int(data["id"])}))

    @app.route("/unpin", methods=["POST"])
    def unpin():
        return jsonify(_send_ctrl({"cmd": "unpin"}))

    @app.route("/intensity", methods=["POST"])
    def intensity():
        data = request.get_json(silent=True) or {}
        return jsonify(_send_ctrl({"cmd": "intensity", "value": float(data.get("value", 0.78))}))

    @app.route("/white_light", methods=["POST"])
    def white_light():
        data = request.get_json(silent=True) or {}
        return jsonify(_send_ctrl({"cmd": "white_light", "value": bool(data.get("value", False))}))

    # ---- remote probe --------------------------------------------------
    # GET /probe/<id>?hold=2&size=300 -- flash a bright projector square
    # at the given object's current cam centroid (parallax-compensated),
    # then return the latest annotated snapshot as JPEG so a remote user
    # can verify wash alignment from anywhere. The hold suppresses
    # perception's own projector messages for `hold` seconds.
    @app.route("/probe/<int:obj_id>")
    def probe(obj_id):
        from flask import Response
        hold = float(request.args.get("hold", "1.6"))
        size = int(request.args.get("size", "300"))
        st = _send_ctrl({"cmd": "list"})
        if not st.get("ok"):
            return jsonify({"ok": False, "reason": "list failed",
                            "detail": st}), 502
        target = None
        for o in (st.get("objects") or []):
            if int(o.get("id")) == obj_id:
                target = o
                break
        if target is None:
            return jsonify({"ok": False, "reason": f"no object id {obj_id}"}), 404
        cx, cy = target["centroid_cam"]
        depth = float(target.get("depth_m") or 0.0)
        rep = _send_ctrl({
            "cmd": "test_point",
            "cam_x": float(cx), "cam_y": float(cy),
            "size_px": size, "hold_s": hold,
            "parallax": True, "depth_m": depth,
        })
        time.sleep(0.6)
        # Grab the latest annotated frame via our own /snapshot.jpg route.
        import urllib.request as _ur
        try:
            jpeg_bytes = _ur.urlopen(
                "http://127.0.0.1:5070/snapshot.jpg", timeout=3,
            ).read()
        except Exception as e:  # noqa: BLE001
            return jsonify({"ok": True, "test_point": rep, "target": target,
                            "no_snapshot": True,
                            "snapshot_error": str(e)})
        return Response(
            jpeg_bytes,
            mimetype="image/jpeg",
            headers={
                "X-Probe-Target-Id": str(obj_id),
                "X-Probe-Target-Name": str(target.get("name", "")),
                "X-Probe-Proj-Xy": str(rep.get("proj_xy") or ""),
                "X-Probe-Method": str(rep.get("method") or ""),
                "Cache-Control": "no-store",
            },
        )

    @app.route("/probe", methods=["GET"])
    def probe_index():
        """One link per current object -- useful on a phone to fire-and-screenshot."""
        st = _send_ctrl({"cmd": "list"})
        objs = (st.get("objects") or []) if st.get("ok") else []
        rows = "\n".join(
            f'<li><a href="/probe/{o["id"]}?hold=2">id={o["id"]} '
            f'{o["name"]} (d={o.get("depth_m", 0):.2f}m)</a></li>'
            for o in objs
        ) or "<li><em>no objects</em></li>"
        return (
            "<!doctype html><title>LiveTracking remote probe</title>"
            "<style>body{font-family:system-ui;padding:1em}"
            "a{display:block;padding:0.6em;font-size:1.1em}</style>"
            "<h2>Probe -- tap an object to fire the projector + capture</h2>"
            f"<ul>{rows}</ul>"
        )


    # ---- re-calibrate --------------------------------------------------
    # Calibration must run in the user's desktop session (it grabs display
    # 1 + the RealSense). Flask runs as LocalSystem in Session 0, so we
    # trigger a pre-registered scheduled task (LiveTrackingCalibrate) and
    # let it orchestrate stop -> calibrate -> restart of the other tasks.
    CALIB_STATUS_FILE = os.path.join(RUNTIME_DIR, "calibration_status.json")

    @app.route("/recalibrate", methods=["POST"])
    def recalibrate():
        try:
            # Reset status so the UI sees fresh state immediately.
            os.makedirs(os.path.dirname(CALIB_STATUS_FILE), exist_ok=True)
            with open(CALIB_STATUS_FILE, "w") as f:
                json.dump({"phase": "starting", "ok": None,
                           "detail": "", "t": time.time()}, f)
            r = subprocess.run(
                ["schtasks", "/run", "/tn", "LiveTrackingCalibrate"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                return jsonify({
                    "ok": False,
                    "reason": (r.stdout + r.stderr).strip()[-400:],
                })
            return jsonify({"ok": True, "started": True})
        except Exception as e:
            return jsonify({"ok": False, "reason": repr(e)})

    @app.route("/calibrate_status")
    def calibrate_status():
        try:
            with open(CALIB_STATUS_FILE) as f:
                return jsonify({"ok": True, "status": json.load(f)})
        except FileNotFoundError:
            return jsonify({"ok": True, "status": None})
        except Exception as e:
            return jsonify({"ok": False, "reason": repr(e)})

    PARALLAX_STATUS_FILE = os.path.join(
        RUNTIME_DIR, "parallax_calibration_status.json"
    )

    @app.route("/parallax_calibrate", methods=["POST"])
    def parallax_calibrate():
        """Kick off scripts/run_parallax_calibration.py via the
        LiveTrackingParallaxCalibrate scheduled task (Interactive, Highest).

        The task itself stops perception+projector, runs the manual two-
        plane alignment UI on the projector, writes H_wall.npy / H_near.npy
        / parallax_depths.json under runtime/calibration/, and restarts the
        daemons.
        """
        try:
            os.makedirs(os.path.dirname(PARALLAX_STATUS_FILE), exist_ok=True)
            with open(PARALLAX_STATUS_FILE, "w") as f:
                json.dump({"phase": "starting", "ok": None,
                           "detail": "", "t": time.time()}, f)
            r = subprocess.run(
                ["schtasks", "/run", "/tn", "LiveTrackingParallaxCalibrate"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                err = (r.stdout + r.stderr).strip()
                # Friendly hint when the task hasn't been registered yet.
                if ("cannot find" in err.lower()
                        or "not found" in err.lower()):
                    err = ("LiveTrackingParallaxCalibrate task is not "
                           "registered. Run scripts/install_services.ps1 "
                           "from an admin PowerShell to register it.")
                return jsonify({"ok": False, "reason": err[-400:]})
            return jsonify({"ok": True, "started": True})
        except Exception as e:
            return jsonify({"ok": False, "reason": repr(e)})

    @app.route("/parallax_status")
    def parallax_status():
        try:
            with open(PARALLAX_STATUS_FILE) as f:
                return jsonify({"ok": True, "status": json.load(f)})
        except FileNotFoundError:
            return jsonify({"ok": True, "status": None})
        except Exception as e:
            return jsonify({"ok": False, "reason": repr(e)})

    @app.route("/hide", methods=["POST"])
    def hide():
        data = request.get_json(silent=True) or {}
        return jsonify(_send_ctrl({"cmd": "hide", "id": int(data["id"])}))

    @app.route("/unhide", methods=["POST"])
    def unhide():
        data = request.get_json(silent=True) or {}
        return jsonify(_send_ctrl({"cmd": "unhide", "id": int(data["id"])}))

    @app.route("/unhide_all", methods=["POST"])
    def unhide_all():
        return jsonify(_send_ctrl({"cmd": "unhide_all"}))

    @app.route("/hidden_list")
    def hidden_list():
        return jsonify(_send_ctrl({"cmd": "hidden_list"}))

    @app.route("/detector", methods=["GET"])
    def detector_get():
        """Return both the persisted selection and the running daemon's
        selection. They differ briefly during a restart."""
        from livetracking.perception.recognize import (
            VALID_DETECTORS, read_active_detector,
        )
        persisted = read_active_detector()
        live = _send_ctrl({"cmd": "detector_info"}, timeout_ms=1500)
        return jsonify({
            "ok": True,
            "persisted": persisted,
            "live": (live.get("detector") if live.get("ok") else None),
            "choices": list(VALID_DETECTORS),
        })

    @app.route("/detector", methods=["POST"])
    def detector_set():
        """Persist a new detector choice and restart the perception task."""
        from livetracking.perception.recognize import (
            VALID_DETECTORS, read_active_detector, write_active_detector,
        )
        data = request.get_json(silent=True) or {}
        name = str(data.get("detector", "")).lower()
        if name not in VALID_DETECTORS:
            return jsonify({
                "ok": False,
                "reason": f"detector must be one of {list(VALID_DETECTORS)}",
            }), 400
        current = read_active_detector()
        if name == current:
            return jsonify({
                "ok": True, "detector": name, "restarted": False,
                "reason": "already active",
            })
        try:
            write_active_detector(name)
        except Exception as e:
            return jsonify({"ok": False, "reason": repr(e)}), 500
        # Bounce the perception task. The Windows reality is uglier than
        # `schtasks /end` would suggest: /end reports SUCCESS but routinely
        # leaves the python child orphaned (still holding the D455 + the
        # ctrl socket, but no longer tracked by Task Scheduler — Status:
        # Ready). The next /run then either no-ops or starts a second
        # daemon that immediately fails to bind. To make the switch
        # reliable we explicitly hunt down any python.exe whose command
        # line invokes the perception daemon and taskkill /F /T it before
        # re-running the task.
        try:
            subprocess.run(
                ["schtasks", "/end", "/tn", "LiveTrackingPerception"],
                capture_output=True, text=True, timeout=10,
            )
            # Kill any orphaned perception python.exe by command-line
            # match. We pull the list via PowerShell's CIM (wmic is
            # deprecated on modern Windows) and pipe each PID through
            # taskkill /F /T so the whole tree dies.
            ps_cmd = (
                "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" "
                "| Where-Object { $_.CommandLine -like "
                "'*livetracking.daemon.perception*' } "
                "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force "
                "-ErrorAction SilentlyContinue }"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True, text=True, timeout=15,
            )
            time.sleep(2)
            r = subprocess.run(
                ["schtasks", "/run", "/tn", "LiveTrackingPerception"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                return jsonify({
                    "ok": False,
                    "detector": name,
                    "restarted": False,
                    "reason": (r.stdout + r.stderr).strip()[-400:],
                })
        except Exception as e:
            return jsonify({
                "ok": False,
                "detector": name,
                "restarted": False,
                "reason": repr(e),
            })
        return jsonify({"ok": True, "detector": name, "restarted": True})

    @app.route("/healthz")
    def healthz():
        return jsonify({
            "ok": True,
            "has_objects": STATE.get_objects() is not None,
            "has_frame": STATE.get_jpeg() is not None,
        })

    # ---- remote-ops endpoints ---------------------------------------
    # Added 2026-06-02 because Ted is away from the rig for a week and
    # needs to tune K live, restart a hung daemon, and tail logs without
    # being physically present.

    @app.route("/parallax", methods=["GET"])
    def parallax_get():
        """Return the live parallax config from the perception pipeline."""
        return jsonify(_send_ctrl({"cmd": "parallax_get"}))

    @app.route("/parallax", methods=["POST"])
    def parallax_tune():
        """Mutate the live parallax config WITHOUT restarting perception.
        Body: {compensate?: bool, sign?: float[-1,1], scale?: float[0,10],
               k_px_m?: float[0,10000]}. All keys optional; missing keys
        are left untouched. Returns {ok, changed, current}."""
        data = request.get_json(silent=True) or {}
        payload = {"cmd": "parallax_tune"}
        for k in ("compensate", "sign", "scale", "k_px_m"):
            if k in data:
                payload[k] = data[k]
        return jsonify(_send_ctrl(payload))

    @app.route("/mask", methods=["GET"])
    def mask_get():
        """Return the live mask-smoothing config from the perception
        pipeline."""
        return jsonify(_send_ctrl({"cmd": "mask_get"}))

    @app.route("/mask", methods=["POST"])
    def mask_tune():
        """Mutate the live mask-smoothing config WITHOUT restarting
        perception. Body: {smooth_px?: int[0,25]}. smooth_px controls
        the Gaussian kernel half-width applied to SAM masks before
        warping; 0 = sharp/pixelated, 3 = soft, 7 = very soft, 12+ =
        airy glow. Returns {ok, changed, current}."""
        data = request.get_json(silent=True) or {}
        payload = {"cmd": "mask_tune"}
        if "smooth_px" in data:
            payload["smooth_px"] = data["smooth_px"]
        return jsonify(_send_ctrl(payload))

    @app.route("/perception/restart", methods=["POST"])
    def perception_restart():
        """End + re-run the LiveTrackingPerception scheduled task. The
        existing detector-change endpoint already does this dance; we
        expose it standalone for remote 'kick the daemon' moments."""
        try:
            subprocess.run(
                ["schtasks", "/end", "/tn", "LiveTrackingPerception"],
                capture_output=True, text=True, timeout=10,
            )
            time.sleep(2)
            r = subprocess.run(
                ["schtasks", "/run", "/tn", "LiveTrackingPerception"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                return jsonify({
                    "ok": False,
                    "reason": (r.stdout + r.stderr).strip()[-400:],
                })
        except Exception as e:  # noqa: BLE001
            return jsonify({"ok": False, "reason": repr(e)})
        return jsonify({"ok": True, "restarted": True})

    @app.route("/logs/<service>")
    def logs_tail(service: str):
        """Tail the last ?n=200 lines of a service log. Allowed services:
        perception, projector, flame_web. Reads from runtime/service-logs/.
        Capped at 1000 lines / 200 KB to keep responses sane over Cloudflare."""
        import os
        from livetracking.paths import RUNTIME_DIR
        ALLOWED = {
            "perception": "perception.log",
            "projector": "projector.log",
            "flame_web": "flame_web.log",
            "calibrate": "calibrate.log",
            "parallax_calibrate": "parallax_calibrate.log",
        }
        if service not in ALLOWED:
            return jsonify({"ok": False,
                            "reason": f"unknown service; allowed: "
                                       f"{sorted(ALLOWED)}"}), 400
        try:
            n = min(int(request.args.get("n", 200)), 1000)
        except ValueError:
            n = 200
        log_dir = os.path.join(RUNTIME_DIR, "service-logs")
        log_path = os.path.join(log_dir, ALLOWED[service])
        if not os.path.exists(log_path):
            # Fall back to the most-recent stderr-dated log for this service
            # (NSSM rotates with timestamps for flame_web).
            cands = []
            try:
                for fn in os.listdir(log_dir):
                    if fn.startswith(service):
                        cands.append((os.path.getmtime(
                            os.path.join(log_dir, fn)),
                            os.path.join(log_dir, fn)))
            except OSError:
                cands = []
            if not cands:
                return jsonify({"ok": False,
                                "reason": f"no log file for {service}"}), 404
            cands.sort(reverse=True)
            log_path = cands[0][1]
        # Read last n lines (capped at 200 KB).
        try:
            with open(log_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                chunk_size = min(size, 200_000)
                f.seek(max(0, size - chunk_size))
                tail = f.read().decode("utf-8", errors="replace")
            lines = tail.splitlines()[-n:]
        except OSError as e:
            return jsonify({"ok": False, "reason": repr(e)}), 500
        return jsonify({
            "ok": True,
            "service": service,
            "path": log_path,
            "lines": len(lines),
            "content": "\n".join(lines),
        })

    @app.route("/services/status")
    def services_status():
        """Compact view of the three scheduled-task lifecycles + Flask itself.
        Useful for a remote dashboard: hit one URL, see what's running."""
        out = {}
        for task in ("LiveTrackingPerception", "LiveTrackingProjector",
                     "LiveTrackingCalibrate", "LiveTrackingParallaxCalibrate"):
            try:
                r = subprocess.run(
                    ["schtasks", "/query", "/tn", task, "/fo", "LIST", "/v"],
                    capture_output=True, text=True, timeout=5,
                )
                if r.returncode != 0:
                    out[task] = {"present": False}
                    continue
                lines = (r.stdout or "").splitlines()
                info = {"present": True}
                for ln in lines:
                    if ":" not in ln:
                        continue
                    k, _, v = ln.partition(":")
                    k = k.strip()
                    if k in ("Status", "Last Result", "Last Run Time"):
                        info[k.lower().replace(" ", "_")] = v.strip()
                out[task] = info
            except Exception as e:  # noqa: BLE001
                out[task] = {"present": False, "error": repr(e)}
        out["flame_web"] = {"ok": True}
        return jsonify({"ok": True, "tasks": out})

    return app


def main() -> int:
    print(f"[web] starting on :{WEB_UI_PORT}, subscribing to {ZMQ_OBJECTS_PUB}")
    t = threading.Thread(target=_zmq_subscriber, daemon=True)
    t.start()
    app = create_app()
    # threaded=True so MJPEG + SSE + POSTs can coexist
    app.run(host="0.0.0.0", port=WEB_UI_PORT, threaded=True, debug=False)
    return 0


if __name__ == "__main__":
    main()

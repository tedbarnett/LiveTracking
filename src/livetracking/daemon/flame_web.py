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
import threading
import time
from typing import Optional

import zmq
from flask import Flask, Response, jsonify, render_template, request

from livetracking.paths import WEB_UI_PORT, ZMQ_OBJECTS_PUB


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

    @app.route("/healthz")
    def healthz():
        return jsonify({
            "ok": True,
            "has_objects": STATE.get_objects() is not None,
            "has_frame": STATE.get_jpeg() is not None,
        })

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

"""Flame web UI - one-button trigger for the guitar flame pipeline.

Serves a single page on http://localhost:5070 (or $LIVETRACKING_WEB_PORT). The
page has a "Run Flame" button that launches ``scripts/flame_on_mask.py`` as a
subprocess. Live stdout streams to the page; result images render below when
the run finishes.

Single-process, subprocess-based: no daemon, no shared camera. Each click
re-grabs the D455 and re-opens the projector window. Trade-off matches the
hardware reality: D455 + projector are exclusive resources, and re-init takes
~3 sec on top of the ~40 sec script run. Re-runnable from the web without
restarting anything on the laptop.
"""
from __future__ import annotations

import os
import sys
import subprocess
import threading
import time
from collections import deque

from flask import Flask, jsonify, request, send_file, abort, Response

_HERE = os.path.dirname(os.path.abspath(__file__))
# src/livetracking/daemon/flame_web.py -> repo root is 3 levels up.
_PRELIM_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if os.path.join(_PRELIM_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_PRELIM_ROOT, "src"))
from livetracking import paths as P  # noqa: E402

_REPO_ROOT = P.REPO_ROOT

SCRIPT = os.path.join(_REPO_ROOT, "scripts", "flame_on_mask.py")
OUT_DIR = os.path.join(_REPO_ROOT, "scripts", "out")
PYTHON = os.environ.get(
    "LIVETRACKING_PYTHON",
    os.path.join(_REPO_ROOT, ".venv", "Scripts", "python.exe"),
)

app = Flask(__name__)

# ------------ run state ------------

_state_lock = threading.Lock()
_state = {
    "running": False,
    "started_at": None,
    "ended_at": None,
    "exit_code": None,
    "lines": deque(maxlen=400),  # tail of recent stdout/stderr
    "metrics": None,             # parsed METRICS line if found
    "run_id": 0,                 # bumped each run so the page can detect new artifacts
}


def _reader(proc):
    """Drain stdout in a background thread."""
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\r\n")
            with _state_lock:
                _state["lines"].append(line)
                if line.startswith("METRICS:"):
                    try:
                        import json as _json
                        _state["metrics"] = _json.loads(line[len("METRICS:"):].strip())
                    except Exception:
                        pass
    except Exception as e:
        with _state_lock:
            _state["lines"].append(f"[reader error] {e}")
    finally:
        proc.wait()
        with _state_lock:
            _state["running"] = False
            _state["ended_at"] = time.time()
            _state["exit_code"] = proc.returncode


def _launch():
    """Start flame_on_mask.py as a subprocess. Returns False if already running."""
    with _state_lock:
        if _state["running"]:
            return False
        _state["running"] = True
        _state["started_at"] = time.time()
        _state["ended_at"] = None
        _state["exit_code"] = None
        _state["lines"].clear()
        _state["metrics"] = None
        _state["run_id"] += 1
        _state["lines"].append(f"$ {PYTHON} {SCRIPT}")

    proc = subprocess.Popen(
        [PYTHON, SCRIPT],
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    t = threading.Thread(target=_reader, args=(proc,), daemon=True)
    t.start()
    return True


# ------------ routes ------------

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LiveTracking - Flame on Guitar</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#111111">
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         background: #111; color: #eee; margin: 0; padding: 20px;
         max-width: 1200px; margin: 0 auto; }
  h1 { margin: 0 0 4px 0; font-size: 28px; }
  .sub { color: #888; font-size: 14px; margin-bottom: 20px; }
  .controls { display: flex; gap: 10px; align-items: center; margin: 20px 0; }
  button { background: #c43; color: white; border: none; padding: 14px 28px;
           border-radius: 6px; cursor: pointer; font-size: 18px; font-weight: 600; }
  button:hover:not(:disabled) { background: #e54; }
  button:disabled { background: #555; cursor: not-allowed; }
  .status { font-size: 14px; color: #9cf; font-variant-numeric: tabular-nums; }
  .status .running { color: #fc6; }
  .status .ok { color: #6f6; }
  .status .fail { color: #f66; }
  pre.log { background: #000; border: 1px solid #333; border-radius: 6px;
            padding: 12px; font-size: 12px; color: #cfc; height: 220px;
            overflow-y: auto; white-space: pre-wrap; word-break: break-word;
            font-family: Consolas, Menlo, monospace; }
  .metrics { display: flex; gap: 18px; flex-wrap: wrap; margin: 12px 0;
             font-variant-numeric: tabular-nums; font-size: 14px; }
  .metric { background: #1c1c1c; padding: 8px 14px; border-radius: 4px;
            border: 1px solid #333; }
  .metric .k { color: #888; font-size: 11px; text-transform: uppercase;
               letter-spacing: 0.5px; }
  .metric .v { color: #eee; font-size: 18px; font-weight: 600; }
  .results { display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
             margin-top: 20px; }
  .results figure { margin: 0; }
  .results figcaption { color: #888; font-size: 12px; margin-bottom: 6px; }
  .results img { width: 100%; border: 1px solid #333; border-radius: 6px;
                 background: #000; }
  details { margin-top: 16px; }
  summary { color: #9cf; cursor: pointer; font-size: 13px; }
  .diag { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
          margin-top: 10px; }
  .diag img { width: 100%; border: 1px solid #333; border-radius: 4px;
              background: #000; }
  .diag figcaption { color: #777; font-size: 11px; }
</style>
</head>
<body>
  <h1>🎸🔥 Flame on Guitar</h1>
  <div class="sub">
    Project an animated blue flame onto a guitar. Each run does a 9-dot
    homography calibration, detects the guitar body mask, and projects the
    flame for 20 seconds. Watch the wall.
  </div>

  <div class="controls">
    <button id="runBtn" onclick="run()">▶ Run flame</button>
    <span class="status">
      <span id="phase">idle</span>
      <span id="elapsed"></span>
    </span>
  </div>

  <div class="metrics" id="metrics"></div>

  <pre class="log" id="log">(no run yet — click "Run flame")</pre>

  <div class="results">
    <figure>
      <figcaption>Result (camera view of the projection on the guitar)</figcaption>
      <img id="resultImg" src="" alt="" style="display:none">
    </figure>
    <figure>
      <figcaption>Montage (camera mask + warped projector mask)</figcaption>
      <img id="montageImg" src="" alt="" style="display:none">
    </figure>
  </div>

  <details>
    <summary>Diagnostics</summary>
    <div class="diag">
      <figure><figcaption>baseline (camera, projector black)</figcaption>
        <img id="diag_baseline" src=""></figure>
      <figure><figcaption>projector footprint in camera space</figcaption>
        <img id="diag_proj_quad" src=""></figure>
      <figure><figcaption>selected guitar body (camera frame)</figcaption>
        <img id="diag_cam_overlay" src=""></figure>
      <figure><figcaption>cam mask (binary)</figcaption>
        <img id="diag_cam_mask" src=""></figure>
      <figure><figcaption>mask warped to projector space</figcaption>
        <img id="diag_proj_mask" src=""></figure>
    </div>
  </details>

<script>
let lastRunId = 0;

async function run() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  try {
    await fetch('/api/run', {method: 'POST'});
  } catch (e) {
    document.getElementById('phase').textContent = 'error: ' + e;
  }
}

function fmtElapsed(s) {
  s = Math.floor(s);
  return s + 's';
}

async function refresh() {
  try {
    const r = await fetch('/api/status', {cache: 'no-store'});
    const s = await r.json();
    const phase = document.getElementById('phase');
    const elapsed = document.getElementById('elapsed');
    const btn = document.getElementById('runBtn');

    if (s.running) {
      phase.innerHTML = '<span class="running">▶ RUNNING</span>';
      elapsed.textContent = ' • ' + fmtElapsed(s.elapsed_s);
      btn.disabled = true;
    } else {
      btn.disabled = false;
      if (s.exit_code === null) {
        phase.textContent = 'idle';
        elapsed.textContent = '';
      } else if (s.exit_code === 0) {
        phase.innerHTML = '<span class="ok">✓ done</span>';
        elapsed.textContent = ' • ' + fmtElapsed(s.elapsed_s);
      } else {
        phase.innerHTML = '<span class="fail">✗ exit ' + s.exit_code + '</span>';
        elapsed.textContent = ' • ' + fmtElapsed(s.elapsed_s);
      }
    }

    document.getElementById('log').textContent = (s.lines || []).join('\\n');

    // Auto-scroll log to bottom while running.
    if (s.running) {
      const log = document.getElementById('log');
      log.scrollTop = log.scrollHeight;
    }

    // Metrics
    const mDiv = document.getElementById('metrics');
    if (s.metrics) {
      mDiv.innerHTML = Object.entries(s.metrics).map(([k, v]) =>
        '<div class="metric"><div class="k">' + k.replace(/_/g, ' ') +
        '</div><div class="v">' + v + '</div></div>'
      ).join('');
    } else {
      mDiv.innerHTML = '';
    }

    // Refresh images when a new run completes.
    if (s.run_id !== lastRunId && s.exit_code === 0) {
      lastRunId = s.run_id;
      const stamp = '?t=' + Date.now();
      const setImg = (id, name) => {
        const im = document.getElementById(id);
        im.src = '/api/out/' + name + stamp;
        im.style.display = '';
      };
      setImg('resultImg', 'flamemask_result.png');
      setImg('montageImg', 'flamemask.png');
      setImg('diag_baseline', 'fom_baseline.png');
      setImg('diag_proj_quad', 'fom_proj_quad_in_cam.png');
      setImg('diag_cam_overlay', 'fom_cam_mask_overlay.png');
      setImg('diag_cam_mask', 'fom_cam_mask.png');
      setImg('diag_proj_mask', 'fom_proj_mask.png');
    }
  } catch (e) {
    document.getElementById('phase').textContent = 'server offline';
  }
}

setInterval(refresh, 700);
refresh();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/run", methods=["POST"])
def api_run():
    ok = _launch()
    return jsonify({"ok": ok, "running": True if ok else "already running"})


@app.route("/api/status")
def api_status():
    with _state_lock:
        started = _state["started_at"]
        ended = _state["ended_at"]
        now = time.time()
        if started is None:
            elapsed = 0.0
        elif _state["running"]:
            elapsed = now - started
        else:
            elapsed = (ended or now) - started
        return jsonify({
            "running": _state["running"],
            "started_at": started,
            "ended_at": ended,
            "elapsed_s": round(elapsed, 1),
            "exit_code": _state["exit_code"],
            "lines": list(_state["lines"]),
            "metrics": _state["metrics"],
            "run_id": _state["run_id"],
        })


# Serve scripts/out/<filename> safely.
_ALLOWED = {
    "flamemask.png", "flamemask_result.png",
    "fom_baseline.png", "fom_proj_quad_in_cam.png",
    "fom_cam_mask.png", "fom_cam_mask_overlay.png",
    "fom_proj_mask.png",
}


@app.route("/api/out/<name>")
def api_out(name):
    if name not in _ALLOWED:
        abort(404)
    p = os.path.join(OUT_DIR, name)
    if not os.path.exists(p):
        abort(404)
    return send_file(p, mimetype="image/png", max_age=0)


if __name__ == "__main__":
    port = P.WEB_UI_PORT
    print(f"Flame web UI on http://localhost:{port}")
    print(f"  script: {SCRIPT}")
    print(f"  python: {PYTHON}")
    print(f"  out:    {OUT_DIR}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

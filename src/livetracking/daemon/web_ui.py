"""LiveTracking web UI.

Reads state.json + latest_frame.jpg from the daemon's runtime directory,
serves a single-page web UI on http://localhost:5060.

Endpoints:
  GET  /                  - HTML page
  GET  /api/state         - JSON: current targets, mode, heal count, uptime
  GET  /api/frame.jpg     - latest RealSense RGB frame (refreshed ~2x/sec by daemon)
  POST /api/command       - {"command": "restart"|"mode_plus"|"mode_fill"|"screenshot"|"quit"}

Run:
    python src/livetracking/daemon/web_ui.py
"""
import json
import os
import time

from flask import Flask, jsonify, request, send_file, abort

RUNTIME_DIR = r"D:\Github-D\LiveTracking\runtime"
STATE_FILE = os.path.join(RUNTIME_DIR, "state.json")
FRAME_FILE = os.path.join(RUNTIME_DIR, "latest_frame.jpg")
COMMAND_FILE = os.path.join(RUNTIME_DIR, "command.txt")

ALLOWED_COMMANDS = {"restart", "recalibrate", "mode_plus", "mode_fill",
                     "plus", "fill", "screenshot", "heal_now", "quit"}

app = Flask(__name__)


INDEX_HTML = """<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>LiveTracking</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         background: #111; color: #eee; margin: 0; padding: 20px; }
  h1 { margin: 0 0 10px 0; font-size: 22px; }
  .row { display: flex; gap: 20px; flex-wrap: wrap; }
  .col { flex: 1; min-width: 320px; }
  img.frame { width: 100%; max-width: 848px; border: 1px solid #333;
              border-radius: 6px; background: #000; }
  table { border-collapse: collapse; width: 100%; }
  th, td { padding: 6px 10px; border-bottom: 1px solid #333; text-align: left;
           font-variant-numeric: tabular-nums; }
  th { background: #1c1c1c; color: #9cf; font-weight: 500; }
  .controls { display: flex; gap: 10px; margin: 16px 0; flex-wrap: wrap; }
  button { background: #2d4a8a; color: white; border: none; padding: 10px 18px;
           border-radius: 5px; cursor: pointer; font-size: 14px; }
  button:hover { background: #3a5fb0; }
  button.active { background: #4a8acc; }
  button.danger { background: #8a3a3a; }
  button.danger:hover { background: #b04a4a; }
  .status { color: #9cf; font-size: 13px; margin-top: 6px;
            font-variant-numeric: tabular-nums; }
  .status .label { color: #888; margin-right: 6px; }
  .toast { position: fixed; right: 20px; bottom: 20px; background: #2d4a8a;
           color: white; padding: 10px 16px; border-radius: 5px; opacity: 0;
           transition: opacity 0.3s; pointer-events: none; }
  .toast.show { opacity: 1; }
</style>
</head>
<body>
  <h1>LiveTracking</h1>
  <div class=\"status\">
    <span class=\"label\">Mode:</span><span id=\"mode\">-</span>
    <span class=\"label\" style=\"margin-left:14px\">Uptime:</span><span id=\"uptime\">-</span>
    <span class=\"label\" style=\"margin-left:14px\">Heals:</span><span id=\"heals\">-</span>
    <span class=\"label\" style=\"margin-left:14px\">Targets:</span><span id=\"target_count\">-</span>
    <span class=\"label\" style=\"margin-left:14px\">Status:</span><span id=\"status\">-</span>
  </div>

  <div class=\"controls\">
    <button id=\"btn_fill\" onclick=\"sendCmd('mode_fill')\">Fill mode</button>
    <button id=\"btn_plus\" onclick=\"sendCmd('mode_plus')\">+ sign mode</button>
    <button onclick=\"sendCmd('screenshot')\">Screenshot</button>
    <button class=\"danger\" onclick=\"if (confirm('Re-detect all targets from scratch?')) sendCmd('restart')\">Restart / Recalibrate</button>
  </div>

  <div class=\"row\">
    <div class=\"col\">
      <h3 style=\"margin-top:0\">Camera view</h3>
      <img class=\"frame\" id=\"frame\" src=\"/api/frame.jpg\" alt=\"camera frame\">
    </div>
    <div class=\"col\">
      <h3 style=\"margin-top:0\">Tracked targets</h3>
      <table>
        <thead>
          <tr><th>#</th><th>Cam (x, y)</th><th>Proj (x, y)</th><th>Err px</th><th>Angle</th></tr>
        </thead>
        <tbody id=\"targets\"></tbody>
      </table>
    </div>
  </div>

  <div id=\"toast\" class=\"toast\">ok</div>

<script>
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 1500);
}

async function sendCmd(cmd) {
  try {
    const r = await fetch('/api/command', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({command: cmd}),
    });
    const j = await r.json();
    if (j.ok) showToast(cmd + ' \u2713'); else showToast('error: ' + (j.error || 'unknown'));
  } catch (e) {
    showToast('error: ' + e);
  }
}

function fmtUptime(s) {
  s = Math.floor(s);
  const m = Math.floor(s / 60);
  const ss = s % 60;
  if (m === 0) return ss + 's';
  return m + 'm ' + ss + 's';
}

async function refresh() {
  try {
    const r = await fetch('/api/state', {cache: 'no-store'});
    const s = await r.json();
    document.getElementById('mode').textContent = s.render_mode || '-';
    document.getElementById('uptime').textContent = fmtUptime(s.uptime_s || 0);
    document.getElementById('heals').textContent = s.heal_count != null ? s.heal_count : '-';
    document.getElementById('target_count').textContent = s.target_count != null ? s.target_count : '-';
    document.getElementById('status').textContent = s.last_status || '-';

    document.getElementById('btn_fill').classList.toggle('active', s.render_mode === 'fill');
    document.getElementById('btn_plus').classList.toggle('active', s.render_mode === 'plus');

    const tbody = document.getElementById('targets');
    tbody.innerHTML = '';
    (s.targets || []).forEach(t => {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td>' + t.index + '</td>' +
                      '<td>(' + t.cam_xy[0] + ', ' + t.cam_xy[1] + ')</td>' +
                      '<td>(' + t.proj_xy[0] + ', ' + t.proj_xy[1] + ')</td>' +
                      '<td>' + t.err_px + '</td>' +
                      '<td>' + t.angle_deg + '\u00b0</td>';
      tbody.appendChild(tr);
    });
  } catch (e) {
    document.getElementById('status').textContent = 'daemon offline';
  }
  // Refresh frame by bumping a cache-buster query.
  const img = document.getElementById('frame');
  img.src = '/api/frame.jpg?t=' + Date.now();
}

setInterval(refresh, 700);
refresh();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/api/state")
def api_state():
    if not os.path.exists(STATE_FILE):
        return jsonify({"error": "daemon not running"}), 503
    try:
        with open(STATE_FILE) as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/frame.jpg")
def api_frame():
    if not os.path.exists(FRAME_FILE):
        abort(404)
    return send_file(FRAME_FILE, mimetype="image/jpeg",
                      max_age=0)


@app.route("/api/command", methods=["POST"])
def api_command():
    data = request.get_json(silent=True) or {}
    cmd = (data.get("command") or "").strip().lower()
    if cmd not in ALLOWED_COMMANDS:
        return jsonify({"ok": False, "error": f"unknown command: {cmd}"}), 400
    try:
        with open(COMMAND_FILE, "w") as f:
            f.write(cmd)
        return jsonify({"ok": True, "command": cmd, "queued_at": time.time()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    print(f"LiveTracking web UI on http://localhost:5070")
    print(f"  state: {STATE_FILE}")
    print(f"  frame: {FRAME_FILE}")
    print(f"  cmd:   {COMMAND_FILE}")
    app.run(host="0.0.0.0", port=5070, debug=False, threaded=True)

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
import re
import time

from flask import Flask, jsonify, request, send_file, abort, send_from_directory, Response

RUNTIME_DIR = r"D:\Github-D\LiveTracking\runtime"
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
STATE_FILE = os.path.join(RUNTIME_DIR, "state.json")
FRAME_FILE = os.path.join(RUNTIME_DIR, "latest_frame.jpg")
COMMAND_FILE = os.path.join(RUNTIME_DIR, "command.txt")

ALLOWED_COMMANDS = {"restart", "recalibrate", "mode_plus", "mode_fill",
                     "plus", "fill", "screenshot", "heal_now", "quit",
                     "reset_nudges"}
NUDGE_RE = re.compile(r"^nudge_[123]_(left|right|up|down|reset)(?:_\d+)?$")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")


WEBMANIFEST = {
    "name": "LiveTracking",
    "short_name": "LiveTrack",
    "description": "Self-calibrating projection mapping (Cobblestone Labs).",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#111111",
    "theme_color": "#111111",
    "icons": [
        {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png"},
        {"src": "/static/icon-256.png", "sizes": "256x256", "type": "image/png"},
        {"src": "/static/icon-384.png", "sizes": "384x384", "type": "image/png"},
        {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png"},
    ],
}


INDEX_HTML = """<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>LiveTracking</title>
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<meta name=\"theme-color\" content=\"#111111\">
<link rel=\"icon\" type=\"image/x-icon\" href=\"/static/favicon.ico\">
<link rel=\"icon\" type=\"image/png\" sizes=\"32x32\" href=\"/static/favicon-32.png\">
<link rel=\"apple-touch-icon\" sizes=\"180x180\" href=\"/static/icon-180.png\">
<link rel=\"manifest\" href=\"/manifest.webmanifest\">
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         background: #111; color: #eee; margin: 0; padding: 8px 10px; font-size: 13px; }
  .topbar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
            margin-bottom: 6px; }
  .topbar h1 { margin: 0; font-size: 15px; color: #9cf; font-weight: 500;
               letter-spacing: 0.5px; }
  .status { color: #9cf; font-size: 11px;
            font-variant-numeric: tabular-nums; }
  .status .label { color: #666; margin-right: 3px; }
  .status .sep { color: #333; margin: 0 6px; }
  .controls { display: flex; gap: 6px; margin-left: auto; flex-wrap: wrap; }
  .controls button { min-width: 40px; height: 36px; font-size: 18px;
                     padding: 0 12px; }
  button { background: #2d4a8a; color: white; border: none; padding: 8px 12px;
           border-radius: 4px; cursor: pointer; font-size: 14px; }
  button:hover { background: #3a5fb0; }
  button.active { background: #4a8acc; }
  button.danger { background: #8a3a3a; }
  button.danger:hover { background: #b04a4a; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: flex-start; }
  .col-cam { flex: 1 1 480px; min-width: 320px; }
  .col-tbl { flex: 0 1 auto; min-width: 320px; }
  img.frame { width: 100%; max-width: 720px; border: 1px solid #333;
              border-radius: 4px; background: #000; display: block; }
  table { border-collapse: collapse; font-size: 11px; }
  th, td { padding: 2px 5px; border-bottom: 1px solid #222; text-align: left;
           font-variant-numeric: tabular-nums; white-space: nowrap; }
  th { background: #1c1c1c; color: #888; font-weight: 500; font-size: 10px;
       text-transform: uppercase; letter-spacing: 0.5px; }
  tr.target-1 td { color: #ff7070; }
  tr.target-2 td { color: #70ff70; }
  tr.target-3 td { color: #80a0ff; }
  tr.target-1 td:first-child,
  tr.target-2 td:first-child,
  tr.target-3 td:first-child { font-weight: 700; }
  .nudge-grid { display: inline-grid;
                grid-template-columns: 28px 28px 28px;
                grid-template-rows: 22px 22px 22px;
                gap: 2px; vertical-align: middle; }
  .nudge-grid button { padding: 0; font-size: 14px; line-height: 1;
                       background: #2d4a8a; color: #eee; border: 0;
                       border-radius: 3px; cursor: pointer; min-width: 0;
                       height: 22px; }
  .nudge-grid button:hover { background: #3a5fb0; }
  .nudge-grid .empty { visibility: hidden; }
  .nudge-grid .nudge-reset { background: #444; font-size: 11px; }
  .nudge-label { font-size: 9px; color: #666; margin-left: 4px;
                 font-variant-numeric: tabular-nums; display: inline-block;
                 vertical-align: middle; }
  .toast { position: fixed; right: 12px; bottom: 12px; background: #2d4a8a;
           color: white; padding: 6px 10px; font-size: 11px; border-radius: 4px;
           opacity: 0; transition: opacity 0.3s; pointer-events: none; }
  .toast.show { opacity: 1; }
</style>
</head>
<body>
  <div class=\"topbar\">
    <h1>LiveTracking</h1>
    <div class=\"status\">
      <span id=\"status\">-</span>
      <span class=\"sep\">·</span>
      <span class=\"label\">mode</span><span id=\"mode\">-</span>
      <span class=\"sep\">·</span>
      <span class=\"label\">up</span><span id=\"uptime\">-</span>
      <span class=\"sep\">·</span>
      <span class=\"label\">heals</span><span id=\"heals\">-</span>
      <span class=\"sep\">·</span>
      <span class=\"label\">n</span><span id=\"target_count\">-</span>
    </div>
    <div class=\"controls\">
      <button id=\"btn_plus\" onclick=\"sendCmd('mode_plus')\" title=\"+ sign mode\">+</button>
      <button id=\"btn_fill\" onclick=\"sendCmd('mode_fill')\" title=\"Fill mode\">■</button>
      <button onclick=\"sendCmd('screenshot')\" title=\"Screenshot\">📷</button>
      <button class=\"danger\" onclick=\"if (confirm('Re-detect all targets from scratch?')) sendCmd('restart')\" title=\"Restart / Recalibrate\">↻</button>
    </div>
  </div>

  <div class=\"row\">
    <div class=\"col-cam\">
      <img class=\"frame\" id=\"frame\" src=\"/api/frame.jpg\" alt=\"camera frame\">
    </div>
    <div class=\"col-tbl\">
      <table>
        <thead>
          <tr><th>#</th><th>Cam</th><th>Proj</th><th>Err</th><th>°</th><th>Nudge</th></tr>
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
      tr.className = 'target-' + t.index;
      const idx = t.index;
      const nx = (t.nudge ? t.nudge[0] : 0);
      const ny = (t.nudge ? t.nudge[1] : 0);
      const grid = '<div class=\"nudge-grid\">' +
        '<span class=\"empty\"></span>' +
          '<button onclick=\"sendCmd(\\'nudge_' + idx + '_up\\')\">\u25b2</button>' +
          '<span class=\"empty\"></span>' +
        '<button onclick=\"sendCmd(\\'nudge_' + idx + '_left\\')\">\u25c0</button>' +
          '<button class=\"nudge-reset\" onclick=\"sendCmd(\\'nudge_' + idx + '_reset\\')\">0</button>' +
          '<button onclick=\"sendCmd(\\'nudge_' + idx + '_right\\')\">\u25b6</button>' +
        '<span class=\"empty\"></span>' +
          '<button onclick=\"sendCmd(\\'nudge_' + idx + '_down\\')\">\u25bc</button>' +
          '<span class=\"empty\"></span>' +
        '</div>';
      const cx = Math.round(t.cam_xy[0]), cy = Math.round(t.cam_xy[1]);
      const px = Math.round(t.proj_xy[0]), py = Math.round(t.proj_xy[1]);
      const err = (typeof t.err_px === 'number') ? t.err_px.toFixed(1) : t.err_px;
      const ang = (typeof t.angle_deg === 'number') ? Math.round(t.angle_deg) : t.angle_deg;
      tr.innerHTML = '<td>' + t.index + '</td>' +
                      '<td>' + cx + ',' + cy + '</td>' +
                      '<td>' + px + ',' + py + '</td>' +
                      '<td>' + err + '</td>' +
                      '<td>' + ang + '</td>' +
                      '<td>' + grid + '<span class=\"nudge-label\">(' + nx + ',' + ny + ')</span></td>';
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


@app.route("/manifest.webmanifest")
def manifest():
    return Response(json.dumps(WEBMANIFEST), mimetype="application/manifest+json")


@app.route("/favicon.ico")
def favicon_root():
    return send_from_directory(STATIC_DIR, "favicon.ico",
                                  mimetype="image/x-icon")


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
    if cmd not in ALLOWED_COMMANDS and not NUDGE_RE.match(cmd):
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

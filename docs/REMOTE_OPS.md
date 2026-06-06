# Remote operations — quick reference

For when Ted is away from the rig. All commands hit
`https://livetracking.barnettlabs.tech/` (or `http://127.0.0.1:5070`
when on the laptop).

## Status / health

```bash
# Is everything alive?
curl -s https://livetracking.barnettlabs.tech/state
# -> {"detector":"yoloworld","ok":true,"paused":false}

# Detailed task state
curl -s https://livetracking.barnettlabs.tech/services/status | jq .

# Live frame
curl -o /tmp/snap.jpg https://livetracking.barnettlabs.tech/snapshot.jpg
open /tmp/snap.jpg

# Live objects (with depth, score, color)
curl -s https://livetracking.barnettlabs.tech/objects.json | jq .
```

## Tune parallax LIVE (no restart)

```bash
# See current config
curl -s https://livetracking.barnettlabs.tech/parallax | jq .

# Bump K (effective baseline; pixels·meters). Default 1200.
curl -X POST -H 'Content-Type: application/json' \
     -d '{"k_px_m": 1500}' \
     https://livetracking.barnettlabs.tech/parallax

# Flip parallax direction
curl -X POST -H 'Content-Type: application/json' \
     -d '{"sign": -1}' \
     https://livetracking.barnettlabs.tech/parallax

# Disable parallax entirely (sanity test)
curl -X POST -H 'Content-Type: application/json' \
     -d '{"compensate": false}' \
     https://livetracking.barnettlabs.tech/parallax
```

Values are clamped to safe ranges (k_px_m ∈ [0, 10000], sign ∈ [-1, 1],
scale ∈ [0, 10]). Changes take effect on the next frame, no restart.

## Tune mask edge softness LIVE (no restart)

```bash
# See current value
curl -s https://livetracking.barnettlabs.tech/mask
# -> {"ok":true,"smooth_px":3}

# Sharper (pixelated; SAM mask edges raw)
curl -X POST -H 'Content-Type: application/json' \
     -d '{"smooth_px": 0}' \
     https://livetracking.barnettlabs.tech/mask

# Softer / glowy
curl -X POST -H 'Content-Type: application/json' \
     -d '{"smooth_px": 10}' \
     https://livetracking.barnettlabs.tech/mask
```

`smooth_px` is the Gaussian kernel half-width (`(2N+1)²` kernel) applied
to SAM masks BEFORE the warp to projector space. Clamped to [0, 25].
Browser UI exposes the same control as the "Edge softness" slider.

Both `/mask` and `/parallax` POSTs trigger an immediate re-warp + re-push
of the currently-shown highlight, so the projected wash updates on the
slider release rather than waiting for the next user hover.

## Tail a service log

```bash
# Last 200 lines of perception
curl -s 'https://livetracking.barnettlabs.tech/logs/perception?n=200' \
     | jq -r .content

# Other services
curl -s 'https://livetracking.barnettlabs.tech/logs/projector?n=100'
curl -s 'https://livetracking.barnettlabs.tech/logs/flame_web?n=100'
curl -s 'https://livetracking.barnettlabs.tech/logs/parallax_calibrate?n=200'
```

## Restart perception remotely

```bash
# Kicks the LiveTrackingPerception scheduled task; daemon comes back
# in ~15-20 s. Use when perception hangs or gets stuck.
curl -X POST https://livetracking.barnettlabs.tech/perception/restart
```

Note: this does NOT restart Flask itself (NSSM-managed, needs admin).
If Flask hangs, you'll need to ask someone with physical access to run
`nssm restart LiveTrackingFlameWeb` from an admin PowerShell.

## Pause / resume the demo

```bash
curl -X POST https://livetracking.barnettlabs.tech/pause
curl -X POST https://livetracking.barnettlabs.tech/run
```

## Run parallax calibration remotely

```bash
# Stops perception+projector, runs the manual two-plane UI on the JMGO,
# restarts everything. Someone WITH PHYSICAL ACCESS to the projector
# still has to drive the keyboard (arrows / +-/ [] / click / Enter).
curl -X POST https://livetracking.barnettlabs.tech/parallax_calibrate

# Poll status
curl -s https://livetracking.barnettlabs.tech/parallax_status | jq .
```

## Run the live-rig integration tests from afar

```bash
# Logs into the laptop via Cloudflare Access (or SSH) first, then:
cd /c/Users/timew/Github/LiveTracking
./.venv/Scripts/python.exe -m pytest -m hardware -v
```

This pokes Flask end-to-end: snapshot freshness, plausible object data,
rename round-trip, pause/resume cycle, calibration files present, and
all the remote-ops endpoints above.

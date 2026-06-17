"""GPU shader effects via moderngl (offscreen, headless).

Shadertoy fragment shaders run here against a standalone GL context, render
to an offscreen framebuffer at the object's (capped) bbox size, and read back
as a numpy RGB array on black — the same shape the numpy effects in
effects.py return, so they drop into the existing render_effect dispatch and
the projector's mask/alpha compositing unchanged.

Design notes:
* Black background is mandatory (projector adds light; dark texels vanish on
  the wall). The ported shaders already render flame on black.
* One lazily-created standalone context, shared across calls, guarded by a
  lock. The projector renders effects from a single thread, but the lock
  keeps us safe if that ever changes.
* The framebuffer is recreated only when the requested size changes (size is
  capped to ~360px upstream, and most objects hold a steady bbox, so this is
  rare). GL's origin is bottom-left, so we flip rows on readback.
* All GL work is wrapped: if the context can't be created (no GPU / headless
  driver missing), the renderer returns black and the caller falls back to a
  flat color. A shader effect must never crash the projector.
"""
from __future__ import annotations

import threading

import numpy as np

try:
    import moderngl
except Exception:  # pragma: no cover - optional dep; fall back to black
    moderngl = None


# Shadertoy-compatible vertex shader: a single full-screen triangle.
_VERT = """
#version 330
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

# Fragment-shader wrapper that supplies the Shadertoy uniforms (iResolution,
# iTime) and calls mainImage() with the pixel coordinate, exactly like the
# Shadertoy harness. SHADER bodies below define mainImage().
_FRAG_HEADER = """
#version 330
uniform vec3 iResolution;
uniform float iTime;
out vec4 fragColor_out;
"""

_FRAG_FOOTER = """
void main() {
    vec4 c = vec4(0.0);
    mainImage(c, gl_FragCoord.xy);
    fragColor_out = vec4(c.rgb, 1.0);
}
"""

# --- ported Shadertoy bodies (mainImage only) -------------------------------

# https://www.shadertoy.com/view/4ttGWM — raymarched flame on black.
# Authored by the Shadertoy community; renders fire that fades to pure black
# at the edges, so on the wall only the flame lights up the object.
_FIRE2_BODY = """
float noise(vec3 p) {
    vec3 i = floor(p);
    vec4 a = dot(i, vec3(1., 57., 21.)) + vec4(0., 57., 21., 78.);
    vec3 f = cos((p - i) * acos(-1.)) * (-.5) + .5;
    a = mix(sin(cos(a) * a), sin(cos(1. + a) * (1. + a)), f.x);
    a.xy = mix(a.xz, a.yw, f.y);
    return mix(a.x, a.y, f.z);
}
float sphere(vec3 p, vec4 spr) { return length(spr.xyz - p) - spr.w; }
float flame(vec3 p) {
    float d = sphere(p * vec3(1., .5, 1.), vec4(.0, -1., .0, 1.));
    return d + (noise(p + vec3(.0, iTime * 2., .0)) + noise(p * 3.) * .5) * .25 * (p.y);
}
float scene(vec3 p) { return min(100. - length(p), abs(flame(p))); }
vec4 raymarch(vec3 org, vec3 dir) {
    float d = 0.0, glow = 0.0, eps = 0.02;
    vec3 p = org;
    bool glowed = false;
    for (int i = 0; i < 64; i++) {
        d = scene(p) + eps;
        p += d * dir;
        if (d > eps) {
            if (flame(p) < .0) glowed = true;
            if (glowed) glow = float(i) / 64.;
        }
    }
    return vec4(p, glow);
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 v = -1.0 + 2.0 * fragCoord.xy / iResolution.xy;
    v.x *= iResolution.x / iResolution.y;
    vec3 org = vec3(0., -2., 4.);
    vec3 dir = normalize(vec3(v.x * 1.6, -v.y, -1.5));
    vec4 p = raymarch(org, dir);
    float glow = p.w;
    vec4 col = mix(vec4(1., .5, .1, 1.), vec4(0.1, .5, 1., 1.), p.y * .02 + .4);
    fragColor = mix(vec4(0.), col, pow(glow * 2., 4.));
}
"""

# Registry of shader-backed effects: name -> mainImage body.
_SHADER_BODIES = {
    "fire2": _FIRE2_BODY,
}

SHADER_EFFECTS = tuple(_SHADER_BODIES.keys())


class _ShaderRenderer:
    """Holds the standalone GL context + per-shader program, FBO, and VAO.
    Lazily initialized; all access serialized by an external lock."""

    def __init__(self) -> None:
        self.ctx = None
        self.programs: dict = {}
        self.vbo = None
        self.fbo = None
        self.fbo_size = (0, 0)
        self.failed = False

    def _ensure_ctx(self) -> bool:
        if self.failed:
            return False
        if self.ctx is not None:
            return True
        if moderngl is None:
            self.failed = True
            return False
        try:
            self.ctx = moderngl.create_standalone_context()
            # Full-screen triangle (covers clip space with one tri).
            verts = np.array(
                [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0], dtype="f4"
            )
            self.vbo = self.ctx.buffer(verts.tobytes())
            return True
        except Exception:
            self.failed = True
            self.ctx = None
            return False

    def _program(self, name: str):
        prog = self.programs.get(name)
        if prog is not None:
            return prog
        body = _SHADER_BODIES[name]
        frag = _FRAG_HEADER + body + _FRAG_FOOTER
        prog = self.ctx.program(vertex_shader=_VERT, fragment_shader=frag)
        vao = self.ctx.vertex_array(prog, [(self.vbo, "2f4", "in_pos")])
        self.programs[name] = (prog, vao)
        return self.programs[name]

    def _ensure_fbo(self, w: int, h: int) -> None:
        if self.fbo is not None and self.fbo_size == (w, h):
            return
        if self.fbo is not None:
            self.fbo.release()
        tex = self.ctx.texture((w, h), 3)
        self.fbo = self.ctx.framebuffer(color_attachments=[tex])
        self.fbo_size = (w, h)

    def render(self, name: str, w: int, h: int, t: float):
        """Render shader `name` at (w, h) for time `t`. Returns (h, w, 3)
        uint8 RGB on black, or None on any GL failure."""
        if not self._ensure_ctx():
            return None
        try:
            prog, vao = self._program(name)
            self._ensure_fbo(w, h)
            self.fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0)
            if "iResolution" in prog:
                prog["iResolution"].value = (float(w), float(h), 1.0)
            if "iTime" in prog:
                prog["iTime"].value = float(t)
            vao.render(mode=moderngl.TRIANGLES, vertices=3)
            data = self.fbo.read(components=3, dtype="f1")
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
            # GL origin is bottom-left; flip to image (top-left) orientation.
            return np.ascontiguousarray(arr[::-1])
        except Exception:
            return None


_renderer = _ShaderRenderer()
_lock = threading.Lock()


def is_shader_effect(name: str) -> bool:
    return name in _SHADER_BODIES


def render_shader(name: str, w: int, h: int, t: float):
    """Thread-safe shader render. Returns (h, w, 3) uint8 RGB on black, or
    None if the effect is unknown or GL is unavailable (caller falls back)."""
    if name not in _SHADER_BODIES:
        return None
    w, h = int(w), int(h)
    if w <= 0 or h <= 0:
        return None
    with _lock:
        return _renderer.render(name, w, h, t)

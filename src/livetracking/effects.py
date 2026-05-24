"""Effect renderer (ModernGL).

A single fragment shader composites:
  layer 0  background pattern  (RGB,   uploaded each frame)
  layer 1  selection overlay   (RGBA,  empty unless in selection mode)
  layer 2  effect on mask      (R8,    silhouette of the selected object)
  layer 3  text overlay        (RGBA,  FPS + mode banner)

Effect modes (u_effect):
   0 = none (background passes through)
   1 = fire        (animated fbm noise * mask, warm color ramp)
   2 = glow        (soft halo via mask box-blur, cool color, pulsing)
   3 = colorshift  (mask filled with hue rotating over time)

All inputs are in display resolution. Pygame surfaces upload top-down,
so we flip Y in the shader.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import moderngl


EFFECT_NONE = 0
EFFECT_FIRE = 1
EFFECT_GLOW = 2
EFFECT_COLORSHIFT = 3

EFFECT_NAMES = {
    EFFECT_NONE: "none",
    EFFECT_FIRE: "fire",
    EFFECT_GLOW: "glow",
    EFFECT_COLORSHIFT: "colorshift",
}

CYCLE_ORDER = [EFFECT_FIRE, EFFECT_GLOW, EFFECT_COLORSHIFT]


VERT_SHADER = """
#version 330
in vec2 in_vert;
out vec2 v_uv;
void main() {
    v_uv = (in_vert + 1.0) * 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

FRAG_SHADER = """
#version 330

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D pattern_tex;     // RGB  (background pattern)
uniform sampler2D selection_tex;   // RGBA (object highlight halos)
uniform sampler2D mask_tex;        // R8   (selected-object silhouette)
uniform sampler2D overlay_tex;     // RGBA (text overlay)

uniform float u_time;
uniform int   u_effect;
uniform int   u_has_mask;
uniform vec2  u_inv_res;

// --- noise helpers --------------------------------------------------
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += a * vnoise(p);
        p *= 2.03;
        a *= 0.5;
    }
    return v;
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Pygame surface is top-down; flip Y when sampling.
vec2 flip(vec2 uv) { return vec2(uv.x, 1.0 - uv.y); }

void main() {
    vec2 uv = v_uv;
    vec2 fuv = flip(uv);

    vec3 col = texture(pattern_tex, fuv).rgb;

    vec4 sel = texture(selection_tex, fuv);
    col = mix(col, sel.rgb, sel.a);

    if (u_has_mask == 1) {
        float m = texture(mask_tex, fuv).r;

        if (u_effect == 1) {
            // FIRE: noise scrolling upward, modulated by mask.
            vec2 fp = uv * vec2(6.0, 4.0);
            fp.y -= u_time * 1.4;
            float n = fbm(fp + vec2(u_time * 0.25, 0.0));
            float grad = 0.4 + 0.6 * (1.0 - uv.y);
            float intensity = n * grad * m * 1.6;
            vec3 fire = vec3(1.0, 0.45, 0.08) * intensity;
            fire += vec3(1.0, 0.95, 0.55) * pow(intensity, 2.2);
            col = col + fire * m;
        } else if (u_effect == 2) {
            // GLOW: soft box-blurred halo around the mask.
            float halo = 0.0;
            const int R = 5;
            for (int i = -R; i <= R; i++) {
                for (int j = -R; j <= R; j++) {
                    vec2 off = vec2(float(i), float(j)) * u_inv_res * 4.0;
                    halo += texture(mask_tex, fuv + off).r;
                }
            }
            halo /= float((2*R+1)*(2*R+1));
            float pulse = 0.75 + 0.25 * sin(u_time * 2.2);
            vec3 glow = vec3(0.35, 0.85, 1.6) * halo * pulse;
            col += glow * 0.9;
            col = mix(col, vec3(0.85, 1.0, 1.6), m * 0.55);
        } else if (u_effect == 3) {
            // COLORSHIFT: rotate hue over time inside the silhouette.
            float h = fract(u_time * 0.15 + uv.x * 0.25 + uv.y * 0.12);
            vec3 shift = hsv2rgb(vec3(h, 0.85, 1.0));
            col = mix(col, shift, m * 0.85);
        }
    }

    vec4 ovr = texture(overlay_tex, fuv);
    col = mix(col, ovr.rgb, ovr.a);

    frag_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


class EffectRenderer:
    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height

        self.prog = ctx.program(vertex_shader=VERT_SHADER, fragment_shader=FRAG_SHADER)

        verts = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype="f4")
        self.vbo = ctx.buffer(verts.tobytes())
        self.vao = ctx.simple_vertex_array(self.prog, self.vbo, "in_vert")

        self.pattern_tex   = ctx.texture((width, height), 3)        # RGB8
        self.selection_tex = ctx.texture((width, height), 4)        # RGBA8
        self.mask_tex      = ctx.texture((width, height), 1)        # R8
        self.overlay_tex   = ctx.texture((width, height), 4)        # RGBA8

        for tex in (self.pattern_tex, self.selection_tex, self.mask_tex, self.overlay_tex):
            tex.repeat_x = False
            tex.repeat_y = False
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Bind sampler units once.
        self.prog["pattern_tex"].value = 0
        self.prog["selection_tex"].value = 1
        self.prog["mask_tex"].value = 2
        self.prog["overlay_tex"].value = 3
        self.prog["u_inv_res"].value = (1.0 / width, 1.0 / height)
        self.prog["u_effect"].value = EFFECT_NONE
        self.prog["u_has_mask"].value = 0
        self.prog["u_time"].value = 0.0

    # ---- per-frame uploads --------------------------------------------------
    def upload_pattern_bgr(self, bgr: np.ndarray):
        """Pattern from cv2/pygame as BGR uint8 (H, W, 3). Convert to RGB."""
        if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
            import cv2
            bgr = cv2.resize(bgr, (self.width, self.height))
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        self.pattern_tex.write(rgb.tobytes())

    def upload_selection_rgba(self, rgba: Optional[np.ndarray]):
        if rgba is None:
            blank = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            self.selection_tex.write(blank.tobytes())
            return
        if rgba.shape[1] != self.width or rgba.shape[0] != self.height:
            import cv2
            rgba = cv2.resize(rgba, (self.width, self.height))
        self.selection_tex.write(np.ascontiguousarray(rgba).tobytes())

    def upload_mask(self, mask: Optional[np.ndarray]):
        """Single-channel uint8 mask, 0/255 or 0/1. None disables the effect."""
        if mask is None:
            self.prog["u_has_mask"].value = 0
            return
        if mask.dtype != np.uint8:
            mask = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask.astype(np.uint8)
        if mask.shape != (self.height, self.width):
            import cv2
            mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        self.mask_tex.write(np.ascontiguousarray(mask).tobytes())
        self.prog["u_has_mask"].value = 1

    def upload_overlay_rgba(self, rgba: np.ndarray):
        if rgba.shape[1] != self.width or rgba.shape[0] != self.height:
            import cv2
            rgba = cv2.resize(rgba, (self.width, self.height))
        self.overlay_tex.write(np.ascontiguousarray(rgba).tobytes())

    # ---- draw --------------------------------------------------------------
    def render(self, t: float, effect: int, target=None):
        """Render the composited frame.

        target: a moderngl.Framebuffer to render into. Defaults to ctx.screen
                (the windowed surface). Headless callers pass an offscreen FBO.
        """
        self.prog["u_time"].value = float(t)
        self.prog["u_effect"].value = int(effect)
        self.pattern_tex.use(0)
        self.selection_tex.use(1)
        self.mask_tex.use(2)
        self.overlay_tex.use(3)
        (target if target is not None else self.ctx.screen).use()
        self.ctx.clear(0.02, 0.02, 0.03)
        self.vao.render(moderngl.TRIANGLE_STRIP)

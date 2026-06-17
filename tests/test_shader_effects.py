"""Tests for GPU shader effects (fire2) and their integration into the effect
dispatch. GL may be unavailable in CI; those paths assert graceful fallback
to black rather than a crash.
"""
import numpy as np

from livetracking.daemon import effects
from livetracking.daemon import shader_effects


def test_fire2_registered():
    assert "fire2" in effects.EFFECTS
    assert effects.is_effect("fire2")
    assert "fire2" in shader_effects.SHADER_EFFECTS
    assert shader_effects.is_shader_effect("fire2")


def test_render_effect_fire2_shape_and_black_bg():
    img = effects.render_effect("fire2", 160, 240, 1.0)
    assert img.shape == (240, 160, 3)
    assert img.dtype == np.uint8
    # Whatever the GL outcome, the result is a valid RGB-on-black frame:
    # either real flame (some bright pixels) or all-black fallback. It must
    # never be uniformly lit (that would mean no black background to vanish
    # on the wall).
    assert int(img.min()) == 0


def test_render_effect_rgba_fire2_alpha_is_luminance():
    rgba = effects.render_effect_rgba("fire2", 160, 240, 2.0)
    assert rgba.shape == (240, 160, 4)
    # Alpha is the per-pixel luminance (max over RGB), so dark texels are
    # transparent. Black pixels => zero alpha.
    lum = rgba[..., :3].max(axis=2)
    assert np.array_equal(rgba[..., 3], lum)


def test_render_shader_unknown_returns_none():
    assert shader_effects.render_shader("nope", 64, 64, 0.0) is None


def test_render_shader_bad_size_returns_none():
    assert shader_effects.render_shader("fire2", 0, 64, 0.0) is None


def test_shader_failure_falls_back_to_black(monkeypatch):
    # Simulate GL unavailable: render_shader returns None -> dispatch emits
    # an all-black frame, never raises.
    monkeypatch.setattr(
        effects._shaders, "render_shader", lambda *a, **k: None)
    img = effects.render_effect("fire2", 100, 100, 0.0)
    assert img.shape == (100, 100, 3)
    assert int(img.max()) == 0

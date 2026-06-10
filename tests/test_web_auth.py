"""Tests for the flame_web auth gate.

The Flask app is built with create_app(); we don't need a perception
daemon — auth runs in before_request, before any ZMQ call. Routes that
DO trampoline to perception will return their normal "daemon timeout"
payloads when authorized, which is fine: we only assert on the 401/200
split and cookie behavior.
"""
from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture()
def web(tmp_path, monkeypatch):
    """Fresh flame_web module with token persisted under tmp_path."""
    monkeypatch.setenv("LIVETRACKING_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("LIVETRACKING_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("LIVETRACKING_AUTH_DISABLED", raising=False)
    # paths.py computes RUNTIME_DIR at import; force a clean re-import
    # chain so AUTH_TOKEN_FILE lands in tmp_path.
    for mod in ("livetracking.daemon.flame_web", "livetracking.paths"):
        sys.modules.pop(mod, None)
    fw = importlib.import_module("livetracking.daemon.flame_web")
    app = fw.create_app()
    app.config["TESTING"] = True
    return fw, app


def _token(fw):
    return fw._load_or_create_token()


class TestAuthGate:
    def test_unauthorized_json_401(self, web):
        fw, app = web
        c = app.test_client()
        r = c.get("/objects.json")
        assert r.status_code == 401
        assert r.get_json()["ok"] is False

    def test_unauthorized_index_gets_html_hint(self, web):
        fw, app = web
        r = app.test_client().get("/")
        assert r.status_code == 401
        assert b"token" in r.data

    def test_healthz_exempt(self, web):
        fw, app = web
        r = app.test_client().get("/healthz")
        assert r.status_code == 200
        assert r.get_json()["ok"] is True

    def test_header_token_accepted(self, web):
        fw, app = web
        r = app.test_client().get(
            "/objects.json",
            headers={"X-LiveTracking-Token": _token(fw)})
        assert r.status_code == 200

    def test_bad_header_token_rejected(self, web):
        fw, app = web
        r = app.test_client().get(
            "/objects.json",
            headers={"X-LiveTracking-Token": "wrong"})
        assert r.status_code == 401

    def test_query_token_sets_cookie_and_redirects(self, web):
        fw, app = web
        c = app.test_client()
        r = c.get(f"/?token={_token(fw)}")
        assert r.status_code == 302
        cookie = r.headers.get("Set-Cookie", "")
        assert fw.AUTH_COOKIE in cookie
        # follow-up request rides the cookie, no token in URL
        r2 = c.get("/")
        assert r2.status_code == 200

    def test_post_route_requires_auth(self, web):
        fw, app = web
        r = app.test_client().post("/clear")
        assert r.status_code == 401

    def test_disabled_via_env(self, web, monkeypatch):
        fw, app = web
        monkeypatch.setenv("LIVETRACKING_AUTH_DISABLED", "1")
        r = app.test_client().get("/objects.json")
        assert r.status_code == 200

    def test_token_persisted_and_stable(self, web):
        fw, app = web
        assert _token(fw) == _token(fw)
        with open(fw.AUTH_TOKEN_FILE) as f:
            assert f.read().strip() == _token(fw)

    def test_env_token_overrides_file(self, web, monkeypatch):
        fw, app = web
        monkeypatch.setenv("LIVETRACKING_AUTH_TOKEN", "env-tok")
        assert fw._load_or_create_token() == "env-tok"


class TestPostBodyValidation:
    """Bad/missing ids should 400, not 500."""

    def _client(self, web):
        fw, app = web
        c = app.test_client()
        c.environ_base = {}
        tok = _token(fw)
        return c, {"X-LiveTracking-Token": tok}

    @pytest.mark.parametrize("route", ["/highlight", "/hide", "/unhide",
                                       "/pin", "/cycle_color"])
    def test_missing_id_400(self, web, route):
        c, h = self._client(web)
        r = c.post(route, json={}, headers=h)
        assert r.status_code == 400

    @pytest.mark.parametrize("route", ["/highlight", "/hide"])
    def test_garbage_id_400(self, web, route):
        c, h = self._client(web)
        r = c.post(route, json={"id": "couch"}, headers=h)
        assert r.status_code == 400

    def test_rename_missing_name_400(self, web):
        c, h = self._client(web)
        r = c.post("/rename", json={"id": 1}, headers=h)
        assert r.status_code == 400

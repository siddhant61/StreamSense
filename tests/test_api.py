"""Headless API tests using FastAPI's TestClient (REST + WebSocket), mock manager."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api.app import create_app
from tests.platform_mocks import make_manager


def _client():
    return TestClient(create_app(manager=make_manager()))


def test_health():
    c = _client()
    r = c.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_discover_then_list_devices():
    c = _client()
    r = c.post("/api/discover", json={"types": ["muse"]})
    assert r.status_code == 200
    assert [d["id"] for d in r.json()] == ["muse:AA"]

    r2 = c.get("/api/devices")
    assert any(d["id"] == "muse:AA" for d in r2.json())


def test_connect_and_disconnect_flow():
    c = _client()
    c.post("/api/discover", json={})
    r = c.post("/api/devices/muse:AA/connect")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["device"]["state"] == "connected"

    r2 = c.post("/api/devices/muse:AA/disconnect")
    assert r2.json()["device"]["state"] == "disconnected"


def test_connect_unknown_device_returns_400():
    c = _client()
    r = c.post("/api/devices/ghost/connect")
    assert r.status_code == 400


def test_recording_endpoints():
    c = _client()
    r = c.post("/api/recording/start")
    assert r.json()["recording"]["active"] is True
    r2 = c.post("/api/recording/stop")
    assert r2.json()["recording"]["active"] is False


def test_import_e4_endpoint(tmp_path):
    (tmp_path / "BVP.csv").write_text("1551434400.0\n64.0\n-0.1\n0.2\n")
    (tmp_path / "tags.csv").write_text("1551434412.0\n")
    c = _client()
    r = c.post("/api/import/e4", json={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["signals"]["BVP"]["sample_rate"] == 64.0
    assert body["tags"] == [1551434412.0]


def test_import_e4_missing_path_returns_404():
    c = _client()
    r = c.post("/api/import/e4", json={"path": "/no/such/e4/session"})
    assert r.status_code == 404


def test_status_endpoint_shape():
    c = _client()
    r = c.get("/api/status")
    body = r.json()
    assert "devices" in body and "recording" in body and "driver_availability" in body


def test_websocket_receives_initial_status_and_live_event():
    mgr = make_manager()
    client = TestClient(create_app(manager=mgr))
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["type"] == "status"
        # Emit a device event from the test thread; the listener hops it onto the loop.
        mgr.discover(["muse"])
        msg = ws.receive_json()
        assert msg["type"] in ("device_update", "log")

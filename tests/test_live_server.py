"""Real-socket tests: the API served by uvicorn and reached over TCP.

``tests/test_api.py`` drives the app in-process through Starlette's ``TestClient``, which
speaks ASGI directly. That cannot see two things a browser depends on:

* whether the server can accept a WebSocket upgrade at all — uvicorn needs the
  ``websockets`` or ``wsproto`` package, and without one ``/ws`` answers 404, so the
  dashboard never goes live even though every in-process test passes;
* whether the ``/ws`` handler lets go of a client that has left, so its listener is
  released and the server can still shut down (Ctrl+C) or ``--reload``.

The server runs in a thread on an ephemeral loopback port; no processes are spawned.
"""

import importlib.util
import json
import socket
import threading
import time

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from api.app import create_app  # noqa: E402
from tests.platform_mocks import make_manager  # noqa: E402


def test_uvicorn_has_a_websocket_implementation():
    """The documented install (requirements-dev.txt) must let uvicorn serve ``/ws``."""
    assert any(importlib.util.find_spec(name) for name in ("websockets", "wsproto")), (
        "uvicorn has no WebSocket implementation installed, so /ws answers 404 and the "
        "dashboard never goes live. Install `websockets` (pinned in requirements-dev.txt) "
        "or `wsproto`."
    )


class LiveServer:
    """``uvicorn`` serving the platform API with a mock manager, in a background thread."""

    def __init__(self):
        self.manager = make_manager()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self.port = self._sock.getsockname()[1]
        config = uvicorn.Config(create_app(manager=self.manager), log_level="warning",
                                lifespan="off")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(
            target=self.server.run, kwargs={"sockets": [self._sock]}, daemon=True)

    def start(self) -> "LiveServer":
        self.thread.start()
        assert _eventually(lambda: self.server.started, timeout=10), "uvicorn did not start"
        return self

    def stop(self, timeout: float = 5.0) -> bool:
        """Ask for a graceful shutdown; True if the server thread finished in time."""
        self.server.should_exit = True
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def kill(self) -> None:
        self.server.force_exit = True
        self.thread.join(5)


def _eventually(predicate, timeout: float = 3.0, step: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return bool(predicate())


@pytest.fixture
def live():
    server = LiveServer().start()
    yield server
    if not server.stop():
        server.kill()


def _connect(port: int):
    ws_client = pytest.importorskip("websockets.sync.client")
    url = f"ws://127.0.0.1:{port}/ws"
    try:
        # Loopback must never be routed through an HTTP proxy from the environment.
        return ws_client.connect(url, open_timeout=5, proxy=None)
    except TypeError:  # websockets < 15 has no proxy support (and no such argument)
        return ws_client.connect(url, open_timeout=5)


def test_websocket_handshake_and_live_events_over_a_real_socket(live):
    with _connect(live.port) as ws:
        first = json.loads(ws.recv(timeout=5))
        assert first["type"] == "status"
        assert set(first["payload"]) == {"devices", "recording", "driver_availability"}

        # Emitted from this (non-event-loop) thread, as a device worker would.
        live.manager.discover(["muse"])
        event = json.loads(ws.recv(timeout=5))
        assert event["type"] == "device_update"
        assert event["payload"]["id"] == "muse:AA"


def test_listener_is_released_when_the_client_leaves(live):
    with _connect(live.port) as ws:
        ws.recv(timeout=5)
        assert _eventually(lambda: len(live.manager._listeners) == 1)
    # No event is emitted here: the handler has to notice the disconnect on its own.
    assert _eventually(lambda: live.manager._listeners == []), (
        "the /ws handler kept its listener after the client disconnected")


def test_every_client_gets_events_and_only_the_leaver_is_released(live):
    with _connect(live.port) as staying:
        staying.recv(timeout=5)
        with _connect(live.port) as leaving:
            leaving.recv(timeout=5)
            assert _eventually(lambda: len(live.manager._listeners) == 2)
        assert _eventually(lambda: len(live.manager._listeners) == 1)

        live.manager.discover(["muse"])
        assert json.loads(staying.recv(timeout=5))["type"] == "device_update"


def test_server_stops_promptly_after_a_client_has_left(live):
    with _connect(live.port) as ws:
        ws.recv(timeout=5)
    assert live.stop(timeout=5), (
        "uvicorn hung in graceful shutdown after a WebSocket client had connected and left")


def test_server_stops_promptly_with_a_client_still_connected(live):
    ws = _connect(live.port)
    try:
        ws.recv(timeout=5)
        assert live.stop(timeout=5), (
            "uvicorn hung in graceful shutdown while a WebSocket client was connected")
    finally:
        ws.close()


def test_pump_failure_closes_the_socket_so_the_client_can_reconnect(live):
    """An event that cannot be serialized must not leave a silent, half-dead socket."""
    with _connect(live.port) as ws:
        ws.recv(timeout=5)
        live.manager._emit("log", {"level": "info", "message": object()})  # not JSON-able
        with pytest.raises(Exception) as closed:
            ws.recv(timeout=5)
        assert type(closed.value).__name__.startswith("ConnectionClosed")
    assert _eventually(lambda: live.manager._listeners == [])

    # A fresh connection works and starts from a full status again.
    with _connect(live.port) as again:
        assert json.loads(again.recv(timeout=5))["type"] == "status"

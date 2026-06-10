"""Headless unit tests for core.DeviceManager (mock drivers/recorder, no hardware)."""

import pytest

from core import DeviceManager, DeviceType, ConnectionState, DeviceInfo, DeviceManagerError
from core.drivers import E4ImportDriver
from tests.platform_mocks import make_manager


def test_discover_adds_devices_and_emits_events():
    events = []
    m = make_manager()
    m.add_listener(events.append)
    found = m.discover(["muse"])
    assert [d.id for d in found] == ["muse:AA"]
    assert "muse:AA" in m.devices
    assert any(e["type"] == "device_update" for e in events)


def test_connect_success_marks_connected():
    m = make_manager()
    m.discover()
    assert m.connect("muse:AA") is True
    assert m.devices["muse:AA"].state == ConnectionState.CONNECTED


def test_connect_failure_marks_error():
    m = make_manager(fail=True)
    m.discover()
    assert m.connect("muse:AA") is False
    assert m.devices["muse:AA"].state == ConnectionState.ERROR


def test_disconnect_stops_streamer_and_updates_state():
    m = make_manager()
    m.discover()
    m.connect("muse:AA")
    streamer = m._streamers["muse:AA"]
    assert m.disconnect("muse:AA") is True
    assert streamer.stopped is True
    assert m.devices["muse:AA"].state == ConnectionState.DISCONNECTED
    assert "muse:AA" not in m._streamers


def test_unknown_device_raises():
    m = make_manager()
    with pytest.raises(DeviceManagerError):
        m.connect("does-not-exist")


def test_import_only_driver_cannot_connect_live():
    m = DeviceManager(drivers={DeviceType.E4: E4ImportDriver()}, output_root="/tmp/ss")
    m.devices["e4:x"] = DeviceInfo(id="e4:x", name="E4", type=DeviceType.E4, address="x")
    with pytest.raises(DeviceManagerError):
        m.connect("e4:x")


def test_recording_lifecycle():
    m = make_manager()
    assert m.start_recording(timeout=5) is True
    assert m.recording.active is True
    assert m.recording.session_id
    assert m.stop_recording() is True
    assert m.recording.active is False


def test_status_reports_driver_availability_and_no_fake_quality():
    m = make_manager()
    m.discover()
    status = m.get_status().to_dict()
    assert "muse" in status["driver_availability"]
    assert status["recording"]["active"] is False
    # Honest: signal quality is unknown (None), never a fabricated number.
    assert status["devices"][0]["signal_quality"] is None


def test_unavailable_driver_is_skipped_in_discovery():
    m = make_manager(available=False)
    assert m.discover(["muse"]) == []

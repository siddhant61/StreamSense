"""Unit tests for the BITalino streamer (streamer/stream_bitalino.py).

The `bitalino` hardware library is not installed in CI, so StreamBioTalino.__init__
raises ImportError by design. We mock the library (and the pylsl outlet factories) to
exercise the real configuration/outlet logic without hardware. These are pure unit
tests (no process spawn), so they count toward coverage.
"""

import pytest
from unittest.mock import MagicMock

import streamer.stream_bitalino as sb
from streamer.stream_bitalino import StreamBioTalino, DEFAULT_SAMPLING_RATE


@pytest.fixture
def bitalino_available(monkeypatch):
    """Pretend the bitalino library is installed so __init__ does not raise."""
    monkeypatch.setattr(sb, "bitalino", MagicMock())


def test_missing_library_raises_importerror(monkeypatch):
    monkeypatch.setattr(sb, "bitalino", None)
    with pytest.raises(ImportError):
        StreamBioTalino("00:13:01:05:12:34", 1.0, "/tmp/bita")


def test_default_config_active_channels_and_name(bitalino_available):
    s = StreamBioTalino("00:13:01:05:12:34", 1.0, "/tmp/bita")
    # Default config enables channels 0,1,2,4,5 (EEG ch 3 disabled).
    assert s.active_channels == [0, 1, 2, 4, 5]
    assert s.device_name == "BITalino_001301051234"
    assert s.sampling_rate == DEFAULT_SAMPLING_RATE


def test_invalid_sampling_rate_falls_back_to_default(bitalino_available):
    s = StreamBioTalino("00:13", 1.0, "/tmp/bita", sampling_rate=999)
    assert s.sampling_rate == DEFAULT_SAMPLING_RATE


def test_valid_sampling_rate_is_kept(bitalino_available):
    s = StreamBioTalino("00:13", 1.0, "/tmp/bita", sampling_rate=100)
    assert s.sampling_rate == 100


def test_no_enabled_channels_raises_valueerror(bitalino_available):
    cfg = {0: {"name": "ECG", "type": "ECG", "unit": "mV", "enabled": False}}
    with pytest.raises(ValueError):
        StreamBioTalino("00:13", 1.0, "/tmp/bita", sensor_config=cfg)


def test_setup_lsl_outlets_one_per_enabled_channel(bitalino_available, monkeypatch):
    monkeypatch.setattr(sb, "StreamInfo", MagicMock())
    monkeypatch.setattr(sb, "StreamOutlet", MagicMock(return_value=MagicMock()))

    cfg = {
        0: {"name": "ECG", "type": "ECG", "unit": "mV", "enabled": True},
        1: {"name": "EDA", "type": "EDA", "unit": "uS", "enabled": True},
        2: {"name": "EMG", "type": "EMG", "unit": "mV", "enabled": False},
    }
    s = StreamBioTalino("00:13", 1.0, "/tmp/bita", sensor_config=cfg)
    s._setup_lsl_outlets()

    assert set(s.outlets.keys()) == {0, 1}
    assert sb.StreamOutlet.call_count == 2


def test_get_battery_level(bitalino_available):
    s = StreamBioTalino("00:13", 1.0, "/tmp/bita")

    s.device = MagicMock()
    s.device.state.return_value = {"battery": 88}
    assert s.get_battery_level() == 88

    s.device = None
    assert s.get_battery_level() is None


def test_repr_mentions_class_and_mac(bitalino_available):
    s = StreamBioTalino("00:13:01:05:12:34", 1.0, "/tmp/bita")
    text = repr(s)
    assert "StreamBioTalino" in text
    assert "00:13:01:05:12:34" in text

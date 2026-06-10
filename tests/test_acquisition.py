"""Headless tests for the acquisition-quality core: clock, backoff, signal quality,
and their wiring into DeviceManager."""

import math

import pytest

from core.clock import SessionClock
from core.backoff import ExponentialBackoff, retry_with_backoff
from core.signal_quality import assess, QualityScore
from core import DeviceType, ConnectionState
from tests.platform_mocks import make_manager, MockDriver


# ----- SessionClock --------------------------------------------------------- #
def test_session_clock_epoch_and_elapsed():
    t = {"v": 100.0}
    clk = SessionClock(time_fn=lambda: t["v"])
    assert clk.epoch == 100.0
    t["v"] = 105.5
    assert clk.elapsed() == pytest.approx(5.5)
    assert clk.stamp(107.0) == pytest.approx(7.0)


def test_session_clock_explicit_epoch():
    clk = SessionClock(epoch=0.0, time_fn=lambda: 42.0)
    assert clk.stamp() == pytest.approx(42.0)


# ----- ExponentialBackoff --------------------------------------------------- #
def test_backoff_grows_and_caps():
    b = ExponentialBackoff(base=1.0, factor=2.0, max_delay=10.0, jitter=0.0)
    assert [b.delay(i) for i in range(5)] == [1.0, 2.0, 4.0, 8.0, 10.0]  # capped at 10


def test_backoff_jitter_within_bounds():
    b = ExponentialBackoff(base=1.0, factor=2.0, max_delay=100.0, jitter=0.5)
    for i in range(4):
        nominal = 1.0 * 2 ** i
        d = b.delay(i)
        assert 0.5 * nominal <= d <= 1.5 * nominal


def test_retry_with_backoff_succeeds_after_failures():
    calls = {"n": 0}
    slept = []

    def op():
        calls["n"] += 1
        return calls["n"] >= 3  # fail twice, then succeed

    ok = retry_with_backoff(op, max_attempts=5,
                            backoff=ExponentialBackoff(base=1, jitter=0.0),
                            sleep=slept.append)
    assert ok is True
    assert calls["n"] == 3
    assert slept == [1.0, 2.0]  # two backoff waits before the 3rd attempt


def test_retry_with_backoff_respects_stop():
    ok = retry_with_backoff(lambda: False, max_attempts=10,
                            should_stop=lambda: True, sleep=lambda d: None)
    assert ok is False


# ----- signal quality ------------------------------------------------------- #
def test_quality_unknown_when_insufficient():
    s = assess([1.0, 2.0])
    assert s.value is None and s.label == "unknown"


def test_quality_good_for_varied_finite_signal():
    samples = [math.sin(i / 5.0) for i in range(200)]
    s = assess(samples)
    assert s.value is not None and s.value >= 0.8 and s.label == "good"


def test_quality_poor_for_flatline():
    s = assess([1.0] * 100)
    assert s.value == 0.0 and s.label == "poor"


def test_quality_penalized_by_nans_and_rate():
    samples = [float("nan") if i % 2 == 0 else math.sin(i) for i in range(100)]
    s = assess(samples, expected_rate=100, actual_rate=50)
    assert s.value is not None and s.value < 0.5  # half NaN * half rate
    assert s.detail["rate_ratio"] == 0.5


def test_quality_multichannel():
    samples = [[math.sin(i), math.cos(i)] for i in range(100)]
    s = assess(samples)
    assert s.detail["channels"] == 2 and s.value >= 0.8


# ----- DeviceManager wiring ------------------------------------------------- #
def test_manager_uses_session_clock_epoch_as_sync_time():
    m = make_manager()
    assert m.sync_time == m.clock.epoch


def test_manager_reconnect_with_backoff(monkeypatch):
    # Driver whose streamer fails the first two start attempts, then connects.
    attempts = {"n": 0}

    class FlakyStreamer:
        def start_streaming(self, timeout=15):
            attempts["n"] += 1
            return attempts["n"] >= 3

        def stop_streaming(self):
            pass

    drv = MockDriver()
    drv.create_streamer = lambda device, out, sync: FlakyStreamer()
    from core import DeviceManager
    m = DeviceManager(drivers={DeviceType.MUSE: drv},
                      recorder_factory=lambda f: None, output_root="/tmp/ss")
    m.discover()
    ok = m.reconnect("muse:AA", max_attempts=5, sleep_fn=lambda d: None)
    assert ok is True
    assert attempts["n"] == 3
    assert m.devices["muse:AA"].state == ConnectionState.CONNECTED


def test_reconnect_succeeds_after_disconnect_all():
    # disconnect_all() sets the stop flag; a later reconnect() must still proceed
    # (regression for the flag never being cleared).
    m = make_manager()
    m.discover()
    m.disconnect_all()  # sets _stop_flag
    assert m._stop_flag.is_set()
    ok = m.reconnect("muse:AA", sleep_fn=lambda d: None)
    assert ok is True
    assert m.devices["muse:AA"].state == ConnectionState.CONNECTED


def test_manager_assess_device_sets_quality_and_emits():
    events = []
    m = make_manager()
    m.discover()
    m.add_listener(events.append)
    score = m.assess_device("muse:AA", [1.0] * 100)  # flatline -> poor
    assert score.label == "poor"
    assert m.devices["muse:AA"].signal_quality == 0.0
    assert any(e["type"] == "device_update" for e in events)

"""Shared mocks for platform (core + api) tests — no hardware, no real LSL."""

import threading

from core import DeviceManager, DeviceType, DeviceInfo
from core.drivers import DeviceDriver


class MockStreamer:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.stopped = False

    def start_streaming(self, timeout: int = 15) -> bool:
        return not self.fail

    def stop_streaming(self) -> None:
        self.stopped = True


class MockDriver(DeviceDriver):
    live = True

    def __init__(self, dtype=DeviceType.MUSE, devices=None, fail=False, available=True):
        self.device_type = dtype
        self._devices = devices if devices is not None else [
            DeviceInfo(id="muse:AA", name="Muse-AA", type=DeviceType.MUSE, address="AA")
        ]
        self.fail = fail
        self._available = available

    def available(self):
        return (self._available, "" if self._available else "mock unavailable")

    def discover(self):
        return [DeviceInfo(**{**d.__dict__}) for d in self._devices]

    def create_streamer(self, device, output_folder, sync_time):
        return MockStreamer(fail=self.fail)


class MockRecorder:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.started_event = threading.Event()
        self._stop = threading.Event()

    def record_streams(self):
        self.started_event.set()
        self._stop.wait()

    def stop(self):
        self._stop.set()


def make_manager(fail=False, devices=None, available=True):
    drv = MockDriver(devices=devices, fail=fail, available=available)
    return DeviceManager(
        drivers={DeviceType.MUSE: drv},
        output_root="/tmp/ss_platform_test",
        recorder_factory=lambda f: MockRecorder(f),
    )

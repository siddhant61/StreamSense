"""Framework-agnostic device manager for the StreamSense platform.

Owns device state, connection lifecycle, and recording. Emits events to registered
listeners (the API layer bridges these to a WebSocket). Contains no FastAPI imports and
no eager hardware imports.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .models import (
    DeviceType, ConnectionState, DeviceInfo, RecordingState, SystemStatus,
)
from .drivers import DeviceDriver, default_drivers
from .clock import SessionClock
from .backoff import ExponentialBackoff, retry_with_backoff
from .signal_quality import assess, QualityScore

logger = logging.getLogger("core.device_manager")

Listener = Callable[[dict], None]


class DeviceManagerError(Exception):
    pass


class DeviceManager:
    def __init__(
        self,
        drivers: Optional[Dict[DeviceType, DeviceDriver]] = None,
        output_root: Optional[str] = None,
        recorder_factory: Optional[Callable[[str], object]] = None,
    ):
        self.drivers: Dict[DeviceType, DeviceDriver] = drivers or default_drivers()
        self.output_root = output_root
        self._recorder_factory = recorder_factory  # (output_folder) -> recorder
        self.devices: Dict[str, DeviceInfo] = {}
        self._streamers: Dict[str, object] = {}
        self._recorder = None
        self._recorder_thread: Optional[threading.Thread] = None
        self.recording = RecordingState()
        self._lock = threading.RLock()
        self._listeners: List[Listener] = []
        # One reference clock per session; streamers stamp against its epoch.
        self.clock = SessionClock()
        self.sync_time = self.clock.epoch
        self.backoff = ExponentialBackoff()
        self._stop_flag = threading.Event()

    # ----- events ------------------------------------------------------------
    def add_listener(self, cb: Listener) -> None:
        self._listeners.append(cb)

    def remove_listener(self, cb: Listener) -> None:
        if cb in self._listeners:
            self._listeners.remove(cb)

    def _emit(self, event_type: str, payload: dict) -> None:
        event = {"type": event_type, "payload": payload, "ts": time.time()}
        for cb in list(self._listeners):
            try:
                cb(event)
            except Exception:  # a bad listener must not break device logic
                logger.exception("listener error")

    def _emit_device(self, device: DeviceInfo) -> None:
        self._emit("device_update", device.to_dict())

    # ----- discovery ---------------------------------------------------------
    def discover(self, types: Optional[List[str]] = None) -> List[DeviceInfo]:
        wanted = self._resolve_types(types)
        discovered: List[DeviceInfo] = []
        for dtype in wanted:
            driver = self.drivers.get(dtype)
            if driver is None or not driver.live:
                continue
            ok, reason = driver.available()
            if not ok:
                self._emit("log", {"level": "warning",
                                   "message": f"{dtype.value} unavailable: {reason}"})
                continue
            try:
                found = driver.discover()
            except Exception as exc:
                logger.exception("discovery failed for %s", dtype)
                self._emit("log", {"level": "error",
                                   "message": f"{dtype.value} discovery error: {exc}"})
                continue
            for dev in found:
                with self._lock:
                    self.devices[dev.id] = dev
                discovered.append(dev)
                self._emit_device(dev)
        return discovered

    # ----- connection --------------------------------------------------------
    def connect(self, device_id: str, timeout: int = 15) -> bool:
        device = self._require_device(device_id)
        driver = self.drivers[device.type]
        if not driver.live:
            raise DeviceManagerError(f"{device.type.value} is import-only; cannot connect live")
        if device.state == ConnectionState.CONNECTED:
            return True

        device.state = ConnectionState.CONNECTING
        self._emit_device(device)
        try:
            streamer = driver.create_streamer(device, self._ensure_output_folder(), self.sync_time)
            ok = streamer.start_streaming(timeout=timeout)
        except Exception as exc:
            logger.exception("connect failed for %s", device_id)
            device.state = ConnectionState.ERROR
            device.detail = str(exc)
            self._emit_device(device)
            return False

        if ok:
            with self._lock:
                self._streamers[device_id] = streamer
            device.state = ConnectionState.CONNECTED
            self._emit_device(device)
            return True

        device.state = ConnectionState.ERROR
        device.detail = "start_streaming timed out / failed"
        self._emit_device(device)
        return False

    def disconnect(self, device_id: str) -> bool:
        device = self._require_device(device_id)
        device.state = ConnectionState.DISCONNECTING
        self._emit_device(device)
        streamer = self._streamers.pop(device_id, None)
        try:
            if streamer is not None:
                streamer.stop_streaming()
        except Exception as exc:
            logger.exception("disconnect error for %s", device_id)
            device.state = ConnectionState.ERROR
            device.detail = str(exc)
            self._emit_device(device)
            return False
        device.state = ConnectionState.DISCONNECTED
        device.signal_quality = None
        self._emit_device(device)
        return True

    def disconnect_all(self) -> None:
        self._stop_flag.set()  # halt any in-flight reconnect loops
        if self.recording.active:
            self.stop_recording()
        for device_id in list(self._streamers.keys()):
            self.disconnect(device_id)

    def reconnect(self, device_id: str, max_attempts: int = 5,
                  sleep_fn: Callable[[float], None] = time.sleep) -> bool:
        """Reconnect a device with exponential backoff (stop-aware).

        Replaces fixed-delay retry loops: waits base*factor**attempt (capped, jittered)
        between attempts and aborts promptly if the manager is shutting down.
        """
        # A reconnect is a fresh intent to connect: clear any stop set by a prior
        # disconnect_all() so the retry loop isn't aborted before it starts. A concurrent
        # disconnect_all() during the loop will re-set the flag and abort it.
        self._stop_flag.clear()
        return retry_with_backoff(
            lambda: self.connect(device_id),
            max_attempts=max_attempts, backoff=self.backoff,
            should_stop=self._stop_flag.is_set, sleep=sleep_fn,
            on_retry=lambda n, d: self._emit("log", {
                "level": "info",
                "message": f"reconnect {device_id}: attempt {n}, next in {d:.1f}s"}),
        )

    # ----- signal quality ----------------------------------------------------
    def update_signal_quality(self, device_id: str, score: QualityScore) -> None:
        """Set a device's signal quality from a computed score and notify listeners."""
        device = self._require_device(device_id)
        device.signal_quality = score.value
        self._emit_device(device)

    def assess_device(self, device_id: str, samples, *, expected_rate=None,
                      actual_rate=None, amplitude_range=None) -> QualityScore:
        """Compute a real signal-quality score from recent samples and apply it."""
        score = assess(samples, expected_rate=expected_rate, actual_rate=actual_rate,
                       amplitude_range=amplitude_range)
        self.update_signal_quality(device_id, score)
        return score

    # ----- recording ---------------------------------------------------------
    def start_recording(self, timeout: int = 10) -> bool:
        if self.recording.active:
            return True
        output_folder = self._ensure_output_folder()
        recorder = self._make_recorder(output_folder)
        thread = threading.Thread(target=recorder.record_streams, daemon=True)
        thread.start()
        started = recorder.started_event.wait(timeout=timeout)
        if not started:
            self._emit("log", {"level": "error",
                               "message": "Recorder failed to start within timeout"})
            return False
        self._recorder = recorder
        self._recorder_thread = thread
        self.recording = RecordingState(
            active=True,
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            output_folder=output_folder,
            started_at=time.time(),
        )
        self._emit("recording", self.recording.to_dict())
        return True

    def stop_recording(self) -> bool:
        if not self.recording.active or self._recorder is None:
            return True
        try:
            self._recorder.stop()
        finally:
            self._recorder = None
            self._recorder_thread = None
            self.recording = RecordingState(active=False)
            self._emit("recording", self.recording.to_dict())
        return True

    # ----- queries -----------------------------------------------------------
    def get_status(self) -> SystemStatus:
        availability = {
            dtype.value: {"available": (av := drv.available())[0], "reason": av[1],
                          "live": drv.live}
            for dtype, drv in self.drivers.items()
        }
        return SystemStatus(
            devices=list(self.devices.values()),
            recording=self.recording,
            driver_availability=availability,
        )

    def list_streams(self) -> List[str]:
        try:
            from pylsl import resolve_streams  # lazy
            return sorted({s.name() for s in resolve_streams(wait_time=1.0)})
        except Exception:
            return []

    # ----- helpers -----------------------------------------------------------
    def _resolve_types(self, types: Optional[List[str]]) -> List[DeviceType]:
        if not types:
            return [t for t, d in self.drivers.items() if d.live]
        out = []
        for t in types:
            try:
                out.append(DeviceType(t))
            except ValueError:
                raise DeviceManagerError(f"unknown device type: {t}")
        return out

    def _require_device(self, device_id: str) -> DeviceInfo:
        device = self.devices.get(device_id)
        if device is None:
            raise DeviceManagerError(f"unknown device: {device_id}")
        return device

    def _ensure_output_folder(self) -> str:
        if not self.output_root:
            base = Path.home() / "StreamSense" / str(time.time()).replace(".", "_")
            base.mkdir(parents=True, exist_ok=True)
            self.output_root = str(base)
        return self.output_root

    def _make_recorder(self, output_folder: str):
        if self._recorder_factory is not None:
            return self._recorder_factory(output_folder)
        from recorder.stream_recorder import StreamRecorder  # lazy
        return StreamRecorder(output_folder)

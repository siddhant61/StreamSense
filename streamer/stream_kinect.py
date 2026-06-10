"""Azure Kinect streamer (body-tracking joints + IMU -> LSL, RGB/depth -> .mkv).

Design: every hardware call lives behind a ``KinectBackend``. The default
``PyK4ABackend`` lazily imports ``pyk4a`` (cameras/IMU/recording) and, when body
tracking is requested, ``pykinect_azure`` (Body Tracking SDK). Tests inject a mock
backend, so the streaming loop, sample shaping and LSL wiring are verified headless;
only the thin ``PyK4ABackend`` awaits on-device verification.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from streamer.base_streamer import BaseStreamer
from streamer.kinect_support import (
    EMPTY_JOINTS_SAMPLE, skeleton_to_sample, imu_to_sample, stream_specs,
)

logger = logging.getLogger("stream_kinect")


@dataclass
class KinectFrame:
    """One poll from the device: a skeleton (or None) and an IMU reading (or None)."""
    joints: Optional[Sequence[Any]] = None
    imu: Any = None


class KinectBackend(ABC):
    """Hardware abstraction. Implementations isolate all device-SDK calls."""

    @abstractmethod
    def start(self, record_path: Optional[str], body_tracking: bool) -> None: ...

    @abstractmethod
    def poll(self) -> KinectFrame:
        """Block for the next capture; write video if recording; return joints+imu."""

    @abstractmethod
    def stop(self) -> None: ...


class PyK4ABackend(KinectBackend):
    """Real backend over pyk4a (+ pykinect_azure for body tracking).

    NOTE: the device-SDK call sequence here awaits verification on physical hardware;
    the surrounding loop/shaping logic is unit-tested. Imports are lazy so this module
    imports cleanly without the Kinect SDKs installed.
    """

    def __init__(self, device_id: int = 0, camera_fps: int = 30):
        self.device_id = device_id
        self.camera_fps = camera_fps
        self._k4a = None
        self._record = None
        self._tracker = None

    def start(self, record_path: Optional[str], body_tracking: bool) -> None:
        from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS  # lazy

        fps = {30: FPS.FPS_30, 15: FPS.FPS_15, 5: FPS.FPS_5}.get(self.camera_fps, FPS.FPS_30)
        config = Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            camera_fps=fps,
            synchronized_images_only=True,
        )
        self._k4a = PyK4A(config=config, device_id=self.device_id)
        self._k4a.start()

        if record_path:
            from pyk4a import PyK4ARecord  # lazy
            self._record = PyK4ARecord(device=self._k4a, config=config, path=record_path)
            self._record.create()

        if body_tracking:
            import pykinect_azure as pykinect  # lazy; Body Tracking SDK
            self._tracker = pykinect.start_body_tracker()

    def poll(self) -> KinectFrame:
        capture = self._k4a.get_capture()
        if self._record is not None:
            self._record.write_capture(capture)
        joints = None
        if self._tracker is not None:
            body_frame = self._tracker.update(capture)
            joints = self._first_body_joints(body_frame)
        imu = None
        try:
            imu = self._k4a.get_imu_sample()
        except Exception:  # IMU read is best-effort
            pass
        return KinectFrame(joints=joints, imu=imu)

    @staticmethod
    def _first_body_joints(body_frame: Any) -> Optional[Sequence[Any]]:
        """Extract the first detected body's joints, or None if no body."""
        if body_frame is None:
            return None
        try:
            if body_frame.get_num_bodies() < 1:
                return None
            return body_frame.get_body(0).joints
        except Exception:
            return None

    def stop(self) -> None:
        try:
            if self._record is not None:
                self._record.flush()
                self._record.close()
        finally:
            if self._k4a is not None:
                self._k4a.stop()


class StreamKinect(BaseStreamer):
    def __init__(
        self,
        device_name: str = "Kinect",
        root_output_folder: str = ".",
        synchronized_start_time: float = 0.0,
        device_id: int = 0,
        camera_fps: int = 30,
        record_video: bool = True,
        enable_body_tracking: bool = True,
        backend_factory: Optional[Callable[[], KinectBackend]] = None,
    ):
        super().__init__(
            device_name=device_name,
            synchronized_start_time=synchronized_start_time,
            root_output_folder=root_output_folder,
        )
        self.device_id = device_id
        self.camera_fps = camera_fps
        self.record_video = record_video
        self.enable_body_tracking = enable_body_tracking
        self._backend_factory = backend_factory or (
            lambda: PyK4ABackend(device_id=device_id, camera_fps=camera_fps)
        )
        specs = stream_specs(device_name, camera_fps)
        if not enable_body_tracking:
            # Without body tracking there is no skeleton — don't advertise a JOINTS stream.
            specs = [s for s in specs if s.key != "joints"]
        self._specs = specs
        self._outlets: Dict[str, Any] = {}

    @property
    def stream_names(self) -> List[str]:
        return [s.name for s in self._specs]

    @property
    def video_path(self) -> str:
        return os.path.join(self.root_output_folder, f"{self.device_name}.mkv")

    def _setup_lsl_outlets(self) -> None:
        import pylsl  # lazy
        self._outlets = {}
        for spec in self._specs:
            info = pylsl.StreamInfo(
                spec.name, spec.stype, spec.channel_count,
                spec.nominal_srate, spec.channel_format, spec.source_id,
            )
            channels = info.desc().append_child("channels")
            for label in spec.channel_labels:
                channels.append_child("channel").append_child_value("label", label)
            self._outlets[spec.key] = pylsl.StreamOutlet(info)

    def _run_loop(
        self,
        backend: KinectBackend,
        outlets: Dict[str, Any],
        clock: Callable[[], float],
        should_stop: Callable[[], bool],
        max_iters: Optional[int] = None,
    ) -> int:
        """Core capture loop (pure logic; no hardware/LSL specifics). Returns frame count.

        For each capture: stamp a SYNC marker on the LSL clock, push the skeleton (or a
        zero-filled sample when no body is detected) and the IMU reading.
        """
        idx = 0
        while not should_stop():
            frame = backend.poll()
            ts = clock()
            if "sync" in outlets:
                outlets["sync"].push_sample([idx], timestamp=ts)
            if "joints" in outlets:
                sample = skeleton_to_sample(frame.joints) if frame.joints else list(EMPTY_JOINTS_SAMPLE)
                outlets["joints"].push_sample(sample, timestamp=ts)
            if "imu" in outlets and frame.imu is not None:
                outlets["imu"].push_sample(imu_to_sample(frame.imu), timestamp=ts)
            idx += 1
            if max_iters is not None and idx >= max_iters:
                break
        return idx

    def _stream_wrapper(self) -> None:
        from pylsl import local_clock  # lazy
        backend = self._backend_factory()
        try:
            record_path = self.video_path if self.record_video else None
            backend.start(record_path=record_path, body_tracking=self.enable_body_tracking)
            self._setup_lsl_outlets()
            self.queue.put("connected")
            self._run_loop(backend, self._outlets, local_clock, self.stop_signal.is_set)
        except Exception:
            logger.exception("Kinect streaming error")
            try:
                self.queue.put("error")
            except Exception:
                pass
        finally:
            try:
                backend.stop()
            except Exception:
                logger.exception("Kinect backend stop error")

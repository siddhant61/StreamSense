"""Streaming-loop + driver tests for StreamKinect (mock backend, no hardware/LSL)."""

from streamer.stream_kinect import StreamKinect, KinectFrame, KinectBackend
from streamer.kinect_support import JOINTS_TOTAL, IMU_CHANNELS


class MockOutlet:
    def __init__(self):
        self.samples = []

    def push_sample(self, x, timestamp=None):
        self.samples.append((list(x), timestamp))


class MockJoint:
    def __init__(self):
        self.position = (1, 2, 3)
        self.orientation = (1, 0, 0, 0)
        self.confidence_level = 2


class MockImu:
    acc = (0.0, 0.0, 0.0)
    gyro = (0.0, 0.0, 0.0)


class MockBackend(KinectBackend):
    def __init__(self, frames):
        self.frames = frames
        self.i = 0
        self.started = False
        self.stopped = False
        self.record_path = None
        self.body_tracking = None

    def start(self, record_path, body_tracking):
        self.started = True
        self.record_path = record_path
        self.body_tracking = body_tracking

    def poll(self):
        frame = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return frame

    def stop(self):
        self.stopped = True


def _streamer():
    return StreamKinect(device_name="K", root_output_folder="/tmp/k",
                        backend_factory=lambda: MockBackend([]))


def test_run_loop_pushes_sync_joints_and_imu():
    frames = [KinectFrame(joints=[MockJoint()] * 32, imu=MockImu()) for _ in range(3)]
    outlets = {"sync": MockOutlet(), "joints": MockOutlet(), "imu": MockOutlet()}
    n = _streamer()._run_loop(MockBackend(frames), outlets, lambda: 1.0, lambda: False, max_iters=3)
    assert n == 3
    assert [s[0][0] for s in outlets["sync"].samples] == [0, 1, 2]
    assert len(outlets["joints"].samples[0][0]) == JOINTS_TOTAL
    assert len(outlets["imu"].samples) == 3
    assert len(outlets["imu"].samples[0][0]) == IMU_CHANNELS


def test_run_loop_no_body_pushes_zeros_and_skips_missing_imu():
    frames = [KinectFrame(joints=None, imu=None)]
    outlets = {"sync": MockOutlet(), "joints": MockOutlet(), "imu": MockOutlet()}
    _streamer()._run_loop(MockBackend(frames), outlets, lambda: 0.0, lambda: False, max_iters=1)
    assert outlets["joints"].samples[0][0] == [0.0] * JOINTS_TOTAL
    assert outlets["imu"].samples == []  # imu None -> skipped


def test_stream_names_and_video_path():
    s = StreamKinect(device_name="K", root_output_folder="/tmp/k")
    assert s.stream_names == ["K_JOINTS", "K_IMU", "K_SYNC"]
    assert s.video_path.endswith("K.mkv")


def test_driver_creates_streamer_with_sync_time():
    from core.drivers import KinectDriver
    from core.models import DeviceInfo, DeviceType
    d = DeviceInfo(id="kinect:0", name="Azure Kinect 0", type=DeviceType.KINECT, address="0")
    streamer = KinectDriver().create_streamer(d, "/tmp/k", 123.0)
    assert isinstance(streamer, StreamKinect)
    assert streamer.synchronized_start_time == 123.0
    assert streamer.device_id == 0

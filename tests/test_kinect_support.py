"""Pure shaping/spec tests for the Kinect support module (no hardware)."""

from streamer.kinect_support import (
    JOINT_COUNT, JOINTS_TOTAL, IMU_CHANNELS,
    skeleton_to_sample, imu_to_sample, stream_specs, EMPTY_JOINTS_SAMPLE,
)


class _Joint:
    def __init__(self, p, o, c):
        self.position = p
        self.orientation = o
        self.confidence_level = c


def test_empty_skeleton_is_zeros():
    assert skeleton_to_sample(None) == [0.0] * JOINTS_TOTAL
    assert skeleton_to_sample([]) == [0.0] * JOINTS_TOTAL
    assert EMPTY_JOINTS_SAMPLE == [0.0] * JOINTS_TOTAL


def test_skeleton_shaping_length_and_first_joint():
    joints = [_Joint((1, 2, 3), (0.1, 0.2, 0.3, 0.4), 2) for _ in range(JOINT_COUNT)]
    s = skeleton_to_sample(joints)
    assert len(s) == JOINTS_TOTAL
    assert s[:8] == [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 2.0]


def test_skeleton_pads_short_and_truncates_long():
    assert len(skeleton_to_sample([_Joint((1, 1, 1), (1, 0, 0, 0), 1)])) == JOINTS_TOTAL
    many = [_Joint((1, 1, 1), (1, 0, 0, 0), 1)] * (JOINT_COUNT + 5)
    assert len(skeleton_to_sample(many)) == JOINTS_TOTAL


def test_imu_shaping():
    class _Imu:
        acc = (0.1, 0.2, 0.3)
        gyro = (1.0, 2.0, 3.0)
    assert imu_to_sample(_Imu()) == [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]
    assert imu_to_sample(None) == [0.0] * IMU_CHANNELS


def test_stream_specs_shapes_and_names():
    specs = {s.key: s for s in stream_specs("Kinect")}
    assert set(specs) == {"joints", "imu", "sync"}
    assert specs["joints"].channel_count == JOINTS_TOTAL
    assert len(specs["joints"].channel_labels) == JOINTS_TOTAL
    assert specs["imu"].channel_count == IMU_CHANNELS
    assert specs["sync"].channel_count == 1
    assert specs["joints"].name == "Kinect_JOINTS"
    assert specs["sync"].stype == "Markers"

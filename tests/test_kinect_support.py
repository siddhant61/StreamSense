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


class _Vec3:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _Quat:
    def __init__(self, w, x, y, z):
        self.w, self.x, self.y, self.z = w, x, y, z


class _AttrJoint:
    """Joint whose position/orientation are attribute structs (no indexing) — the
    representation the Body Tracking SDK (pykinect_azure) uses."""
    def __init__(self):
        self.position = _Vec3(1, 2, 3)
        self.orientation = _Quat(0.1, 0.2, 0.3, 0.4)
        self.confidence_level = 2


def test_skeleton_shaping_with_attribute_structs():
    s = skeleton_to_sample([_AttrJoint()])
    assert s[:8] == [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 2.0]


def test_imu_shaping_with_attribute_structs():
    class _Imu:
        acc = _Vec3(0.1, 0.2, 0.3)
        gyro = _Vec3(1.0, 2.0, 3.0)
    assert imu_to_sample(_Imu()) == [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]


def test_stream_specs_shapes_and_names():
    specs = {s.key: s for s in stream_specs("Kinect")}
    assert set(specs) == {"joints", "imu", "sync"}
    assert specs["joints"].channel_count == JOINTS_TOTAL
    assert len(specs["joints"].channel_labels) == JOINTS_TOTAL
    assert specs["imu"].channel_count == IMU_CHANNELS
    assert specs["sync"].channel_count == 1
    assert specs["joints"].name == "Kinect_JOINTS"
    assert specs["sync"].stype == "Markers"

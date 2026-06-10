"""Tests for the skeleton 2D projection + DeviceManager joints broadcast (headless)."""

from streamer.kinect_support import (
    BONES, JOINT_COUNT, JOINTS_TOTAL, project_skeleton_sample,
)
from tests.platform_mocks import make_manager


def test_project_sample_returns_one_point_per_joint():
    sample = [float(i) for i in range(JOINTS_TOTAL)]
    points = project_skeleton_sample(sample)
    assert len(points) == JOINT_COUNT
    # joint 0: px=0, py=1, conf=7 (channels px,py,pz,qw,qx,qy,qz,conf)
    assert points[0] == [0.0, 1.0, 7.0]
    # joint 1 starts at channel 8: px=8, py=9, conf=15
    assert points[1] == [8.0, 9.0, 15.0]


def test_project_short_sample_pads_zeros():
    assert project_skeleton_sample([]) == [[0.0, 0.0, 0.0]] * JOINT_COUNT


def test_bones_reference_valid_joint_indices():
    assert BONES, "expected a non-empty bone list"
    for a, b in BONES:
        assert 0 <= a < JOINT_COUNT and 0 <= b < JOINT_COUNT


def test_broadcast_joints_emits_projected_points():
    events = []
    m = make_manager()
    m.add_listener(events.append)
    m.broadcast_joints("kinect:0", [float(i) for i in range(JOINTS_TOTAL)])
    joints = [e for e in events if e["type"] == "joints"]
    assert len(joints) == 1
    payload = joints[0]["payload"]
    assert payload["device_id"] == "kinect:0"
    assert len(payload["points"]) == JOINT_COUNT

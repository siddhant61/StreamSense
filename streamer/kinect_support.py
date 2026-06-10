"""Pure (hardware-free) support for the Azure Kinect streamer.

Everything here is testable without `pyk4a` or the Body Tracking SDK: joint/IMU sample
shaping and the LSL stream specifications. The hardware-touching code lives in
``streamer.stream_kinect`` behind an injectable backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Any

# Azure Kinect Body Tracking emits a fixed 32-joint skeleton.
JOINT_NAMES: List[str] = [
    "PELVIS", "SPINE_NAVEL", "SPINE_CHEST", "NECK",
    "CLAVICLE_LEFT", "SHOULDER_LEFT", "ELBOW_LEFT", "WRIST_LEFT",
    "HAND_LEFT", "HANDTIP_LEFT", "THUMB_LEFT",
    "CLAVICLE_RIGHT", "SHOULDER_RIGHT", "ELBOW_RIGHT", "WRIST_RIGHT",
    "HAND_RIGHT", "HANDTIP_RIGHT", "THUMB_RIGHT",
    "HIP_LEFT", "KNEE_LEFT", "ANKLE_LEFT", "FOOT_LEFT",
    "HIP_RIGHT", "KNEE_RIGHT", "ANKLE_RIGHT", "FOOT_RIGHT",
    "HEAD", "NOSE", "EYE_LEFT", "EAR_LEFT", "EYE_RIGHT", "EAR_RIGHT",
]
JOINT_COUNT = len(JOINT_NAMES)              # 32
# Per joint: position (x,y,z) + orientation quaternion (w,x,y,z) + confidence.
JOINT_CHANNELS = 8
JOINTS_TOTAL = JOINT_COUNT * JOINT_CHANNELS  # 256
IMU_CHANNELS = 6                             # acc xyz + gyro xyz

EMPTY_JOINTS_SAMPLE: List[float] = [0.0] * JOINTS_TOTAL


_VEC3_ATTRS = ("x", "y", "z")
_QUAT_ATTRS = ("w", "x", "y", "z")


def _coerce_vector(value: Any, attrs: Sequence[str]) -> List[float]:
    """Coerce a position/orientation-like value to a fixed-length float list.

    Tolerant of both representations the Kinect stacks use: index access
    (numpy arrays / tuples, e.g. pyk4a) and attribute access (SDK structs exposing
    ``.x/.y/.z`` or ``.w/.x/.y/.z``, e.g. pykinect_azure). Missing components -> 0.0.
    """
    if value is None:
        return [0.0] * len(attrs)
    out: List[float] = []
    for i, attr in enumerate(attrs):
        try:
            component = value[i]
        except (TypeError, KeyError, IndexError):
            component = getattr(value, attr, None)
        try:
            out.append(float(component))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _joint_to_channels(joint: Any) -> List[float]:
    """One joint -> [px,py,pz, qw,qx,qy,qz, confidence] (8 floats).

    Tolerant of the exact pyk4a/pykinect_azure joint representation: reads
    ``position``, ``orientation`` and ``confidence_level`` attributes when present.
    """
    position = getattr(joint, "position", None)
    orientation = getattr(joint, "orientation", None)
    confidence = getattr(joint, "confidence_level", getattr(joint, "confidence", 0))
    chans = _coerce_vector(position, _VEC3_ATTRS) + _coerce_vector(orientation, _QUAT_ATTRS)
    try:
        chans.append(float(confidence))
    except (TypeError, ValueError):
        chans.append(0.0)
    return chans


def skeleton_to_sample(joints: Optional[Sequence[Any]]) -> List[float]:
    """Flatten a 32-joint skeleton to a 256-float LSL sample.

    ``None`` / empty (no body detected) -> zeros, so the stream stays continuous.
    Pads or truncates to exactly JOINT_COUNT joints.
    """
    if not joints:
        return list(EMPTY_JOINTS_SAMPLE)
    sample: List[float] = []
    for i in range(JOINT_COUNT):
        if i < len(joints):
            sample.extend(_joint_to_channels(joints[i]))
        else:
            sample.extend([0.0] * JOINT_CHANNELS)
    return sample


def imu_to_sample(imu: Any) -> List[float]:
    """IMU reading -> [ax,ay,az, gx,gy,gz]. Tolerant of acc/gyro attribute shapes."""
    if imu is None:
        return [0.0] * IMU_CHANNELS
    acc = getattr(imu, "acc", getattr(imu, "acc_sample", None))
    gyro = getattr(imu, "gyro", getattr(imu, "gyro_sample", None))
    return _coerce_vector(acc, _VEC3_ATTRS) + _coerce_vector(gyro, _VEC3_ATTRS)


@dataclass
class StreamSpec:
    key: str                      # internal handle: "joints" | "imu" | "sync"
    name: str                     # LSL stream name
    stype: str                    # LSL stream type
    channel_count: int
    nominal_srate: float
    channel_format: str           # "float32" | "int32"
    source_id: str
    channel_labels: List[str] = field(default_factory=list)


def _joint_labels() -> List[str]:
    suffixes = ["px", "py", "pz", "qw", "qx", "qy", "qz", "conf"]
    return [f"{name}_{s}" for name in JOINT_NAMES for s in suffixes]


def stream_specs(device_name: str, camera_fps: int = 30) -> List[StreamSpec]:
    """LSL stream specs for a Kinect device: JOINTS, IMU, and a SYNC marker.

    Raw RGB/depth video is NOT an LSL stream (bandwidth) — it is written to an `.mkv`
    sidecar; the SYNC stream carries the per-frame index stamped on the LSL clock so the
    video can be aligned to the other modalities post-hoc.
    """
    return [
        StreamSpec(
            key="joints", name=f"{device_name}_JOINTS", stype="MoCap",
            channel_count=JOINTS_TOTAL, nominal_srate=camera_fps,
            channel_format="float32", source_id=f"kinect-joints-{device_name}",
            channel_labels=_joint_labels(),
        ),
        StreamSpec(
            key="imu", name=f"{device_name}_IMU", stype="IMU",
            channel_count=IMU_CHANNELS, nominal_srate=camera_fps,
            channel_format="float32", source_id=f"kinect-imu-{device_name}",
            channel_labels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        ),
        StreamSpec(
            key="sync", name=f"{device_name}_SYNC", stype="Markers",
            channel_count=1, nominal_srate=camera_fps,
            channel_format="int32", source_id=f"kinect-sync-{device_name}",
            channel_labels=["video_frame_index"],
        ),
    ]

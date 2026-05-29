"""Plain-dataclass models for the platform core.

Deliberately uses stdlib dataclasses (not Pydantic) so the core has no web-stack
dependency and unit-tests in the slim environment. The API layer defines its own
serialization models.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any


class DeviceType(str, Enum):
    MUSE = "muse"
    BITALINO = "bitalino"
    KINECT = "kinect"
    E4 = "e4"  # import-only (Empatica withdrew the live streaming server)


class ConnectionState(str, Enum):
    DISCOVERED = "discovered"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class DeviceInfo:
    """A device known to the manager (discovered and/or connected)."""

    id: str
    name: str
    type: DeviceType
    address: str = ""
    state: ConnectionState = ConnectionState.DISCOVERED
    # 0.0..1.0, or None when unknown. We never fabricate a value.
    signal_quality: Optional[float] = None
    detail: Optional[str] = None
    # Names of the LSL streams this device publishes once connected.
    streams: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        d["state"] = self.state.value
        return d


@dataclass
class RecordingState:
    active: bool = False
    session_id: Optional[str] = None
    output_folder: Optional[str] = None
    started_at: Optional[float] = None  # epoch seconds

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SystemStatus:
    devices: List[DeviceInfo]
    recording: RecordingState
    # type -> (available, reason) so the UI can show why a modality is unusable.
    driver_availability: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "devices": [d.to_dict() for d in self.devices],
            "recording": self.recording.to_dict(),
            "driver_availability": self.driver_availability,
        }

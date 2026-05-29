"""StreamSense platform core (framework-agnostic).

This package contains the UI-independent device-management logic. It must NOT import
FastAPI or any hardware library at module load time; hardware imports live inside drivers
and are performed lazily so the core imports and unit-tests cleanly in a headless CI.
"""

from .models import (
    DeviceType,
    ConnectionState,
    DeviceInfo,
    RecordingState,
    SystemStatus,
)
from .device_manager import DeviceManager, DeviceManagerError

__all__ = [
    "DeviceType",
    "ConnectionState",
    "DeviceInfo",
    "RecordingState",
    "SystemStatus",
    "DeviceManager",
    "DeviceManagerError",
]

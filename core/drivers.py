"""Device drivers for the platform core.

A driver knows how to (a) report whether its hardware/SW stack is *available* in the
current environment, (b) *discover* devices, and (c) *create a streamer* for a device.

Hardware libraries (pygatt, muselsl, bitalino, pyk4a, …) are imported **lazily inside
methods** so that importing this module never requires any device SDK. This is what lets
the core + API import and unit-test in a headless CI.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from .models import DeviceType, DeviceInfo, ConnectionState

logger = logging.getLogger("core.drivers")


class DeviceDriver(ABC):
    device_type: DeviceType
    #: True if the driver supports live streaming; False for import-only (e.g. E4).
    live: bool = True

    @abstractmethod
    def available(self) -> Tuple[bool, str]:
        """Return (usable, human-reason). Must never raise."""

    @abstractmethod
    def discover(self) -> List[DeviceInfo]:
        """Return discovered devices. Live drivers only."""

    def create_streamer(self, device: DeviceInfo, output_folder: str, sync_time: float):
        """Create (but do not start) a BaseStreamer for the device."""
        raise NotImplementedError


def _module_importable(module: str) -> Tuple[bool, str]:
    try:
        importlib.import_module(module)
        return True, ""
    except Exception as exc:  # ImportError or deeper init failure
        return False, f"{module} unavailable: {exc}"


# --------------------------------------------------------------------------- #
# Real drivers (lazy hardware imports)
# --------------------------------------------------------------------------- #
class MuseDriver(DeviceDriver):
    device_type = DeviceType.MUSE
    live = True

    def available(self) -> Tuple[bool, str]:
        ok, reason = _module_importable("muselsl")
        if not ok:
            return False, reason
        return _module_importable("pygatt")

    def discover(self) -> List[DeviceInfo]:
        from helper.find_devices import FindDevices  # lazy
        muses, com_ports = FindDevices().find_muses_with_ports()
        out: List[DeviceInfo] = []
        n = min(len(com_ports), len(muses)) if com_ports and muses else 0
        for i in range(n):
            name, address = muses[i]
            out.append(DeviceInfo(
                id=f"muse:{address}", name=name, type=DeviceType.MUSE,
                address=address, detail=f"interface={com_ports[n - i - 1]}",
            ))
        return out

    def create_streamer(self, device: DeviceInfo, output_folder: str, sync_time: float):
        from streamer.stream_muse import StreamMuse  # lazy
        interface = (device.detail or "").replace("interface=", "") or None
        return StreamMuse(
            name=device.name, address=device.address, interface=interface,
            root_output_folder=output_folder, synchronized_start_time=sync_time,
        )


class BitalinoDriver(DeviceDriver):
    device_type = DeviceType.BITALINO
    live = True

    def available(self) -> Tuple[bool, str]:
        return _module_importable("bitalino")

    def discover(self) -> List[DeviceInfo]:
        from helper.find_devices import FindDevices  # lazy
        out: List[DeviceInfo] = []
        for dev in FindDevices().scan_bluetooth():
            name = dev.get("name", "")
            if "bitalino" in name.lower():
                addr = dev["address"]
                out.append(DeviceInfo(
                    id=f"bitalino:{addr}", name=name,
                    type=DeviceType.BITALINO, address=addr,
                ))
        return out

    def create_streamer(self, device: DeviceInfo, output_folder: str, sync_time: float):
        from streamer.stream_bitalino import StreamBioTalino  # lazy
        return StreamBioTalino(
            mac_address=device.address, synchronized_start_time=sync_time,
            root_output_folder=output_folder,
        )


class KinectDriver(DeviceDriver):
    """Placeholder until PR-2 implements the pyk4a streamer."""

    device_type = DeviceType.KINECT
    live = True

    def available(self) -> Tuple[bool, str]:
        ok, reason = _module_importable("pyk4a")
        if not ok:
            return False, reason
        return False, "Kinect streamer not yet implemented (planned in PR-2)"

    def discover(self) -> List[DeviceInfo]:
        return []


class E4ImportDriver(DeviceDriver):
    """E4 is import-only: Empatica withdrew the live streaming server.

    This driver never streams live; it represents the E4 as a data source to be imported
    from E4 Connect session archives (implemented in PR-3).
    """

    device_type = DeviceType.E4
    live = False

    def available(self) -> Tuple[bool, str]:
        return False, "E4 real-time streaming withdrawn by Empatica; offline import only (PR-3)"

    def discover(self) -> List[DeviceInfo]:
        return []


def default_drivers() -> dict:
    """The drivers registered in production."""
    return {
        DeviceType.MUSE: MuseDriver(),
        DeviceType.BITALINO: BitalinoDriver(),
        DeviceType.KINECT: KinectDriver(),
        DeviceType.E4: E4ImportDriver(),
    }

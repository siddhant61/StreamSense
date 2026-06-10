"""
StreamSense UI Controller - Backend Integration

Bridges the PyQt5 UI to StreamSense core functionality (device discovery,
streaming, recording). Designed for clean separation of concerns and
robust error handling for job demonstrations.

Author: StreamSense Team
Date: November 5, 2025
"""

import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import userpaths
from pylsl import local_clock
from PyQt5.QtCore import QObject, pyqtSignal

from helper.find_devices import FindDevices
from streamer.stream_muse import StreamMuse
from streamer.stream_e4 import StreamE4
from streamer.stream_bitalino import StreamBioTalino
from recorder.stream_recorder import StreamRecorder


# Setup logging
logger = logging.getLogger("streamsense_controller")
logger.setLevel(logging.INFO)
fh = logging.FileHandler('Logs/streamsense_controller.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


@dataclass
class DeviceInfo:
    """Information about a discovered device."""
    name: str
    device_type: str
    address: str
    interface: Optional[str] = None  # For Muse (COM port)
    mac_address: Optional[str] = None  # For BITalino
    connected: bool = False
    streamer: Optional[object] = None  # Reference to actual streamer


class StreamSenseController(QObject):
    """
    Backend controller for StreamSense UI.

    Manages device discovery, connection, streaming, and recording.
    Emits Qt signals to update UI in real-time.
    """

    # Qt signals for UI updates
    device_discovered = pyqtSignal(str, str, str)  # (name, type, address)
    device_connected = pyqtSignal(str, bool, int)  # (name, connected, signal_quality)
    recording_started = pyqtSignal(str)  # (session_id)
    recording_stopped = pyqtSignal()
    error_occurred = pyqtSignal(str, str)  # (title, message)
    status_message = pyqtSignal(str)  # (message)

    def __init__(self):
        super().__init__()

        # State management
        self.devices: Dict[str, DeviceInfo] = {}
        self.muse_streamers: Dict[str, StreamMuse] = {}
        self.e4_streamers: Dict[str, StreamE4] = {}
        self.bitalino_streamers: Dict[str, StreamBioTalino] = {}

        self.recorder: Optional[StreamRecorder] = None
        self.recording = False

        self.root_output_folder: Optional[str] = None
        self.synchronized_start_time = local_clock()

        logger.info("StreamSenseController initialized")

    def _ensure_output_folder(self) -> str:
        """Ensure output directory exists and return its path."""
        if not self.root_output_folder:
            documents = userpaths.get_my_documents().replace("\\", "/")
            folder = f"{documents}/StreamSense/{str(datetime.today().timestamp()).replace('.', '_')}"
            path = Path(folder)
            path.mkdir(parents=True, exist_ok=True)
            self.root_output_folder = folder
            logger.info(f"Created output folder: {folder}")
        return self.root_output_folder

    def discover_devices(self, device_types: List[str] = None) -> Dict[str, DeviceInfo]:
        """
        Discover available devices.

        Args:
            device_types: List of device types to discover.
                         Options: ['muse', 'e4', 'bitalino']
                         If None, discovers all types.

        Returns:
            Dictionary of discovered devices {device_name: DeviceInfo}
        """
        if device_types is None:
            device_types = ['muse', 'e4', 'bitalino']

        self.status_message.emit("Discovering devices...")
        logger.info(f"Starting device discovery for: {device_types}")

        discovered = {}

        # Discover Muse devices
        if 'muse' in device_types:
            try:
                self.status_message.emit("Searching for Muse headbands...")
                finder = FindDevices()
                muses, com_ports = finder.find_muses_with_ports()

                if len(com_ports) > 0 and len(muses) > 0:
                    n = min(len(com_ports), len(muses))
                    for i in range(n):
                        name, address = muses[i]
                        interface = com_ports[n - i - 1]

                        device_info = DeviceInfo(
                            name=name,
                            device_type="Muse Headband",
                            address=address,
                            interface=interface
                        )
                        discovered[name] = device_info
                        self.devices[name] = device_info
                        self.device_discovered.emit(name, "Muse Headband", address)
                        logger.info(f"Discovered Muse: {name} at {address}")

                self.status_message.emit(f"Found {len([d for d in discovered.values() if 'Muse' in d.device_type])} Muse device(s)")

            except Exception as e:
                error_msg = f"Error discovering Muse devices: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                self.error_occurred.emit("Muse Discovery Error", error_msg)

        # Discover E4 devices
        if 'e4' in device_types:
            try:
                self.status_message.emit("Searching for Empatica E4 devices...")
                finder = FindDevices()
                e4s = finder.find_empatica()

                for e4_id in e4s:
                    device_info = DeviceInfo(
                        name=f"E4-{e4_id}",
                        device_type="Empatica E4",
                        address=e4_id
                    )
                    discovered[f"E4-{e4_id}"] = device_info
                    self.devices[f"E4-{e4_id}"] = device_info
                    self.device_discovered.emit(f"E4-{e4_id}", "Empatica E4", e4_id)
                    logger.info(f"Discovered E4: {e4_id}")

                self.status_message.emit(f"Found {len(e4s)} E4 device(s)")

            except Exception as e:
                error_msg = f"Error discovering E4 devices: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                self.error_occurred.emit("E4 Discovery Error", error_msg)

        # Discover BITalino devices
        if 'bitalino' in device_types:
            try:
                self.status_message.emit("Searching for BITalino devices...")
                finder = FindDevices()
                bluetooth_devices = finder.scan_bluetooth()

                # Filter for BITalino devices
                for device in bluetooth_devices:
                    if 'BITalino' in device.get('name', '') or 'bitalino' in device.get('name', '').lower():
                        name = device['name']
                        address = device['address']

                        device_info = DeviceInfo(
                            name=name,
                            device_type="BITalino (r)evolution",
                            address=address,
                            mac_address=address
                        )
                        discovered[name] = device_info
                        self.devices[name] = device_info
                        self.device_discovered.emit(name, "BITalino (r)evolution", address)
                        logger.info(f"Discovered BITalino: {name} at {address}")

                bitalino_count = len([d for d in discovered.values() if 'BITalino' in d.device_type])
                self.status_message.emit(f"Found {bitalino_count} BITalino device(s)")

            except Exception as e:
                error_msg = f"Error discovering BITalino devices: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                self.error_occurred.emit("BITalino Discovery Error", error_msg)

        total = len(discovered)
        self.status_message.emit(f"Discovery complete! Found {total} device(s) total")
        logger.info(f"Device discovery complete. Found {total} devices")

        return discovered

    def connect_device(self, device_name: str) -> bool:
        """
        Connect to a specific device and start streaming.

        Args:
            device_name: Name of the device to connect

        Returns:
            True if connection successful, False otherwise
        """
        if device_name not in self.devices:
            self.error_occurred.emit("Connection Error", f"Device {device_name} not found")
            return False

        device_info = self.devices[device_name]

        if device_info.connected:
            logger.warning(f"Device {device_name} already connected")
            return True

        self.status_message.emit(f"Connecting to {device_name}...")
        logger.info(f"Attempting to connect to {device_name}")

        try:
            output_folder = self._ensure_output_folder()

            # Connect based on device type
            if 'Muse' in device_info.device_type:
                streamer = StreamMuse(
                    name=device_info.name,
                    address=device_info.address,
                    interface=device_info.interface,
                    root_output_folder=output_folder,
                    synchronized_start_time=self.synchronized_start_time
                )

                if streamer.start_streaming(timeout=15):
                    self.muse_streamers[device_name] = streamer
                    device_info.streamer = streamer
                    device_info.connected = True
                    # Quality unknown until measured (0 = unknown). Real per-stream SQI
                    # is computed by core.signal_quality in the web platform.
                    self.device_connected.emit(device_name, True, 0)
                    self.status_message.emit(f"✓ {device_name} connected")
                    logger.info(f"Successfully connected to Muse: {device_name}")
                    return True
                else:
                    raise Exception("Connection timeout")

            elif 'E4' in device_info.device_type:
                # StreamE4.__init__ signature is (e4, root_output_folder,
                # synchronized_start_time). The previous device_id=/output_path=
                # kwargs did not exist and raised TypeError on every E4 connect.
                streamer = StreamE4(
                    e4=device_info.address,
                    root_output_folder=output_folder,
                    synchronized_start_time=self.synchronized_start_time
                )

                if streamer.start_streaming(timeout=15):
                    self.e4_streamers[device_name] = streamer
                    device_info.streamer = streamer
                    device_info.connected = True
                    # Quality unknown until measured (0 = unknown); see core.signal_quality.
                    self.device_connected.emit(device_name, True, 0)
                    self.status_message.emit(f"✓ {device_name} connected")
                    logger.info(f"Successfully connected to E4: {device_name}")
                    return True
                else:
                    raise Exception("Connection timeout")

            elif 'BITalino' in device_info.device_type:
                # Default BITalino configuration
                sensor_config = {
                    0: {'name': 'ECG', 'type': 'ECG', 'unit': 'mV', 'enabled': True},
                    1: {'name': 'EDA', 'type': 'EDA', 'unit': 'uS', 'enabled': True},
                    2: {'name': 'EMG', 'type': 'EMG', 'unit': 'mV', 'enabled': True},
                }

                streamer = StreamBioTalino(
                    mac_address=device_info.mac_address,
                    synchronized_start_time=self.synchronized_start_time,
                    root_output_folder=output_folder,
                    sampling_rate=1000,
                    sensor_config=sensor_config
                )

                if streamer.start_streaming(timeout=15):
                    self.bitalino_streamers[device_name] = streamer
                    device_info.streamer = streamer
                    device_info.connected = True
                    # Quality unknown until measured (0 = unknown); see core.signal_quality.
                    self.device_connected.emit(device_name, True, 0)
                    self.status_message.emit(f"✓ {device_name} connected")
                    logger.info(f"Successfully connected to BITalino: {device_name}")
                    return True
                else:
                    raise Exception("Connection timeout")

            else:
                raise Exception(f"Unsupported device type: {device_info.device_type}")

        except Exception as e:
            error_msg = f"Failed to connect to {device_name}: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.error_occurred.emit("Connection Error", error_msg)
            self.device_connected.emit(device_name, False, 0)
            return False

    def disconnect_device(self, device_name: str) -> bool:
        """
        Disconnect a specific device and stop streaming.

        Args:
            device_name: Name of the device to disconnect

        Returns:
            True if disconnection successful, False otherwise
        """
        if device_name not in self.devices:
            return False

        device_info = self.devices[device_name]

        if not device_info.connected:
            logger.warning(f"Device {device_name} already disconnected")
            return True

        self.status_message.emit(f"Disconnecting {device_name}...")
        logger.info(f"Attempting to disconnect {device_name}")

        try:
            if device_info.streamer:
                device_info.streamer.stop_streaming()

            # Remove from streamer dictionaries
            self.muse_streamers.pop(device_name, None)
            self.e4_streamers.pop(device_name, None)
            self.bitalino_streamers.pop(device_name, None)

            device_info.connected = False
            device_info.streamer = None
            self.device_connected.emit(device_name, False, 0)
            self.status_message.emit(f"✗ {device_name} disconnected")
            logger.info(f"Successfully disconnected {device_name}")
            return True

        except Exception as e:
            error_msg = f"Error disconnecting {device_name}: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.error_occurred.emit("Disconnection Error", error_msg)
            return False

    def start_recording(self) -> bool:
        """
        Start recording all active LSL streams.

        Returns:
            True if recording started successfully, False otherwise
        """
        if self.recording:
            logger.warning("Recording already in progress")
            return True

        self.status_message.emit("Starting recording...")
        logger.info("Attempting to start recording")

        try:
            output_folder = self._ensure_output_folder()

            self.recorder = StreamRecorder(output_folder)

            # Start recording in a separate thread (StreamRecorder handles this)
            import threading
            recorder_thread = threading.Thread(target=self.recorder.record_streams)
            recorder_thread.start()

            # Wait for recorder to be ready
            if not self.recorder.started_event.wait(timeout=10):
                raise Exception("Recorder failed to start within 10 seconds")

            self.recording = True
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_started.emit(session_id)
            self.status_message.emit(f"🔴 Recording started: {session_id}")
            logger.info(f"Recording started successfully: {session_id}")
            return True

        except Exception as e:
            error_msg = f"Failed to start recording: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.error_occurred.emit("Recording Error", error_msg)
            self.recorder = None
            return False

    def stop_recording(self) -> bool:
        """
        Stop the active recording.

        Returns:
            True if recording stopped successfully, False otherwise
        """
        if not self.recording or not self.recorder:
            logger.warning("No active recording to stop")
            return True

        self.status_message.emit("Stopping recording...")
        logger.info("Attempting to stop recording")

        try:
            self.recorder.stop()
            self.recording = False
            self.recorder = None
            self.recording_stopped.emit()
            self.status_message.emit("⏹️  Recording stopped")
            logger.info("Recording stopped successfully")
            return True

        except Exception as e:
            error_msg = f"Error stopping recording: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.error_occurred.emit("Recording Error", error_msg)
            return False

    def disconnect_all(self):
        """Disconnect all devices and stop recording."""
        logger.info("Disconnecting all devices")

        # Stop recording first
        if self.recording:
            self.stop_recording()

        # Disconnect all devices
        for device_name in list(self.devices.keys()):
            if self.devices[device_name].connected:
                self.disconnect_device(device_name)

        self.status_message.emit("All devices disconnected")
        logger.info("All devices disconnected")

    def get_connected_devices(self) -> List[DeviceInfo]:
        """Get list of currently connected devices."""
        return [d for d in self.devices.values() if d.connected]

    def get_discovered_devices(self) -> List[DeviceInfo]:
        """Get list of all discovered devices."""
        return list(self.devices.values())

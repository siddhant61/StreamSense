"""
Mock Muse headband hardware for testing without physical devices.

Provides realistic mocks for:
- Muse device discovery (Bluetooth scanning)
- BGAPI backend (BLED112 USB dongle)
- Muse device connection and streaming
- EEG, PPG, accelerometer, and gyroscope data
"""

import time
import threading
from typing import List, Dict, Optional, Any
from unittest.mock import Mock
from multiprocessing import Queue

from .data_generators import (
    EEGDataGenerator,
    PPGDataGenerator,
    AccelerometerDataGenerator,
    GyroscopeDataGenerator
)


class MockBGAPIBackend:
    """
    Mock BGAPI backend for BLED112 USB dongle communication.

    Simulates device discovery without requiring physical hardware.
    """

    def __init__(self, serial_port: str = "COM3"):
        """
        Initialize mock BGAPI backend.

        Parameters
        ----------
        serial_port : str
            Simulated serial port identifier
        """
        self.serial_port = serial_port
        self._is_started = False
        self._mock_devices = [
            {
                'name': 'Muse-1A2B',
                'address': '00:55:DA:B1:1A:2B'
            },
            {
                'name': 'Muse-3C4D',
                'address': '00:55:DA:B3:3C:4D'
            }
        ]

    def start(self):
        """Start the backend adapter."""
        self._is_started = True

    def stop(self):
        """Stop the backend adapter."""
        self._is_started = False

    def scan(self, timeout: int = 3) -> List[Dict[str, str]]:
        """
        Scan for Muse devices.

        Parameters
        ----------
        timeout : int
            Scan timeout in seconds

        Returns
        -------
        List[Dict[str, str]]
            List of discovered devices with 'name' and 'address' keys
        """
        if not self._is_started:
            raise RuntimeError("Backend not started")

        time.sleep(0.1)  # Simulate scan delay
        return self._mock_devices.copy()

    def connect(self, address: str, **kwargs):
        """Mock device connection."""
        if not self._is_started:
            raise RuntimeError("Backend not started")

        # Verify address exists in mock devices
        if not any(d['address'] == address for d in self._mock_devices):
            raise ConnectionError(f"Device {address} not found")

        return MockMuseAdapter(address)


class MockMuseAdapter:
    """
    Mock adapter for connecting to a Muse device.

    Simulates pygatt adapter behavior.
    """

    def __init__(self, address: str):
        """
        Initialize mock Muse adapter.

        Parameters
        ----------
        address : str
            Device Bluetooth MAC address
        """
        self.address = address
        self._connected = False
        self._subscriptions = {}

    def connect(self, address: Optional[str] = None, **kwargs):
        """
        Connect to the Muse device.

        Parameters
        ----------
        address : str, optional
            Device address (uses constructor address if None)
        """
        if address is None:
            address = self.address

        time.sleep(0.2)  # Simulate connection delay
        self._connected = True
        return self

    def disconnect(self):
        """Disconnect from the device."""
        self._connected = False
        self._subscriptions.clear()

    def char_write_handle(self, handle: int, value: bytearray, wait_for_response: bool = False):
        """
        Mock characteristic write.

        Parameters
        ----------
        handle : int
            Characteristic handle
        value : bytearray
            Value to write
        wait_for_response : bool
            Whether to wait for response
        """
        if not self._connected:
            raise RuntimeError("Not connected")
        # Silently accept writes

    def subscribe(self, uuid: str, callback: callable, indication: bool = False):
        """
        Subscribe to a characteristic.

        Parameters
        ----------
        uuid : str
            Characteristic UUID
        callback : callable
            Callback function for notifications
        indication : bool
            Whether to use indications instead of notifications
        """
        if not self._connected:
            raise RuntimeError("Not connected")

        self._subscriptions[uuid] = callback

    def unsubscribe(self, uuid: str):
        """Unsubscribe from a characteristic."""
        self._subscriptions.pop(uuid, None)


class MockMuse:
    """
    Mock Muse headband device with realistic data streaming.

    Provides complete mock implementation of the Muse helper class
    for testing without physical hardware.
    """

    def __init__(self,
                 address: str,
                 shared_eeg: Queue,
                 shared_ppg: Queue,
                 shared_acc: Queue,
                 shared_gyro: Queue,
                 shared_tel: Queue,
                 shared_con: Queue,
                 synchronized_start_time: float,
                 backend: str = 'bgapi',
                 interface: Optional[str] = None,
                 time_func: callable = time.time,
                 name: Optional[str] = None,
                 preset: Optional[int] = None,
                 disable_light: bool = False,
                 enable_eeg: bool = True,
                 enable_control: bool = True,
                 enable_telemetry: bool = True,
                 enable_acc: bool = True,
                 enable_gyro: bool = True,
                 enable_ppg: bool = True):
        """
        Initialize mock Muse device.

        Parameters match the real Muse helper class for drop-in replacement.
        """
        self.address = address
        self.name = name or f"MockMuse_{address[-5:]}"
        self.interface = interface
        self.time_func = time_func
        self.backend = backend
        self.preset = preset
        self.disable_light = disable_light
        self.connected = False

        # Queues for inter-process communication
        self.shared_eeg = shared_eeg
        self.shared_ppg = shared_ppg
        self.shared_acc = shared_acc
        self.shared_gyro = shared_gyro
        self.shared_tel = shared_tel
        self.shared_con = shared_con

        # Enable flags
        self.enable_eeg = enable_eeg
        self.enable_control = enable_control
        self.enable_telemetry = enable_telemetry
        self.enable_acc = enable_acc
        self.enable_gyro = enable_gyro
        self.enable_ppg = enable_ppg

        # Data generators
        self.eeg_generator = EEGDataGenerator(num_channels=5, sampling_rate=256.0)
        self.ppg_generator = PPGDataGenerator(num_channels=3, sampling_rate=64.0)
        self.acc_generator = AccelerometerDataGenerator(sampling_rate=52.0)
        self.gyro_generator = GyroscopeDataGenerator(sampling_rate=52.0)

        # Streaming control
        self._streaming_thread = None
        self._stop_streaming = threading.Event()
        self.last_timestamp = synchronized_start_time

        # Mock adapter
        self.adapter = None

    def connect(self, reconnect: bool = False) -> bool:
        """
        Connect to the mock Muse device.

        Parameters
        ----------
        reconnect : bool
            Whether this is a reconnection attempt

        Returns
        -------
        bool
            True if connection successful
        """
        try:
            if reconnect:
                time.sleep(0.5)  # Simulate reconnection delay

            self.adapter = MockMuseAdapter(self.address)
            self.adapter.connect()
            self.connected = True
            return True

        except Exception as e:
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from the device."""
        if self.adapter:
            self.adapter.disconnect()
        self.connected = False

    def start(self):
        """Start data streaming."""
        if not self.connected:
            raise RuntimeError("Device not connected")

        self._stop_streaming.clear()
        self._streaming_thread = threading.Thread(target=self._stream_data, daemon=True)
        self._streaming_thread.start()

    def stop(self):
        """Stop data streaming."""
        self._stop_streaming.set()
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=2.0)

    def start_keep_alive(self):
        """Start keep-alive thread (no-op in mock)."""
        pass

    def _stream_data(self):
        """
        Internal method to stream mock data to queues.

        Runs in a background thread and generates realistic physiological data.
        """
        # Streaming rates
        eeg_interval = 12 / 256.0  # 12 samples at 256 Hz
        ppg_interval = 6 / 64.0    # 6 samples at 64 Hz
        acc_interval = 1 / 52.0     # 1 sample at 52 Hz
        gyro_interval = 1 / 52.0    # 1 sample at 52 Hz

        last_eeg_time = time.time()
        last_ppg_time = time.time()
        last_acc_time = time.time()
        last_gyro_time = time.time()

        while not self._stop_streaming.is_set():
            current_time = time.time()

            # Generate EEG data
            if self.enable_eeg and (current_time - last_eeg_time) >= eeg_interval:
                eeg_data, eeg_timestamps = self.eeg_generator.generate(num_samples=12)
                self.shared_eeg.put((eeg_data, eeg_timestamps))
                last_eeg_time = current_time
                self.last_timestamp = current_time

            # Generate PPG data
            if self.enable_ppg and (current_time - last_ppg_time) >= ppg_interval:
                ppg_data, ppg_timestamps = self.ppg_generator.generate(num_samples=6)
                self.shared_ppg.put((ppg_data, ppg_timestamps))
                last_ppg_time = current_time

            # Generate accelerometer data
            if self.enable_acc and (current_time - last_acc_time) >= acc_interval:
                acc_data, acc_timestamps = self.acc_generator.generate(num_samples=1)
                self.shared_acc.put((acc_data, acc_timestamps))
                last_acc_time = current_time

            # Generate gyroscope data
            if self.enable_gyro and (current_time - last_gyro_time) >= gyro_interval:
                gyro_data, gyro_timestamps = self.gyro_generator.generate(num_samples=1)
                self.shared_gyro.put((gyro_data, gyro_timestamps))
                last_gyro_time = current_time

            # Sleep briefly to avoid busy-waiting
            time.sleep(0.01)


def create_mock_muse_device(name: str = "MockMuse", address: str = "00:55:DA:B0:00:01",
                            enable_all: bool = True) -> MockMuse:
    """
    Convenience function to create a mock Muse device for testing.

    Parameters
    ----------
    name : str
        Device name
    address : str
        Bluetooth MAC address
    enable_all : bool
        Enable all data streams

    Returns
    -------
    MockMuse
        Configured mock Muse device

    Example
    -------
    >>> from multiprocessing import Queue
    >>> muse = create_mock_muse_device()
    >>> # Set up queues
    >>> muse.shared_eeg = Queue()
    >>> muse.connect()
    >>> muse.start()
    >>> # Get data
    >>> eeg_data = muse.shared_eeg.get(timeout=1)
    >>> muse.stop()
    >>> muse.disconnect()
    """
    return MockMuse(
        address=address,
        shared_eeg=Queue(),
        shared_ppg=Queue(),
        shared_acc=Queue(),
        shared_gyro=Queue(),
        shared_tel=Queue(),
        shared_con=Queue(),
        synchronized_start_time=time.time(),
        name=name,
        enable_eeg=enable_all,
        enable_ppg=enable_all,
        enable_acc=enable_all,
        enable_gyro=enable_all,
        enable_telemetry=enable_all,
        enable_control=enable_all
    )

"""
Mock Empatica E4 wearable hardware for testing without physical devices.

Provides realistic mocks for:
- Empatica BLE Server socket communication
- E4 device discovery
- E4 connection and streaming
- BVP, GSR, temperature, accelerometer, and tag data
"""

import socket
import time
import threading
from typing import List, Optional, Dict
from queue import Queue
from multiprocessing import Event

from .data_generators import (
    BVPDataGenerator,
    GSRDataGenerator,
    TemperatureDataGenerator,
    AccelerometerDataGenerator
)


class MockEmpaticaServer:
    """
    Mock Empatica BLE Server for E4 device communication.

    Simulates the socket-based protocol used by Empatica's Windows application.
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 28000):
        """
        Initialize mock Empatica server.

        Parameters
        ----------
        host : str
            Server host address
        port : int
            Server port
        """
        self.host = host
        self.port = port
        self.connected_event = Event()
        self.stop_signal = Event()

        # Mock devices
        self._mock_devices = [
            "A01234",
            "A05678"
        ]

        # Connected devices
        self._connected_devices = set()

        # Mock server socket
        self._server_socket = None

    def start_server(self):
        """Start the mock BLE server (no-op in testing)."""
        pass

    def find_e4s(self) -> List[str]:
        """
        Discover available E4 devices.

        Returns
        -------
        List[str]
            List of device IDs
        """
        time.sleep(0.2)  # Simulate discovery delay
        return self._mock_devices.copy()

    def send_command(self, sock: socket.socket, command: str) -> str:
        """
        Send a command and receive response (mocked).

        Parameters
        ----------
        sock : socket.socket
            Socket connection (ignored in mock)
        command : str
            Command to send

        Returns
        -------
        str
            Server response
        """
        # Parse command
        if command == "device_discover_list":
            # Return device list
            device_list = " | ".join(self._mock_devices)
            return f"R device_discover_list | {device_list} | "

        elif command.startswith("device_connect_btle"):
            # Extract device ID
            parts = command.split()
            if len(parts) >= 2:
                device_id = parts[1]
                if device_id in self._mock_devices:
                    self._connected_devices.add(device_id)
                    self.connected_event.set()
                    return "R device_connect_btle OK"
                else:
                    return f"R device_connect_btle ERR device {device_id} not found"
            return "R device_connect_btle ERR invalid command"

        elif command == "device_list":
            # Return connected devices
            if self._connected_devices:
                device_list = " | ".join(self._connected_devices)
                return f"R device_list | {device_list} | "
            return "R device_list | "

        elif command.startswith("device_disconnect"):
            parts = command.split()
            if len(parts) >= 2:
                device_id = parts[1]
                self._connected_devices.discard(device_id)
                return "R device_disconnect OK"
            return "R device_disconnect ERR"

        elif command.startswith("device_subscribe"):
            # Subscribe to data stream
            return "R device_subscribe OK"

        elif command == "pause ON":
            return "R pause ON"

        elif command == "pause OFF":
            return "R pause OFF"

        else:
            return f"R {command} ERR unknown command"

    def connect_and_monitor_e4(self, device_id: str):
        """
        Connect to and monitor an E4 device.

        Parameters
        ----------
        device_id : str
            Device identifier
        """
        if device_id not in self._mock_devices:
            raise ValueError(f"Device {device_id} not found")

        self._connected_devices.add(device_id)
        self.connected_event.set()


class MockEmpaticaE4:
    """
    Mock Empatica E4 device with realistic data streaming.

    Simulates the EmpaticaE4 helper class for testing.
    """

    def __init__(self, device_id: str):
        """
        Initialize mock E4 device.

        Parameters
        ----------
        device_id : str
            Device identifier (e.g., "A01234")
        """
        self.device_id = device_id
        self.connected = False

        # Data queue for LSL streaming
        self.lsl_data_queue = Queue()

        # Data generators
        self.acc_generator = AccelerometerDataGenerator(sampling_rate=32.0)
        self.bvp_generator = BVPDataGenerator(sampling_rate=64.0)
        self.gsr_generator = GSRDataGenerator(sampling_rate=4.0)
        self.temp_generator = TemperatureDataGenerator(sampling_rate=4.0)

        # Streaming control
        self._streaming_thread = None
        self._stop_streaming = threading.Event()
        self._suspended = False

        # Subscription flags
        self._subscribed_streams = {
            'acc': False,
            'bvp': False,
            'gsr': False,
            'tmp': False,
            'tag': False,
            'ibi': False,
            'hr': False,
            'bat': False
        }

    def connect(self):
        """Connect to the mock E4 device."""
        time.sleep(0.2)  # Simulate connection delay
        self.connected = True

    def disconnect(self):
        """Disconnect from the device."""
        self.stop_streaming()
        self.connected = False

    def subscribe_to_stream(self, stream_type: str):
        """
        Subscribe to a data stream.

        Parameters
        ----------
        stream_type : str
            Stream type ('acc', 'bvp', 'gsr', 'tmp', 'tag', 'ibi', 'hr', 'bat')
        """
        if stream_type in self._subscribed_streams:
            self._subscribed_streams[stream_type] = True

    def suspend_streaming(self):
        """Suspend data streaming."""
        self._suspended = True

    def start_streaming(self):
        """Start data streaming."""
        if not self.connected:
            raise RuntimeError("Device not connected")

        self._suspended = False
        self._stop_streaming.clear()
        self._streaming_thread = threading.Thread(target=self._stream_data, daemon=True)
        self._streaming_thread.start()

    def stop_streaming(self):
        """Stop data streaming."""
        self._stop_streaming.set()
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=2.0)

    def _stream_data(self):
        """
        Internal method to stream mock data to queue.

        Generates realistic E4 data in the format expected by the streamer.
        """
        # Streaming intervals
        acc_interval = 1 / 32.0   # 32 Hz
        bvp_interval = 1 / 64.0   # 64 Hz
        gsr_interval = 1 / 4.0    # 4 Hz
        tmp_interval = 1 / 4.0    # 4 Hz

        last_acc_time = time.time()
        last_bvp_time = time.time()
        last_gsr_time = time.time()
        last_tmp_time = time.time()

        device_start_time = time.time()

        while not self._stop_streaming.is_set() and not self._suspended:
            current_time = time.time()
            device_timestamp = current_time - device_start_time

            messages = []

            # Generate accelerometer data
            if self._subscribed_streams['acc'] and (current_time - last_acc_time) >= acc_interval:
                acc_data, _ = self.acc_generator.generate(num_samples=1)
                # Format: E4_Acc timestamp x y z
                messages.append(
                    f"E4_Acc {device_timestamp:.3f} "
                    f"{int(acc_data[0, 0] * 64)} "
                    f"{int(acc_data[1, 0] * 64)} "
                    f"{int(acc_data[2, 0] * 64)}"
                )
                last_acc_time = current_time

            # Generate BVP data
            if self._subscribed_streams['bvp'] and (current_time - last_bvp_time) >= bvp_interval:
                bvp_data, _ = self.bvp_generator.generate(num_samples=1)
                # Format: E4_Bvp timestamp value
                messages.append(f"E4_Bvp {device_timestamp:.3f} {bvp_data[0, 0]:.2f}")
                last_bvp_time = current_time

            # Generate GSR data
            if self._subscribed_streams['gsr'] and (current_time - last_gsr_time) >= gsr_interval:
                gsr_data, _ = self.gsr_generator.generate(num_samples=1)
                # Format: E4_Gsr timestamp value
                messages.append(f"E4_Gsr {device_timestamp:.3f} {gsr_data[0, 0]:.3f}")
                last_gsr_time = current_time

            # Generate temperature data
            if self._subscribed_streams['tmp'] and (current_time - last_tmp_time) >= tmp_interval:
                tmp_data, _ = self.temp_generator.generate(num_samples=1)
                # Format: E4_Temperature timestamp value
                messages.append(f"E4_Temperature {device_timestamp:.3f} {tmp_data[0, 0]:.2f}")
                last_tmp_time = current_time

            # Put messages in queue
            if messages:
                response = "\n".join(messages) + "\n"
                self.lsl_data_queue.put(response)

            # Sleep briefly
            time.sleep(0.01)

    def is_connected(self) -> bool:
        """Check if device is connected."""
        return self.connected


class MockE4:
    """
    High-level mock E4 interface for testing.

    Combines server and device mocking for convenient test setup.
    """

    def __init__(self, device_id: str = "A01234"):
        """
        Initialize mock E4.

        Parameters
        ----------
        device_id : str
            Device identifier
        """
        self.device_id = device_id
        self.server = MockEmpaticaServer()
        self.device = MockEmpaticaE4(device_id)

    def discover(self) -> List[str]:
        """Discover E4 devices."""
        return self.server.find_e4s()

    def connect(self) -> bool:
        """Connect to the E4 device."""
        try:
            self.device.connect()
            return True
        except Exception:
            return False

    def disconnect(self):
        """Disconnect from the device."""
        self.device.disconnect()

    def subscribe_all(self):
        """Subscribe to all available data streams."""
        for stream in ['acc', 'bvp', 'gsr', 'tmp', 'tag']:
            self.device.subscribe_to_stream(stream)

    def start_streaming(self):
        """Start data streaming."""
        self.device.start_streaming()

    def stop_streaming(self):
        """Stop data streaming."""
        self.device.stop_streaming()

    def get_data_queue(self) -> Queue:
        """Get the data queue for reading streamed data."""
        return self.device.lsl_data_queue


def create_mock_e4_device(device_id: str = "A01234") -> MockE4:
    """
    Convenience function to create a mock E4 device for testing.

    Parameters
    ----------
    device_id : str
        Device identifier

    Returns
    -------
    MockE4
        Configured mock E4 device

    Example
    -------
    >>> e4 = create_mock_e4_device("A01234")
    >>> e4.connect()
    >>> e4.subscribe_all()
    >>> e4.start_streaming()
    >>> # Get data
    >>> data = e4.get_data_queue().get(timeout=1)
    >>> print(data)
    >>> e4.stop_streaming()
    >>> e4.disconnect()
    """
    return MockE4(device_id)

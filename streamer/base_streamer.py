"""
Base Streamer Interface for StreamSense Hardware Devices

This module provides a standardized base class for all hardware streamers that use
multiprocessing for device isolation. It implements common patterns for:
- Process-based streaming
- Connection management
- LSL outlet setup
- Lifecycle management
- Context manager protocol for proper resource cleanup

Design Principles:
- Use multiprocessing.Process for device isolation and true parallelism
- Use multiprocessing.Event for reliable stop signaling across processes
- Use multiprocessing.Queue for inter-process status communication
- Implement context manager protocol for automatic cleanup
- Provide clear lifecycle hooks for subclass customization

Author: StreamSense Team
Date: November 5, 2025
"""

import logging
from abc import ABC, abstractmethod
from multiprocessing import Process, Event, Queue
from queue import Empty
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class BaseStreamer(ABC):
    """
    Abstract base class for all hardware streamers.

    This class provides common functionality for device streaming using multiprocessing:
    - Process lifecycle management (start, stop, cleanup)
    - Connection status tracking
    - Queue-based status communication
    - Context manager support for automatic cleanup

    Subclasses must implement:
    - _stream_wrapper(): The main streaming logic that runs in the process
    - _setup_lsl_outlets(): LSL outlet creation for the specific device

    Example usage:
        >>> with MyStreamer(device_name="Device1", ...) as streamer:
        >>>     streamer.start_streaming()
        >>>     # ... streaming happens ...
        >>>     # Automatic cleanup on context exit
    """

    def __init__(
        self,
        device_name: str,
        synchronized_start_time: float,
        root_output_folder: str,
        **kwargs
    ):
        """
        Initialize the base streamer.

        Args:
            device_name: Unique identifier for the device
            synchronized_start_time: Synchronized timestamp for multi-device coordination
            root_output_folder: Output directory for logs and data
            **kwargs: Additional device-specific parameters
        """
        self.device_name = device_name
        self.synchronized_start_time = synchronized_start_time
        self.root_output_folder = root_output_folder

        # Process management
        self.process: Optional[Process] = None
        self.stop_signal = Event()  # Use Event for cross-process reliability
        self.connected_event = Event()
        self.queue = Queue()  # For status communication from process to main

        # State tracking
        self.streaming = False
        self.connected = False

        logger.info(f"Initialized {self.__class__.__name__} for device: {device_name}")

    def start_streaming(self, timeout: int = 10) -> bool:
        """
        Start the streaming process.

        This method:
        1. Creates a new Process running _stream_wrapper()
        2. Waits for connection confirmation via the queue
        3. Sets streaming state on success

        Args:
            timeout: Maximum seconds to wait for connection confirmation

        Returns:
            True if streaming started successfully, False otherwise
        """
        if self.streaming:
            logger.warning(f"{self.device_name}: Streaming already in progress, ignoring start_streaming")
            return False

        logger.info(f"{self.device_name}: Starting streaming process")

        # Reset state
        self.connected_event.clear()
        self.stop_signal.clear()

        # Create and start the process
        self.process = Process(target=self._stream_wrapper, name=f"Streamer-{self.device_name}")
        self.process.start()

        # Wait for connection confirmation
        try:
            result = self.queue.get(timeout=timeout)
            if result == 'connected':
                self.connected_event.set()
                self.streaming = True
                self.connected = True
                logger.info(f"{self.device_name}: Streaming started successfully")
                print(f"✓ {self.device_name}: Streaming started successfully")
                return True
            else:
                logger.warning(f"{self.device_name}: Unexpected queue result: {result}")
                self._cleanup()
                return False
        except Empty:
            logger.error(f"{self.device_name}: Timed out waiting for connection confirmation after {timeout}s")
            print(f"✗ {self.device_name}: Connection timeout")
            self._cleanup()
            return False

    def stop_streaming(self, timeout: int = 5) -> None:
        """
        Stop the streaming process.

        This method:
        1. Sets the stop signal (Event)
        2. Waits for the process to terminate gracefully
        3. Forces termination if needed
        4. Cleans up resources

        Args:
            timeout: Maximum seconds to wait for graceful termination
        """
        if not self.streaming and self.process is None:
            logger.debug(f"{self.device_name}: stop_streaming called but not streaming")
            return

        logger.info(f"{self.device_name}: Stopping streaming")
        self.stop_signal.set()

        if self.process and self.process.is_alive():
            logger.debug(f"{self.device_name}: Waiting for process to terminate (timeout={timeout}s)")
            self.process.join(timeout=timeout)

            # Force termination if still alive
            if self.process.is_alive():
                logger.warning(f"{self.device_name}: Process did not terminate gracefully, forcing termination")
                self.process.terminate()
                self.process.join(timeout=2)

                if self.process.is_alive():
                    logger.error(f"{self.device_name}: Process still alive after terminate, sending SIGKILL")
                    self.process.kill()
                    self.process.join(timeout=1)

        self._cleanup()
        logger.info(f"{self.device_name}: Streaming stopped")
        print(f"✓ {self.device_name}: Streaming stopped")

    def _cleanup(self) -> None:
        """
        Clean up resources after streaming stops.

        This method resets all state and clears events. Subclasses can override
        to add device-specific cleanup, but should call super()._cleanup().
        """
        self.process = None
        self.streaming = False
        self.connected = False
        self.connected_event.clear()
        logger.debug(f"{self.device_name}: Cleanup completed")

    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self.streaming and self.process is not None and self.process.is_alive()

    def is_connected(self) -> bool:
        """Check if device is connected."""
        return self.connected_event.is_set()

    # Abstract methods that subclasses must implement

    @abstractmethod
    def _stream_wrapper(self) -> None:
        """
        Main streaming logic that runs in the process.

        This method should:
        1. Connect to the device
        2. Set up LSL outlets via _setup_lsl_outlets()
        3. Send 'connected' to self.queue when ready
        4. Enter main streaming loop, checking self.stop_signal.is_set()
        5. Handle reconnection on connection loss
        6. Clean up on exit

        Example implementation:
            def _stream_wrapper(self):
                try:
                    self._connect_device()
                    self._setup_lsl_outlets()
                    self.queue.put('connected')

                    while not self.stop_signal.is_set():
                        data = self._read_device_data()
                        self._push_to_lsl(data)
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                finally:
                    self._disconnect_device()
        """
        pass

    @abstractmethod
    def _setup_lsl_outlets(self) -> None:
        """
        Set up LSL outlets for the device.

        This method should create pylsl.StreamInfo and pylsl.StreamOutlet
        objects for each data stream the device provides (e.g., EEG, ACC, PPG).

        Example implementation:
            def _setup_lsl_outlets(self):
                info = StreamInfo('DeviceName_EEG', 'EEG', 5, 256, 'float32', 'device123')
                self.eeg_outlet = StreamOutlet(info, chunk_size=32)
        """
        pass

    # Context manager protocol

    def __enter__(self):
        """
        Context manager entry.

        Usage:
            with MyStreamer(...) as streamer:
                streamer.start_streaming()
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - ensures cleanup on exit.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred

        Returns:
            False to propagate any exceptions
        """
        try:
            self.stop_streaming()
        except Exception as e:
            logger.error(f"{self.device_name}: Error during context manager cleanup: {e}")
        return False

    def __repr__(self) -> str:
        """String representation of the streamer."""
        return (f"{self.__class__.__name__}(device_name='{self.device_name}', "
                f"streaming={self.streaming}, connected={self.connected})")


class StreamerError(Exception):
    """Base exception for streamer-related errors."""
    pass


class ConnectionError(StreamerError):
    """Raised when device connection fails."""
    pass


class StreamingError(StreamerError):
    """Raised when streaming operation fails."""
    pass

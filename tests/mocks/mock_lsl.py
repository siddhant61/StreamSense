"""
Mock Lab Streaming Layer (LSL) components for testing.

Provides lightweight mocks for pylsl components to test stream creation
and data publishing without requiring LSL runtime.
"""

from typing import List, Optional, Any
from collections import deque
import time


class MockLSLStreamInfo:
    """Mock pylsl.StreamInfo for testing stream configuration."""

    def __init__(self, name: str, type: str, channel_count: int,
                 nominal_srate: float, channel_format: str, source_id: str):
        """
        Initialize mock stream info.

        Parameters
        ----------
        name : str
            Stream name
        type : str
            Stream type (e.g., 'EEG', 'PPG', 'ACC')
        channel_count : int
            Number of channels
        nominal_srate : float
            Nominal sampling rate in Hz
        channel_format : str
            Data format (e.g., 'float32', 'int32')
        source_id : str
            Unique source identifier
        """
        self.name = name
        self.type = type
        self.channel_count = channel_count
        self.nominal_srate = nominal_srate
        self.channel_format = channel_format
        self.source_id = source_id
        self._desc = MockLSLXMLElement("info")

    def desc(self):
        """Return XML description element."""
        return self._desc

    def __repr__(self):
        return (f"MockLSLStreamInfo(name={self.name}, type={self.type}, "
                f"channels={self.channel_count}, rate={self.nominal_srate} Hz)")


class MockLSLXMLElement:
    """Mock XML element for LSL stream metadata."""

    def __init__(self, name: str):
        self.name = name
        self.children = []
        self.value = None

    def append_child(self, name: str):
        """Append a child element."""
        child = MockLSLXMLElement(name)
        self.children.append(child)
        return child

    def append_child_value(self, name: str, value: str):
        """Append a child element with a value."""
        child = MockLSLXMLElement(name)
        child.value = value
        self.children.append(child)
        return self


class MockLSLOutlet:
    """
    Mock pylsl.StreamOutlet for testing data publication.

    Captures published samples for verification in tests.
    """

    def __init__(self, stream_info: MockLSLStreamInfo, chunk_size: int = 0,
                 max_buffered: int = 360):
        """
        Initialize mock LSL outlet.

        Parameters
        ----------
        stream_info : MockLSLStreamInfo
            Stream configuration
        chunk_size : int
            Desired chunk granularity (0 for default)
        max_buffered : int
            Maximum buffered samples
        """
        self.stream_info = stream_info
        self.chunk_size = chunk_size
        self.max_buffered = max_buffered

        # Storage for published data
        self.samples = deque(maxlen=max_buffered)
        self.timestamps = deque(maxlen=max_buffered)

        self._is_active = True

    def push_sample(self, sample: List[float], timestamp: Optional[float] = None,
                    pushthrough: bool = True):
        """
        Push a single sample to the outlet.

        Parameters
        ----------
        sample : List[float]
            Sample data (length must match channel_count)
        timestamp : float, optional
            Sample timestamp (uses current time if None)
        pushthrough : bool
            Whether to push immediately (ignored in mock)
        """
        if not self._is_active:
            raise RuntimeError("Cannot push to inactive outlet")

        if len(sample) != self.stream_info.channel_count:
            raise ValueError(
                f"Sample length {len(sample)} does not match "
                f"channel count {self.stream_info.channel_count}"
            )

        if timestamp is None:
            timestamp = time.time()

        self.samples.append(list(sample))
        self.timestamps.append(timestamp)

    def push_chunk(self, samples: List[List[float]], timestamps: Optional[List[float]] = None,
                   pushthrough: bool = True):
        """
        Push a chunk of samples to the outlet.

        Parameters
        ----------
        samples : List[List[float]]
            List of samples to push
        timestamps : List[float], optional
            Timestamps for each sample
        pushthrough : bool
            Whether to push immediately (ignored in mock)
        """
        if not self._is_active:
            raise RuntimeError("Cannot push to inactive outlet")

        if timestamps is None:
            timestamps = [time.time() + i * (1.0 / self.stream_info.nominal_srate)
                         for i in range(len(samples))]

        if len(samples) != len(timestamps):
            raise ValueError("Number of samples must match number of timestamps")

        for sample, timestamp in zip(samples, timestamps):
            self.push_sample(sample, timestamp, pushthrough=False)

    def have_consumers(self) -> bool:
        """Check if any consumers are connected."""
        return True  # Always return True in mock

    def wait_for_consumers(self, timeout: float = 0.0) -> bool:
        """Wait for consumers to connect."""
        return True  # Immediately return True in mock

    def get_samples(self, clear: bool = False) -> List[List[float]]:
        """
        Retrieve captured samples.

        Parameters
        ----------
        clear : bool
            Whether to clear the sample buffer after retrieval

        Returns
        -------
        List[List[float]]
            List of captured samples
        """
        samples = list(self.samples)
        if clear:
            self.samples.clear()
            self.timestamps.clear()
        return samples

    def get_timestamps(self, clear: bool = False) -> List[float]:
        """
        Retrieve captured timestamps.

        Parameters
        ----------
        clear : bool
            Whether to clear the timestamp buffer after retrieval

        Returns
        -------
        List[float]
            List of captured timestamps
        """
        timestamps = list(self.timestamps)
        if clear:
            self.samples.clear()
            self.timestamps.clear()
        return timestamps

    def get_sample_count(self) -> int:
        """Get number of captured samples."""
        return len(self.samples)

    def clear(self):
        """Clear all captured samples and timestamps."""
        self.samples.clear()
        self.timestamps.clear()

    def close(self):
        """Close the outlet."""
        self._is_active = False

    def __repr__(self):
        return (f"MockLSLOutlet({self.stream_info.name}, "
                f"samples_captured={len(self.samples)})")


def create_mock_lsl_outlet(name: str, stream_type: str, num_channels: int,
                           sampling_rate: float, data_format: str = 'float32',
                           source_id: Optional[str] = None) -> MockLSLOutlet:
    """
    Convenience function to create a mock LSL outlet.

    Parameters
    ----------
    name : str
        Stream name
    stream_type : str
        Stream type (e.g., 'EEG', 'PPG')
    num_channels : int
        Number of channels
    sampling_rate : float
        Sampling rate in Hz
    data_format : str
        Data format (default: 'float32')
    source_id : str, optional
        Source identifier (auto-generated if None)

    Returns
    -------
    MockLSLOutlet
        Configured mock outlet ready for testing

    Example
    -------
    >>> outlet = create_mock_lsl_outlet('TestEEG', 'EEG', 5, 256.0)
    >>> outlet.push_sample([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> assert outlet.get_sample_count() == 1
    """
    if source_id is None:
        source_id = f"mock_{name}_{id(name)}"

    info = MockLSLStreamInfo(
        name=name,
        type=stream_type,
        channel_count=num_channels,
        nominal_srate=sampling_rate,
        channel_format=data_format,
        source_id=source_id
    )

    return MockLSLOutlet(info)


# Mock the local_clock function from pylsl
def mock_local_clock() -> float:
    """
    Mock replacement for pylsl.local_clock().

    Returns current time in seconds with high precision.
    """
    return time.time()

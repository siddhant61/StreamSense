"""
Pytest configuration and shared fixtures for StreamSense testing.

Provides fixtures for:
- Mock hardware devices (Muse, E4)
- Mock LSL outlets
- Synthetic data generators
- Temporary directories for test outputs
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from multiprocessing import Queue
import time

from tests.mocks import (
    MockMuse,
    MockE4,
    MockEmpaticaServer,
    create_mock_lsl_outlet,
    mock_local_clock
)


@pytest.fixture
def temp_output_dir():
    """
    Create a temporary directory for test outputs.

    Yields
    ------
    Path
        Temporary directory path

    The directory is automatically cleaned up after the test.
    """
    temp_dir = tempfile.mkdtemp(prefix="streamsense_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_muse_device():
    """
    Create a mock Muse headband device.

    Yields
    ------
    MockMuse
        Configured mock Muse device with all queues initialized

    Example
    -------
    def test_muse_connection(mock_muse_device):
        assert mock_muse_device.connect()
        assert mock_muse_device.connected
        mock_muse_device.disconnect()
    """
    muse = MockMuse(
        address="00:55:DA:B0:00:01",
        shared_eeg=Queue(),
        shared_ppg=Queue(),
        shared_acc=Queue(),
        shared_gyro=Queue(),
        shared_tel=Queue(),
        shared_con=Queue(),
        synchronized_start_time=time.time(),
        name="TestMuse",
        enable_eeg=True,
        enable_ppg=True,
        enable_acc=True,
        enable_gyro=True
    )

    yield muse

    # Cleanup
    if muse.connected:
        muse.stop()
        muse.disconnect()


@pytest.fixture
def mock_e4_device():
    """
    Create a mock Empatica E4 wearable device.

    Yields
    ------
    MockE4
        Configured mock E4 device

    Example
    -------
    def test_e4_connection(mock_e4_device):
        assert mock_e4_device.connect()
        mock_e4_device.subscribe_all()
        mock_e4_device.start_streaming()
        data = mock_e4_device.get_data_queue().get(timeout=1)
        assert data is not None
        mock_e4_device.stop_streaming()
        mock_e4_device.disconnect()
    """
    e4 = MockE4(device_id="A01234")

    yield e4

    # Cleanup
    if e4.device.connected:
        e4.stop_streaming()
        e4.disconnect()


@pytest.fixture
def mock_empatica_server():
    """
    Create a mock Empatica BLE server.

    Yields
    ------
    MockEmpaticaServer
        Mock server for E4 device discovery and connection

    Example
    -------
    def test_e4_discovery(mock_empatica_server):
        devices = mock_empatica_server.find_e4s()
        assert len(devices) > 0
    """
    server = MockEmpaticaServer()
    yield server


@pytest.fixture
def mock_lsl_eeg_outlet():
    """
    Create a mock LSL outlet for EEG data.

    Yields
    ------
    MockLSLOutlet
        5-channel EEG outlet at 256 Hz

    Example
    -------
    def test_eeg_streaming(mock_lsl_eeg_outlet):
        sample = [1.0, 2.0, 3.0, 4.0, 5.0]
        mock_lsl_eeg_outlet.push_sample(sample)
        assert mock_lsl_eeg_outlet.get_sample_count() == 1
    """
    outlet = create_mock_lsl_outlet(
        name="TestEEG",
        stream_type="EEG",
        num_channels=5,
        sampling_rate=256.0
    )

    yield outlet

    outlet.close()


@pytest.fixture
def mock_lsl_ppg_outlet():
    """
    Create a mock LSL outlet for PPG data.

    Yields
    ------
    MockLSLOutlet
        3-channel PPG outlet at 64 Hz
    """
    outlet = create_mock_lsl_outlet(
        name="TestPPG",
        stream_type="PPG",
        num_channels=3,
        sampling_rate=64.0
    )

    yield outlet

    outlet.close()


@pytest.fixture
def mock_lsl_e4_bvp_outlet():
    """
    Create a mock LSL outlet for E4 BVP data.

    Yields
    ------
    MockLSLOutlet
        1-channel BVP outlet at 64 Hz
    """
    outlet = create_mock_lsl_outlet(
        name="TestBVP",
        stream_type="BVP",
        num_channels=1,
        sampling_rate=64.0
    )

    yield outlet

    outlet.close()


@pytest.fixture
def mock_lsl_e4_gsr_outlet():
    """
    Create a mock LSL outlet for E4 GSR data.

    Yields
    ------
    MockLSLOutlet
        1-channel GSR outlet at 4 Hz
    """
    outlet = create_mock_lsl_outlet(
        name="TestGSR",
        stream_type="GSR",
        num_channels=1,
        sampling_rate=4.0
    )

    yield outlet

    outlet.close()


@pytest.fixture
def synchronized_start_time():
    """
    Provide a synchronized start time for device coordination.

    Returns
    -------
    float
        Current timestamp for synchronization
    """
    return time.time()


@pytest.fixture
def mock_queues():
    """
    Create a set of multiprocessing queues for device communication.

    Returns
    -------
    dict
        Dictionary with queues for 'eeg', 'ppg', 'acc', 'gyro', 'tel', 'con'

    Example
    -------
    def test_data_flow(mock_queues):
        mock_queues['eeg'].put(([1, 2, 3], [0.1, 0.2, 0.3]))
        data, timestamps = mock_queues['eeg'].get()
        assert len(data) == 3
    """
    return {
        'eeg': Queue(),
        'ppg': Queue(),
        'acc': Queue(),
        'gyro': Queue(),
        'tel': Queue(),
        'con': Queue()
    }


# Monkeypatch helper for replacing real hardware with mocks
@pytest.fixture
def patch_hardware_imports(monkeypatch):
    """
    Fixture to monkeypatch hardware dependencies with mocks.

    Use this to replace real hardware imports with mocks in tests.

    Example
    -------
    def test_with_mocked_hardware(patch_hardware_imports, monkeypatch):
        from tests.mocks import MockBGAPIBackend
        monkeypatch.setattr('helper.serial_helper.BGAPIBackend', MockBGAPIBackend)
        # Now code importing BGAPIBackend will get the mock
    """
    pass  # This is a marker fixture, actual patching done in tests


# Configuration for test execution
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "hardware: marks tests that interact with hardware mocks"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests that test multiple components"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests that are slow to execute"
    )

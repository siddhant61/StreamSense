  # Hardware Mocking Infrastructure

This directory contains comprehensive hardware mocking utilities for testing StreamSense without physical devices.

## Overview

The mocking infrastructure provides lightweight, realistic simulations of:
- **Muse EEG headbands** - Bluetooth device with EEG, PPG, accelerometer, gyroscope
- **Empatica E4 wearables** - Wrist device with BVP, GSR, temperature, accelerometer
- **Lab Streaming Layer (LSL)** - Data publication outlets
- **Physiological data generators** - Synthetic EEG, PPG, BVP, GSR, temperature signals

## Architecture

```
tests/mocks/
├── __init__.py              # Package exports
├── data_generators.py       # Synthetic physiological signal generators
├── mock_lsl.py             # LSL outlet and stream mocking
├── mock_muse.py            # Muse headband device mocking
├── mock_e4.py              # Empatica E4 wearable mocking
└── README.md               # This file

tests/
├── conftest.py             # Pytest fixtures for easy test setup
└── test_hardware_mocks.py  # Validation tests and usage examples
```

## Quick Start

### Using Pytest Fixtures (Recommended)

The easiest way to use mocks is through pytest fixtures:

```python
def test_muse_connection(mock_muse_device):
    """Test with a pre-configured mock Muse device."""
    assert mock_muse_device.connect()
    assert mock_muse_device.connected
    mock_muse_device.disconnect()


def test_eeg_streaming(mock_muse_device, mock_lsl_eeg_outlet):
    """Test EEG data flow."""
    mock_muse_device.connect()
    mock_muse_device.start()

    # Get EEG data
    eeg_data, timestamps = mock_muse_device.shared_eeg.get(timeout=1.0)

    # Push to LSL
    for i in range(eeg_data.shape[1]):
        mock_lsl_eeg_outlet.push_sample(eeg_data[:, i].tolist())

    assert mock_lsl_eeg_outlet.get_sample_count() > 0

    mock_muse_device.stop()
    mock_muse_device.disconnect()
```

### Creating Mocks Manually

You can also create mocks directly:

```python
from tests.mocks import MockMuse, create_mock_lsl_outlet
from multiprocessing import Queue
import time

# Create mock Muse
muse = MockMuse(
    address="00:55:DA:B0:00:01",
    shared_eeg=Queue(),
    shared_ppg=Queue(),
    shared_acc=Queue(),
    shared_gyro=Queue(),
    shared_tel=Queue(),
    shared_con=Queue(),
    synchronized_start_time=time.time(),
    name="TestMuse"
)

# Connect and stream
muse.connect()
muse.start()

# Get data
eeg_data, eeg_timestamps = muse.shared_eeg.get(timeout=1.0)
print(f"Received EEG: {eeg_data.shape}")

# Cleanup
muse.stop()
muse.disconnect()
```

## Available Fixtures

All fixtures are defined in `tests/conftest.py`:

| Fixture | Description |
|---------|-------------|
| `mock_muse_device` | Pre-configured Muse headband with queues |
| `mock_e4_device` | Pre-configured E4 wearable |
| `mock_empatica_server` | Mock Empatica BLE server |
| `mock_lsl_eeg_outlet` | LSL outlet for 5-channel EEG at 256 Hz |
| `mock_lsl_ppg_outlet` | LSL outlet for 3-channel PPG at 64 Hz |
| `mock_lsl_e4_bvp_outlet` | LSL outlet for BVP at 64 Hz |
| `mock_lsl_e4_gsr_outlet` | LSL outlet for GSR at 4 Hz |
| `temp_output_dir` | Temporary directory (auto-cleanup) |
| `synchronized_start_time` | Synchronized timestamp for multi-device tests |
| `mock_queues` | Set of multiprocessing queues |

## Data Generators

Generate realistic synthetic physiological signals:

```python
from tests.mocks import (
    generate_eeg_sample,
    generate_ppg_sample,
    generate_bvp_sample,
    generate_gsr_sample
)

# Generate EEG data (5 channels, 12 samples)
eeg_data, eeg_timestamps = generate_eeg_sample(num_channels=5, num_samples=12)
assert eeg_data.shape == (5, 12)

# Generate PPG data (3 channels, 6 samples)
ppg_data, ppg_timestamps = generate_ppg_sample(num_channels=3, num_samples=6)
assert ppg_data.shape == (3, 6)

# Generate BVP data (1 channel, 10 samples)
bvp_data, bvp_timestamps = generate_bvp_sample(num_samples=10)
assert bvp_data.shape == (1, 10)
```

### Data Characteristics

All generators produce physiologically plausible signals:

| Signal | Range | Frequency Components |
|--------|-------|---------------------|
| **EEG** | ±200 μV | Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-100 Hz) |
| **PPG** | 0-200 | Cardiac pulse (~1 Hz) with harmonics |
| **BVP** | Variable | Heart rate (~70 bpm default) |
| **GSR** | 0-20 μS | Tonic level + phasic responses |
| **Temperature** | 30-35°C | Slow drift with noise |
| **Accelerometer** | ±2 g | Gravity + motion components |
| **Gyroscope** | ±10 dps | Rotational movements |

## Mock Components

### MockMuse

Simulates a Muse headband with all sensors:

```python
from tests.mocks import MockMuse
from multiprocessing import Queue

muse = MockMuse(
    address="00:55:DA:B0:00:01",
    shared_eeg=Queue(),
    # ... other queues
    name="MyMuse",
    enable_eeg=True,
    enable_ppg=True,
    enable_acc=True,
    enable_gyro=True
)

# Connect
assert muse.connect()

# Start streaming (runs in background thread)
muse.start()

# Get data from queues
eeg_data, timestamps = muse.shared_eeg.get(timeout=1.0)
ppg_data, timestamps = muse.shared_ppg.get(timeout=1.0)

# Stop
muse.stop()
muse.disconnect()
```

**Key methods:**
- `connect()` - Establish connection (returns bool)
- `disconnect()` - Close connection
- `start()` - Begin data streaming in background
- `stop()` - Stop streaming
- `start_keep_alive()` - No-op (for compatibility)

**Data queues:**
- `shared_eeg` - 5-channel EEG at 256 Hz (12-sample chunks)
- `shared_ppg` - 3-channel PPG at 64 Hz (6-sample chunks)
- `shared_acc` - 3-axis accelerometer at 52 Hz
- `shared_gyro` - 3-axis gyroscope at 52 Hz

### MockE4

Simulates an Empatica E4 wearable:

```python
from tests.mocks import MockE4

e4 = MockE4(device_id="A01234")

# Connect
assert e4.connect()

# Subscribe to streams
e4.subscribe_all()

# Start streaming
e4.start_streaming()

# Get data (formatted as E4 server protocol)
data_queue = e4.get_data_queue()
data_str = data_queue.get(timeout=1.0)

# Parse data
# Format: "E4_Bvp 123.456 789.0\nE4_Gsr 123.456 2.5\n..."

# Stop
e4.stop_streaming()
e4.disconnect()
```

**Key methods:**
- `connect()` - Connect to device
- `disconnect()` - Disconnect
- `subscribe_all()` - Subscribe to all data streams
- `start_streaming()` - Begin data generation
- `stop_streaming()` - Stop streaming
- `get_data_queue()` - Get queue for reading data

**Data streams:**
- `E4_Acc` - 3-axis accelerometer at 32 Hz
- `E4_Bvp` - Blood volume pulse at 64 Hz
- `E4_Gsr` - Galvanic skin response at 4 Hz
- `E4_Temperature` - Skin temperature at 4 Hz
- `E4_Tag` - Event markers (when enabled)

### MockLSLOutlet

Mock LSL outlet that captures published data:

```python
from tests.mocks import create_mock_lsl_outlet

# Create outlet
outlet = create_mock_lsl_outlet(
    name="TestEEG",
    stream_type="EEG",
    num_channels=5,
    sampling_rate=256.0
)

# Push samples
outlet.push_sample([1.0, 2.0, 3.0, 4.0, 5.0], timestamp=123.456)

# Push chunks
chunk = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
outlet.push_chunk(chunk)

# Retrieve captured data
samples = outlet.get_samples()
timestamps = outlet.get_timestamps()
count = outlet.get_sample_count()

# Clear buffer
outlet.clear()

# Close outlet
outlet.close()
```

**Key methods:**
- `push_sample(data, timestamp)` - Publish single sample
- `push_chunk(data_list, timestamps)` - Publish multiple samples
- `get_samples(clear=False)` - Retrieve captured samples
- `get_timestamps(clear=False)` - Retrieve timestamps
- `get_sample_count()` - Count captured samples
- `clear()` - Clear buffer
- `close()` - Close outlet

## Monkeypatching for Integration Tests

Replace real hardware with mocks in integration tests:

```python
import pytest
from tests.mocks import MockBGAPIBackend, MockEmpaticaServer

def test_device_discovery(monkeypatch):
    """Test device discovery with mocked hardware."""

    # Replace real BGAPI backend with mock
    monkeypatch.setattr(
        'helper.serial_helper.BGAPIBackend',
        MockBGAPIBackend
    )

    # Replace Empatica server
    monkeypatch.setattr(
        'helper.e4_helper.EmpaticaServer',
        MockEmpaticaServer
    )

    # Now test code using these classes will get mocks
    from helper.find_devices import FindDevices

    muses = FindDevices.find_muse()
    assert len(muses) > 0

    e4s = FindDevices.find_empatica()
    assert len(e4s) > 0
```

## Writing New Tests

### Template for Device Tests

```python
import pytest
from tests.mocks import MockMuse, create_mock_lsl_outlet
from multiprocessing import Queue
import time

class TestMyFeature:
    """Test my new feature with mocked hardware."""

    def test_feature_with_muse(self, mock_muse_device):
        """Test using pytest fixture."""
        # Setup
        mock_muse_device.connect()
        mock_muse_device.start()

        # Test your feature
        # ...

        # Cleanup (handled by fixture)

    def test_feature_manual_mock(self):
        """Test with manually created mock."""
        # Create mock
        muse = MockMuse(
            address="00:55:DA:B0:00:01",
            shared_eeg=Queue(),
            shared_ppg=Queue(),
            shared_acc=Queue(),
            shared_gyro=Queue(),
            shared_tel=Queue(),
            shared_con=Queue(),
            synchronized_start_time=time.time()
        )

        try:
            # Test
            assert muse.connect()
            # ...
        finally:
            # Cleanup
            if muse.connected:
                muse.stop()
                muse.disconnect()
```

### Best Practices

1. **Use fixtures when possible** - They handle setup and cleanup automatically
2. **Test with realistic data** - The generators produce physiologically plausible signals
3. **Verify data shapes** - Check channel counts and sample counts
4. **Test error conditions** - Mock wrong channel counts, connection failures, etc.
5. **Use integration markers** - Mark slow/integration tests: `@pytest.mark.integration`

## Extending the Mocks

### Adding New Sensors

To add a new sensor to MockMuse:

1. Add generator in `data_generators.py`:
```python
class NewSensorGenerator(PhysiologicalDataGenerator):
    def generate(self, num_samples=1):
        # Generate synthetic data
        ...
```

2. Add to MockMuse in `mock_muse.py`:
```python
class MockMuse:
    def __init__(self, ..., enable_newsensor=True):
        self.newsensor_generator = NewSensorGenerator()
        self.enable_newsensor = enable_newsensor
```

3. Update streaming in `_stream_data()`:
```python
if self.enable_newsensor and ...:
    data, timestamps = self.newsensor_generator.generate()
    self.shared_newsensor.put((data, timestamps))
```

### Adding New Device Types

1. Create new file `mock_newdevice.py`
2. Implement data generators
3. Implement mock device class
4. Add to `__init__.py` exports
5. Add pytest fixture in `conftest.py`
6. Write validation tests

## Troubleshooting

**No data in queue:**
- Ensure `start()` was called after `connect()`
- Check that streaming is enabled for that sensor
- Increase timeout: `queue.get(timeout=2.0)`

**Wrong data shape:**
- Verify generator parameters match expectations
- Check num_samples matches sampling rate expectations

**Import errors:**
- Ensure you're importing from `tests.mocks`
- Check that pytest can find the tests directory

**Fixtures not found:**
- Make sure `conftest.py` is in the `tests/` directory
- Run pytest from project root: `pytest tests/`

## Performance

The mocks are designed to be lightweight:
- **Memory**: ~1MB per device (buffered data)
- **CPU**: <1% per streaming device
- **Startup**: <50ms per device connection
- **Teardown**: <100ms per device cleanup

Mocks run in background threads and generate data at realistic rates without blocking tests.

## Future Enhancements

Potential improvements to the mocking infrastructure:

- [ ] Add event injection (button presses, connection drops)
- [ ] Support for multi-device synchronization testing
- [ ] Configurable data corruption/dropout simulation
- [ ] Recording/playback of real device data
- [ ] Mock for recorder LSL inlet (complementing outlets)
- [ ] Mock for visualization components

## References

- [Lab Streaming Layer Documentation](https://labstreaminglayer.readthedocs.io/)
- [Muse Direct Documentation](https://mind-monitor.com/FAQ.php)
- [Empatica E4 Documentation](https://support.empatica.com/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)

---

**Last Updated**: November 2025
**Maintainer**: StreamSense Development Team

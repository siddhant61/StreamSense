# StreamSense Streamer Module

## Overview

The streamer module provides standardized interfaces for streaming physiological data from hardware devices to LSL (Lab Streaming Layer) using multiprocessing for device isolation.

## Architecture

### Base Streamer Pattern

All hardware streamers inherit from `BaseStreamer`, which provides:

- **Process-based isolation**: Each device runs in a separate process for crash isolation and true parallelism
- **Standardized lifecycle**: Common start/stop/cleanup patterns
- **Connection management**: Queue-based status communication and Event-based synchronization
- **Context manager support**: Automatic cleanup using `with` statements
- **Robust error handling**: Proper timeout and cleanup on failures

### Design Principles

1. **Use multiprocessing.Process** for device isolation (not threading.Thread)
2. **Use multiprocessing.Event** for stop signaling (reliable across processes)
3. **Use multiprocessing.Queue** for status communication from child to parent
4. **Implement context managers** for guaranteed resource cleanup
5. **Follow common lifecycle patterns** for consistency

## Module Structure

```
streamer/
├── base_streamer.py          # Abstract base class for all streamers
├── stream_muse.py            # Muse headband streamer
├── stream_e4.py              # Empatica E4 wearable streamer
└── README.md                 # This file
```

## BaseStreamer API

### Class: `BaseStreamer`

Abstract base class providing common streaming functionality.

#### Methods

**`__init__(device_name, synchronized_start_time, root_output_folder, **kwargs)`**
- Initialize the streamer with device information
- Sets up process management infrastructure (queues, events, etc.)

**`start_streaming(timeout=10) -> bool`**
- Start the streaming process
- Returns `True` on success, `False` on failure
- Waits up to `timeout` seconds for connection confirmation

**`stop_streaming(timeout=5) -> None`**
- Stop the streaming process gracefully
- Waits up to `timeout` seconds for graceful termination
- Forces termination if needed

**`is_streaming() -> bool`**
- Check if currently streaming

**`is_connected() -> bool`**
- Check if device is connected

#### Abstract Methods (must be implemented by subclasses)

**`_stream_wrapper() -> None`**
- Main streaming logic that runs in the process
- Should connect to device, set up LSL outlets, signal connection, enter streaming loop

**`_setup_lsl_outlets() -> None`**
- Create LSL outlets for the device's data streams

#### Context Manager Support

```python
with MyStreamer(...) as streamer:
    streamer.start_streaming()
    # ... streaming happens ...
# Automatic cleanup on exit
```

## Creating a New Streamer

To create a new hardware streamer:

1. **Inherit from BaseStreamer**:
```python
from streamer.base_streamer import BaseStreamer

class MyStreamer(BaseStreamer):
    def __init__(self, device_name, synchronized_start_time, root_output_folder):
        super().__init__(device_name, synchronized_start_time, root_output_folder)
        # Add device-specific initialization
```

2. **Implement `_stream_wrapper()`**:
```python
def _stream_wrapper(self):
    try:
        # 1. Connect to device
        self._connect_device()

        # 2. Set up LSL outlets
        self._setup_lsl_outlets()

        # 3. Signal successful connection
        self.queue.put('connected')

        # 4. Main streaming loop
        while not self.stop_signal.is_set():
            data = self._read_device_data()
            self._push_to_lsl(data)

    except Exception as e:
        logger.error(f"Streaming error: {e}")
    finally:
        self._disconnect_device()
```

3. **Implement `_setup_lsl_outlets()`**:
```python
def _setup_lsl_outlets(self):
    from pylsl import StreamInfo, StreamOutlet

    info = StreamInfo(
        f'{self.device_name}_EEG',  # Stream name
        'EEG',                       # Stream type
        5,                           # Channel count
        256,                         # Sampling rate
        'float32',                   # Data type
        f'device_{self.device_name}' # Source ID
    )
    self.eeg_outlet = StreamOutlet(info, chunk_size=32)
```

4. **Use the streamer**:
```python
# With context manager (recommended)
with MyStreamer("Device1", time.time(), "/output") as streamer:
    if streamer.start_streaming(timeout=10):
        # Streaming is active
        time.sleep(60)
    # Automatic cleanup

# Or manually
streamer = MyStreamer("Device1", time.time(), "/output")
streamer.start_streaming()
# ... do work ...
streamer.stop_streaming()
```

## Example Implementations

### Simple Streamer (Minimal)

```python
from streamer.base_streamer import BaseStreamer
from pylsl import StreamInfo, StreamOutlet
import time

class SimpleStreamer(BaseStreamer):
    def _stream_wrapper(self):
        self._setup_lsl_outlets()
        self.queue.put('connected')

        while not self.stop_signal.is_set():
            # Generate dummy data
            data = [1.0, 2.0, 3.0]
            self.outlet.push_sample(data)
            time.sleep(0.01)

    def _setup_lsl_outlets(self):
        info = StreamInfo('Simple_Data', 'Misc', 3, 100, 'float32', 'simple123')
        self.outlet = StreamOutlet(info)
```

### Complex Streamer (With Reconnection)

```python
from streamer.base_streamer import BaseStreamer
from pylsl import StreamInfo, StreamOutlet
import time

class ComplexStreamer(BaseStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = 3

    def _stream_wrapper(self):
        retry_count = 0

        while retry_count < self.max_retries and not self.stop_signal.is_set():
            try:
                # Connect
                self._connect_device()
                self._setup_lsl_outlets()
                self.queue.put('connected')

                # Stream
                while not self.stop_signal.is_set():
                    try:
                        data = self._read_device_data()
                        self.outlet.push_sample(data)
                    except ConnectionLost:
                        # Try to reconnect
                        retry_count += 1
                        break

            except Exception as e:
                logger.error(f"Error: {e}")
                retry_count += 1
                time.sleep(2 ** retry_count)  # Exponential backoff

        self._disconnect_device()

    def _setup_lsl_outlets(self):
        info = StreamInfo('Complex_Data', 'Misc', 5, 256, 'float32', 'complex123')
        self.outlet = StreamOutlet(info)
```

## Migration Guide

### From Old Pattern to BaseStreamer

**Old pattern (mixed threading/multiprocessing):**
```python
class OldStreamer:
    def __init__(self, device):
        self.device = device
        self.process = None
        self.stop_signal = False  # ❌ Bool not reliable across processes

    def start_streaming(self):
        thread = threading.Thread(target=self._start_process)  # ❌ Unnecessary wrapper
        thread.start()
```

**New pattern (BaseStreamer):**
```python
class NewStreamer(BaseStreamer):
    def __init__(self, device, synchronized_start_time, root_output_folder):
        super().__init__(device, synchronized_start_time, root_output_folder)
        # ✓ stop_signal is Event (reliable)
        # ✓ No threading wrapper needed

    def _stream_wrapper(self):
        # Implement device-specific logic
        pass
```

## Testing

All streamers should have comprehensive tests covering:

1. **Initialization**: Verify all attributes set correctly
2. **Lifecycle**: Test start/stop/cleanup
3. **State tracking**: Test is_streaming(), is_connected()
4. **Context manager**: Test automatic cleanup
5. **Error handling**: Test timeouts, connection failures
6. **Integration**: Test full workflow

See `tests/test_base_streamer.py` for examples.

## Best Practices

### DO ✓

- Inherit from `BaseStreamer` for all new streamers
- Use context managers (`with` statement) when possible
- Implement proper cleanup in `_stream_wrapper()`
- Use `self.stop_signal.is_set()` to check for stop signal
- Log important events for debugging
- Test with hardware mocks (see `tests/mocks/`)

### DON'T ✗

- Don't use `threading.Thread` to wrap `multiprocessing.Process`
- Don't use bool for `stop_signal` (use Event)
- Don't forget to call `self.queue.put('connected')` after successful setup
- Don't block indefinitely in `_stream_wrapper()` (check stop_signal regularly)
- Don't access parent process attributes from child process (use Queues)
- Don't forget to call `super().__init__()` in subclass constructors

## Common Patterns

### Pattern 1: Simple Device Connection

```python
def _stream_wrapper(self):
    device = DeviceSDK.connect(self.device_name)
    self._setup_lsl_outlets()
    self.queue.put('connected')

    while not self.stop_signal.is_set():
        sample = device.read()
        self.outlet.push_sample(sample)

    device.disconnect()
```

### Pattern 2: With Reconnection Logic

```python
def _stream_wrapper(self):
    while not self.stop_signal.is_set():
        try:
            device = self._connect_with_retry()
            self._setup_lsl_outlets()
            self.queue.put('connected')

            while not self.stop_signal.is_set():
                sample = device.read()
                self.outlet.push_sample(sample)

        except ConnectionLost:
            logger.warning("Connection lost, reconnecting...")
            time.sleep(2)
```

### Pattern 3: Multi-Stream Device

```python
def _setup_lsl_outlets(self):
    # EEG stream
    eeg_info = StreamInfo(f'{self.device_name}_EEG', 'EEG', 5, 256, 'float32', ...)
    self.eeg_outlet = StreamOutlet(eeg_info)

    # ACC stream
    acc_info = StreamInfo(f'{self.device_name}_ACC', 'ACC', 3, 32, 'float32', ...)
    self.acc_outlet = StreamOutlet(acc_info)

def _stream_wrapper(self):
    device = self._connect()
    self._setup_lsl_outlets()
    self.queue.put('connected')

    while not self.stop_signal.is_set():
        eeg_data, acc_data = device.read_all()
        self.eeg_outlet.push_sample(eeg_data)
        self.acc_outlet.push_sample(acc_data)
```

## Troubleshooting

### Problem: Streamer hangs on start

**Cause**: `_stream_wrapper()` never sends 'connected' to queue

**Solution**: Ensure `self.queue.put('connected')` is called after successful setup

### Problem: Streamer doesn't stop

**Cause**: `_stream_wrapper()` doesn't check `stop_signal` regularly

**Solution**: Add `while not self.stop_signal.is_set()` to main loop

### Problem: Process becomes zombie

**Cause**: Process not properly joined/terminated

**Solution**: Use `BaseStreamer.stop_streaming()` which handles this

### Problem: Changes in child process not visible in parent

**Cause**: Multiprocessing copies memory, doesn't share it

**Solution**: Use Queues or other multiprocessing primitives for communication

## Performance Considerations

### Memory Usage

- Each Process uses ~50-100MB of memory
- Use Process pooling for many devices
- Consider asyncio for I/O-bound workloads (future enhancement)

### CPU Usage

- Processes provide true parallelism (no GIL)
- Good for CPU-bound signal processing
- Each device can use a separate CPU core

### Data Throughput

- LSL can handle 100k+ samples/second
- Bottleneck is usually device connection (Bluetooth, USB)
- Use appropriate chunk sizes for LSL outlets

## Future Enhancements

1. **Asyncio Support**: Migrate to asyncio for I/O-bound operations
2. **Process Pooling**: Reuse processes for multiple devices
3. **Health Monitoring**: Built-in connection health checks
4. **Auto-Reconnection**: Standardized reconnection strategies
5. **Metrics Collection**: Built-in performance monitoring

## References

- [Lab Streaming Layer (LSL) Documentation](https://labstreaminglayer.readthedocs.io/)
- [Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
- [StreamSense Architecture](../audit/PHASE3_CONCURRENCY_ANALYSIS.md)

---

**Author**: StreamSense Team
**Last Updated**: November 5, 2025
**Version**: 1.0

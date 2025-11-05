# StreamSense Concurrency Guidelines

**Author:** StreamSense Team
**Date:** November 5, 2025
**Status:** Active
**Phase:** 3.1 - Architecture Refactoring

---

## Overview

This document provides clear guidelines for when to use different concurrency patterns in StreamSense. Following these guidelines ensures consistency, maintainability, and optimal performance.

## Decision Tree

```
Does the task involve I/O operations (network, serial, LSL)?
├─ YES → Is it device streaming?
│  ├─ YES → Use multiprocessing.Process (device isolation)
│  └─ NO → Consider asyncio (future enhancement)
└─ NO → Is it CPU-intensive?
   ├─ YES → Use multiprocessing.Process (avoid GIL)
   └─ NO → Use threading.Thread or direct execution
```

---

## Pattern 1: multiprocessing.Process (Recommended for Streamers)

### When to Use

✓ **Device streaming** (Muse, E4, etc.)
✓ **CPU-intensive operations** (signal processing, heavy computation)
✓ **Crash isolation required** (one device failure shouldn't crash others)
✓ **True parallelism needed** (utilize multiple CPU cores)

### Benefits

- ✅ True parallelism (bypasses Python GIL)
- ✅ Crash isolation (process failures don't affect parent)
- ✅ Separate memory space (clean separation)
- ✅ Good for long-running tasks

### Drawbacks

- ⚠️ Higher memory overhead (~50-100MB per process)
- ⚠️ More complex inter-process communication (Queues, Pipes)
- ⚠️ Slower startup time
- ⚠️ Cannot share state easily (must use IPC primitives)

### Implementation

**Use BaseStreamer for all new device streamers:**

```python
from streamer.base_streamer import BaseStreamer

class MyStreamer(BaseStreamer):
    def __init__(self, device_name, synchronized_start_time, root_output_folder):
        super().__init__(device_name, synchronized_start_time, root_output_folder)

    def _stream_wrapper(self):
        # Runs in separate process
        self._connect_device()
        self._setup_lsl_outlets()
        self.queue.put('connected')

        while not self.stop_signal.is_set():
            data = self._read_data()
            self.outlet.push_sample(data)

    def _setup_lsl_outlets(self):
        # Create LSL outlets
        pass
```

**Usage:**

```python
# With context manager (recommended)
with MyStreamer("Device1", time.time(), "/output") as streamer:
    streamer.start_streaming()
    # ... do work ...
# Automatic cleanup

# Or manually
streamer = MyStreamer("Device1", time.time(), "/output")
streamer.start_streaming()
# ... do work ...
streamer.stop_streaming()
```

### Key Rules

1. **Always inherit from BaseStreamer** for device streamers
2. **Use Event() for stop signaling** (not bool - reliable across processes)
3. **Use Queue() for status communication** from child to parent
4. **Never wrap Process in Thread** (anti-pattern, adds unnecessary layer)
5. **Implement context managers** for guaranteed cleanup

---

## Pattern 2: threading.Thread (For Coordination Only)

### When to Use

✓ **I/O-bound coordination tasks** (waiting for multiple streamers)
✓ **Lightweight background tasks** (monitoring, logging)
✓ **Shared memory needed** (efficient data sharing)
✓ **Legacy compatibility** (existing thread-based code)

### Benefits

- ✅ Lower memory overhead
- ✅ Shared memory (easy state sharing)
- ✅ Faster startup
- ✅ Simpler communication

### Drawbacks

- ⚠️ Subject to GIL (no true parallelism for CPU work)
- ⚠️ Can cause deadlocks if not careful
- ⚠️ Harder to debug
- ⚠️ No crash isolation

### Implementation

```python
import threading

def background_task():
    while not stop_event.is_set():
        # Do lightweight I/O work
        time.sleep(1)

stop_event = threading.Event()
thread = threading.Thread(target=background_task, daemon=True)
thread.start()

# ... do work ...

stop_event.set()
thread.join(timeout=5)
```

### Key Rules

1. **Use daemon=True** for background threads that should die with main
2. **Always join() threads** with timeout to prevent hanging
3. **Use threading.Event** for stop signaling (not bool)
4. **Use threading.Lock** for shared state protection
5. **Prefer queues over locks** when possible

---

## Pattern 3: asyncio (Future Enhancement)

### When to Use (Future)

✓ **High I/O concurrency** (many network connections)
✓ **Modern async libraries available**
✓ **Single-threaded event loop preferred**

### Current Status

⏳ **Not yet implemented** - planned for Phase 3.2

### Benefits (When Implemented)

- ✅ Efficient I/O handling
- ✅ Low memory overhead
- ✅ No GIL issues (single-threaded)
- ✅ Modern Python idiom

### Migration Path

1. Phase 3.1: Standardize current patterns ✅ (in progress)
2. Phase 3.2: Create asyncio prototypes ⏳
3. Phase 3.3: Evaluate and migrate if beneficial ⏳

---

## Anti-Patterns (DO NOT USE)

### ❌ Anti-Pattern 1: Thread Wrapping Process

**WRONG:**
```python
streamer = StreamMuse(...)
thread = threading.Thread(target=streamer.start_streaming)
thread.start()
# Creates: Thread → Process → does work (unnecessary layer!)
```

**CORRECT:**
```python
streamer = StreamMuse(...)
streamer.start_streaming()  # Directly creates Process
# Creates: Process → does work
```

**Why it's wrong:**
- Adds unnecessary complexity
- Thread just waits for Process
- Harder to debug
- More resource overhead

**Fixed in:** Phase 3.1 (main.py refactoring)

---

### ❌ Anti-Pattern 2: Using bool for stop_signal in multiprocessing

**WRONG:**
```python
class MyStreamer:
    def __init__(self):
        self.stop_signal = False  # ❌ Won't work across processes!

    def stream(self):
        while not self.stop_signal:  # Parent setting this won't affect child
            # ...
```

**CORRECT:**
```python
from multiprocessing import Event

class MyStreamer:
    def __init__(self):
        self.stop_signal = Event()  # ✓ Works across processes

    def stream(self):
        while not self.stop_signal.is_set():  # ✓ Child sees parent's signal
            # ...
```

**Why it's wrong:**
- Processes have separate memory spaces
- Bool changes in parent don't affect child
- Can't stop process reliably

**Fixed in:** BaseStreamer implementation

---

### ❌ Anti-Pattern 3: Mixing threading and multiprocessing primitives

**WRONG:**
```python
import threading
from multiprocessing import Event

class MyStreamer:
    def __init__(self):
        self.lock = threading.Lock()  # ❌ Thread primitive
        self.stop_signal = Event()     # ✓ Process primitive
        # Mixing patterns causes confusion!
```

**CORRECT:**
```python
# Either use all threading primitives:
import threading
class MyRecorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_signal = threading.Event()

# Or use all multiprocessing primitives:
from multiprocessing import Lock, Event
class MyStreamer:
    def __init__(self):
        self.lock = Lock()
        self.stop_signal = Event()
```

**Why it's wrong:**
- Causes confusion
- threading primitives don't work across processes
- Hard to debug

**Status:** Partially addressed in Phase 3.1

---

### ❌ Anti-Pattern 4: Forgetting to clean up resources

**WRONG:**
```python
streamer = MyStreamer(...)
streamer.start_streaming()
# ... program exits, process becomes zombie
```

**CORRECT:**
```python
# Option 1: Context manager (best)
with MyStreamer(...) as streamer:
    streamer.start_streaming()
    # ... do work ...
# Automatic cleanup

# Option 2: Try/finally (manual)
streamer = MyStreamer(...)
try:
    streamer.start_streaming()
    # ... do work ...
finally:
    streamer.stop_streaming()
```

**Why it's wrong:**
- Leaves zombie processes
- Resource leaks
- Hard to debug

**Fixed in:** BaseStreamer context manager

---

## Module-Specific Guidelines

### Streamers (stream_muse.py, stream_e4.py)

**Pattern:** `multiprocessing.Process`

**Rationale:**
- Device I/O isolation (Bluetooth, USB)
- Crash isolation (one device failure doesn't crash system)
- True parallelism for data processing

**Implementation:**
- Inherit from `BaseStreamer`
- Use `Event()` for stop signaling
- Use `Queue()` for status communication
- Implement context managers

**Current Status:**
- BaseStreamer created ✅
- Need to migrate existing streamers ⏳

---

### Recorder (stream_recorder.py)

**Pattern:** `threading.Thread`

**Rationale:**
- I/O-bound (LSL inlet polling)
- Needs shared memory for data buffers
- Fine-grained locking with `threading.Lock()`

**Implementation:**
- Keep threading-based
- Use `threading.Event()` for stop (not `multiprocessing.Event`)
- Use `threading.Lock()` for data protection
- Consider asyncio in Phase 3.2

**Current Status:**
- Threading-based implementation ✅
- Works correctly ✅
- Consider asyncio migration later ⏳

---

### Viewer (view_streams.py)

**Pattern:** `multiprocessing.Process` (per plot)

**Rationale:**
- GUI rendering isolation (PyQt, Vispy)
- Each plot is independent
- Crash isolation for plots

**Concerns:**
- High memory overhead (one process per plot)
- Consider process pooling

**Recommendations:**
- Keep current pattern for Phase 3.1 ✅
- Consider process pooling in Phase 3.2 ⏳
- Consider shared rendering context ⏳

**Current Status:**
- Process-based working ✅
- Optimization deferred to Phase 3.2 ⏳

---

### Main Orchestrator (main.py)

**Pattern:** Direct function calls + Process management

**Rationale:**
- Coordination layer (not computation)
- Manages lifecycle of streamers
- No heavy lifting

**Implementation:**
- Call `streamer.start_streaming()` directly (no Thread wrapper) ✅
- Track streamers in `AppState` ✅
- Use context managers when possible ⏳

**Current Status:**
- Thread wrappers removed ✅
- Direct Process management ✅

---

## Quick Reference

| Use Case | Pattern | Primitive | Example |
|----------|---------|-----------|---------|
| Device streaming | `multiprocessing.Process` | `Event(), Queue()` | StreamMuse, StreamE4 |
| Data recording | `threading.Thread` | `threading.Event(), threading.Lock()` | StreamRecorder |
| Visualization | `multiprocessing.Process` | `Manager()` | ViewStreams |
| Coordination | Direct calls | N/A | main.py |
| Background monitoring | `threading.Thread` | `threading.Event()` | Connection monitors |
| CPU-intensive work | `multiprocessing.Process` | `Event(), Queue()` | Signal processing |

---

## Migration Checklist

When refactoring existing code:

### For Process-Based Streamers:

- [ ] Inherit from `BaseStreamer`
- [ ] Replace `self.stop_signal = False` with inherited `Event()`
- [ ] Implement `_stream_wrapper()` method
- [ ] Implement `_setup_lsl_outlets()` method
- [ ] Use `self.queue.put('connected')` to signal success
- [ ] Check `self.stop_signal.is_set()` in main loop
- [ ] Remove any `threading.Thread` wrappers in calling code
- [ ] Add context manager usage where possible
- [ ] Update tests

### For Thread-Based Components:

- [ ] Ensure using `threading.Event()` (not bool)
- [ ] Use `threading.Lock()` for shared state
- [ ] Set `daemon=True` for background threads
- [ ] Always `join()` with timeout
- [ ] Implement proper cleanup in `finally` blocks
- [ ] Avoid mixing threading and multiprocessing primitives

---

## Testing Guidelines

### Process-Based Components:

```python
def test_streamer_lifecycle():
    streamer = MyStreamer(...)

    # Start
    assert streamer.start_streaming(timeout=5)
    assert streamer.is_streaming()

    # Stop
    streamer.stop_streaming()
    assert not streamer.is_streaming()
```

### Thread-Based Components:

```python
def test_threaded_component():
    component = MyComponent()
    thread = threading.Thread(target=component.run)
    thread.start()

    # Work
    time.sleep(1)

    # Stop
    component.stop()
    thread.join(timeout=5)
    assert not thread.is_alive()
```

---

## Common Pitfalls

### 1. Forgetting Process memory isolation

```python
# ❌ WRONG: Changes in parent don't affect child
self.counter = 0
process = Process(target=self.work)
process.start()
self.counter += 1  # Child still sees 0!

# ✓ CORRECT: Use Queue or shared memory
queue = Queue()
process = Process(target=self.work, args=(queue,))
process.start()
queue.put(1)  # Child receives via queue
```

### 2. Blocking indefinitely on Queue.get()

```python
# ❌ WRONG: Hangs if queue empty
data = queue.get()  # Blocks forever

# ✓ CORRECT: Use timeout
try:
    data = queue.get(timeout=1.0)
except Empty:
    # Handle timeout
```

### 3. Not joining processes/threads

```python
# ❌ WRONG: Leaves zombies
process.terminate()
# Process becomes zombie

# ✓ CORRECT: Always join
process.terminate()
process.join(timeout=5)
if process.is_alive():
    process.kill()
```

---

## Performance Considerations

### Memory Usage

| Pattern | Memory per Unit | Startup Time | Use When |
|---------|----------------|--------------|----------|
| Process | ~50-100MB | ~100-500ms | Isolation needed |
| Thread | ~8MB | ~1-10ms | Shared memory needed |
| Asyncio | ~1-2MB | <1ms | High I/O concurrency |

### Throughput

- **Process**: Best for CPU-bound, scales with cores
- **Thread**: Good for I/O-bound, limited by GIL for CPU
- **Asyncio**: Best for I/O-bound with many connections

### Latency

- **Process**: Higher startup latency (~100ms)
- **Thread**: Medium startup latency (~10ms)
- **Asyncio**: Lowest latency (<1ms)

---

## Future Direction

### Phase 3.2: Asyncio Exploration (Weeks 2-6)

1. Create asyncio-based streamer prototype
2. Benchmark against Process-based approach
3. Migrate device discovery to asyncio
4. Evaluate LSL asyncio integration

### Phase 3.3: Optimization (Weeks 7-8)

1. Process pooling for viewers
2. Shared memory for data buffers
3. Performance benchmarking
4. Final integration testing

---

## Resources

- **Code Examples**: `streamer/README.md`
- **Base Implementation**: `streamer/base_streamer.py`
- **Tests**: `tests/test_base_streamer.py`
- **Analysis**: `audit/PHASE3_CONCURRENCY_ANALYSIS.md`
- **Python Docs**:
  - [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
  - [threading](https://docs.python.org/3/library/threading.html)
  - [asyncio](https://docs.python.org/3/library/asyncio.html)

---

## Questions?

For questions about concurrency patterns:
1. Check this document first
2. Review `streamer/README.md` for examples
3. Check `audit/PHASE3_CONCURRENCY_ANALYSIS.md` for rationale
4. Consult the team

---

**Document Version:** 1.0
**Last Updated:** November 5, 2025
**Status:** Active
**Next Review:** Phase 3.2 kickoff

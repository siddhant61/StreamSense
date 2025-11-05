# Phase 3.1: Concurrency Model Analysis

**Date:** November 5, 2025
**Status:** Analysis Complete ✅

## Current Concurrency Patterns

### Overview
StreamSense uses a **mixed concurrency model** combining `threading` and `multiprocessing`, leading to complexity and potential issues.

### Module-by-Module Analysis

#### 1. **Streamers** (stream_e4.py, stream_muse.py)
**Pattern:** `multiprocessing.Process`

**Usage:**
```python
# stream_e4.py
from multiprocessing import Process, Event, Queue

class StreamE4:
    def start_streaming(self):
        self.process = Process(target=self.e4_streamer)
        self.process.start()
```

**Rationale:**
- ✅ **Good:** Process isolation prevents GIL contention
- ✅ **Good:** True parallelism for I/O-bound streaming
- ✅ **Good:** Crash isolation (one stream failure doesn't crash others)
- ⚠️ **Issue:** High memory overhead (separate Python interpreters)
- ⚠️ **Issue:** Complex inter-process communication via Queues

**Impact:** Critical - handles real-time data streaming

---

#### 2. **Recorder** (stream_recorder.py)
**Pattern:** `threading.Thread`

**Usage:**
```python
import threading
from multiprocessing import Event  # Mixed!

class StreamRecorder:
    def __init__(self):
        self.locks = {}
        for key in self.stream_sample_rates.keys():
            self.locks[key] = threading.Lock()

        self.stream_update_thread = threading.Thread(
            target=self.update_streams
        )
```

**Rationale:**
- ✅ **Good:** Shared memory for data buffers (efficient)
- ✅ **Good:** Fine-grained locking with `threading.Lock()`
- ⚠️ **Issue:** GIL contention for CPU-bound processing
- ⚠️ **Issue:** Mixed with `multiprocessing.Event` (confusing)

**Impact:** High - manages all data recording

---

#### 3. **Viewer** (view_streams.py)
**Pattern:** `multiprocessing.Process`

**Usage:**
```python
from multiprocessing import Process, Manager

def start_viewing(self):
    with Manager() as manager:
        processes = []
        for stream in validated_streams.values():
            process = Process(
                target=self.plot_stream_with_canvas,
                args=(stream.as_xml(), shared_canvases_statuses)
            )
            processes.append(process)
            process.start()
```

**Rationale:**
- ✅ **Good:** Isolates GUI rendering (Vispy/Qt)
- ✅ **Good:** Multiple plots can render in parallel
- ⚠️ **Issue:** High memory overhead for each plot
- ⚠️ **Issue:** Complex coordination with Manager

**Impact:** Medium - visualization is non-critical path

---

#### 4. **Main Orchestrator** (main.py)
**Pattern:** `threading.Thread`

**Usage:**
```python
import threading

# Coordinates all components
recorder_thread: Optional[threading.Thread] = None
muse_threads: Dict[str, threading.Thread] = field(default_factory=dict)

# Starts streamers
thread = threading.Thread(target=streamer.start_streaming)
thread.start()
```

**Rationale:**
- ✅ **Good:** Lightweight coordination
- ⚠️ **Issue:** Threads wrapping Process-based streamers (unnecessary layer)
- ⚠️ **Issue:** Inconsistent with streamer architecture

**Impact:** High - core orchestration

---

## Issues Identified

### 1. **Mixed Concurrency Models** 🔴 CRITICAL
**Problem:** Threading and multiprocessing mixed without clear boundaries

**Evidence:**
- Recorder uses `threading.Thread` + `multiprocessing.Event`
- Main uses `threading.Thread` to start `multiprocessing.Process`-based streamers
- No clear architectural decision documented

**Impact:**
- Developer confusion
- Harder debugging
- Potential deadlocks
- Resource leaks

---

### 2. **GIL Contention** 🟡 MODERATE
**Problem:** Threading used for CPU-bound tasks

**Evidence:**
- `data_processor.py` performs heavy computation in threads
- Recorder processing in threads limits throughput

**Impact:**
- Limited CPU utilization
- Slower data processing
- Can't scale to multi-core

---

### 3. **Memory Overhead** 🟡 MODERATE
**Problem:** Excessive process creation

**Evidence:**
- One process per plot (viewer)
- One process per device stream

**Impact:**
- High memory usage (~50-100MB per process)
- Slower startup time
- Limited scalability

---

### 4. **Complex IPC** 🟡 MODERATE
**Problem:** Queue-based communication is fragile

**Evidence:**
```python
# stream_muse.py
self.shared_eeg = Queue()  # 6 queues per device!
self.shared_ppg = Queue()
self.shared_acc = Queue()
# ...
```

**Impact:**
- Queue blocking issues
- Data loss on queue overflow
- Hard to debug

---

## Recommendations

### Strategy 1: **Asyncio-First** (Recommended) 🌟

**Approach:** Use `asyncio` for I/O-bound tasks, `multiprocessing` only for CPU-bound

**Architecture:**
```
┌─────────────────────────────────────────┐
│         Main Event Loop (asyncio)       │
│  - Device discovery                     │
│  - Stream coordination                  │
│  - User interaction                     │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│   Streamer  │  │   Recorder  │
│  (asyncio)  │  │  (asyncio)  │
│  coroutines │  │  coroutines │
└─────────────┘  └─────────────┘
       │
       ▼ (only for heavy processing)
┌─────────────┐
│  Process    │
│   Pool      │
│ (CPU work)  │
└─────────────┘
```

**Benefits:**
- ✅ Single-threaded (no GIL issues)
- ✅ Low memory overhead
- ✅ Better error handling
- ✅ Modern Python idiom
- ✅ Great for I/O (LSL, network)

**Challenges:**
- ⚠️ Requires significant refactoring
- ⚠️ Learning curve for team
- ⚠️ Need to integrate with existing libs (PyQt, Vispy)

---

### Strategy 2: **Consistent Multiprocessing** (Moderate)

**Approach:** Use `multiprocessing` everywhere for isolation

**Benefits:**
- ✅ Clear separation
- ✅ Crash isolation
- ✅ True parallelism

**Challenges:**
- ⚠️ High memory usage
- ⚠️ Complex IPC
- ⚠️ Harder debugging

---

### Strategy 3: **Hybrid (Current + Cleanup)** (Quick Win) ⚡

**Approach:** Keep current model but standardize interfaces

**Changes:**
1. Document when to use threading vs multiprocessing
2. Create standard base classes
3. Eliminate mixed patterns (e.g., threading.Thread wrapping Process)
4. Add proper context managers

**Benefits:**
- ✅ Minimal changes
- ✅ Keeps working code
- ✅ Quick to implement

**Challenges:**
- ⚠️ Doesn't solve fundamental issues
- ⚠️ Still complex

---

## Decision Matrix

| Criterion | Asyncio-First | Consistent MP | Hybrid Cleanup |
|-----------|---------------|---------------|----------------|
| **Effort** | High (8 weeks) | Medium (4 weeks) | Low (1 week) |
| **Risk** | Medium | Low | Very Low |
| **Performance** | Excellent | Good | Same |
| **Maintainability** | Excellent | Good | Fair |
| **Scalability** | Excellent | Fair | Fair |
| **Modern** | Yes | No | No |

---

## Proposed Roadmap

### Phase 3.1: Quick Wins (Week 1) ✅
1. ✅ Document concurrency patterns
2. ⏳ Create base classes for streamers
3. ⏳ Eliminate threading.Thread wrapping Process
4. ⏳ Add proper context managers

### Phase 3.2: Asyncio Migration (Weeks 2-6)
1. Create asyncio-based streamer prototype
2. Migrate device discovery to asyncio
3. Migrate recorder to asyncio
4. Integrate with PyQt event loop
5. Comprehensive testing

### Phase 3.3: Cleanup (Weeks 7-8)
1. Remove old threading code
2. Update documentation
3. Performance benchmarking
4. Final integration testing

---

## Recommendation: **Hybrid Approach with Asyncio Exploration**

**Phase 1 (Immediate):** Implement Hybrid Cleanup for quick stability
**Phase 2 (Future):** Explore asyncio migration in parallel branch
**Phase 3 (Long-term):** Evaluate and potentially adopt asyncio

This balanced approach:
- ✅ Delivers immediate value
- ✅ Reduces risk
- ✅ Keeps future options open
- ✅ Maintains test coverage

---

## Next Steps

1. ✅ **Create base streamer interface** (standardize Process-based streamers)
2. **Eliminate threading wrappers** (main.py cleanup)
3. **Document concurrency guidelines** (when to use what)
4. **Add context managers** (proper resource cleanup)
5. **Create asyncio prototype** (exploration)

---

**Analysis Complete:** November 5, 2025
**Recommendation:** Hybrid cleanup now, asyncio exploration for future
**Next Task:** Create base streamer interface


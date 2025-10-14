# Deliverable 5: Actionable Stabilization Roadmap

## Introduction

This document provides a prioritized, actionable roadmap for stabilizing the StreamSense project. The project currently suffers from significant technical debt at all levels, and this roadmap is designed to address the most critical issues first, creating a stable foundation for future development.

---

## Phase 1: Foundational Stability (Highest Priority)

**Goal**: Establish a stable, reproducible, and testable development environment. This phase is the prerequisite for all future work.

1.  **Task: Create a `requirements.txt` file.**
    -   **Action**: Based on the 19 libraries identified in the `audit/tech_stack.md` report, create a `requirements.txt` file.
    -   **Justification**: This is the most critical first step. Without it, no developer can create a working environment to run, test, or improve the code.

2.  **Task: Create a comprehensive and accurate `README.md`.**
    -   **Action**: Replace the current `README.md` with one that accurately reflects the project's current state. It should include correct setup instructions (referencing the new `requirements.txt`), a description of the actual architecture, and a list of the orphaned scripts and their status.
    -   **Justification**: The current `README.md` is misleading and a barrier to new developers.

3.  **Task: Establish a basic testing framework.**
    -   **Action**: Introduce `pytest` to the project. Create a `tests/` directory and add a single, simple test that imports a module to ensure the framework is configured correctly.
    -   **Justification**: This establishes the foundation for all future testing efforts, which are critical for stabilizing the codebase.

4.  **Task: Create a root-level `.gitignore` file.**
    -   **Action**: Add a standard Python `.gitignore` file at the root of the project to prevent common artifacts (`__pycache__`, `.log` files, IDE files) from being committed.
    -   **Justification**: This is a basic best practice for any version-controlled project.

5.  **Task: Archive Orphaned Code.**
    -   **Action**: Move the orphaned files (`e4_basic_flow.py`, `helper/data_helper.py`) to a new `archive/` directory to clearly separate them from the main application codebase.
    -   **Justification**: This cleans up the project structure and makes it clear what code is part of the core application.

---

## Phase 2: Architectural Refactoring

**Goal**: Address the most significant architectural flaws to improve stability and maintainability.

1.  **Task: Refactor Global State Management.**
    -   **Action**: Remove the use of global variables in `main.py`. Encapsulate the application's state (e.g., active threads, streamers, output folder) within a dedicated state management class.
    -   **Justification**: This will make the application's state predictable, easier to manage, and will reduce the risk of bugs related to concurrency.

2.  **Task: Implement a Robust Application Lifecycle Manager.**
    -   **Action**: Replace the manual threading and process management in `main.py` with a dedicated lifecycle manager. This manager should be responsible for starting, stopping, and monitoring the health of all components (streamers, recorder, etc.).
    -   **Justification**: This will fix the fragile startup/shutdown process and provide a central point of control for the application.

3.  **Task: Replace `time.sleep()` with Event-Driven Synchronization.**
    -   **Action**: Systematically search the codebase for `time.sleep()` calls used for synchronization and replace them with proper primitives like `threading.Event` and `multiprocessing.Event`.
    -   **Justification**: This will eliminate a major source of race conditions and unreliability, making the application more robust and performant.

4.  **Task: Integrate Orphaned Utilities.**
    -   **Action**: Integrate the `stream_info.py` stream monitoring dashboard into the main CLI as a menu option.
    -   **Justification**: This will make a key feature accessible to users and unify the application's toolset.

---

## Phase 3: Code-Level Refactoring and Feature Hardening

**Goal**: Improve the quality of the code and the reliability of the individual features.

1.  **Task: Add Comprehensive Error Handling.**
    -   **Action**: Systematically review the codebase and replace `pass` statements in `except` blocks with proper logging and, where appropriate, state-updating error handling logic.
    -   **Justification**: This will make the application more resilient to common errors, such as device disconnections.

2.  **Task: Refactor High-Complexity Modules.**
    -   **Action**: Break down large, complex classes and functions into smaller, more focused units with a single responsibility. The `StreamRecorder` and `Muse` helper classes are prime candidates.
    -   **Justification**: This will improve code readability, maintainability, and testability.

3.  **Task: Add Unit and Integration Tests.**
    -   **Action**: With the testing framework in place, begin writing unit tests for business logic and integration tests for critical workflows like the data recording and processing pipelines.
    -   **Justification**: This is the only way to ensure the reliability of the features and prevent regressions.

4.  **Task: Harden Device Connections.**
    -   **Action**: Refactor the device connection logic in the `helper` modules to be more resilient to errors and disconnections. This includes improving the reconnection logic and providing clearer feedback to the user.
    -   **Justification**: The current connection logic is a major source of fragility and a poor user experience.

---

## Milestone A: Controller & Platform Refactor

**Goal**: Decouple lifecycle management from the CLI and remove Windows-specific assumptions so the platform can run reliably across environments.

1.  **Task: Introduce a lifecycle controller.**
    -   **Action**: Extract discovery, streaming, recording, and viewing orchestration from `main.py` into a dedicated controller/service layer that owns shared state instead of relying on module-level dictionaries.
    -   **Justification**: Eliminates fragile global state, enables reuse by headless or GUI clients, and clarifies ownership of background threads.

2.  **Task: Isolate device discovery workflows.**
    -   **Action**: Move device scanning concerns behind controller-managed interfaces so discovery can be invoked deterministically and its errors surfaced to operators.
    -   **Justification**: Prevents discovery from polluting global state and allows the visualization layer to report issues promptly.

3.  **Task: Replace Windows-only utilities.**
    -   **Action**: Abstract platform-dependent helpers (e.g., WMI lookups for the Empatica server) behind interchangeable providers with Linux and macOS implementations.
    -   **Justification**: Removes the current Windows lock-in and prepares the controller for remote or containerized deployments.

4.  **Task: Establish minimal CI coverage.**
    -   **Action**: Lock dependencies, add smoke tests that start the new controller in headless mode, and gate merges on those checks.
    -   **Justification**: Prevents regressions while the refactor is underway and documents the supported execution path.

---

## Milestone B: Visualization Architecture

**Goal**: Stabilize the viewing experience by consolidating processes and improving error visibility.

1.  **Task: Redesign `ViewStreams`.**
    -   **Action**: Replace the current process-per-canvas approach with a single manager that tracks active canvases, controls their lifecycle, and surfaces validation errors deterministically.
    -   **Justification**: Reduces resource usage, simplifies synchronization, and ensures operators receive feedback when discovery fails.

2.  **Task: Provide a unified visualization shell.**
    -   **Action**: Implement a Qt- or web-based host that can display multiple streams, health indicators, and controls within one event loop.
    -   **Justification**: Lays the groundwork for dashboards, remote GUIs, and future automation hooks.

---

## Milestone C: Clock Synchronization & Data Integrity

**Goal**: Align device timestamps and capture quality metrics so downstream processing can trust recorded data.

1.  **Task: Persist LSL clock offsets.**
    -   **Action**: When a device connection is established, capture its offset from the local LSL clock and reuse it on reconnects so elapsed (Empatica) and raw (Muse) timestamps align with the shared wall clock.
    -   **Justification**: Prevents drift and makes multi-device correlation possible without manual adjustments.

2.  **Task: Record latency and jitter metadata.**
    -   **Action**: Instrument streamers and the recorder to track latency/jitter metrics, store them alongside recordings, and feed them into gap detection and resampling routines.
    -   **Justification**: Provides the recorder with the context it needs to apply corrections consistently and audit stream quality.

---

## Milestone D: Sensor Plugin Framework

**Goal**: Make the platform extensible to new sensors by standardizing transport abstractions and stream descriptors.

1.  **Task: Define transport adapters.**
    -   **Action**: Build reusable adapters for BLE, serial, and socket transports and move Empatica/Muse specifics into device plugins.
    -   **Justification**: Enables onboarding additional wearables with minimal cross-cutting changes and encourages code reuse.

2.  **Task: Generalize recorder/viewer inputs.**
    -   **Action**: Update the recorder and viewer to consume generic stream descriptors rather than device-specific assumptions so new plugins can integrate without bespoke wiring.
    -   **Justification**: Ensures future sensors benefit from the same stability investments and keeps the controller architecture coherent.

---

## Milestone E: Testing & Tooling Enhancements

**Goal**: Provide automated guardrails that cover the data-processing pipeline and forthcoming controller abstractions.

1.  **Task: Expand `DataProcessor` coverage.**
    -   **Action**: Add regression tests for `highlight_gaps`, resampling, and export paths, building on the existing `detect_gaps` tests.
    -   **Justification**: Protects critical data workflows from regressions as synchronization logic evolves.

2.  **Task: Mock hardware dependencies.**
    -   **Action**: Introduce mocks for hardware interfaces so CI can exercise the controller, discovery flows, and streamers without physical devices.
    -   **Justification**: Makes the test suite reliable, unlocks headless verification, and shortens feedback loops for contributors.

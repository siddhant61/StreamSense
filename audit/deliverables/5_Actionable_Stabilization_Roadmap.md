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

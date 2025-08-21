# Deliverable 4: Technical Debt Report

This document summarizes the major areas of technical debt identified in the StreamSense project. The debt is substantial and exists at all levels of the project, from architecture to code quality and development process.

## 1. Process and Documentation Debt

This is the most critical category of technical debt and the root cause of many other issues.

-   **No Test Coverage**: The complete absence of automated tests (unit, integration, or E2E) is the single largest risk factor. It makes any code change unsafe and the current functionality impossible to verify.
-   **No Dependency Management**: The lack of a `requirements.txt` or similar file makes the project's environment non-reproducible. It is impossible to know which versions of the 19+ external libraries are required, making it extremely difficult to set up a working development environment.
-   **No Version Control History**: The Git history contains only one meaningful commit. This indicates that the code was not developed under version control, making it impossible to review changes, understand the evolution of features, or revert to previous versions.
-   **Inaccurate and Incomplete Documentation**: The `README.md` is the only piece of documentation, and it is severely out of date. It references multiple non-existent files (`requirements.txt`, `docs/`, `LICENSE`, etc.), which is highly misleading.

## 2. Architectural Debt

The application's high-level design contains several significant flaws that contribute to its instability.

-   **Fragile State Management**: The application's state is managed by a set of global variables in `main.py`. This is not a robust or scalable solution. It makes the application's state hard to track and debug, especially in a multi-threaded context.
-   **Lack of Robust Lifecycle Management**: The application's startup and shutdown sequences are not well-defined. The system relies on `time.sleep()` calls to wait for components to initialize, which is unreliable. The `stop` command attempts to clean up resources, but without proper state management, its effectiveness is questionable.
-   **Fragile External Integrations**:
    -   The Empatica E4 integration depends on an external, proprietary executable (`EmpaticaBLEServer.exe`) and communicates via a raw TCP socket, which is a brittle approach.
    -   The Muse integration uses a complex stack of libraries and a custom serial backend, making it difficult to maintain.
-   **Orphaned Components**: Key features, like the stream monitoring dashboard (`stream_info.py`), exist as standalone, un-integrated scripts rather than being part of a cohesive application.

## 3. Code-Level Debt (Code Smells)

The source code itself exhibits numerous code smells and anti-patterns.

-   **Excessive Use of `time.sleep()`**: `time.sleep()` is used pervasively throughout the codebase for synchronization, especially in the device connection and streaming logic. This is a major anti-pattern that leads to race conditions, slow performance, and unreliable behavior. Asynchronous events should be handled with proper synchronization primitives like Events, Queues, and Conditions.
-   **High Complexity and Low Cohesion**: Many files are extremely long and complex, with classes and functions that have too many responsibilities. For example, `stream_recorder.py` handles not only recording but also data interpolation, conversion to MNE format, and saving processed datasets. This violates the Single Responsibility Principle and makes the code hard to understand and maintain.
-   **Minimal Error Handling**: Many `try...except` blocks either contain a `pass` statement (silently ignoring errors) or simply log the error without any attempt at recovery. In a multi-threaded application dealing with unreliable hardware connections, this is a recipe for instability.
-   **Improper Use of Multithreading/Multiprocessing**: The application liberally spins up new threads and processes without a clear strategy for managing them, communicating between them, or ensuring they are properly terminated. This can lead to resource leaks and zombie processes.
-   **Hardcoded Paths and Configuration**: The path to the `EmpaticaBLEServer.exe` is hardcoded in `e4_helper.py`, making the application non-portable. Other configuration values are scattered throughout the code as global constants.

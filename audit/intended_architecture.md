# Intended Architecture and Features

This document outlines the intended architecture and features of the StreamSense project, as reconstructed from the `README.md` file and associated images. It also notes significant discrepancies between the documentation and the repository's actual contents.

## Project Overview

StreamSense is intended to be an open-source platform for the real-time acquisition, processing, and visualization of physiological data from Muse and Empatica E4 sensors. The project is explicitly marked as a "Work in Progress" (WIP).

## Intended System Architecture

The system is designed around a modular framework, as depicted in the architecture diagram (`Logs/Slide20.JPG`).

- **User Interface**: A **Command Line Interface (CLI)** serves as the primary point of user interaction, supporting both direct arguments and an interactive menu.
- **Hardware Layer**: The system is designed to interface with **Sensors** (Muse, E4) via a **BLE Adapter**.
- **Core Backend**:
    - **Device and Stream Management**: This component is responsible for connecting to the hardware and managing the data streams. It consists of a `Sensor Backend` (interfacing with the BLE adapter) and a `Sensor Streamer`.
    - **Lab Streaming Layer (LSL)**: LSL acts as the central data bus, receiving data from the `Sensor Streamer` and making it available to other components.
- **Data Handling Layer**: This layer consumes data from LSL.
    - **Stream Recording**: For saving data to files for offline analysis.
    - **Stream Visualization**: For real-time plotting of sensor data.
    - **Stream Monitoring**: For observing the status and quality of active data streams.
- **Utilities**: A collection of `Helper Scripts` supports the data handling layer.

## Intended Features

Based on the `README.md` and the provided slides, the following key features were intended:

1.  **Multi-Device Support**:
    -   Empatica E4
    -   Muse

2.  **Multi-Stream Data Acquisition**:
    -   **EEG** (Muse @ 256 Hz)
    -   **PPG** (Muse @ 64 Hz)
    -   **BVP** (E4 @ 64 Hz)
    -   **Accelerometer** (Muse @ 50 Hz, E4 @ 32 Hz)
    -   **Gyroscope** (Muse @ 50 Hz)
    -   **GSR** (E4 @ 4 Hz)
    -   **Temperature** (E4 @ 4 Hz)
    -   **Event Markers** (E4 @ 32 Hz)

3.  **Real-Time Visualization & Monitoring**:
    -   Simultaneous plotting of synchronized streams from multiple devices.
    -   Visualization of multi-channel data (e.g., EEG) to assess signal quality.
    -   A dashboard view to monitor the status and sampling rates of all active LSL streams.

4.  **Data Recording & Post-Processing**:
    -   Recording of all active LSL streams.
    -   Offline data quality analysis, including plotting signals, highlighting gaps, and calculating data integrity metrics (e.g., "% Intact").

5.  **Event Logging**:
    -   A console for users to log events in real-time, synchronized with the physiological data.

## Discrepancies between Documentation and Repository

There are several significant discrepancies between the `README.md` file and the actual contents of the repository:

-   **Missing `requirements.txt`**: The installation instructions refer to a `requirements.txt` file that does not exist.
-   **Missing `docs/` Folder**: The README refers to a `docs` folder for detailed documentation, which is not present.
-   **Missing `methodology.docx`**: The "Research Methodology" section refers to a `.docx` file that does not exist.
-   **Missing `CONTRIBUTING.md`**: The "Contributing" section refers to a `CONTRIBUTING.md` file that does not exist.
-   **Missing `LICENSE` File**: The "License" section refers to a `LICENSE` file that does not exist.

These omissions indicate that the project documentation is incomplete and out of date, which is a major contributor to the project's disorganized state.

# Integration Analysis

This document outlines the internal and external integration points of the StreamSense project and assesses their status based on static code analysis.

## Internal Integration: Lab Streaming Layer (LSL)

The primary mechanism for internal integration is the **Lab Streaming Layer (LSL)**. LSL functions as a real-time pub/sub message bus, decoupling the data producers from the data consumers.

-   **Status**: `Needs Improvement (Technical Debt)`
-   **Assessment**: The use of LSL as a central bus is architecturally sound. The code correctly implements LSL outlets (producers) and inlets (consumers). However, the overall integration is fragile. There is no central mechanism for error handling or state management; if a single producer or consumer fails, other components of the system are not designed to handle it gracefully. The system relies on components being manually started in the correct order and does not have a robust lifecycle management system.

### LSL Producers (Publishers)

-   **`streamer/stream_muse.py`**: Publishes EEG, PPG, Accelerometer, and Gyroscope data from Muse devices.
-   **`streamer/stream_e4.py`**: Publishes Accelerometer, Blood Volume Pulse (BVP), Galvanic Skin Response (GSR), Temperature, and Tag data from Empatica E4 devices.

### LSL Consumers (Subscribers)

-   **`recorder/stream_recorder.py`**: Subscribes to all available LSL streams to record the data into HDF5 files.
-   **`viewer/view_streams.py`**: Subscribes to user-selected LSL streams for real-time data visualization.
-   **`stream_info.py`**: Subscribes to all streams to monitor their status and sampling rates.

## External Integrations

The system's external integrations are with the physical sensor hardware. There are no integrations with external web services or APIs.

### Hardware Device: Muse Headbands

-   **Integration Method**: Bluetooth Low Energy (BLE).
-   **Managing Code**: `helper/muse_helper.py`, `streamer/stream_muse.py`, `helper/serial_helper.py`.
-   **Core Libraries**: `muselsl`, `pygatt`, `pyserial`.
-   **Status**: `Needs Improvement (Technical Debt)`
-   **Assessment**: The integration is highly complex, relying on multiple libraries and a custom BGAPI backend implementation. The code is difficult to follow, heavily reliant on threading, and uses fixed `time.sleep()` calls for synchronization, which is unreliable. Error handling is minimal, making the connection prone to failure without a clear path to recovery.

### Hardware Device: Empatica E4

-   **Integration Method**: TCP socket connection to a proprietary desktop application (`EmpaticaBLEServer.exe`).
-   **Managing Code**: `helper/e4_helper.py`, `streamer/stream_e4.py`.
-   **Core Libraries**: `socket` (Python standard library).
-   **Status**: `Needs Improvement (Technical Debt)`
-   **Assessment**: This integration is fragile due to its dependency on an external, proprietary executable. The communication over a raw TCP socket is managed by complex, multi-threaded code that lacks robust error handling and, like the Muse integration, uses unreliable `time.sleep()` calls for timing and synchronization.

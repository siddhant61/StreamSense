# StreamSense

StreamSense is a research prototype for acquiring, recording, and visualising physiological data from Muse EEG headbands and Empatica E4 wearables. The system glues together multiple device-specific streamers, a recorder, and a Vispy-based viewer around a command-line workflow defined in `main.py`.

## Project Status & Limitations

- **Windows-focused runtime** – The main entry point (`main.py`) depends on `wmi`, direct serial access, and Empatica's BLE server checks, so the end-to-end workflow currently operates only on Windows hosts with the vendor tools installed.
- **Manual lifecycle management** – Device threads and processes are created directly in the CLI and synchronised with `threading.Event` objects, leaving limited error handling and no headless automation interface.
- **Prototype-grade tooling** – Visualisation (`viewer/view_streams.py`) and experiments (`experiments/visual_oddball.py`) run as separate processes with minimal cleanup, and only a small pytest suite exists today.

If you plan to evaluate the project on other platforms or extend it for production use, expect to invest in cross-platform abstractions, centralised lifecycle management, and broader automated test coverage.

## Requirements

### Python Version

**Supported Python versions**: 3.8, 3.9, 3.10, 3.11

**Recommended**: Python 3.10 or 3.11 for best compatibility

**Testing matrix**:
- Windows: Python 3.8, 3.10, 3.11
- macOS: Python 3.10, 3.11 (limited feature set)
- Linux: Python 3.9, 3.11 (limited feature set)

### Platform Support

- **Full support**: Windows 10/11 (64-bit)
- **Partial support**: macOS, Linux (some features unavailable due to `wmi` dependency)

### Hardware Requirements

- Bluetooth LE adapter (for Muse headbands and Empatica E4)
- Minimum 4GB RAM (8GB recommended for visualization)
- Display with OpenGL 2.1+ support (for Vispy viewer)

### External Dependencies

- **Empatica E4**: Requires [Empatica BLE Server](https://support.empatica.com/hc/en-us/articles/202581999-E4-streaming-server-for-Windows) installed separately
- **Muse headbands**: Compatible with Muse 2, Muse S (original Muse may have limited support)

See `DEPENDENCY_MANAGEMENT.md` for detailed dependency information and troubleshooting.

## Getting Started

1. **Create a Python environment** (Python 3.8+ is recommended) and activate it.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Several packages (e.g., `PyQt5`, `psychopy`, `pybluez`, `pywifi`, `wmi`) have platform-specific prerequisites. Consult their upstream documentation if installation fails.
3. **Connect supported hardware** (Muse headbands via Bluetooth and Empatica E4 via the Empatica BLE Server) before launching the CLI on Windows.

### Running the command-line workflow

```bash
python main.py
```

The CLI offers menu-driven options for discovering devices, starting LSL streams, launching the Vispy viewer, recording streams, and running the visual oddball experiment. Logs are written to `Logs/`.

### Running tests

```bash
pytest
```

The `tests/` directory currently provides limited coverage focused on the data processing utilities. Additional tests are encouraged as you extend the project.

## Repository Layout

- `main.py` – interactive CLI orchestrator that coordinates discovery, streaming, recording, and experiments.
- `streamer/` – device-specific streamers for Muse (`stream_muse.py`) and Empatica E4 (`stream_e4.py`) that publish data through Lab Streaming Layer (LSL).
- `recorder/` – logic for persisting active LSL streams to disk.
- `viewer/` – Vispy viewer processes for monitoring available streams.
- `experiments/` – experimental protocols such as the visual oddball paradigm.
- `helper/` – shared helpers for device discovery and plotting.
- `tests/` – pytest-based regression tests.
- `audit/` – artefacts from the codebase audit that identified current gaps and prioritised follow-up work.
- `archive/` – orphaned scripts retained for reference.

## Archived scripts

- `archive/e4_basic_flow.py` – legacy Empatica E4 streaming prototype kept for historical reference.
- `archive/data_helper.py` – deprecated data helper functions no longer used by the main application.

## Contributing

Contributions that improve stability, cross-platform support, and test coverage are welcome. Please open an issue outlining the proposed change, especially if it alters device streaming or lifecycle management.

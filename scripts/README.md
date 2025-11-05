# Scripts

Utility scripts for StreamSense development and documentation.

## Screenshot Capture

**`capture_ui_screenshots.py`** - Automated UI screenshot generation for documentation

### Purpose
Automatically captures professional screenshots of the StreamSense UI in various states for the README and documentation.

### Usage

```bash
python scripts/capture_ui_screenshots.py
```

### What It Does

The script will:
1. Launch the StreamSense UI
2. Populate with demo devices
3. Simulate different states (connected, recording, etc.)
4. Capture high-quality PNG screenshots
5. Save to `docs/screenshots/`

### Screenshots Captured

1. **01_initial_state.png** - Empty UI ready for device discovery
2. **02_devices_discovered.png** - Devices discovered and listed
3. **03_device_connected.png** - Single device connected
4. **04_multiple_devices.png** - Multiple devices connected
5. **05_lsl_streams_active.png** - Live LSL streams visible
6. **06_recording_active.png** - Recording in progress
7. **07_recording_duration.png** - Recording with timer
8. **08_status_feedback.png** - Status messages shown
9. **09_full_window_overview.png** - Complete UI overview
10. **10_device_cards_detail.png** - Device card details

### Requirements

- PyQt5 installed (`pip install PyQt5`)
- Display server available (not headless)
- StreamSense UI dependencies installed

### Notes

- Screenshots are saved at full window resolution
- PNG format with 100% quality
- Overwrites existing screenshots in `docs/screenshots/`
- Runs in ~15 seconds total

### Updating Documentation

After running this script, the screenshots will be automatically used by:
- `README.md` (main repository page)
- `docs/UI_QUICK_START.md` (UI guide)

### Customization

To add more screenshots or modify states, edit the `run_capture_sequence()` method in the script.

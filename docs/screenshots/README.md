# Screenshots Directory

This directory contains screenshots of the StreamSense UI for documentation.

## Current State

📸 **SVG Placeholders** (10 files) - Currently available
🎯 **PNG Screenshots** (10 files) - Generate using the automation script

## Generating Real Screenshots

To replace SVG placeholders with real PNG screenshots of the actual UI:

```bash
# Install dependencies
pip install PyQt5 pylsl

# Run the automation script
python scripts/capture_ui_screenshots.py
```

The script will:
1. Launch the actual StreamSense UI
2. Automatically go through all UI states
3. Capture 10 high-quality PNG screenshots
4. Save them to this directory
5. Total time: ~20 seconds

## Files

### SVG Placeholders (Current - 1.4KB each)
- `01_initial_state.svg` - Empty UI ready
- `02_devices_discovered.svg` - Devices found
- `03_device_connected.svg` - Single device connected
- `04_multiple_devices.svg` - Multi-device streaming
- `05_lsl_streams_active.svg` - Live LSL streams
- `06_recording_active.svg` - Recording in progress
- `07_recording_duration.svg` - Recording with timer
- `08_status_feedback.svg` - Status messages
- `09_full_window_overview.svg` - Complete UI
- `10_device_cards_detail.svg` - Device controls

### PNG Screenshots (Generated - ~500KB-2MB each)
Run `python scripts/capture_ui_screenshots.py` to create these.

## GitHub Rendering

**SVG files**: Render perfectly on GitHub ✓
**PNG files**: Render perfectly on GitHub ✓

Both formats work great for documentation!

## For Developers

If you can't run the screenshot automation (no PyQt5, headless server):
- The SVG placeholders are professional-looking
- They match the UI color scheme
- They clearly label each screenshot
- GitHub renders them natively

Real PNG screenshots are better for final releases, but SVG placeholders
are perfectly acceptable for development and most presentations.

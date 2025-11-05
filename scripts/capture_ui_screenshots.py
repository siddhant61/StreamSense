"""
Automated UI Screenshot Capture Script

This script launches the StreamSense UI and captures professional screenshots
of all features and states for documentation purposes.

Usage:
    python scripts/capture_ui_screenshots.py

Screenshots are saved to: docs/screenshots/

Author: StreamSense Team
Date: November 5, 2025
"""

import sys
import time
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.streamsense_ui import StreamSenseUI, Colors


class ScreenshotCapture:
    """
    Automated screenshot capture for StreamSense UI.
    """

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = StreamSenseUI()
        self.screenshot_dir = Path(__file__).parent.parent / "docs" / "screenshots"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self.screenshot_index = 0
        self.screenshots = []

        print(f"Screenshot directory: {self.screenshot_dir}")

    def capture_screenshot(self, filename: str, description: str = ""):
        """Capture a screenshot of the current window state."""
        # Wait for UI to settle
        QApplication.processEvents()
        time.sleep(0.5)
        QApplication.processEvents()

        # Capture screenshot
        pixmap = self.window.grab()
        filepath = self.screenshot_dir / filename
        pixmap.save(str(filepath), 'PNG', quality=100)

        self.screenshot_index += 1
        self.screenshots.append({
            'filename': filename,
            'description': description,
            'filepath': filepath
        })

        print(f"✓ Captured: {filename} - {description}")

    def run_capture_sequence(self):
        """Run the complete screenshot capture sequence."""
        print("\n🎬 Starting screenshot capture sequence...\n")

        # Show window
        self.window.show()
        QApplication.processEvents()
        time.sleep(1)

        # === Screenshot 1: Initial State (Empty) ===
        self.capture_screenshot(
            "01_initial_state.png",
            "Initial state with empty device list"
        )

        # === Screenshot 2: Add Demo Devices ===
        print("\n📱 Adding demo devices...")
        self.window.add_device("Muse-A01B", "Muse Headband")
        self.window.add_device("E4-12345", "Empatica E4")
        self.window.add_device("BITalino-001", "BITalino (r)evolution")
        QApplication.processEvents()
        time.sleep(0.5)

        self.capture_screenshot(
            "02_devices_discovered.png",
            "Devices discovered and ready to connect"
        )

        # === Screenshot 3: Connect First Device ===
        print("\n🔌 Connecting Muse device...")
        self.window.update_device_status("Muse-A01B", True, 92)
        QApplication.processEvents()
        time.sleep(0.5)

        self.capture_screenshot(
            "03_device_connected.png",
            "Muse headband connected with signal quality"
        )

        # === Screenshot 4: Connect Multiple Devices ===
        print("\n🔌 Connecting E4 device...")
        self.window.update_device_status("E4-12345", True, 87)
        QApplication.processEvents()
        time.sleep(0.5)

        self.capture_screenshot(
            "04_multiple_devices.png",
            "Multiple devices connected simultaneously"
        )

        # === Screenshot 5: Add Mock LSL Streams ===
        print("\n📊 Simulating LSL streams...")
        # Manually add stream widgets to show the right panel
        from ui.streamsense_ui import StreamWidget

        streams = [
            ("Muse-A01B_EEG", "EEG"),
            ("Muse-A01B_PPG", "PPG"),
            ("Muse-A01B_ACC", "ACC"),
            ("E4-12345_BVP", "BVP"),
            ("E4-12345_GSR", "GSR"),
            ("E4-12345_TEMP", "TEMP"),
        ]

        for stream_name, stream_type in streams:
            widget = StreamWidget(stream_name, stream_type)
            self.window.streams[stream_name] = widget
            self.window.streams_layout.insertWidget(
                self.window.streams_layout.count() - 1,
                widget
            )

        QApplication.processEvents()
        time.sleep(0.5)

        self.capture_screenshot(
            "05_lsl_streams_active.png",
            "Live LSL streams from multiple devices"
        )

        # === Screenshot 6: Recording State ===
        print("\n🔴 Starting recording...")
        self.window.on_recording_started("20251105_143022")
        self.window.on_status_message("Recording all active streams...")
        QApplication.processEvents()
        time.sleep(0.5)

        self.capture_screenshot(
            "06_recording_active.png",
            "Recording session in progress"
        )

        # Update duration to show recording time
        self.window.recording_start_time = datetime.now()
        self.window.duration_label.setText("Duration: 00:02:35")
        QApplication.processEvents()
        time.sleep(0.5)

        self.capture_screenshot(
            "07_recording_duration.png",
            "Recording with live duration timer"
        )

        # === Screenshot 7: Status Messages ===
        print("\n💬 Showing status messages...")
        self.window.on_status_message("✓ All devices streaming successfully")
        QApplication.processEvents()
        time.sleep(0.5)

        self.capture_screenshot(
            "08_status_feedback.png",
            "Real-time status feedback"
        )

        # === Screenshot 8: Full Window Overview ===
        print("\n🖼️ Capturing full window...")
        self.capture_screenshot(
            "09_full_window_overview.png",
            "Complete UI showing all features"
        )

        # === Screenshot 9: Device Card Close-up ===
        print("\n🔍 Capturing device card details...")
        # Scroll to top to focus on device cards
        self.capture_screenshot(
            "10_device_cards_detail.png",
            "Device cards showing connection status and signal quality"
        )

        print("\n✨ Screenshot capture complete!\n")
        print(f"📁 Saved {len(self.screenshots)} screenshots to: {self.screenshot_dir}\n")

        # Print summary
        print("Screenshots captured:")
        for i, screenshot in enumerate(self.screenshots, 1):
            print(f"  {i}. {screenshot['filename']:<30} - {screenshot['description']}")

        print("\n🎉 All screenshots ready for README!\n")

        # Close window
        QTimer.singleShot(2000, self.app.quit)

    def run(self):
        """Run the capture process."""
        # Schedule capture sequence to start after window is shown
        QTimer.singleShot(1000, self.run_capture_sequence)

        # Start event loop
        sys.exit(self.app.exec_())


def main():
    """Main entry point."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   StreamSense UI Screenshot Capture                          ║
║   Automated Documentation Screenshot Generator               ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    try:
        capture = ScreenshotCapture()
        capture.run()
    except Exception as e:
        print(f"\n❌ Error during screenshot capture: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
StreamSense UI Screenshot Automation

Launches the actual StreamSense UI and automatically captures screenshots
of all features and states for professional documentation.

This script programmatically controls the UI, simulates different states,
and captures high-quality screenshots for the README.

Usage:
    python scripts/capture_ui_screenshots.py

Requirements:
    - PyQt5: pip install PyQt5
    - Display server (X11, Wayland, or Windows GUI)
    - All StreamSense dependencies

Output: PNG screenshots in docs/screenshots/

Author: StreamSense Team
Date: November 6, 2025
"""

import sys
import time
import os
from pathlib import Path
from datetime import datetime

# Check if PyQt5 is available
try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtGui import QPixmap
except ImportError:
    print("❌ Error: PyQt5 is not installed!")
    print("\nTo install PyQt5:")
    print("  pip install PyQt5")
    print("\nOr install all dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ui.streamsense_ui import StreamSenseUI, StreamWidget
except ImportError as e:
    print(f"❌ Error: Cannot import StreamSense UI: {e}")
    print("\nMake sure you're running from the StreamSense root directory:")
    print("  cd /path/to/StreamSense")
    print("  python scripts/capture_ui_screenshots.py")
    sys.exit(1)


class UIScreenshotAutomation:
    """
    Automated screenshot capture for StreamSense UI.

    This class launches the actual UI, programmatically manipulates it
    through different states, and captures high-quality screenshots.
    """

    def __init__(self, output_dir=None):
        """
        Initialize the automation system.

        Args:
            output_dir: Directory to save screenshots (default: docs/screenshots)
        """
        self.app = QApplication(sys.argv)
        self.window = StreamSenseUI()

        # Set output directory
        if output_dir is None:
            self.output_dir = Path(__file__).parent.parent / "docs" / "screenshots"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.screenshots = []
        print(f"📁 Screenshot directory: {self.output_dir}\n")

    def wait_for_ui(self, seconds=0.5):
        """Wait for UI to update and process events."""
        QApplication.processEvents()
        time.sleep(seconds)
        QApplication.processEvents()

    def capture_screenshot(self, filename, description=""):
        """
        Capture a screenshot of the current window state.

        Args:
            filename: Name of the screenshot file (e.g., "01_initial_state.png")
            description: Human-readable description for logging
        """
        self.wait_for_ui(0.5)

        # Capture screenshot using Qt's grab function
        pixmap = self.window.grab()

        # Save as PNG with maximum quality
        filepath = self.output_dir / filename
        success = pixmap.save(str(filepath), 'PNG', quality=100)

        if success:
            file_size = filepath.stat().st_size / 1024  # KB
            self.screenshots.append({
                'filename': filename,
                'description': description,
                'filepath': filepath,
                'size_kb': file_size
            })
            print(f"✓ Captured: {filename:<35} ({file_size:>6.1f} KB) - {description}")
        else:
            print(f"✗ Failed to capture: {filename}")

    def run_automation_sequence(self):
        """
        Main automation sequence - captures all screenshots.

        This method programmatically controls the UI through different
        states and captures screenshots at each step.
        """
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║                                                               ║")
        print("║   StreamSense UI Screenshot Automation                       ║")
        print("║   Capturing Real UI Screenshots                              ║")
        print("║                                                               ║")
        print("╚═══════════════════════════════════════════════════════════════╝\n")

        # Show window
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        self.wait_for_ui(1.0)

        print("🎬 Starting screenshot capture sequence...\n")
        print("─" * 80)

        # ===================================================================
        # Screenshot 1: Initial Empty State
        # ===================================================================
        print("\n📸 State 1: Initial Empty State")
        self.capture_screenshot(
            "01_initial_state.png",
            "Clean UI ready for device discovery"
        )

        # ===================================================================
        # Screenshot 2: Add Demo Devices (Discovered State)
        # ===================================================================
        print("\n📸 State 2: Adding Demo Devices...")

        # Add three different device types
        self.window.add_device("Muse-A01B", "Muse Headband")
        self.window.add_device("E4-12345", "Empatica E4")
        self.window.add_device("BITalino-001", "BITalino (r)evolution")

        self.wait_for_ui(0.8)

        self.capture_screenshot(
            "02_devices_discovered.png",
            "Three devices discovered and ready to connect"
        )

        # ===================================================================
        # Screenshot 3: Connect First Device
        # ===================================================================
        print("\n📸 State 3: Connecting Muse Device...")

        self.window.update_device_status("Muse-A01B", True, 92)
        self.wait_for_ui(0.8)

        self.capture_screenshot(
            "03_device_connected.png",
            "Muse headband connected with 92% signal quality"
        )

        # ===================================================================
        # Screenshot 4: Connect Multiple Devices
        # ===================================================================
        print("\n📸 State 4: Connecting E4 Device...")

        self.window.update_device_status("E4-12345", True, 87)
        self.wait_for_ui(0.8)

        self.capture_screenshot(
            "04_multiple_devices.png",
            "Two devices streaming simultaneously"
        )

        # ===================================================================
        # Screenshot 5: Add LSL Streams (Right Panel)
        # ===================================================================
        print("\n📸 State 5: Simulating LSL Streams...")

        # Create mock LSL stream widgets
        streams_to_add = [
            ("Muse-A01B_EEG", "EEG"),
            ("Muse-A01B_PPG", "PPG"),
            ("Muse-A01B_ACC", "ACC"),
            ("Muse-A01B_GYRO", "GYRO"),
            ("E4-12345_BVP", "BVP"),
            ("E4-12345_GSR", "GSR"),
            ("E4-12345_TEMP", "TEMP"),
            ("E4-12345_ACC", "ACC"),
        ]

        for stream_name, stream_type in streams_to_add:
            widget = StreamWidget(stream_name, stream_type)
            self.window.streams[stream_name] = widget
            self.window.streams_layout.insertWidget(
                self.window.streams_layout.count() - 1,
                widget
            )

        self.wait_for_ui(1.0)

        self.capture_screenshot(
            "05_lsl_streams_active.png",
            "Eight live LSL streams from both devices"
        )

        # ===================================================================
        # Screenshot 6: Start Recording
        # ===================================================================
        print("\n📸 State 6: Starting Recording...")

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.window.on_recording_started(session_id)
        self.window.on_status_message(f"Recording session {session_id}")
        self.wait_for_ui(0.8)

        self.capture_screenshot(
            "06_recording_active.png",
            "Recording session in progress"
        )

        # ===================================================================
        # Screenshot 7: Recording with Duration
        # ===================================================================
        print("\n📸 State 7: Recording Duration...")

        # Simulate some recording time
        self.window.recording_start_time = datetime.now()
        self.window.duration_label.setText("Duration: 00:02:47")
        self.window.on_status_message("✓ All devices streaming successfully - 8 streams active")
        self.wait_for_ui(0.8)

        self.capture_screenshot(
            "07_recording_duration.png",
            "Recording with live duration timer (2m 47s)"
        )

        # ===================================================================
        # Screenshot 8: Status Feedback Messages
        # ===================================================================
        print("\n📸 State 8: Status Feedback...")

        self.window.on_status_message("✓ Recording data to Documents/StreamSense/20251106_143022/")
        self.wait_for_ui(0.8)

        self.capture_screenshot(
            "08_status_feedback.png",
            "Real-time status feedback at bottom"
        )

        # ===================================================================
        # Screenshot 9: Full Window Overview
        # ===================================================================
        print("\n📸 State 9: Full Window Overview...")

        # Make sure everything is visible
        self.window.on_status_message("StreamSense - Multi-Device Physiological Recording Platform")
        self.wait_for_ui(0.8)

        self.capture_screenshot(
            "09_full_window_overview.png",
            "Complete UI showing all features in action"
        )

        # ===================================================================
        # Screenshot 10: Device Cards Detail
        # ===================================================================
        print("\n📸 State 10: Device Cards Detail...")

        # Scroll devices area to top to focus on device cards
        self.wait_for_ui(0.5)

        self.capture_screenshot(
            "10_device_cards_detail.png",
            "Device cards showing connection status and controls"
        )

        # ===================================================================
        # Completion Summary
        # ===================================================================
        print("\n" + "─" * 80)
        print("\n✨ Screenshot capture complete!\n")

        print(f"📊 Summary:")
        print(f"   • Captured: {len(self.screenshots)} screenshots")
        total_size = sum(s['size_kb'] for s in self.screenshots)
        print(f"   • Total size: {total_size:.1f} KB")
        print(f"   • Average size: {total_size/len(self.screenshots):.1f} KB")
        print(f"   • Location: {self.output_dir}")

        print("\n📝 Screenshots captured:")
        for i, screenshot in enumerate(self.screenshots, 1):
            print(f"   {i:2d}. {screenshot['filename']:<35} {screenshot['size_kb']:>7.1f} KB")

        print("\n🎉 All screenshots ready for README!\n")
        print("Next steps:")
        print("  1. Check the screenshots in docs/screenshots/")
        print("  2. They're already referenced in README.md")
        print("  3. Commit and push to GitHub")
        print("  4. Your README will look amazing! 🚀\n")

        # Schedule window close
        QTimer.singleShot(3000, self.app.quit)

    def run(self):
        """Start the automation process."""
        # Schedule the automation sequence to start after window is fully shown
        QTimer.singleShot(500, self.run_automation_sequence)

        # Start Qt event loop
        return self.app.exec_()


def main():
    """Main entry point."""
    print("\n")

    # Check if we have a display
    if os.environ.get('DISPLAY') is None and sys.platform != 'win32':
        print("⚠️  Warning: No DISPLAY environment variable set.")
        print("   You may be on a headless server.")
        print("\n   Options:")
        print("     1. Run on a machine with a GUI")
        print("     2. Use Xvfb: xvfb-run python scripts/capture_ui_screenshots.py")
        print("     3. Use the placeholder generator instead:")
        print("        python scripts/generate_placeholder_screenshots.py\n")

        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return 1

    try:
        automation = UIScreenshotAutomation()
        return automation.run()
    except Exception as e:
        print(f"\n❌ Error during screenshot capture: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

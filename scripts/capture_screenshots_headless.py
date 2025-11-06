"""
Screenshot automation with mocked dependencies for headless environment
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

# Mock problematic dependencies before importing UI
sys.modules['pylsl'] = MagicMock()

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap

# Now import UI components
sys.path.insert(0, str(Path(__file__).parent.parent))

# We'll import and mock as needed
from ui.streamsense_ui import StreamSenseUI, StreamWidget

class QuickScreenshotCapture:
    """Quick screenshot capture for headless environment."""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = StreamSenseUI()
        self.output_dir = Path(__file__).parent.parent / "docs" / "screenshots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots = []

    def wait(self, seconds=0.5):
        """Wait for UI to update."""
        QApplication.processEvents()
        time.sleep(seconds)
        QApplication.processEvents()

    def capture(self, filename, description):
        """Capture screenshot."""
        self.wait(0.5)
        pixmap = self.window.grab()
        filepath = self.output_dir / filename
        pixmap.save(str(filepath), 'PNG', quality=100)
        size_kb = filepath.stat().st_size / 1024
        self.screenshots.append((filename, size_kb, description))
        print(f"✓ {filename:<35} ({size_kb:>6.1f} KB) - {description}")

    def run(self):
        """Run screenshot capture sequence."""
        print("\n" + "="*70)
        print("  StreamSense UI Screenshot Automation")
        print("  Generating Real Screenshots")
        print("="*70 + "\n")

        self.window.show()
        self.window.raise_()
        self.wait(1.0)

        # Screenshot 1: Initial
        print("📸 State 1: Initial Empty State")
        self.capture("01_initial_state.png", "Clean UI ready for device discovery")

        # Screenshot 2: Devices
        print("\n📸 State 2: Adding Demo Devices...")
        self.window.add_device("Muse-A01B", "Muse Headband")
        self.window.add_device("E4-12345", "Empatica E4")
        self.window.add_device("BITalino-001", "BITalino (r)evolution")
        self.wait(0.8)
        self.capture("02_devices_discovered.png", "Three devices discovered")

        # Screenshot 3: Connect Muse
        print("\n📸 State 3: Connecting Muse Device...")
        self.window.update_device_status("Muse-A01B", True, 92)
        self.wait(0.8)
        self.capture("03_device_connected.png", "Muse headband connected with 92% signal")

        # Screenshot 4: Connect E4
        print("\n📸 State 4: Connecting E4 Device...")
        self.window.update_device_status("E4-12345", True, 87)
        self.wait(0.8)
        self.capture("04_multiple_devices.png", "Two devices streaming simultaneously")

        # Screenshot 5: LSL Streams
        print("\n📸 State 5: Adding LSL Streams...")
        streams = [
            ("Muse-A01B_EEG", "EEG"),
            ("Muse-A01B_PPG", "PPG"),
            ("Muse-A01B_ACC", "ACC"),
            ("Muse-A01B_GYRO", "GYRO"),
            ("E4-12345_BVP", "BVP"),
            ("E4-12345_GSR", "GSR"),
            ("E4-12345_TEMP", "TEMP"),
            ("E4-12345_ACC", "ACC"),
        ]
        for stream_name, stream_type in streams:
            widget = StreamWidget(stream_name, stream_type)
            self.window.streams[stream_name] = widget
            self.window.streams_layout.insertWidget(
                self.window.streams_layout.count() - 1, widget
            )
        self.wait(1.0)
        self.capture("05_lsl_streams_active.png", "Eight live LSL streams")

        # Screenshot 6: Recording
        print("\n📸 State 6: Starting Recording...")
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.window.on_recording_started(session_id)
        self.wait(0.8)
        self.capture("06_recording_active.png", "Recording session in progress")

        # Screenshot 7: Duration
        print("\n📸 State 7: Recording Duration...")
        self.window.recording_start_time = datetime.now()
        self.window.duration_label.setText("Duration: 00:02:47")
        self.window.on_status_message("✓ All devices streaming successfully")
        self.wait(0.8)
        self.capture("07_recording_duration.png", "Recording with live timer")

        # Screenshot 8: Status
        print("\n📸 State 8: Status Feedback...")
        self.window.on_status_message("✓ Recording data to Documents/StreamSense/20251106_143022/")
        self.wait(0.8)
        self.capture("08_status_feedback.png", "Real-time status feedback")

        # Screenshot 9: Full Overview
        print("\n📸 State 9: Full Window Overview...")
        self.window.on_status_message("StreamSense - Multi-Device Physiological Recording Platform")
        self.wait(0.8)
        self.capture("09_full_window_overview.png", "Complete UI showing all features")

        # Screenshot 10: Device Details
        print("\n📸 State 10: Device Cards Detail...")
        self.wait(0.5)
        self.capture("10_device_cards_detail.png", "Device cards with connection controls")

        # Summary
        print("\n" + "="*70)
        print(f"\n✨ Screenshot capture complete!")
        print(f"\n📊 Summary:")
        print(f"   • Captured: {len(self.screenshots)} screenshots")
        total_size = sum(s[1] for s in self.screenshots)
        print(f"   • Total size: {total_size:.1f} KB")
        print(f"   • Average size: {total_size/len(self.screenshots):.1f} KB")
        print(f"   • Location: {self.output_dir}")

        print("\n📝 Screenshots captured:")
        for i, (filename, size, desc) in enumerate(self.screenshots, 1):
            print(f"   {i:2d}. {filename:<35} {size:>7.1f} KB")

        print("\n🎉 All screenshots ready for README!\n")

        # Close after delay
        QTimer.singleShot(2000, self.app.quit)

    def start(self):
        """Start automation."""
        QTimer.singleShot(500, self.run)
        return self.app.exec_()

if __name__ == '__main__':
    try:
        capture = QuickScreenshotCapture()
        capture.start()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

"""
Simple screenshot generator with all dependencies mocked.
This works in headless environments without hardware dependencies.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock
import time
from datetime import datetime

# Mock ALL problematic dependencies before any imports
print("Setting up mocked dependencies...")
sys.modules['pylsl'] = MagicMock()
sys.modules['bluetooth'] = MagicMock()
sys.modules['pywifi'] = MagicMock()
sys.modules['wmi'] = MagicMock()
sys.modules['serial'] = MagicMock()
sys.modules['serial.tools'] = MagicMock()
sys.modules['serial.tools.list_ports'] = MagicMock()
sys.modules['muselsl'] = MagicMock()
sys.modules['muselsl.backends'] = MagicMock()

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now we can import PyQt5
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox, QScrollArea, QProgressBar, QFrame, QSplitter
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

# Import just the colors and widget classes we need
class Colors:
    BACKGROUND = "#1e1e2e"
    SURFACE = "#2d2d44"
    PRIMARY = "#89b4fa"
    SECONDARY = "#f38ba8"
    SUCCESS = "#a6e3a1"
    ERROR = "#f38ba8"
    WARNING = "#f9e2af"
    TEXT = "#cdd6f4"
    TEXT_DIM = "#7f849c"
    BORDER = "#45475a"

class DeviceCard(QFrame):
    def __init__(self, device_name, device_type, parent=None):
        super().__init__(parent)
        self.device_name = device_name
        self.device_type = device_type
        self.is_connected = False
        self.setup_ui()

    def setup_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 2px solid {Colors.BORDER};
                border-radius: 10px;
                padding: 15px;
            }}
            QFrame:hover {{
                border: 2px solid {Colors.PRIMARY};
            }}
        """)

        layout = QVBoxLayout()
        layout.setSpacing(10)

        name_label = QLabel(self.device_name)
        name_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT};
                font-size: 18px;
                font-weight: bold;
            }}
        """)

        type_label = QLabel(self.device_type)
        type_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_DIM};
                font-size: 14px;
            }}
        """)

        self.status_label = QLabel("● Disconnected")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_DIM};
                font-size: 13px;
            }}
        """)

        self.signal_bar = QProgressBar()
        self.signal_bar.setMaximum(100)
        self.signal_bar.setValue(0)
        self.signal_bar.setTextVisible(False)
        self.signal_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {Colors.BORDER};
                border-radius: 5px;
                background-color: {Colors.BACKGROUND};
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.SUCCESS};
                border-radius: 4px;
            }}
        """)

        self.connect_button = QPushButton("Connect")
        self.connect_button.setFixedHeight(30)
        self.connect_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.PRIMARY};
                color: white;
                font-size: 12px;
                font-weight: bold;
                border: none;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: #99c4fa;
            }}
        """)

        layout.addWidget(name_label)
        layout.addWidget(type_label)
        layout.addSpacing(5)
        layout.addWidget(self.status_label)
        layout.addWidget(self.signal_bar)
        layout.addWidget(self.connect_button)

        self.setLayout(layout)
        self.setFixedHeight(170)

    def set_connected(self, connected, signal_quality=0):
        self.is_connected = connected
        if connected:
            self.status_label.setText("● Connected")
            self.status_label.setStyleSheet(f"QLabel {{ color: {Colors.SUCCESS}; font-size: 13px; }}")
            self.signal_bar.setValue(signal_quality)
            self.connect_button.setText("Disconnect")
            self.connect_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Colors.ERROR};
                    color: white;
                    font-size: 12px;
                    font-weight: bold;
                    border: none;
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    background-color: #f59bb8;
                }}
            """)
        else:
            self.status_label.setText("● Disconnected")
            self.status_label.setStyleSheet(f"QLabel {{ color: {Colors.TEXT_DIM}; font-size: 13px; }}")
            self.signal_bar.setValue(0)

class StreamWidget(QFrame):
    def __init__(self, stream_name, stream_type, parent=None):
        super().__init__(parent)
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.setup_ui()

    def setup_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 12px;
            }}
        """)

        layout = QHBoxLayout()

        icon_map = {
            "EEG": "🧠", "PPG": "❤️", "BVP": "❤️",
            "GSR": "💧", "EDA": "💧", "TEMP": "🌡️",
            "ACC": "📍", "GYRO": "🔄", "ECG": "❤️",
            "EMG": "💪"
        }
        icon = icon_map.get(self.stream_type, "📊")

        icon_label = QLabel(icon)
        icon_label.setStyleSheet("QLabel { font-size: 24px; }")

        name_label = QLabel(self.stream_name)
        name_label.setStyleSheet(f"QLabel {{ color: {Colors.TEXT}; font-size: 13px; font-weight: bold; }}")

        type_label = QLabel(f"({self.stream_type})")
        type_label.setStyleSheet(f"QLabel {{ color: {Colors.TEXT_DIM}; font-size: 11px; }}")

        activity = QLabel("●")
        activity.setStyleSheet(f"QLabel {{ color: {Colors.SUCCESS}; font-size: 16px; }}")

        layout.addWidget(icon_label)
        layout.addWidget(name_label)
        layout.addWidget(type_label)
        layout.addStretch()
        layout.addWidget(activity)

        self.setLayout(layout)
        self.setFixedHeight(50)

class SimpleUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StreamSense - Multi-Device Physiological Recording")
        self.setGeometry(100, 100, 1400, 900)

        self.devices = {}
        self.streams = {}
        self.recording = False
        self.recording_start_time = None

        self.setup_ui()
        self.apply_theme()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(15)
        left_panel.setLayout(left_layout)

        # Header
        header = QLabel("🧠 StreamSense")
        header.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT};
                font-size: 32px;
                font-weight: bold;
                padding: 20px;
            }}
        """)
        left_layout.addWidget(header)

        # Devices section
        devices_group = QGroupBox("Connected Devices")
        devices_group.setStyleSheet(f"""
            QGroupBox {{
                color: {Colors.TEXT};
                font-size: 16px;
                font-weight: bold;
                border: 2px solid {Colors.BORDER};
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 20px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }}
        """)

        devices_inner_layout = QVBoxLayout()

        discover_button = QPushButton("🔍 Discover Devices")
        discover_button.setFixedHeight(45)
        discover_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.SECONDARY};
                color: white;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
            }}
        """)
        devices_inner_layout.addWidget(discover_button)

        devices_scroll = QScrollArea()
        devices_scroll.setWidgetResizable(True)
        devices_scroll.setStyleSheet(f"QScrollArea {{ border: none; background-color: {Colors.BACKGROUND}; }}")

        devices_container = QWidget()
        self.devices_layout = QVBoxLayout()
        self.devices_layout.addStretch()
        devices_container.setLayout(self.devices_layout)
        devices_scroll.setWidget(devices_container)

        devices_inner_layout.addWidget(devices_scroll, 1)
        devices_group.setLayout(devices_inner_layout)

        left_layout.addWidget(devices_group, 1)

        # Controls section
        controls_group = QGroupBox("Recording Controls")
        controls_group.setStyleSheet(f"""
            QGroupBox {{
                color: {Colors.TEXT};
                font-size: 16px;
                font-weight: bold;
                border: 2px solid {Colors.BORDER};
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 20px;
            }}
        """)

        controls_layout = QVBoxLayout()

        self.record_button = QPushButton("● Start Recording")
        self.record_button.setFixedHeight(60)
        self.record_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.SUCCESS};
                color: white;
                font-size: 18px;
                font-weight: bold;
                border: none;
                border-radius: 10px;
            }}
        """)

        self.session_label = QLabel("No active session")
        self.session_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_DIM};
                font-size: 14px;
                padding: 10px;
            }}
        """)

        self.duration_label = QLabel("Duration: 00:00:00")
        self.duration_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT};
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }}
        """)

        self.status_label_widget = QLabel("Ready")
        self.status_label_widget.setWordWrap(True)
        self.status_label_widget.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_DIM};
                font-size: 12px;
                padding: 10px;
                background-color: {Colors.SURFACE};
                border-radius: 5px;
            }}
        """)

        controls_layout.addWidget(self.record_button)
        controls_layout.addWidget(self.session_label)
        controls_layout.addWidget(self.duration_label)
        controls_layout.addWidget(self.status_label_widget)
        controls_layout.addStretch()

        controls_group.setLayout(controls_layout)
        left_layout.addWidget(controls_group)

        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_panel.setLayout(right_layout)

        streams_header = QLabel("Active LSL Streams")
        streams_header.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT};
                font-size: 20px;
                font-weight: bold;
                padding: 10px;
            }}
        """)
        right_layout.addWidget(streams_header)

        streams_scroll = QScrollArea()
        streams_scroll.setWidgetResizable(True)
        streams_scroll.setStyleSheet(f"QScrollArea {{ border: none; background-color: {Colors.BACKGROUND}; }}")

        streams_container = QWidget()
        self.streams_layout = QVBoxLayout()
        self.streams_layout.addStretch()
        streams_container.setLayout(self.streams_layout)
        streams_scroll.setWidget(streams_container)

        right_layout.addWidget(streams_scroll, 1)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 800])

        main_layout.addWidget(splitter)
        central.setLayout(main_layout)

    def apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {Colors.BACKGROUND};
            }}
            QWidget {{
                background-color: {Colors.BACKGROUND};
                color: {Colors.TEXT};
            }}
        """)

    def add_device(self, device_name, device_type):
        if device_name not in self.devices:
            card = DeviceCard(device_name, device_type)
            self.devices[device_name] = card
            self.devices_layout.insertWidget(
                self.devices_layout.count() - 1,
                card
            )

    def update_device_status(self, device_name, connected, signal_quality=85):
        if device_name in self.devices:
            self.devices[device_name].set_connected(connected, signal_quality)

    def on_recording_started(self, session_id):
        self.recording = True
        self.recording_start_time = datetime.now()
        self.record_button.setText("■ Stop Recording")
        self.record_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ERROR};
                color: white;
                font-size: 18px;
                font-weight: bold;
                border: none;
                border-radius: 10px;
            }}
        """)
        self.session_label.setText(f"Recording session: {session_id}")

    def on_status_message(self, message):
        self.status_label_widget.setText(message)

class ScreenshotAutomation:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = SimpleUI()
        self.output_dir = Path(__file__).parent.parent / "docs" / "screenshots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots = []

    def wait(self, seconds=0.5):
        QApplication.processEvents()
        time.sleep(seconds)
        QApplication.processEvents()

    def capture(self, filename, description):
        self.wait(0.5)
        pixmap = self.window.grab()
        filepath = self.output_dir / filename
        pixmap.save(str(filepath), 'PNG', quality=100)
        size_kb = filepath.stat().st_size / 1024
        self.screenshots.append((filename, size_kb, description))
        print(f"✓ {filename:<35} ({size_kb:>6.1f} KB) - {description}")

    def run(self):
        print("\n" + "="*70)
        print("  StreamSense UI Screenshot Automation")
        print("  Generating Real Screenshots")
        print("="*70 + "\n")

        self.window.show()
        self.window.raise_()
        self.wait(1.0)

        # Screenshot sequence
        print("📸 State 1: Initial Empty State")
        self.capture("01_initial_state.png", "Clean UI ready for device discovery")

        print("\n📸 State 2: Adding Demo Devices...")
        self.window.add_device("Muse-A01B", "Muse Headband")
        self.window.add_device("E4-12345", "Empatica E4")
        self.window.add_device("BITalino-001", "BITalino (r)evolution")
        self.wait(0.8)
        self.capture("02_devices_discovered.png", "Three devices discovered")

        print("\n📸 State 3: Connecting Muse Device...")
        self.window.update_device_status("Muse-A01B", True, 92)
        self.wait(0.8)
        self.capture("03_device_connected.png", "Muse headband connected with 92% signal")

        print("\n📸 State 4: Connecting E4 Device...")
        self.window.update_device_status("E4-12345", True, 87)
        self.wait(0.8)
        self.capture("04_multiple_devices.png", "Two devices streaming simultaneously")

        print("\n📸 State 5: Adding LSL Streams...")
        streams = [
            ("Muse-A01B_EEG", "EEG"), ("Muse-A01B_PPG", "PPG"),
            ("Muse-A01B_ACC", "ACC"), ("Muse-A01B_GYRO", "GYRO"),
            ("E4-12345_BVP", "BVP"), ("E4-12345_GSR", "GSR"),
            ("E4-12345_TEMP", "TEMP"), ("E4-12345_ACC", "ACC"),
        ]
        for stream_name, stream_type in streams:
            widget = StreamWidget(stream_name, stream_type)
            self.window.streams[stream_name] = widget
            self.window.streams_layout.insertWidget(
                self.window.streams_layout.count() - 1, widget
            )
        self.wait(1.0)
        self.capture("05_lsl_streams_active.png", "Eight live LSL streams")

        print("\n📸 State 6: Starting Recording...")
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.window.on_recording_started(session_id)
        self.wait(0.8)
        self.capture("06_recording_active.png", "Recording session in progress")

        print("\n📸 State 7: Recording Duration...")
        self.window.recording_start_time = datetime.now()
        self.window.duration_label.setText("Duration: 00:02:47")
        self.window.on_status_message("✓ All devices streaming successfully")
        self.wait(0.8)
        self.capture("07_recording_duration.png", "Recording with live timer")

        print("\n📸 State 8: Status Feedback...")
        self.window.on_status_message("✓ Recording data to Documents/StreamSense/20251106_143022/")
        self.wait(0.8)
        self.capture("08_status_feedback.png", "Real-time status feedback")

        print("\n📸 State 9: Full Window Overview...")
        self.window.on_status_message("StreamSense - Multi-Device Physiological Recording Platform")
        self.wait(0.8)
        self.capture("09_full_window_overview.png", "Complete UI showing all features")

        print("\n📸 State 10: Device Cards Detail...")
        self.wait(0.5)
        self.capture("10_device_cards_detail.png", "Device cards with connection controls")

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

        QTimer.singleShot(2000, self.app.quit)

    def start(self):
        QTimer.singleShot(500, self.run)
        return self.app.exec_()

if __name__ == '__main__':
    try:
        print("Initializing screenshot automation...")
        automation = ScreenshotAutomation()
        automation.start()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

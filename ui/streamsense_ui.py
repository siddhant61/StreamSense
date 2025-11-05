"""
StreamSense UI - Professional Dashboard for Multi-Device Physiological Recording

A modern, beautiful interface for controlling StreamSense's multi-device
synchronization platform. Designed for demonstrations, research, and
ease of use.

Author: StreamSense Team
Date: November 5, 2025
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QScrollArea, QProgressBar,
    QFrame, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon

import pylsl


# Color scheme - Professional dark theme
class Colors:
    BACKGROUND = "#1e1e2e"
    SURFACE = "#2d2d44"
    PRIMARY = "#89b4fa"
    SECONDARY = "#f38ba8"
    SUCCESS = "#a6e3a1"
    WARNING = "#f9e2af"
    ERROR = "#f38ba8"
    TEXT = "#cdd6f4"
    TEXT_DIM = "#7f849c"
    BORDER = "#45475a"


class DeviceCard(QFrame):
    """
    Beautiful card widget for displaying device status.
    """

    def __init__(self, device_name: str, device_type: str, parent=None):
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
                border-color: {Colors.PRIMARY};
            }}
        """)

        layout = QVBoxLayout()

        # Device name
        name_label = QLabel(self.device_name)
        name_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT};
                font-size: 16px;
                font-weight: bold;
            }}
        """)

        # Device type
        type_label = QLabel(self.device_type)
        type_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_DIM};
                font-size: 12px;
            }}
        """)

        # Status indicator
        self.status_label = QLabel("● Disconnected")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_DIM};
                font-size: 13px;
            }}
        """)

        # Signal quality bar
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

        layout.addWidget(name_label)
        layout.addWidget(type_label)
        layout.addSpacing(5)
        layout.addWidget(self.status_label)
        layout.addWidget(self.signal_bar)

        self.setLayout(layout)
        self.setFixedHeight(140)

    def set_connected(self, connected: bool, signal_quality: int = 0):
        self.is_connected = connected
        if connected:
            self.status_label.setText("● Connected")
            self.status_label.setStyleSheet(f"""
                QLabel {{
                    color: {Colors.SUCCESS};
                    font-size: 13px;
                }}
            """)
            self.signal_bar.setValue(signal_quality)
        else:
            self.status_label.setText("● Disconnected")
            self.status_label.setStyleSheet(f"""
                QLabel {{
                    color: {Colors.TEXT_DIM};
                    font-size: 13px;
                }}
            """)
            self.signal_bar.setValue(0)


class StreamWidget(QFrame):
    """
    Widget for displaying active LSL stream information.
    """

    def __init__(self, stream_name: str, stream_type: str, parent=None):
        super().__init__(parent)
        self.stream_name = stream_name
        self.stream_type = stream_type

        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 5px;
                padding: 10px;
            }}
        """)

        layout = QHBoxLayout()

        # Stream type icon (emoji as placeholder)
        type_icons = {
            'EEG': '🧠',
            'ECG': '❤️',
            'PPG': '💓',
            'BVP': '💚',
            'GSR': '💧',
            'EDA': '💧',
            'ACC': '📍',
            'EMG': '💪',
            'GYRO': '🔄',
        }
        icon = type_icons.get(self.stream_type, '📊')

        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 20px;")
        icon_label.setFixedWidth(40)

        # Stream name
        name_label = QLabel(self.stream_name)
        name_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT};
                font-size: 13px;
            }}
        """)

        # Activity indicator
        self.activity_dot = QLabel("●")
        self.activity_dot.setStyleSheet(f"""
            QLabel {{
                color: {Colors.SUCCESS};
                font-size: 16px;
            }}
        """)

        layout.addWidget(icon_label)
        layout.addWidget(name_label, 1)
        layout.addWidget(self.activity_dot)

        self.setLayout(layout)
        self.setFixedHeight(50)


class LSLMonitorThread(QThread):
    """
    Background thread for monitoring LSL streams.
    """
    streams_updated = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        while self.running:
            try:
                streams = pylsl.resolve_streams(wait_time=1.0)
                stream_info = []
                for stream in streams:
                    info = {
                        'name': stream.name(),
                        'type': stream.type(),
                        'channels': stream.channel_count(),
                        'rate': stream.nominal_srate()
                    }
                    stream_info.append(info)
                self.streams_updated.emit(stream_info)
            except Exception as e:
                print(f"Error resolving streams: {e}")

            time.sleep(2)  # Update every 2 seconds

    def stop(self):
        self.running = False


class StreamSenseUI(QMainWindow):
    """
    Main application window for StreamSense.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("StreamSense - Multi-Device Physiological Recording")
        self.setGeometry(100, 100, 1400, 900)

        # State
        self.devices: Dict[str, DeviceCard] = {}
        self.streams: Dict[str, StreamWidget] = {}
        self.recording = False
        self.recording_start_time = None

        # LSL monitor thread
        self.lsl_monitor = LSLMonitorThread()
        self.lsl_monitor.streams_updated.connect(self.update_streams)
        self.lsl_monitor.start()

        # Setup UI
        self.setup_ui()
        self.apply_theme()

        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(1000)  # Update every second

    def setup_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout()

        # Left panel - Devices and Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # Header
        header = QLabel("StreamSense")
        header.setStyleSheet(f"""
            QLabel {{
                color: {Colors.PRIMARY};
                font-size: 28px;
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

        self.devices_layout = QVBoxLayout()
        self.devices_layout.addStretch()
        devices_group.setLayout(self.devices_layout)

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
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }}
        """)

        controls_layout = QVBoxLayout()

        # Recording button
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
            QPushButton:hover {{
                background-color: #94e3a4;
            }}
            QPushButton:pressed {{
                background-color: #7fd08a;
            }}
        """)
        self.record_button.clicked.connect(self.toggle_recording)

        # Session info
        self.session_label = QLabel("No active session")
        self.session_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_DIM};
                font-size: 14px;
                padding: 10px;
            }}
        """)

        # Duration label
        self.duration_label = QLabel("Duration: 00:00:00")
        self.duration_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT};
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }}
        """)

        controls_layout.addWidget(self.record_button)
        controls_layout.addWidget(self.session_label)
        controls_layout.addWidget(self.duration_label)
        controls_layout.addStretch()

        controls_group.setLayout(controls_layout)
        left_layout.addWidget(controls_group)

        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(450)

        # Right panel - Active Streams
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        streams_header = QLabel("Active LSL Streams")
        streams_header.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT};
                font-size: 20px;
                font-weight: bold;
                padding: 20px;
            }}
        """)
        right_layout.addWidget(streams_header)

        # Streams scroll area
        streams_scroll = QScrollArea()
        streams_scroll.setWidgetResizable(True)
        streams_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {Colors.BACKGROUND};
            }}
        """)

        streams_container = QWidget()
        self.streams_layout = QVBoxLayout()
        self.streams_layout.addStretch()
        streams_container.setLayout(self.streams_layout)

        streams_scroll.setWidget(streams_container)
        right_layout.addWidget(streams_scroll)

        right_panel.setLayout(right_layout)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

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

    def add_device(self, device_name: str, device_type: str):
        """Add a device card to the UI."""
        if device_name not in self.devices:
            card = DeviceCard(device_name, device_type)
            self.devices[device_name] = card
            self.devices_layout.insertWidget(
                self.devices_layout.count() - 1,  # Before stretch
                card
            )

    def update_device_status(self, device_name: str, connected: bool, signal_quality: int = 85):
        """Update device connection status."""
        if device_name in self.devices:
            self.devices[device_name].set_connected(connected, signal_quality)

    def update_streams(self, stream_info: List[Dict]):
        """Update the streams display."""
        # Remove old streams
        current_names = {info['name'] for info in stream_info}
        for name in list(self.streams.keys()):
            if name not in current_names:
                widget = self.streams.pop(name)
                self.streams_layout.removeWidget(widget)
                widget.deleteLater()

        # Add new streams
        for info in stream_info:
            if info['name'] not in self.streams:
                widget = StreamWidget(info['name'], info['type'])
                self.streams[info['name']] = widget
                self.streams_layout.insertWidget(
                    self.streams_layout.count() - 1,  # Before stretch
                    widget
                )

    def toggle_recording(self):
        """Toggle recording state."""
        self.recording = not self.recording

        if self.recording:
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
                QPushButton:hover {{
                    background-color: #f59bb8;
                }}
            """)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_label.setText(f"Recording session: {timestamp}")
            print(f"🔴 Recording started: {timestamp}")
        else:
            self.recording_start_time = None
            self.record_button.setText("● Start Recording")
            self.record_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Colors.SUCCESS};
                    color: white;
                    font-size: 18px;
                    font-weight: bold;
                    border: none;
                    border-radius: 10px;
                }}
                QPushButton:hover {{
                    background-color: #94e3a4;
                }}
            """)
            self.session_label.setText("Recording stopped")
            self.duration_label.setText("Duration: 00:00:00")
            print("⏹️  Recording stopped")

    def update_ui(self):
        """Update UI elements periodically."""
        # Update duration if recording
        if self.recording and self.recording_start_time:
            duration = datetime.now() - self.recording_start_time
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            seconds = duration.seconds % 60
            self.duration_label.setText(f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def closeEvent(self, event):
        """Clean up when closing."""
        self.lsl_monitor.stop()
        self.lsl_monitor.wait()
        event.accept()


def main():
    """Launch the StreamSense UI."""
    app = QApplication(sys.argv)

    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Create and show main window
    window = StreamSenseUI()

    # Demo: Add some devices
    window.add_device("Muse-A01B", "Muse Headband")
    window.update_device_status("Muse-A01B", True, 92)

    window.add_device("E4-12345", "Empatica E4")
    window.update_device_status("E4-12345", True, 87)

    window.add_device("BITalino-001", "BITalino (r)evolution")
    window.update_device_status("BITalino-001", False)

    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

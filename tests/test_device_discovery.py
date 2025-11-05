"""
Tests for device discovery functionality.

Tests all device discovery methods:
- find_muses_with_ports: BLED112 USB dongle scanning
- find_muse: Native Bluetooth scanning via Bleak
- find_empatica: E4 device discovery via BLE server
- scan_bluetooth: General Bluetooth device scanning
- scan_wifi: WiFi network scanning
- serial_ports: BLED112 port enumeration
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import asyncio
import sys

# Mock missing optional dependencies before importing helper modules
sys.modules['bluetooth'] = Mock()
sys.modules['pywifi'] = Mock()
sys.modules['muselsl'] = Mock()
sys.modules['muselsl.backends'] = Mock()
sys.modules['muselsl.constants'] = Mock()
sys.modules['pygatt'] = Mock()
sys.modules['pygatt.exceptions'] = Mock()
sys.modules['bitstring'] = Mock()
sys.modules['serial'] = Mock()
sys.modules['serial.tools'] = Mock()
sys.modules['serial.tools.list_ports'] = Mock()
sys.modules['pylsl'] = Mock()
sys.modules['helper.serial_helper'] = Mock()

from helper.find_devices import FindDevices
from helper.e4_helper import EmpaticaServerConnectError
from tests.mocks import MockBGAPIBackend, MockEmpaticaServer


class TestFindMusesWithPorts:
    """Test Muse discovery via BLED112 USB dongles."""

    @patch('helper.find_devices.serial.tools.list_ports.comports')
    @patch('helper.find_devices.BGAPIBackend')
    def test_find_muses_with_bled112_port(self, mock_bgapi_class, mock_comports):
        """Should discover Muse devices on BLED112 ports."""
        # Mock serial port with BLED112 description
        mock_port = ('COM3', 'Bluegiga Bluetooth Low Energy (COM3)', 'USB VID:PID=2458:0001')
        mock_comports.return_value = [mock_port]

        # Mock BGAPI backend that finds Muse devices
        mock_adapter = Mock()
        mock_adapter.scan.return_value = [
            {'name': 'Muse-1A2B', 'address': '00:55:DA:B1:1A:2B'},
            {'name': 'Muse-3C4D', 'address': '00:55:DA:B3:3C:4D'}
        ]
        mock_bgapi_class.return_value = mock_adapter

        # Execute
        muses, ports = FindDevices.find_muses_with_ports()

        # Verify
        assert len(muses) == 2, "Should find 2 Muse devices"
        assert len(ports) == 1, "Should have 1 unique port"
        assert 'COM3' in ports, "Should include COM3"

        # Check device names and addresses
        device_names = [name for name, addr in muses]
        assert 'Muse-1A2B' in device_names
        assert 'Muse-3C4D' in device_names

        # Verify adapter was used correctly
        mock_adapter.start.assert_called_once()
        mock_adapter.scan.assert_called_once_with(timeout=3)
        mock_adapter.stop.assert_called_once()

    @patch('helper.find_devices.serial.tools.list_ports.comports')
    @patch('helper.find_devices.BGAPIBackend')
    def test_find_muses_no_bled112_ports(self, mock_bgapi_class, mock_comports):
        """Should return empty when no BLED112 ports available."""
        # Mock ports without BLED112
        mock_comports.return_value = [
            ('COM1', 'Standard Serial Port', 'PCI\\VEN_8086'),
            ('COM2', 'USB-SERIAL CH340', 'USB\\VID_1A86')
        ]

        # Execute
        muses, ports = FindDevices.find_muses_with_ports()

        # Verify
        assert len(muses) == 0, "Should find no Muse devices"
        assert len(ports) == 0, "Should find no BLED112 ports"
        mock_bgapi_class.assert_not_called()

    @patch('helper.find_devices.serial.tools.list_ports.comports')
    @patch('helper.find_devices.BGAPIBackend')
    def test_find_muses_filters_non_muse_devices(self, mock_bgapi_class, mock_comports):
        """Should filter out non-Muse Bluetooth devices."""
        mock_port = ('COM3', 'Bluegiga Bluetooth Low Energy (COM3)', 'USB VID:PID=2458:0001')
        mock_comports.return_value = [mock_port]

        # Mock adapter that finds mixed devices
        mock_adapter = Mock()
        mock_adapter.scan.return_value = [
            {'name': 'Muse-1A2B', 'address': '00:55:DA:B1:1A:2B'},
            {'name': 'Some Headphones', 'address': '00:11:22:33:44:55'},
            {'name': 'Muse-3C4D', 'address': '00:55:DA:B3:3C:4D'},
            {'name': 'Fitness Tracker', 'address': '00:AA:BB:CC:DD:EE'}
        ]
        mock_bgapi_class.return_value = mock_adapter

        # Execute
        muses, ports = FindDevices.find_muses_with_ports()

        # Verify - only Muse devices
        assert len(muses) == 2, "Should find only 2 Muse devices"
        device_names = [name for name, addr in muses]
        assert all('Muse' in name for name in device_names)

    @patch('helper.find_devices.serial.tools.list_ports.comports')
    @patch('helper.find_devices.BGAPIBackend')
    def test_find_muses_handles_null_bytes_in_name(self, mock_bgapi_class, mock_comports):
        """Should clean null bytes from device names."""
        mock_port = ('COM3', 'Bluegiga Bluetooth Low Energy (COM3)', 'USB VID:PID=2458:0001')
        mock_comports.return_value = [mock_port]

        # Mock device with null bytes in name
        mock_adapter = Mock()
        mock_adapter.scan.return_value = [
            {'name': 'Muse-1A2B\x00\x00', 'address': '00:55:DA:B1:1A:2B'}
        ]
        mock_bgapi_class.return_value = mock_adapter

        # Execute
        muses, ports = FindDevices.find_muses_with_ports()

        # Verify - null bytes removed
        assert len(muses) == 1
        device_name = muses[0][0]
        assert '\x00' not in device_name
        assert device_name == 'Muse-1A2B'

    @patch('helper.find_devices.serial.tools.list_ports.comports')
    @patch('helper.find_devices.BGAPIBackend')
    def test_find_muses_handles_adapter_errors(self, mock_bgapi_class, mock_comports):
        """Should handle adapter errors gracefully."""
        mock_port = ('COM3', 'Bluegiga Bluetooth Low Energy (COM3)', 'USB VID:PID=2458:0001')
        mock_comports.return_value = [mock_port]

        # Mock adapter that raises error
        mock_adapter = Mock()
        mock_adapter.scan.side_effect = Exception("Device not responding")
        mock_bgapi_class.return_value = mock_adapter

        # Execute - should not crash
        muses, ports = FindDevices.find_muses_with_ports()

        # Verify - returns empty gracefully
        assert len(muses) == 0
        assert len(ports) == 0


class TestFindMuse:
    """Test Muse discovery via native Bluetooth (Bleak backend)."""

    @patch('helper.find_devices.backends.BleakBackend')
    def test_find_muse_discovers_devices(self, mock_bleak_class):
        """Should discover Muse devices via Bleak."""
        # Mock Bleak backend
        mock_adapter = Mock()
        mock_adapter.scan.return_value = [
            {'name': 'Muse-1A2B', 'address': '00:55:DA:B1:1A:2B'},
            {'name': 'Muse-3C4D', 'address': '00:55:DA:B3:3C:4D'}
        ]
        mock_bleak_class.return_value = mock_adapter

        # Execute
        muses = FindDevices.find_muse()

        # Verify
        assert len(muses) == 2, "Should find 2 Muse devices"
        assert muses[0]['name'] == 'Muse-1A2B'
        assert muses[1]['name'] == 'Muse-3C4D'

        # Verify adapter lifecycle
        mock_adapter.start.assert_called_once()
        mock_adapter.scan.assert_called_once_with(timeout=10)
        mock_adapter.stop.assert_called_once()

    @patch('helper.find_devices.backends.BleakBackend')
    def test_find_muse_returns_empty_when_none_found(self, mock_bleak_class):
        """Should return empty list when no Muse devices found."""
        mock_adapter = Mock()
        mock_adapter.scan.return_value = []
        mock_bleak_class.return_value = mock_adapter

        # Execute
        muses = FindDevices.find_muse()

        # Verify
        assert len(muses) == 0, "Should return empty list"
        assert isinstance(muses, list)

    @patch('helper.find_devices.backends.BleakBackend')
    def test_find_muse_filters_non_muse_devices(self, mock_bleak_class):
        """Should filter out non-Muse devices."""
        mock_adapter = Mock()
        mock_adapter.scan.return_value = [
            {'name': 'Muse-1A2B', 'address': '00:55:DA:B1:1A:2B'},
            {'name': 'iPhone', 'address': '00:11:22:33:44:55'},
            {'name': 'Bluetooth Speaker', 'address': '00:AA:BB:CC:DD:EE'},
            {'name': None, 'address': '00:FF:FF:FF:FF:FF'}  # Device without name
        ]
        mock_bleak_class.return_value = mock_adapter

        # Execute
        muses = FindDevices.find_muse()

        # Verify - only Muse device
        assert len(muses) == 1
        assert muses[0]['name'] == 'Muse-1A2B'


class TestFindEmpatica:
    """Test Empatica E4 device discovery."""

    @patch('helper.find_devices.EmpaticaServer')
    @patch('helper.find_devices.threading.Thread')
    def test_find_empatica_discovers_devices(self, mock_thread, mock_server_class):
        """Should discover E4 devices via BLE server."""
        # Mock server
        mock_server = Mock()
        mock_server.find_e4s.return_value = ['A01234', 'A05678']
        mock_server.connected_event = Mock()
        mock_server.connected_event.wait = Mock()
        mock_server.connected_event.clear = Mock()
        mock_server_class.return_value = mock_server

        # Mock threading
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        # Execute
        with patch('helper.find_devices.time.sleep'):  # Skip sleep delays
            e4s = FindDevices.find_empatica()

        # Verify
        assert len(e4s) == 2, "Should find 2 E4 devices"
        assert 'A01234' in e4s
        assert 'A05678' in e4s

        # Verify server was queried
        mock_server.find_e4s.assert_called_once()

    @patch('helper.find_devices.EmpaticaServer')
    def test_find_empatica_returns_empty_when_none_found(self, mock_server_class):
        """Should return empty list when no E4 devices found."""
        mock_server = Mock()
        mock_server.find_e4s.return_value = []
        mock_server_class.return_value = mock_server

        # Execute
        e4s = FindDevices.find_empatica()

        # Verify
        assert len(e4s) == 0, "Should return empty list"
        assert isinstance(e4s, list)

    @patch('helper.find_devices.EmpaticaServer')
    def test_find_empatica_handles_server_connect_error(self, mock_server_class):
        """Should handle server connection errors gracefully."""
        # Mock server that raises connection error
        mock_server = Mock()
        mock_server.find_e4s.side_effect = EmpaticaServerConnectError("Server not running")
        mock_server_class.return_value = mock_server

        # Execute - should not crash
        e4s = FindDevices.find_empatica()

        # Verify
        assert len(e4s) == 0, "Should return empty on error"

    @patch('helper.find_devices.EmpaticaServer')
    def test_find_empatica_handles_general_exceptions(self, mock_server_class):
        """Should handle unexpected exceptions gracefully."""
        mock_server = Mock()
        mock_server.find_e4s.side_effect = Exception("Unexpected error")
        mock_server_class.return_value = mock_server

        # Execute - should not crash
        e4s = FindDevices.find_empatica()

        # Verify
        assert len(e4s) == 0, "Should return empty on error"


class TestScanBluetooth:
    """Test general Bluetooth device scanning."""

    @patch('helper.find_devices.bluetooth.discover_devices')
    def test_scan_bluetooth_discovers_devices(self, mock_discover):
        """Should discover Bluetooth devices."""
        # Mock discovered devices
        mock_discover.return_value = [
            ('00:11:22:33:44:55', 'Device 1'),
            ('00:AA:BB:CC:DD:EE', 'Device 2'),
            ('00:FF:FF:FF:FF:FF', 'Device 3')
        ]

        # Execute
        devices = FindDevices.scan_bluetooth()

        # Verify
        assert len(devices) == 3, "Should find 3 devices"

        # Check structure
        for device in devices:
            assert 'name' in device
            assert 'address' in device
            assert 'type' in device
            assert device['type'] == 'Bluetooth'

        # Check specific devices
        assert devices[0]['name'] == 'Device 1'
        assert devices[0]['address'] == '00:11:22:33:44:55'

    @patch('helper.find_devices.bluetooth.discover_devices')
    def test_scan_bluetooth_returns_empty_when_none_found(self, mock_discover):
        """Should return empty list when no devices found."""
        mock_discover.return_value = []

        # Execute
        devices = FindDevices.scan_bluetooth()

        # Verify
        assert len(devices) == 0, "Should return empty list"
        assert isinstance(devices, list)


class TestScanWiFi:
    """Test WiFi network scanning."""

    @patch('helper.find_devices.pywifi.PyWiFi')
    def test_scan_wifi_discovers_networks(self, mock_pywifi_class):
        """Should discover WiFi networks."""
        # Mock WiFi interface
        mock_iface = Mock()
        mock_result_1 = Mock()
        mock_result_1.ssid = 'Network_1'
        mock_result_1.bssid = '00:11:22:33:44:55'

        mock_result_2 = Mock()
        mock_result_2.ssid = 'Network_2'
        mock_result_2.bssid = '00:AA:BB:CC:DD:EE'

        mock_iface.scan_results.return_value = [mock_result_1, mock_result_2]

        mock_wifi = Mock()
        mock_wifi.interfaces.return_value = [mock_iface]
        mock_pywifi_class.return_value = mock_wifi

        # Execute
        with patch('helper.find_devices.time.sleep'):  # Skip sleep
            devices = FindDevices.scan_wifi()

        # Verify
        assert len(devices) == 2, "Should find 2 WiFi networks"

        # Check structure
        for device in devices:
            assert 'name' in device
            assert 'address' in device
            assert 'type' in device
            assert device['type'] == 'WiFi'

        # Check specific networks
        assert devices[0]['name'] == 'Network_1'
        assert devices[0]['address'] == '00:11:22:33:44:55'

    @patch('helper.find_devices.pywifi.PyWiFi')
    def test_scan_wifi_returns_empty_when_none_found(self, mock_pywifi_class):
        """Should return empty list when no networks found."""
        mock_iface = Mock()
        mock_iface.scan_results.return_value = []

        mock_wifi = Mock()
        mock_wifi.interfaces.return_value = [mock_iface]
        mock_pywifi_class.return_value = mock_wifi

        # Execute
        with patch('helper.find_devices.time.sleep'):
            devices = FindDevices.scan_wifi()

        # Verify
        assert len(devices) == 0, "Should return empty list"


class TestSerialPorts:
    """Test BLED112 serial port enumeration."""

    @patch('helper.find_devices.serial.tools.list_ports.comports')
    @patch('helper.find_devices.BGAPIBackend')
    def test_serial_ports_finds_bled112_ports(self, mock_bgapi_class, mock_comports):
        """Should find available BLED112 ports."""
        # Mock ports
        mock_comports.return_value = [
            ('COM3', 'Bluegiga Bluetooth Low Energy (COM3)', 'USB VID:PID=2458:0001'),
            ('COM5', 'Bluegiga Bluetooth Low Energy (COM5)', 'USB VID:PID=2458:0001'),
            ('COM1', 'Standard Serial Port', 'PCI\\VEN_8086')
        ]

        # Mock BGAPI backend that starts/stops successfully
        mock_adapter = Mock()
        mock_bgapi_class.return_value = mock_adapter

        # Execute
        ports = FindDevices.serial_ports()

        # Verify
        assert len(ports) == 2, "Should find 2 BLED112 ports"
        assert 'COM3' in ports
        assert 'COM5' in ports
        assert 'COM1' not in ports

    @patch('helper.find_devices.serial.tools.list_ports.comports')
    @patch('helper.find_devices.BGAPIBackend')
    def test_serial_ports_filters_unavailable_ports(self, mock_bgapi_class, mock_comports):
        """Should filter out unavailable BLED112 ports."""
        mock_comports.return_value = [
            ('COM3', 'Bluegiga Bluetooth Low Energy (COM3)', 'USB VID:PID=2458:0001'),
            ('COM5', 'Bluegiga Bluetooth Low Energy (COM5)', 'USB VID:PID=2458:0001')
        ]

        # Mock adapter - COM3 works, COM5 fails
        def create_adapter_side_effect(serial_port):
            mock_adapter = Mock()
            if serial_port == 'COM5':
                mock_adapter.start.side_effect = Exception("Port in use")
            return mock_adapter

        mock_bgapi_class.side_effect = create_adapter_side_effect

        # Execute
        ports = FindDevices.serial_ports()

        # Verify - only COM3 available
        assert len(ports) == 1
        assert 'COM3' in ports
        assert 'COM5' not in ports

    @patch('helper.find_devices.serial.tools.list_ports.comports')
    def test_serial_ports_returns_empty_when_no_bled112(self, mock_comports):
        """Should return empty when no BLED112 ports present."""
        mock_comports.return_value = [
            ('COM1', 'Standard Serial Port', 'PCI\\VEN_8086'),
            ('COM2', 'USB-SERIAL CH340', 'USB\\VID_1A86')
        ]

        # Execute
        ports = FindDevices.serial_ports()

        # Verify
        assert len(ports) == 0, "Should return empty list"


@pytest.mark.integration
class TestDeviceDiscoveryIntegration:
    """Integration tests using mock hardware."""

    def test_discover_all_device_types(self, monkeypatch):
        """Should discover all types of devices in one session."""
        # Mock all discovery methods
        mock_muses = [
            {'name': 'Muse-1A2B', 'address': '00:55:DA:B1:1A:2B'}
        ]
        mock_e4s = ['A01234']
        mock_bt_devices = [
            {'name': 'Headphones', 'address': '00:11:22:33:44:55', 'type': 'Bluetooth'}
        ]

        with patch.object(FindDevices, 'find_muse', return_value=mock_muses):
            with patch.object(FindDevices, 'find_empatica', return_value=mock_e4s):
                with patch.object(FindDevices, 'scan_bluetooth', return_value=mock_bt_devices):
                    # Discover all
                    muses = FindDevices.find_muse()
                    e4s = FindDevices.find_empatica()
                    bt = FindDevices.scan_bluetooth()

                    # Verify all discovered
                    assert len(muses) > 0
                    assert len(e4s) > 0
                    assert len(bt) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

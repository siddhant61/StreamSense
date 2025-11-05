"""
Tests for the hardware mocking infrastructure.

Validates that all mock components work correctly and demonstrates
how to use them in tests for hardware-dependent features.
"""

import pytest
import time
from multiprocessing import Queue

from tests.mocks import (
    MockMuse,
    MockE4,
    MockBGAPIBackend,
    MockEmpaticaServer,
    create_mock_lsl_outlet,
    generate_eeg_sample,
    generate_ppg_sample,
    generate_bvp_sample,
    generate_gsr_sample
)


class TestDataGenerators:
    """Test synthetic physiological data generators."""

    def test_eeg_generator_shape(self):
        """EEG generator should produce correct shape."""
        data, timestamps = generate_eeg_sample(num_channels=5, num_samples=12)

        assert data.shape == (5, 12), "EEG data should be 5 channels x 12 samples"
        assert len(timestamps) == 12, "Should have 12 timestamps"

    def test_eeg_generator_values(self):
        """EEG data should be in physiologically plausible range."""
        data, timestamps = generate_eeg_sample(num_channels=5, num_samples=100)

        # EEG values should be roughly in range -200 to +200 microvolts
        assert data.min() > -300, "EEG values too low"
        assert data.max() < 300, "EEG values too high"

    def test_ppg_generator_shape(self):
        """PPG generator should produce correct shape."""
        data, timestamps = generate_ppg_sample(num_channels=3, num_samples=6)

        assert data.shape == (3, 6), "PPG data should be 3 channels x 6 samples"
        assert len(timestamps) == 6, "Should have 6 timestamps"

    def test_bvp_generator(self):
        """BVP generator should produce single-channel data."""
        data, timestamps = generate_bvp_sample(num_samples=10)

        assert data.shape == (1, 10), "BVP should be single channel"
        assert len(timestamps) == 10, "Should have 10 timestamps"

    def test_gsr_generator_range(self):
        """GSR values should be in physiological range."""
        data, timestamps = generate_gsr_sample(num_samples=10)

        # GSR typically 0-20 microsiemens
        assert data.min() >= 0, "GSR should be non-negative"
        assert data.max() < 50, "GSR values too high"


class TestMockLSL:
    """Test LSL mocking components."""

    def test_outlet_creation(self, mock_lsl_eeg_outlet):
        """Mock LSL outlet should be created with correct config."""
        assert mock_lsl_eeg_outlet.stream_info.name == "TestEEG"
        assert mock_lsl_eeg_outlet.stream_info.type == "EEG"
        assert mock_lsl_eeg_outlet.stream_info.channel_count == 5
        assert mock_lsl_eeg_outlet.stream_info.nominal_srate == 256.0

    def test_push_sample(self, mock_lsl_eeg_outlet):
        """Should be able to push samples to outlet."""
        sample = [1.0, 2.0, 3.0, 4.0, 5.0]
        mock_lsl_eeg_outlet.push_sample(sample)

        assert mock_lsl_eeg_outlet.get_sample_count() == 1
        captured = mock_lsl_eeg_outlet.get_samples()
        assert captured[0] == sample

    def test_push_sample_with_timestamp(self, mock_lsl_eeg_outlet):
        """Should capture timestamp with sample."""
        sample = [1.0, 2.0, 3.0, 4.0, 5.0]
        timestamp = 123.456
        mock_lsl_eeg_outlet.push_sample(sample, timestamp=timestamp)

        timestamps = mock_lsl_eeg_outlet.get_timestamps()
        assert timestamps[0] == timestamp

    def test_push_chunk(self, mock_lsl_eeg_outlet):
        """Should be able to push multiple samples at once."""
        chunk = [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0]
        ]
        mock_lsl_eeg_outlet.push_chunk(chunk)

        assert mock_lsl_eeg_outlet.get_sample_count() == 3

    def test_sample_validation(self, mock_lsl_eeg_outlet):
        """Should reject samples with wrong channel count."""
        invalid_sample = [1.0, 2.0, 3.0]  # Only 3 channels, need 5

        with pytest.raises(ValueError, match="does not match channel count"):
            mock_lsl_eeg_outlet.push_sample(invalid_sample)

    def test_outlet_clear(self, mock_lsl_eeg_outlet):
        """Should be able to clear captured samples."""
        mock_lsl_eeg_outlet.push_sample([1.0, 2.0, 3.0, 4.0, 5.0])
        mock_lsl_eeg_outlet.push_sample([6.0, 7.0, 8.0, 9.0, 10.0])

        assert mock_lsl_eeg_outlet.get_sample_count() == 2

        mock_lsl_eeg_outlet.clear()

        assert mock_lsl_eeg_outlet.get_sample_count() == 0


class TestMockMuse:
    """Test Muse device mocking."""

    def test_muse_creation(self, mock_muse_device):
        """Mock Muse should be created with correct configuration."""
        assert mock_muse_device.name == "TestMuse"
        assert mock_muse_device.address == "00:55:DA:B0:00:01"
        assert not mock_muse_device.connected

    def test_muse_connection(self, mock_muse_device):
        """Should be able to connect to mock Muse."""
        success = mock_muse_device.connect()

        assert success
        assert mock_muse_device.connected

    def test_muse_disconnection(self, mock_muse_device):
        """Should be able to disconnect from mock Muse."""
        mock_muse_device.connect()
        mock_muse_device.disconnect()

        assert not mock_muse_device.connected

    def test_muse_streaming(self, mock_muse_device):
        """Should generate realistic EEG data when streaming."""
        mock_muse_device.connect()
        mock_muse_device.start()

        # Wait for some data
        time.sleep(0.2)

        # Check for EEG data
        try:
            eeg_data, eeg_timestamps = mock_muse_device.shared_eeg.get(timeout=1.0)
            assert eeg_data.shape[0] == 5, "Should have 5 EEG channels"
            assert eeg_data.shape[1] > 0, "Should have samples"
        except:
            pytest.fail("No EEG data generated")

        mock_muse_device.stop()
        mock_muse_device.disconnect()

    def test_bgapi_backend_scan(self):
        """BGAPI backend mock should discover devices."""
        backend = MockBGAPIBackend(serial_port="COM3")
        backend.start()

        devices = backend.scan(timeout=3)

        assert len(devices) > 0, "Should discover at least one device"
        assert all('name' in d and 'address' in d for d in devices)

        backend.stop()


class TestMockE4:
    """Test Empatica E4 device mocking."""

    def test_e4_creation(self, mock_e4_device):
        """Mock E4 should be created with correct configuration."""
        assert mock_e4_device.device_id == "A01234"
        assert not mock_e4_device.device.connected

    def test_e4_connection(self, mock_e4_device):
        """Should be able to connect to mock E4."""
        success = mock_e4_device.connect()

        assert success
        assert mock_e4_device.device.connected

    def test_e4_disconnection(self, mock_e4_device):
        """Should be able to disconnect from mock E4."""
        mock_e4_device.connect()
        mock_e4_device.disconnect()

        assert not mock_e4_device.device.connected

    def test_e4_data_streaming(self, mock_e4_device):
        """Should generate realistic E4 data when streaming."""
        mock_e4_device.connect()
        mock_e4_device.subscribe_all()
        mock_e4_device.start_streaming()

        # Wait for some data
        time.sleep(0.2)

        # Get data from queue
        try:
            data = mock_e4_device.get_data_queue().get(timeout=1.0)
            assert data is not None, "Should receive data"
            assert isinstance(data, str), "Data should be string format"

            # Check for expected stream types
            assert any(stream_type in data for stream_type in
                      ['E4_Acc', 'E4_Bvp', 'E4_Gsr', 'E4_Temperature'])

        except:
            pytest.fail("No E4 data generated")

        mock_e4_device.stop_streaming()
        mock_e4_device.disconnect()

    def test_empatica_server_discovery(self, mock_empatica_server):
        """Empatica server should discover E4 devices."""
        devices = mock_empatica_server.find_e4s()

        assert len(devices) > 0, "Should discover at least one E4"
        assert all(isinstance(d, str) for d in devices)

    def test_empatica_server_commands(self, mock_empatica_server):
        """Server should respond to commands correctly."""
        import socket
        sock = socket.socket()  # Dummy socket

        # Test device list command
        response = mock_empatica_server.send_command(sock, "device_discover_list")
        assert "R device_discover_list" in response

        # Test connection command
        response = mock_empatica_server.send_command(sock, "device_connect_btle A01234")
        assert "OK" in response


@pytest.mark.integration
class TestHardwareMockIntegration:
    """Integration tests using multiple mock components together."""

    def test_muse_with_lsl_outlet(self, mock_muse_device, mock_lsl_eeg_outlet):
        """Test Muse data flow through LSL outlet."""
        # Connect and start Muse
        mock_muse_device.connect()
        mock_muse_device.start()

        # Wait for data
        time.sleep(0.2)

        # Get EEG data from Muse
        try:
            eeg_data, eeg_timestamps = mock_muse_device.shared_eeg.get(timeout=1.0)

            # Push to LSL outlet
            for i in range(eeg_data.shape[1]):
                sample = eeg_data[:, i].tolist()
                mock_lsl_eeg_outlet.push_sample(sample, timestamp=eeg_timestamps[i])

            # Verify data captured by outlet
            assert mock_lsl_eeg_outlet.get_sample_count() > 0

        except:
            pytest.fail("Integration test failed")

        finally:
            mock_muse_device.stop()
            mock_muse_device.disconnect()

    def test_e4_with_lsl_outlet(self, mock_e4_device, mock_lsl_e4_bvp_outlet):
        """Test E4 data flow through LSL outlet."""
        # Connect and start E4
        mock_e4_device.connect()
        mock_e4_device.subscribe_all()
        mock_e4_device.start_streaming()

        # Wait for data
        time.sleep(0.3)

        # Get BVP data from E4
        try:
            data_str = mock_e4_device.get_data_queue().get(timeout=1.0)

            # Parse BVP samples
            lines = data_str.strip().split('\n')
            for line in lines:
                if line.startswith('E4_Bvp'):
                    parts = line.split()
                    timestamp = float(parts[1])
                    value = float(parts[2])
                    mock_lsl_e4_bvp_outlet.push_sample([value], timestamp=timestamp)

            # Verify data captured
            assert mock_lsl_e4_bvp_outlet.get_sample_count() > 0

        except:
            pytest.fail("E4 integration test failed")

        finally:
            mock_e4_device.stop_streaming()
            mock_e4_device.disconnect()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

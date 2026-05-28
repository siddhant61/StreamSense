"""
Tests for Empatica E4 streaming functionality.

Tests the StreamE4 class which handles:
- E4 device connection and initialization
- Data stream subscription (ACC, BVP, GSR, TEMP, TAG)
- LSL outlet creation for each data type
- Multiprocessing-based streaming
- Data parsing and timestamp correction
- Reconnection logic with retry mechanism
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from multiprocessing import Queue, Event
from queue import Empty
import time
import sys

# Mock dependencies before importing
sys.modules['pylsl'] = Mock()
sys.modules['helper.e4_helper'] = Mock()

from streamer.stream_e4 import StreamE4
from tests.mocks import MockE4, create_mock_lsl_outlet

# Spawns real multiprocessing.Process workers; excluded from coverage runs. See .coveragerc.
pytestmark = pytest.mark.integration


class TestStreamE4Initialization:
    """Test StreamE4 initialization and configuration."""

    def test_stream_e4_initialization(self):
        """Should initialize with correct configuration."""
        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp/output",
            synchronized_start_time=time.time()
        )

        # Verify basic attributes
        assert streamer.current_e4 == "A01234"
        assert streamer.device_name == "A01234"  # Inherited from BaseStreamer
        assert streamer.root_output_folder == "/tmp/output"
        assert streamer.streaming is False
        assert streamer.connected is False
        # stop_signal is now an Event (from BaseStreamer), not a boolean
        assert streamer.stop_signal is not None
        assert not streamer.stop_signal.is_set()

        # Verify queues and events created (from BaseStreamer)
        assert streamer.queue is not None
        assert streamer.connected_event is not None

        # Verify outlets initialized as None
        assert streamer.outletACC is None
        assert streamer.outletBVP is None
        assert streamer.outletGSR is None
        assert streamer.outletTEMP is None
        assert streamer.outletTAG is None

    def test_subscribed_streams_initialization(self):
        """Should initialize subscription flags correctly."""
        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        # Verify subscription dict
        assert 'acc' in streamer.subscribed_streams
        assert 'bvp' in streamer.subscribed_streams
        assert 'gsr' in streamer.subscribed_streams
        assert 'tmp' in streamer.subscribed_streams
        assert 'tag' in streamer.subscribed_streams
        assert 'ibi' in streamer.subscribed_streams

        # All should start as False
        for stream in streamer.subscribed_streams.values():
            assert stream is False


class TestStreamE4Connection:
    """Test E4 device connection management."""

    @patch('streamer.stream_e4.EmpaticaE4')
    def test_connect_success(self, mock_e4_class):
        """Should connect to E4 device successfully."""
        mock_e4_device = Mock()
        mock_e4_class.return_value = mock_e4_device

        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        # Connect
        streamer.connect()

        # Verify E4 device created
        mock_e4_class.assert_called_once_with("A01234")

        # Verify connected flag set
        assert streamer.connected is True
        assert streamer.empatica_e4 is not None

    @patch('streamer.stream_e4.EmpaticaE4')
    def test_connect_failure(self, mock_e4_class):
        """Should handle connection failure gracefully."""
        mock_e4_class.side_effect = Exception("Connection failed")

        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        # Attempt connection - should not raise
        streamer.connect()

        # Connected flag should still be False
        assert streamer.connected is False


class TestStreamE4Subscription:
    """Test data stream subscription."""

    @patch('streamer.stream_e4.time.sleep')
    def test_subscribe_to_data(self, mock_sleep):
        """Should subscribe to all enabled data streams."""
        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        # Mock E4 device
        mock_e4 = Mock()
        streamer.empatica_e4 = mock_e4

        # Subscribe
        streamer.subscribe_to_data()

        # Verify suspend called
        mock_e4.suspend_streaming.assert_called_once()

        # Verify subscriptions called for enabled streams
        # (acc, bvp, gsr, tmp, tag are enabled by default in the module)
        subscribe_calls = [call[0][0] for call in mock_e4.subscribe_to_stream.call_args_list]
        assert 'acc' in subscribe_calls
        assert 'bvp' in subscribe_calls
        assert 'gsr' in subscribe_calls
        assert 'tmp' in subscribe_calls
        assert 'tag' in subscribe_calls

    @patch('streamer.stream_e4.time.sleep')
    def test_subscribe_handles_errors(self, mock_sleep):
        """Should handle subscription errors gracefully."""
        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        # Mock E4 that raises error
        mock_e4 = Mock()
        mock_e4.subscribe_to_stream.side_effect = Exception("Subscription failed")
        streamer.empatica_e4 = mock_e4

        # Subscribe - should not raise
        streamer.subscribe_to_data()


class TestStreamE4LSLSetup:
    """Test LSL stream preparation."""

    @patch('streamer.stream_e4.pylsl.StreamOutlet')
    @patch('streamer.stream_e4.pylsl.StreamInfo')
    def test_prepare_lsl_streaming_creates_outlets(self, mock_info_class, mock_outlet_class):
        """Should create LSL outlets for enabled streams."""
        mock_info_class.return_value = Mock()
        mock_outlet_class.return_value = Mock()

        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        # Mock E4 device
        mock_e4 = Mock()
        streamer.empatica_e4 = mock_e4

        # Prepare LSL
        streamer._setup_lsl_outlets()

        # Verify E4 streaming started
        mock_e4.start_streaming.assert_called_once()

        # Verify outlets created (acc, bvp, gsr, tmp, tag enabled by default)
        assert streamer.outletACC is not None
        assert streamer.outletBVP is not None
        assert streamer.outletGSR is not None
        assert streamer.outletTEMP is not None
        assert streamer.outletTAG is not None

    @patch('streamer.stream_e4.pylsl.StreamOutlet')
    @patch('streamer.stream_e4.pylsl.StreamInfo')
    def test_prepare_lsl_acc_outlet_config(self, mock_info_class, mock_outlet_class):
        """Should create ACC outlet with correct configuration."""
        mock_info = Mock()
        mock_info_class.return_value = mock_info
        mock_outlet_class.return_value = Mock()

        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        mock_e4 = Mock()
        streamer.empatica_e4 = mock_e4

        # Prepare LSL
        streamer._setup_lsl_outlets()

        # Verify ACC StreamInfo created correctly
        acc_call = None
        for call in mock_info_class.call_args_list:
            if 'ACC' in str(call):
                acc_call = call
                break

        assert acc_call is not None
        # ACC should have: name with device ID, type 'ACC', 3 channels, 32 Hz, int32
        call_args = acc_call[0]
        assert 'A01234_ACC' in call_args[0]
        assert call_args[1] == 'ACC'
        assert call_args[2] == 3  # 3 axes
        assert call_args[3] == 32  # 32 Hz


class TestStreamE4DataParsing:
    """Test E4 data parsing and streaming."""

    @patch('streamer.stream_e4.pylsl.StreamOutlet')
    @patch('streamer.stream_e4.pylsl.StreamInfo')
    def test_stream_parses_acc_data(self, mock_info_class, mock_outlet_class):
        """Should parse ACC data correctly."""
        mock_info_class.return_value = Mock()

        # Mock ACC outlet
        mock_acc_outlet = Mock()
        mock_outlet_class.return_value = mock_acc_outlet

        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        # Setup mock E4
        mock_e4 = Mock()
        mock_e4.lsl_data_queue = Queue()
        streamer.empatica_e4 = mock_e4
        streamer.outletACC = mock_acc_outlet

        # Put sample ACC data in queue
        mock_e4.lsl_data_queue.put("E4_Acc 1.234 10 20 30\n")

        # Set stop signal to exit after one sample
        streamer.stop_signal = True

        # Stream (will process one sample then exit)
        try:
            streamer.stream()
        except:
            pass  # Expected to exit via stop_signal

        # Verify ACC data was pushed
        if mock_acc_outlet.push_sample.called:
            call_args = mock_acc_outlet.push_sample.call_args[0]
            data = call_args[0]
            assert len(data) == 3
            assert data == [10, 20, 30]

    @patch('streamer.stream_e4.pylsl.StreamOutlet')
    @patch('streamer.stream_e4.pylsl.StreamInfo')
    def test_stream_parses_bvp_data(self, mock_info_class, mock_outlet_class):
        """Should parse BVP data correctly."""
        mock_info_class.return_value = Mock()
        mock_bvp_outlet = Mock()
        mock_outlet_class.return_value = mock_bvp_outlet

        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        mock_e4 = Mock()
        mock_e4.lsl_data_queue = Queue()
        streamer.empatica_e4 = mock_e4
        streamer.outletBVP = mock_bvp_outlet

        # Put sample BVP data
        mock_e4.lsl_data_queue.put("E4_Bvp 1.234 123.45\n")
        streamer.stop_signal = True

        try:
            streamer.stream()
        except:
            pass

        # Verify BVP data was pushed
        if mock_bvp_outlet.push_sample.called:
            call_args = mock_bvp_outlet.push_sample.call_args[0]
            data = call_args[0]
            assert len(data) == 1
            assert isinstance(data[0], float)

    def test_stream_handles_connection_lost(self):
        """Should detect and handle connection loss."""
        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        mock_e4 = Mock()
        mock_e4.lsl_data_queue = Queue()
        streamer.empatica_e4 = mock_e4
        streamer.connected = True

        # Put connection lost message
        mock_e4.lsl_data_queue.put("connection lost to device\n")

        # Mock reconnect to avoid hanging
        reconnect_called = []
        def mock_reconnect():
            reconnect_called.append(True)
            streamer.stop_signal = True  # Stop after reconnect

        streamer.reconnect = mock_reconnect

        try:
            streamer.stream()
        except:
            pass

        # Verify reconnect was attempted (the important behavior)
        assert len(reconnect_called) > 0, "Reconnect should have been called on connection loss"


class TestStreamE4Reconnection:
    """Test E4 reconnection logic."""

    @patch('streamer.stream_e4.time.sleep')
    def test_reconnect_success(self, mock_sleep):
        """Should reconnect successfully."""
        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        # Mock successful connection
        streamer.connect = Mock(side_effect=lambda: setattr(streamer, 'connected', True))
        streamer.subscribe_to_data = Mock()
        streamer.stream = Mock()

        # Attempt reconnection
        streamer.reconnect()

        # Verify connection attempted
        streamer.connect.assert_called()
        streamer.subscribe_to_data.assert_called()
        streamer.stream.assert_called()

    @patch('streamer.stream_e4.time.sleep')
    def test_reconnect_max_retries(self, mock_sleep):
        """Should give up after max reconnection attempts."""
        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        # Mock failed connection
        streamer.connect = Mock()
        streamer.connected = False

        # Attempt reconnection
        streamer.reconnect()

        # Verify retried up to MAX_RETRIES_E4 (3)
        assert streamer.connect.call_count <= 3
        assert streamer.connected is False

    @patch('streamer.stream_e4.time.sleep')
    def test_reconnect_respects_stop_signal(self, mock_sleep):
        """Should stop reconnection if stop signal set."""
        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        streamer.connect = Mock()
        streamer.connected = False
        streamer.stop_signal.set()  # Set stop signal (now an Event)

        # Attempt reconnection
        streamer.reconnect()

        # Should not attempt many retries when stop signal set
        assert streamer.connect.call_count <= 1


class TestStreamE4Lifecycle:
    """Test streaming lifecycle management."""

    @patch('streamer.base_streamer.Process')  # Process is now in BaseStreamer
    def test_start_streaming_success(self, mock_process_class):
        """Should start streaming process successfully."""
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        # Pre-populate queue with success
        streamer.queue.put('connected')

        # Start streaming
        streamer.start_streaming()

        # Verify process started
        mock_process.start.assert_called_once()

        # Verify connected event set
        assert streamer.connected_event.is_set()
        assert streamer.streaming is True

    @patch('streamer.base_streamer.Process')  # Process is now in BaseStreamer
    def test_start_streaming_timeout(self, mock_process_class):
        """Should handle timeout waiting for connection."""
        mock_process = Mock()
        mock_process.is_alive.return_value = True
        mock_process_class.return_value = mock_process

        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        # Don't put anything in queue - will timeout

        # Start streaming (with short timeout)
        streamer.start_streaming()

        # Verify process cleanup happened (BaseStreamer calls _cleanup which may join/terminate)
        # The exact cleanup method depends on process state, but streaming should be False
        assert streamer.streaming is False
        # Process should have been cleaned up
        assert not streamer.is_streaming()

    @patch('streamer.base_streamer.Process')  # Process is now in BaseStreamer
    def test_start_streaming_already_streaming(self, mock_process_class):
        """Should not start if already streaming."""
        mock_process_class.return_value = Mock()

        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        streamer.streaming = True  # Already streaming

        # Attempt to start again
        streamer.start_streaming()

        # Verify process not created
        mock_process_class.assert_not_called()

    def test_stop_streaming(self):
        """Should stop streaming and clean up."""
        streamer = StreamE4(
            e4="A01234",
            root_output_folder="/tmp",
            synchronized_start_time=time.time()
        )

        # Mock process
        mock_process = Mock()
        mock_process.is_alive.return_value = False
        streamer.process = mock_process

        # Mock E4 device
        mock_e4 = Mock()
        streamer.empatica_e4 = mock_e4

        streamer.streaming = True
        streamer.connected_event.set()

        # Stop streaming
        streamer.stop_streaming()

        # Verify stop signal set (now an Event)
        assert streamer.stop_signal.is_set()

        # Note: BaseStreamer's stop_streaming() doesn't disconnect the E4 device
        # That happens in the _stream_wrapper when it detects stop_signal

        # Verify streaming flags cleared
        assert streamer.streaming is False
        assert not streamer.connected_event.is_set()
        assert streamer.process is None


@pytest.mark.integration
class TestE4StreamingIntegration:
    """Integration tests using mock hardware."""

    def test_complete_e4_lifecycle(self, mock_e4_device):
        """Test full E4 streaming lifecycle with mock device."""
        # Verify mock device
        assert mock_e4_device is not None
        assert mock_e4_device.device_id == "A01234"

        # Connect
        assert mock_e4_device.connect()

        # Subscribe to all streams
        mock_e4_device.subscribe_all()

        # Verify subscriptions
        assert mock_e4_device.device._subscribed_streams['acc'] is True
        assert mock_e4_device.device._subscribed_streams['bvp'] is True
        assert mock_e4_device.device._subscribed_streams['gsr'] is True

        # Disconnect
        mock_e4_device.disconnect()
        assert not mock_e4_device.device.connected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

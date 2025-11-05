"""
Tests for the base streamer interface.

Tests the BaseStreamer abstract base class which provides common functionality
for all hardware streamers using multiprocessing.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from multiprocessing import Queue, Event

# Import the base streamer
from streamer.base_streamer import BaseStreamer, StreamerError, ConnectionError, StreamingError


class ConcreteStreamer(BaseStreamer):
    """Concrete implementation of BaseStreamer for testing."""

    def __init__(self, *args, should_fail=False, connection_delay=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.should_fail = should_fail
        self.connection_delay = connection_delay
        self.stream_called = False
        self.setup_lsl_called = False
        self.disconnect_called = False

    def _stream_wrapper(self):
        """Implementation of streaming logic."""
        try:
            self.stream_called = True

            # Simulate connection delay
            if self.connection_delay > 0:
                time.sleep(self.connection_delay)

            # Simulate connection failure
            if self.should_fail:
                return  # Don't send 'connected'

            # Set up outlets
            self._setup_lsl_outlets()

            # Signal successful connection
            self.queue.put('connected')

            # Simulate streaming loop
            while not self.stop_signal.is_set():
                time.sleep(0.1)  # Simulate work

        except Exception as e:
            print(f"Error in _stream_wrapper: {e}")
        finally:
            self.disconnect_called = True

    def _setup_lsl_outlets(self):
        """Implementation of LSL setup."""
        self.setup_lsl_called = True


class TestBaseStreamerInitialization:
    """Test BaseStreamer initialization."""

    def test_initialization_with_required_params(self):
        """Should initialize with required parameters."""
        streamer = ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=1234567890.0,
            root_output_folder="/tmp/test"
        )

        assert streamer.device_name == "TestDevice"
        assert streamer.synchronized_start_time == 1234567890.0
        assert streamer.root_output_folder == "/tmp/test"
        assert streamer.process is None
        assert streamer.streaming is False
        assert streamer.connected is False
        # Note: Events and Queues are multiprocessing objects, just verify they exist
        assert streamer.stop_signal is not None
        assert streamer.connected_event is not None
        assert streamer.queue is not None

    def test_initialization_with_kwargs(self):
        """Should accept additional keyword arguments."""
        streamer = ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=1234567890.0,
            root_output_folder="/tmp/test",
            custom_param="custom_value"
        )

        assert streamer.device_name == "TestDevice"


class TestBaseStreamerLifecycle:
    """Test BaseStreamer lifecycle management."""

    def test_start_streaming_success(self):
        """Should start streaming successfully."""
        streamer = ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=time.time(),
            root_output_folder="/tmp/test"
        )

        result = streamer.start_streaming(timeout=5)

        assert result is True
        assert streamer.streaming is True
        assert streamer.connected is True
        assert streamer.connected_event.is_set()
        assert streamer.process is not None
        assert streamer.process.is_alive()

        # Cleanup
        streamer.stop_streaming()

    def test_start_streaming_timeout(self):
        """Should timeout if connection takes too long."""
        streamer = ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=time.time(),
            root_output_folder="/tmp/test",
            connection_delay=10  # Longer than timeout
        )

        result = streamer.start_streaming(timeout=1)

        assert result is False
        assert streamer.streaming is False
        assert streamer.connected is False

        # Cleanup
        if streamer.process and streamer.process.is_alive():
            streamer.process.terminate()
            streamer.process.join()

    def test_start_streaming_connection_failure(self):
        """Should handle connection failure."""
        streamer = ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=time.time(),
            root_output_folder="/tmp/test",
            should_fail=True
        )

        result = streamer.start_streaming(timeout=2)

        assert result is False
        assert streamer.streaming is False

    def test_start_streaming_already_streaming(self):
        """Should not start if already streaming."""
        streamer = ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=time.time(),
            root_output_folder="/tmp/test"
        )

        # Start once
        result1 = streamer.start_streaming(timeout=5)
        assert result1 is True

        # Try to start again
        result2 = streamer.start_streaming(timeout=5)
        assert result2 is False  # Should return False

        # Cleanup
        streamer.stop_streaming()

    def test_stop_streaming_graceful(self):
        """Should stop streaming gracefully."""
        streamer = ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=time.time(),
            root_output_folder="/tmp/test"
        )

        streamer.start_streaming(timeout=5)
        assert streamer.is_streaming() is True

        streamer.stop_streaming(timeout=5)

        assert streamer.streaming is False
        assert streamer.connected is False
        assert streamer.process is None or not streamer.process.is_alive()

    def test_stop_streaming_when_not_streaming(self):
        """Should handle stop when not streaming."""
        streamer = ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=time.time(),
            root_output_folder="/tmp/test"
        )

        # Should not raise
        streamer.stop_streaming()

        assert streamer.streaming is False


class TestBaseStreamerState:
    """Test BaseStreamer state tracking."""

    def test_is_streaming_true(self):
        """Should return True when streaming."""
        streamer = ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=time.time(),
            root_output_folder="/tmp/test"
        )

        streamer.start_streaming(timeout=5)
        assert streamer.is_streaming() is True

        streamer.stop_streaming()

    def test_is_streaming_false(self):
        """Should return False when not streaming."""
        streamer = ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=time.time(),
            root_output_folder="/tmp/test"
        )

        assert streamer.is_streaming() is False

    def test_is_connected_true(self):
        """Should return True when connected."""
        streamer = ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=time.time(),
            root_output_folder="/tmp/test"
        )

        streamer.start_streaming(timeout=5)
        assert streamer.is_connected() is True

        streamer.stop_streaming()

    def test_is_connected_false(self):
        """Should return False when not connected."""
        streamer = ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=time.time(),
            root_output_folder="/tmp/test"
        )

        assert streamer.is_connected() is False


class TestBaseStreamerContextManager:
    """Test BaseStreamer context manager protocol."""

    def test_context_manager_normal_exit(self):
        """Should cleanup on normal context exit."""
        with ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=time.time(),
            root_output_folder="/tmp/test"
        ) as streamer:
            streamer.start_streaming(timeout=5)
            assert streamer.is_streaming() is True

        # After context exit, should be cleaned up
        assert streamer.streaming is False
        assert streamer.process is None or not streamer.process.is_alive()

    def test_context_manager_with_exception(self):
        """Should cleanup even if exception occurs."""
        try:
            with ConcreteStreamer(
                device_name="TestDevice",
                synchronized_start_time=time.time(),
                root_output_folder="/tmp/test"
            ) as streamer:
                streamer.start_streaming(timeout=5)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still be cleaned up
        assert streamer.streaming is False


class TestBaseStreamerAbstractMethods:
    """Test that abstract methods must be implemented."""

    def test_cannot_instantiate_base_class(self):
        """Should not be able to instantiate BaseStreamer directly."""
        with pytest.raises(TypeError):
            BaseStreamer(
                device_name="Test",
                synchronized_start_time=time.time(),
                root_output_folder="/tmp/test"
            )


class TestBaseStreamerExceptions:
    """Test custom exceptions."""

    def test_streamer_error(self):
        """Should define StreamerError."""
        with pytest.raises(StreamerError):
            raise StreamerError("Test error")

    def test_connection_error(self):
        """Should define ConnectionError."""
        with pytest.raises(ConnectionError):
            raise ConnectionError("Connection failed")

    def test_streaming_error(self):
        """Should define StreamingError."""
        with pytest.raises(StreamingError):
            raise StreamingError("Streaming failed")


class TestBaseStreamerRepr:
    """Test string representation."""

    def test_repr(self):
        """Should have meaningful string representation."""
        streamer = ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=time.time(),
            root_output_folder="/tmp/test"
        )

        repr_str = repr(streamer)

        assert "ConcreteStreamer" in repr_str
        assert "TestDevice" in repr_str
        assert "streaming=False" in repr_str
        assert "connected=False" in repr_str


@pytest.mark.integration
class TestBaseStreamerIntegration:
    """Integration tests for BaseStreamer."""

    def test_full_lifecycle(self):
        """Test complete lifecycle: start -> stream -> stop."""
        streamer = ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=time.time(),
            root_output_folder="/tmp/test"
        )

        # Start
        assert streamer.start_streaming(timeout=5) is True
        # Note: stream_called is set in child process, so we can't check it in parent
        # Instead, verify streaming state which confirms _stream_wrapper executed
        assert streamer.streaming is True
        assert streamer.connected is True

        # Let it stream for a bit
        time.sleep(0.5)
        assert streamer.is_streaming() is True

        # Stop
        streamer.stop_streaming()
        assert streamer.streaming is False
        # Note: disconnect_called is also in child process, can't verify in parent

    def test_multiple_start_stop_cycles(self):
        """Test multiple start/stop cycles."""
        streamer = ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=time.time(),
            root_output_folder="/tmp/test"
        )

        # Cycle 1
        streamer.start_streaming(timeout=5)
        assert streamer.is_streaming() is True
        streamer.stop_streaming()
        assert streamer.streaming is False

        # Need to create new instance for second cycle (process can't be restarted)
        streamer = ConcreteStreamer(
            device_name="TestDevice",
            synchronized_start_time=time.time(),
            root_output_folder="/tmp/test"
        )

        # Cycle 2
        streamer.start_streaming(timeout=5)
        assert streamer.is_streaming() is True
        streamer.stop_streaming()
        assert streamer.streaming is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

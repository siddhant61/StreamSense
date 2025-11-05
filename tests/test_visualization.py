"""
Tests for stream visualization functionality.

Tests the ViewStreams class which handles:
- LSL stream discovery and filtering
- Stream validation (checking if streams produce data)
- Multiple stream type support (EEG, ACC, BVP, GSR, PPG, HR, TEMP)
- Multiprocess canvas creation for visualization
- Stream selection and plotting
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from collections import OrderedDict
import sys

# Mock GUI and visualization dependencies before importing
sys.modules['PyQt5'] = Mock()
sys.modules['PyQt5.QtCore'] = Mock()
sys.modules['pylsl'] = Mock()
sys.modules['muselsl'] = Mock()
sys.modules['muselsl.constants'] = Mock()
sys.modules['viewer.plot_streams'] = Mock()
sys.modules['helper.plot_helper'] = Mock()

# Mock constants
sys.modules['muselsl.constants'].LSL_SCAN_TIMEOUT = 5.0

from viewer.view_streams import ViewStreams


class TestViewStreamsInitialization:
    """Test ViewStreams initialization."""

    def test_view_streams_initialization(self):
        """Should initialize ViewStreams instance."""
        viewer = ViewStreams()
        assert viewer is not None


class TestFindStreams:
    """Test LSL stream discovery and filtering."""

    @patch('viewer.view_streams.resolve_streams')
    def test_find_streams_discovers_matching_streams(self, mock_resolve):
        """Should discover streams of specified type."""
        # Mock stream objects
        mock_eeg_stream_1 = Mock()
        mock_eeg_stream_1.type.return_value = 'EEG'
        mock_eeg_stream_1.name.return_value = 'Muse-1A2B_EEG'
        mock_eeg_stream_1.created_at.return_value = 1000.0

        mock_eeg_stream_2 = Mock()
        mock_eeg_stream_2.type.return_value = 'EEG'
        mock_eeg_stream_2.name.return_value = 'Muse-3C4D_EEG'
        mock_eeg_stream_2.created_at.return_value = 2000.0

        mock_acc_stream = Mock()
        mock_acc_stream.type.return_value = 'ACC'
        mock_acc_stream.name.return_value = 'Muse-1A2B_ACC'
        mock_acc_stream.created_at.return_value = 1500.0

        mock_resolve.return_value = [mock_eeg_stream_1, mock_eeg_stream_2, mock_acc_stream]

        viewer = ViewStreams()
        streams = viewer.find_streams('EEG')

        # Should return only EEG streams
        assert len(streams) == 2
        assert 'Muse-1A2B_EEG' in streams
        assert 'Muse-3C4D_EEG' in streams

        # Verify correct stream objects returned
        assert streams['Muse-1A2B_EEG'] == mock_eeg_stream_1
        assert streams['Muse-3C4D_EEG'] == mock_eeg_stream_2

    @patch('viewer.view_streams.resolve_streams')
    def test_find_streams_returns_empty_when_no_match(self, mock_resolve):
        """Should return empty dict when no matching streams found."""
        mock_eeg_stream = Mock()
        mock_eeg_stream.type.return_value = 'EEG'
        mock_eeg_stream.name.return_value = 'Muse_EEG'
        mock_eeg_stream.created_at.return_value = 1000.0

        mock_resolve.return_value = [mock_eeg_stream]

        viewer = ViewStreams()
        streams = viewer.find_streams('BVP')  # Search for BVP, only EEG available

        # Should return empty
        assert len(streams) == 0
        assert isinstance(streams, dict)

    @patch('viewer.view_streams.resolve_streams')
    def test_find_streams_returns_latest_for_duplicate_names(self, mock_resolve):
        """Should return latest stream when multiple streams have same name."""
        # Two streams with same name, different creation times
        mock_stream_old = Mock()
        mock_stream_old.type.return_value = 'EEG'
        mock_stream_old.name.return_value = 'Muse_EEG'
        mock_stream_old.created_at.return_value = 1000.0

        mock_stream_new = Mock()
        mock_stream_new.type.return_value = 'EEG'
        mock_stream_new.name.return_value = 'Muse_EEG'  # Same name
        mock_stream_new.created_at.return_value = 2000.0  # Newer

        mock_resolve.return_value = [mock_stream_old, mock_stream_new]

        viewer = ViewStreams()
        streams = viewer.find_streams('EEG')

        # Should return only one stream (the newer one)
        assert len(streams) == 1
        assert streams['Muse_EEG'] == mock_stream_new

    @patch('viewer.view_streams.resolve_streams')
    def test_find_streams_handles_resolve_exception(self, mock_resolve):
        """Should handle stream resolution errors gracefully."""
        mock_resolve.side_effect = Exception("Network error")

        viewer = ViewStreams()
        streams = viewer.find_streams('EEG')

        # Should return empty dict on error
        assert len(streams) == 0
        assert isinstance(streams, dict)

    @patch('viewer.view_streams.resolve_streams')
    def test_find_streams_handles_none_created_at(self, mock_resolve):
        """Should handle streams with None created_at timestamp."""
        mock_stream = Mock()
        mock_stream.type.return_value = 'EEG'
        mock_stream.name.return_value = 'Muse_EEG'
        mock_stream.created_at.return_value = None  # No timestamp

        mock_resolve.return_value = [mock_stream]

        viewer = ViewStreams()
        streams = viewer.find_streams('EEG')

        # Should still return the stream (with 0.0 as fallback timestamp)
        assert len(streams) == 1
        assert 'Muse_EEG' in streams


class TestStartViewing:
    """Test stream visualization startup."""

    @patch('viewer.view_streams.resolve_streams')
    @patch('viewer.view_streams.StreamInlet')
    @patch('viewer.view_streams.Process')
    @patch('viewer.view_streams.Manager')
    def test_start_viewing_eeg_streams(self, mock_manager, mock_process_class,
                                       mock_inlet_class, mock_resolve):
        """Should start viewing EEG streams."""
        # Mock stream discovery
        mock_stream = Mock()
        mock_stream.type.return_value = 'EEG'
        mock_stream.name.return_value = 'Muse_EEG'
        mock_stream.created_at.return_value = 1000.0
        mock_stream.as_xml.return_value = '<stream>...</stream>'
        mock_resolve.return_value = [mock_stream]

        # Mock inlet for stream validation
        mock_inlet = Mock()
        mock_inlet.pull_sample.return_value = ([1, 2, 3, 4, 5], 123.456)
        mock_inlet_class.return_value = mock_inlet

        # Mock process and manager
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        mock_mgr = Mock()
        mock_shared_list = []
        mock_mgr.list.return_value = mock_shared_list
        mock_manager.return_value.__enter__.return_value = mock_mgr

        viewer = ViewStreams()
        viewer.start_viewing(choice=1, duration=60)  # Choice 1 = EEG

        # Verify stream was discovered
        mock_resolve.assert_called_once()

        # Verify inlet created for validation
        mock_inlet_class.assert_called_once()
        mock_inlet.pull_sample.assert_called_once()

        # Verify process created for visualization
        mock_process_class.assert_called_once()
        mock_process.start.assert_called_once()
        mock_process.join.assert_called_once()

    @patch('viewer.view_streams.resolve_streams')
    def test_start_viewing_no_streams_found(self, mock_resolve):
        """Should handle case when no streams are discovered."""
        mock_resolve.return_value = []

        viewer = ViewStreams()
        # Should not raise, just return early
        viewer.start_viewing(choice=1, duration=60)

    @patch('viewer.view_streams.resolve_streams')
    @patch('viewer.view_streams.StreamInlet')
    def test_start_viewing_stream_validation_fails(self, mock_inlet_class, mock_resolve):
        """Should handle streams that fail validation."""
        # Mock stream discovery
        mock_stream = Mock()
        mock_stream.type.return_value = 'EEG'
        mock_stream.name.return_value = 'Muse_EEG'
        mock_stream.created_at.return_value = 1000.0
        mock_resolve.return_value = [mock_stream]

        # Mock inlet that times out (no data)
        mock_inlet = Mock()
        mock_inlet.pull_sample.return_value = (None, None)  # No data
        mock_inlet_class.return_value = mock_inlet

        viewer = ViewStreams()
        # Should not create any processes
        viewer.start_viewing(choice=1, duration=60)

        # Stream should be filtered out during validation

    def test_start_viewing_invalid_choice(self):
        """Should handle invalid stream type choice."""
        viewer = ViewStreams()
        # Should not raise, just print message and return
        viewer.start_viewing(choice=99, duration=60)

    @patch('viewer.view_streams.resolve_streams')
    def test_start_viewing_choice_mapping(self, mock_resolve):
        """Should correctly map choice numbers to stream types."""
        mock_resolve.return_value = []

        viewer = ViewStreams()

        # Test all valid choices
        choice_to_type = {
            1: 'EEG',
            2: 'ACC',
            3: 'BVP',
            4: 'GSR',
            5: 'PPG',
            6: 'HR',
            7: 'TEMP'
        }

        for choice, expected_type in choice_to_type.items():
            mock_resolve.reset_mock()
            viewer.start_viewing(choice=choice, duration=60)
            # Verify find_streams was called with correct type
            # (we can't directly verify the parameter, but we verify it was called)
            mock_resolve.assert_called_once()


class TestPlotStreamWithCanvas:
    """Test canvas creation for stream plotting."""

    @patch('viewer.view_streams.plot_stream')
    @patch('viewer.view_streams.run_vispy')
    @patch('viewer.view_streams.StreamInfo')
    @patch('viewer.view_streams.QTimer')
    def test_plot_stream_with_canvas_success(self, mock_qtimer, mock_streaminfo_class,
                                             mock_run_vispy, mock_plot_stream):
        """Should create canvas and start visualization."""
        # Mock canvas
        mock_canvas = Mock()
        mock_plot_stream.return_value = mock_canvas

        # Mock StreamInfo
        mock_streaminfo_class.return_value = Mock()

        viewer = ViewStreams()
        canvases_statuses = []

        viewer.plot_stream_with_canvas('<stream>...</stream>', canvases_statuses, duration=60)

        # Verify canvas created
        mock_plot_stream.assert_called_once()

        # Verify vispy started
        mock_run_vispy.assert_called_once()

        # Verify status appended
        assert len(canvases_statuses) == 1
        assert canvases_statuses[0] is True

        # Verify timer created for auto-close
        mock_qtimer.singleShot.assert_called_once()

    @patch('viewer.view_streams.plot_stream')
    @patch('viewer.view_streams.StreamInfo')
    def test_plot_stream_with_canvas_no_canvas(self, mock_streaminfo_class, mock_plot_stream):
        """Should handle case when canvas creation fails."""
        # Mock plot_stream returning None (failure)
        mock_plot_stream.return_value = None

        mock_streaminfo_class.return_value = Mock()

        viewer = ViewStreams()
        canvases_statuses = []

        viewer.plot_stream_with_canvas('<stream>...</stream>', canvases_statuses, duration=60)

        # Verify status appended as False
        assert len(canvases_statuses) == 1
        assert canvases_statuses[0] is False


@pytest.mark.integration
class TestVisualizationIntegration:
    """Integration tests for visualization workflow."""

    @patch('viewer.view_streams.resolve_streams')
    @patch('viewer.view_streams.StreamInlet')
    def test_full_visualization_workflow(self, mock_inlet_class, mock_resolve):
        """Test complete workflow from discovery to validation."""
        # Mock multiple streams of different types
        mock_eeg_stream = Mock()
        mock_eeg_stream.type.return_value = 'EEG'
        mock_eeg_stream.name.return_value = 'Muse_EEG'
        mock_eeg_stream.created_at.return_value = 1000.0

        mock_acc_stream = Mock()
        mock_acc_stream.type.return_value = 'ACC'
        mock_acc_stream.name.return_value = 'E4_ACC'
        mock_acc_stream.created_at.return_value = 2000.0

        mock_resolve.return_value = [mock_eeg_stream, mock_acc_stream]

        viewer = ViewStreams()

        # Discover EEG streams
        eeg_streams = viewer.find_streams('EEG')
        assert len(eeg_streams) == 1
        assert 'Muse_EEG' in eeg_streams

        # Discover ACC streams
        acc_streams = viewer.find_streams('ACC')
        assert len(acc_streams) == 1
        assert 'E4_ACC' in acc_streams


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

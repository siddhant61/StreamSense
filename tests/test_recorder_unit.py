"""Pure-unit tests for StreamRecorder helper methods (no LSL, no spawned threads).

These complement the spawn-based test_stream_recorder.py (marked integration) and add
real coverage for the recorder's data-handling logic.
"""

import numpy as np
import h5py

from recorder.stream_recorder import StreamRecorder


def _recorder(tmp_path):
    # __init__ only creates the RawData dir and in-memory state (no threads/processes).
    return StreamRecorder(str(tmp_path))


def test_calculate_sfreq_regular_spacing(tmp_path):
    rec = _recorder(tmp_path)
    timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4])  # 10 Hz
    sfreq = rec.calculate_sfreq(timestamps)
    assert abs(sfreq - 10.0) < 1e-3


def test_calculate_sfreq_returns_none_for_degenerate_timestamps(tmp_path):
    rec = _recorder(tmp_path)
    # Identical timestamps -> no positive diffs -> ValueError caught -> None returned.
    assert rec.calculate_sfreq(np.array([5.0, 5.0, 5.0])) is None


def test_save_to_h5_creates_then_appends(tmp_path):
    rec = _recorder(tmp_path)
    stream_id = "EEG_test"
    h5_path = tmp_path / "eeg.h5"
    rec.output_files[stream_id] = str(h5_path)

    rec.save_to_h5(stream_id, np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.0, 0.1]))
    rec.save_to_h5(stream_id, np.array([[5.0, 6.0]]), np.array([0.2]))

    with h5py.File(str(h5_path), "r") as hf:
        assert hf[stream_id].shape == (3, 2)
        assert hf[f"{stream_id}_timestamps"].shape == (3,)
        assert list(hf[f"{stream_id}_timestamps"][:]) == [0.0, 0.1, 0.2]


def test_save_to_h5_replaces_nan_with_zero(tmp_path):
    rec = _recorder(tmp_path)
    stream_id = "BVP_test"
    h5_path = tmp_path / "bvp.h5"
    rec.output_files[stream_id] = str(h5_path)

    rec.save_to_h5(stream_id, np.array([[np.nan, 1.0]]), np.array([0.0]))
    with h5py.File(str(h5_path), "r") as hf:
        assert hf[stream_id][0, 0] == 0.0  # NaN -> 0


def test_convert_to_mne_builds_4ch_raw(tmp_path):
    rec = _recorder(tmp_path)
    data = np.random.randn(4, 100)  # 4 EEG channels x 100 samples
    raw = rec.convert_to_mne(data, 256)
    assert raw.info["sfreq"] == 256
    assert raw.get_data().shape == (4, 100)

import threading
import time

import pytest

pytest.importorskip("h5py")
pytest.importorskip("mne")
pytest.importorskip("numpy")
pytest.importorskip("pandas")

from recorder.stream_recorder import StreamRecorder

# Spawns real background workers; excluded from coverage runs. See .coveragerc.
pytestmark = pytest.mark.integration


def test_stream_recorder_signals_start(tmp_path, monkeypatch):
    recorder = StreamRecorder(str(tmp_path))

    # Avoid interacting with real LSL streams during the test.
    monkeypatch.setattr(recorder, "check_streams", lambda: {})

    def _idle_worker():
        while not recorder.stop_signal:
            time.sleep(0.01)

    monkeypatch.setattr(recorder, "update_streams", _idle_worker)
    monkeypatch.setattr(recorder, "handle_disconnected_streams_thread", _idle_worker)

    thread = threading.Thread(target=recorder.record_streams, daemon=True)
    thread.start()

    assert recorder.started_event.wait(timeout=1.0)

    recorder.stop()
    thread.join(timeout=1.0)

    assert not thread.is_alive()

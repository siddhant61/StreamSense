import logging
from collections import OrderedDict
from typing import Dict, Tuple

from pylsl import StreamInfo, StreamInlet, resolve_streams
from viewer.plot_streams import plot_stream
from muselsl.constants import LSL_SCAN_TIMEOUT
from helper.plot_helper import run_vispy
from PyQt5.QtCore import QTimer
from multiprocessing import Process, Manager

view_logger = logging.getLogger(__name__)

class ViewStreams:
    def __init__(self):
        super(ViewStreams, self).__init__()
        view_logger.info("Initiating Viewer Instance.")


    def find_streams(self, stream_type: str) -> Dict[str, StreamInfo]:
        """Return the latest LSL stream for each discovered stream name.

        The previous implementation attempted to coerce a list into a dictionary
        in ``start_viewing`` and silently dropped duplicate streams.  Instead we
        explicitly track the newest stream per name and return a mapping so the
        caller can address streams deterministically.
        """

        try:
            discovered_streams = resolve_streams(LSL_SCAN_TIMEOUT)
        except Exception:
            view_logger.exception("Failed to resolve streams from the local LSL network.")
            return {}

        matching_streams = [s for s in discovered_streams if s.type() == stream_type]
        latest_streams: Dict[str, Tuple[float, StreamInfo]] = {}

        for stream in matching_streams:
            created_at = stream.created_at() or 0.0
            name = stream.name()
            previous_entry = latest_streams.get(name)
            if previous_entry is None or created_at > previous_entry[0]:
                latest_streams[name] = (created_at, stream)

        ordered_pairs = sorted(latest_streams.items(), key=lambda item: item[1][0])
        ordered_streams = OrderedDict((name, stream) for name, (_created_at, stream) in ordered_pairs)

        return dict(ordered_streams)

    def plot_stream_with_canvas(self, stream_info_xml: str, canvases_statuses, duration = 60):
        stream_info = StreamInfo(xml=stream_info_xml)
        canvas = plot_stream(stream_info)
        if canvas:
            canvases_statuses.append(True)  # Just a simple flag indicating a canvas was created
            run_vispy()

            def close_plots():
                canvas.stop()
                canvas.close()

            QTimer.singleShot(duration * 1000, close_plots)
        else:
            canvases_statuses.append(False)

    def start_viewing(self, choice, duration=60):

        if choice == 1:
            stream_type = 'EEG'
        elif choice == 2:
            stream_type = 'ACC'
        elif choice == 3:
            stream_type = 'BVP'
        elif choice == 4:
            stream_type = 'GSR'
        elif choice == 5:
            stream_type = 'PPG'
        elif choice == 6:
            stream_type = 'HR'
        elif choice == 7:
            stream_type = 'TEMP'
        else:
            print("Invalid choice.")
            return
        streams = self.find_streams(stream_type)
        if not streams:
            view_logger.warning("No %s streams discovered on the LSL network.", stream_type)
            return

        validated_streams = {}
        for name, stream in streams.items():
            inlet = None
            sample = None
            try:
                inlet = StreamInlet(stream)
                sample, _timestamp = inlet.pull_sample(timeout=5)
            except Exception:
                view_logger.exception("Failed to validate stream '%s'.", name)
                continue
            finally:
                if inlet is not None:
                    try:
                        inlet.close_stream()
                    except Exception:
                        view_logger.debug("Stream inlet cleanup failed for '%s'.", name, exc_info=True)

            if sample:
                validated_streams[name] = stream
            else:
                view_logger.warning("Stream '%s' did not return data within the timeout window.", name)

        if not validated_streams:
            view_logger.warning("No %s streams produced data for visualization.", stream_type)
            return

        with Manager() as manager:
            shared_canvases_statuses = manager.list()  # This will allow us to use the statuses across processes

            processes = []
            for stream in validated_streams.values():
                # Start a separate process for each plot_stream_with_canvas call
                process = Process(
                    target=self.plot_stream_with_canvas,
                    args=(stream.as_xml(), shared_canvases_statuses),
                )
                processes.append(process)
                process.start()

            # Wait for all processes to finish
            for process in processes:
                process.join()

        # Now shared_canvases_statuses contains the statuses, and you can use them further in the main process if required.
        print(f"Number of canvases created: {sum(shared_canvases_statuses)}")




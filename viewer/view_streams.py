import logging
from pylsl import resolve_streams
from collections import Counter
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


    def find_streams(self, stream_type):
        all_streams = resolve_streams(LSL_SCAN_TIMEOUT)
        all_streams = [stream for stream in all_streams if stream.type() == stream_type]
        stream_ids = {}
        result = {}
        streams = {}

        for stream in all_streams:
            key = stream.created_at()
            value = stream.name()
            stream_ids[key] = value

        # Create a Counter from the dictionary values
        counts = Counter(stream_ids.values())

        # Create a new dictionary with only the keys whose value has a count greater than 1
        duplicates = {k: v for k, v in stream_ids.items() if counts[v] > 1}

        # Keep values which were created later to access the latest stream
        for key, value in duplicates.items():
            if value not in result or key > result[value]:
                result[value] = key

        result = {v: k for k, v in result.items()}

        # Remove older duplicate streams from the dictionary
        for stream in all_streams:
            if not stream.name() in duplicates.values():
                key = stream.created_at()
                value = stream.name()
                result[key] = value

        # Save latest stream names and objects in the streams dictionary
        for stream in all_streams:
            if stream.created_at() in result.keys():
                streams[stream.name()] = stream


        return streams.values()

    def plot_stream_with_canvas(self, stream_type, i, canvases_statuses, duration = 60):
        canvas = plot_stream(stream_type, i)
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
        canvases = []
        streams = self.find_streams(stream_type)
        for stream in streams:
            try:
                # Attempt to pull a sample of data from the stream
                sample, timestamp = stream.pull_sample(timeout=5)
                if sample:  # Check if any data was retrieved
                    streams[stream.name()] = stream
            except:
                pass

        with Manager() as manager:
            shared_canvases_statuses = manager.list()  # This will allow us to use the statuses across processes

            processes = []
            for i, stream in enumerate(streams):
                # Start a separate process for each plot_stream_with_canvas call
                process = Process(target=self.plot_stream_with_canvas, args=(stream.type(), i, shared_canvases_statuses))
                processes.append(process)
                process.start()

            # Wait for all processes to finish
            for process in processes:
                process.join()

        # Now shared_canvases_statuses contains the statuses, and you can use them further in the main process if required.
        print(f"Number of canvases created: {sum(shared_canvases_statuses)}")




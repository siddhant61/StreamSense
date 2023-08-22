import csv
import logging
import math
import os
import pickle
import time
import numpy as np
import pandas as pd
import threading
from pathlib import Path
from collections import Counter
from muselsl.constants import AUTO_DISCONNECT_DELAY
from datetime import datetime, timedelta
from pylsl import resolve_streams, StreamInlet, local_clock

recorder_logger = logging.getLogger(__name__)

class SaveData:
    def __init__(self, root_output_folder):
        self.root_output_folder = root_output_folder
        self.output_files = {}
        self.stream_inlets = {}
        self.timestamps = {}
        self.connected = {}
        self.streams = {}
        self.sync_timestamp = None
        self.stop_signal = False
        self.output_folder = self.root_output_folder + \
                             "/RAW_Data"
        self.output_folder_path = Path(self.output_folder)
        self.output_folder_path.mkdir(parents=True, exist_ok=True)
        self.stream_update_interval = 2.0
        self.stream_update_thread = threading.Thread(target=self.update_streams)


    def save_to_csv(self, stream_id, data, timestamp):
        timestamp_str = timestamp.isoformat()
        file_path = self.output_files[stream_id]

        # Determine headers based on stream type
        if "EEG" in stream_id:
            headers = ["TimeStamp", "RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10","AUX"]
        elif "BVP" in stream_id:
            headers = ["TimeStamp", "BVP"]
        elif "GSR" in stream_id:
            headers = ["TimeStamp", "GSR"]
        elif "TEMP" in stream_id:
            headers = ["TimeStamp", "TEMP"]
        elif "ACC" in stream_id:
            headers = ["TimeStamp", "ACC_X", "ACC_Y", "ACC_Z"]
        elif "GYRO" in stream_id:
            headers = ["TimeStamp", "GYR_X", "GYR_Y", "GYR_Z"]
        else:
            headers = ["TimeStamp"] + [f"Data{i + 1}" for i in range(len(data))]

        # Check if file exists
        file_exists = os.path.exists(file_path)

        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # If file doesn't exist, write headers
            if not file_exists:
                writer.writerow(headers)
            writer.writerow([timestamp_str] + data)

    def save_data_to_pickle(self):
        output_folder = Path(self.root_output_folder + "/Dataset")
        output_folder.mkdir(parents=True, exist_ok=True)

        datasets = {}
        for file, path in self.output_files.items():
            try:
                df = pd.read_csv(path, header=0, dtype={'TimeStamp': str})
                stream_type = file.split("_")[-1].split(".")[0]  # Assuming the file name ends with the stream type

                if stream_type not in datasets:
                    datasets[stream_type] = {}

                if stream_type in ["EEG", "ACC", "GYRO", "TEMP", "BVP", "GSR"]:
                    sfreq = self.calculate_sfreq(df['TimeStamp'])
                    datasets[stream_type][file] = {'data': df, 'sfreq': sfreq}
                else:
                    datasets[stream_type][file] = {'data': df}
            except Exception as e:
                print(f"Error processing {file}: {e}")

        # Saving the dataset as a pickle file for each stream type
        for stream_type, dataset in datasets.items():
            try:
                pickle_file_path = f"{output_folder}/{stream_type}_dataset.pkl"
                with open(pickle_file_path, 'wb') as f:
                    pickle.dump(dataset, f)
            except Exception as e:
                print(f"Error saving {stream_type} dataset: {e}")


    def calculate_sfreq(self, timestamps):
        try:
            # Convert all timestamps to datetime objects
            timestamps_datetime = pd.to_datetime(timestamps, errors='coerce')

            # Drop invalid timestamps (NaT values)
            timestamps_datetime = timestamps_datetime.dropna()

            time_diff = np.diff(timestamps_datetime)
            # Convert numpy.timedelta64 to milliseconds
            time_diff = time_diff.astype('timedelta64[ms]').astype(int)
            avg_time_diff = np.mean(time_diff)

            # Convert the average time difference back to seconds for the sampling frequency calculation
            avg_time_diff = avg_time_diff / 1000.0

            if pd.isnull(avg_time_diff) or avg_time_diff == 0:
                raise ValueError("Invalid timestamps: Cannot calculate sampling frequency.")

            sfreq = int(1 / avg_time_diff)
            return sfreq

        except ValueError as e:
            print(f"Timestamp conversion error: {e}")
        except Exception as e:
            print(f"Unexpected error in calculate_sfreq: {e}")

    def check_streams(self):
        all_streams = resolve_streams()
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

        return streams

    def is_stream_active(self, stream, retries=3, delay=0.1):
        for _ in range(retries):
            try:
                inlet = StreamInlet(stream)
                inlet.pull_sample(timeout=1)
                return True
            except Exception as e:
                print(f"Error when checking if stream is active: {e}")
                time.sleep(delay)
        return False


    def update_streams(self):
        while True:
            self.streams = self.check_streams()
            time.sleep(self.stream_update_interval)


    def record_streams(self):

        self.stream_update_thread.start()

        streams = self.check_streams()

        for stream in streams.values():
            if self.is_stream_active(stream):
                self.streams[stream.name()] = stream
                print(f"{stream.name()} is active")
            else:
                print(f"{stream.name()} is not active")

        miss_count = {}

        for stream_id, stream in self.streams.items():
            inlet = StreamInlet(stream)
            self.stream_inlets[stream_id] = inlet
            self.connected[stream_id] = True
            print(f"Stream Connected: {stream_id}")
            recorder_logger.info(f"Stream Connected: {stream_id}")
            output_file = f"{self.output_folder}/{stream_id}.csv"
            self.output_files[stream_id] = output_file
            miss_count[stream_id] = 0

        #mainloop_starts = time.time()
        while not self.stop_signal:
            #now = time.time()
            #elapsed_time = now - mainloop_starts #to find out the time taken by each iteration of the loop (T = 0.13s)

            disconnected_streams = []

            for stream_id, inlet in self.stream_inlets.items():
                sample = inlet.pull_sample(timeout=0.0)
                if sample[0] is None:
                    miss_count[stream_id] = miss_count.get(stream_id) + 1
                else:
                    miss_count[stream_id] = 0

                # If there are more than 2 missing samples then the stream is disconnected.
                # SR = 256, each iteration should receive 256*0.13 = 33 samples.
                if "EEG" in stream_id:
                    if miss_count.get(stream_id) > 2:
                        if self.connected[stream_id]:
                            recorder_logger.info(f"Stream disconnected: {stream_id}")
                        disconnected_streams.append(stream_id)
                        self.connected[stream_id] = False

                # If there are more than 10 missing samples then the stream is disconnected.
                # SR = 64, each iteration should receive 64*0.13 = 8 samples.
                elif "BVP" in stream_id:
                    if miss_count.get(stream_id) > 10:
                        if self.connected[stream_id]:
                            recorder_logger.info(f"Stream disconnected: {stream_id}")
                        disconnected_streams.append(stream_id)
                        self.connected[stream_id] = False

                # If there are more than 15 missing samples then the stream is disconnected
                # SR = 50, each iteration should receive 50*0.13 = 6.5 samples
                elif "GYRO" in stream_id:
                    if miss_count.get(stream_id) > 15:
                        if self.connected[stream_id]:
                            recorder_logger.info(f"Stream disconnected: {stream_id}")
                        disconnected_streams.append(stream_id)
                        self.connected[stream_id] = False

                # If there are more than 20 missing samples then the stream is disconnected
                # SR = 32, each iteration should receive 32*0.13 = 4 samples.
                elif "ACC" in stream_id:
                    if miss_count.get(stream_id) > 20:
                        if self.connected[stream_id]:
                            recorder_logger.info(f"Stream disconnected: {stream_id}")
                        disconnected_streams.append(stream_id)
                        self.connected[stream_id] = False

                # If there are more than 30 missing samples then the stream is disconnected
                # SR = 4, each iteration should receive 4*0.13 = 0.5 samples or 1 sample every 2 iterations.
                elif "GSR" in stream_id:
                    if miss_count.get(stream_id) > 40:
                        if self.connected[stream_id]:
                            recorder_logger.info(f"Stream disconnected: {stream_id}")
                        disconnected_streams.append(stream_id)
                        self.connected[stream_id] = False

                # If there are more than 30 missing samples then the stream is disconnected
                # SR = 4, each iteration should receive 4*0.13 = 0.5 samples or 1 sample every 2 iterations.
                elif "TEMP" in stream_id:
                    if miss_count.get(stream_id) > 40:
                        if self.connected[stream_id]:
                            recorder_logger.info(f"Stream disconnected: {stream_id}")
                        disconnected_streams.append(stream_id)
                        self.connected[stream_id] = False

            for stream_id, inlet in self.stream_inlets.items():
                if self.connected[stream_id]:
                    samples, sample_timestamps = inlet.pull_chunk(timeout=0.0)
                    if samples:
                        sample_rate = inlet.info().nominal_srate()
                        time_delta = 1.0 / sample_rate
                        for sample, sample_timestamp in zip(samples, sample_timestamps):
                            if sample_timestamp > self.timestamps.get(stream_id, 0.0):
                                self.timestamps[stream_id] = sample_timestamp
                                if self.sync_timestamp is None or sample_timestamp < self.sync_timestamp:
                                    self.sync_timestamp = sample_timestamp
                                timestamp = datetime.now() + timedelta(seconds=sample_timestamp - self.sync_timestamp)
                                self.save_to_csv(stream_id, sample, timestamp)
                else:
                    if stream_id not in self.timestamps:
                        self.timestamps[stream_id] = self.sync_timestamp
                    timestamp = datetime.now() + timedelta(seconds=self.timestamps[stream_id] - self.sync_timestamp)
                    self.save_to_csv(stream_id, [0] * inlet.info().channel_count(), timestamp)

            for stream_id in disconnected_streams:
                if stream_id in self.stream_inlets and stream_id in self.streams:
                    recorder_logger.info(f"Stream reconnected: {stream_id}")
                    self.stream_inlets[stream_id] = StreamInlet(self.streams[stream_id])
                    self.connected[stream_id] = True
                    miss_count[stream_id] = 0

            time.sleep(0.1)

    def stop_recording(self):
        self.stop_signal = True



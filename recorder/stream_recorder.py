import datetime
import logging
import time
import pickle
import traceback
from multiprocessing import Event

import h5py
import mne
import numpy as np
import pandas as pd
import threading
from pathlib import Path
from pylsl import resolve_streams, StreamInlet, local_clock
from collections import Counter

# Setup logging
from scipy.interpolate import interp1d

logger = logging.getLogger("stream_recorder.py")
logger.setLevel(logging.CRITICAL)
fh = logging.FileHandler("Logs/stream_recorder.log")
fh.setLevel(logging.CRITICAL)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

class StreamRecorder:
    def __init__(self, root_output_folder, backoff_start=1, backoff_factor=2, backoff_limit=10):
        self.root_output_folder = root_output_folder
        self.output_folder = Path(self.root_output_folder + "/RawData")
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.output_files = {}
        self.stream_inlets = {}
        self.timestamps = {}
        self.connected = {}
        self.streams = {}
        self.buffers = {}
        self.locks = {}
        self.sync_timestamp = None
        self.stop_signal = False
        self.stream_sample_rates = {
            "EEG": 256, "BVP": 64, "PPG": 64, "GYRO": 50, "ACC": 32, "GSR": 4, "TEMP": 4, "TAG": 0.1
        }
        self.backoff_start = backoff_start  # Initial backoff time in seconds
        self.backoff_factor = backoff_factor  # Factor by which the backoff time increases
        self.backoff_limit = backoff_limit  # Maximum backoff time in seconds
        self.backoff_times = {}  # Dictionary to track backoff times for each stream
        self.stream_update_interval = 2.0
        self.stream_update_thread = threading.Thread(target=self.update_streams)
        self.disconnect_handler_thread = threading.Thread(target=self.handle_disconnected_streams_thread)
        self.disconnected_streams = set()

        # Initialize the buffers and locks for the predefined stream types
        for key in self.stream_sample_rates.keys():
            self.buffers[key] = []
            self.locks[key] = threading.Lock()

        logger.info(f"StreamRecorder initialized with root_output_folder: {self.root_output_folder}")

    def save_to_h5(self, stream_id, samples, sample_timestamps):
        samples = np.nan_to_num(samples, nan=0)
        with h5py.File(self.output_files[stream_id], 'a') as hf:
            if stream_id in hf:
                dset_data = hf[stream_id]
                dset_timestamps = hf[f"{stream_id}_timestamps"]
                dset_data.resize(dset_data.shape[0] + len(samples), axis=0)
                dset_timestamps.resize(dset_timestamps.shape[0] + len(sample_timestamps), axis=0)
                dset_data[-len(samples):] = samples
                dset_timestamps[-len(sample_timestamps):] = sample_timestamps
            else:
                hf.create_dataset(stream_id, data=samples, maxshape=(None, len(samples[0])))
                hf.create_dataset(f"{stream_id}_timestamps", data=sample_timestamps, maxshape=(None,))

    def calculate_sfreq(self, timestamps):
        try:
            # Ensure timestamps are sorted and unique
            timestamps = np.sort(np.unique(timestamps))

            # Calculate the differences between consecutive timestamps
            time_diff = np.diff(timestamps)

            # Handle any potential anomalies in time differences
            time_diff = time_diff[time_diff > 0]  # Remove non-positive differences

            if len(time_diff) == 0:
                raise ValueError("Invalid timestamps: Cannot calculate sampling frequency.")

            avg_time_diff = np.mean(time_diff)

            sfreq = 1 / avg_time_diff
            return sfreq
        except ValueError as e:
            print(f"Timestamp conversion error: {e}")
        except Exception as e:
            print(f"Unexpected error in calculate_sfreq: {e}")


    def convert_to_mne(self, data, sfreq):
        # Convert data to MNE format
        # Data shape is (n_channels, n_samples)
        raw = mne.io.RawArray(data, info=mne.create_info(ch_names=["AF7", "AF8", "TP9", "TP10"],
                                                         sfreq=sfreq, ch_types='eeg'))
        return raw


    def save_datasets(self):
        output_folder = Path(self.root_output_folder + "/Dataset")
        output_folder.mkdir(parents=True, exist_ok=True)
        eeg_dataset = {}
        acc_dataset = {}
        gyro_dataset = {}
        bvp_dataset = {}
        ibi_dataset = {}
        ppg_dataset = {}
        gsr_dataset = {}
        temp_dataset = {}
        tag_dataset = {}


        for stream_id, filepath in self.output_files.items():
            try:
                with h5py.File(filepath, 'r') as hf:
                    if stream_id in hf:
                        if 'EEG' in stream_id:
                            df_data = pd.DataFrame(hf[stream_id][:]).drop(columns=[4])
                        else:
                            df_data = pd.DataFrame(hf[stream_id][:])
                        timestamps = hf[f"{stream_id}_timestamps"][:]
                        df = pd.DataFrame(df_data.values, index=timestamps)
                        if "EEG" in stream_id:
                            sfreq = self.calculate_sfreq(timestamps)
                            eeg_dataset[stream_id] = {'data':self.convert_to_mne(df.values.T,sfreq), 'sfreq': sfreq}
                        elif "ACC" in stream_id:
                            acc_dataset[stream_id] = {'data':df, 'sfreq': self.calculate_sfreq(timestamps)}
                        elif "GYRO" in stream_id:
                            gyro_dataset[stream_id] = {'data': df, 'sfreq': self.calculate_sfreq(timestamps)}
                        elif "BVP" in stream_id:
                            bvp_dataset[stream_id] = {'data': df, 'sfreq': self.calculate_sfreq(timestamps)}
                        elif "PPG" in stream_id:
                            ppg_dataset[stream_id] = {'data': df, 'sfreq': self.calculate_sfreq(timestamps)}
                        elif "GSR" in stream_id:
                            gsr_dataset[stream_id] = {'data': df, 'sfreq': self.calculate_sfreq(timestamps)}
                        elif "TEMP" in stream_id:
                            temp_dataset[stream_id] = {'data': df, 'sfreq': self.calculate_sfreq(timestamps)}
                        elif "TAG" in stream_id:
                            tag_dataset[stream_id] = {'data': df, 'sfreq': self.calculate_sfreq(timestamps)}
                        elif "IBI" in stream_id:
                            ibi_dataset[stream_id] = {'data': df, 'sfreq': self.calculate_sfreq(timestamps)}
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}. Traceback: {traceback.format_exc()}")

        if len(eeg_dataset) != 0:
            pickle_file_path = output_folder / "eeg_dataset.pkl"
            try:
                with open(pickle_file_path, 'wb') as f:
                    pickle.dump(eeg_dataset, f)
                    logger.info("Processed EEG data and saved to dataset.")
            except Exception as e:
                logger.error(f"Error saving dataset for EEG data: {e}. Traceback: {traceback.format_exc()}")
        if len(acc_dataset) != 0:
            pickle_file_path = output_folder / "acc_dataset.pkl"
            try:
                with open(pickle_file_path, 'wb') as f:
                    pickle.dump(acc_dataset, f)
                    logger.info("Processed ACC data and saved to dataset.")
            except Exception as e:
                logger.error(f"Error saving dataset for ACC data: {e}. Traceback: {traceback.format_exc()}")
        if len(gyro_dataset) != 0:
            pickle_file_path = output_folder / "gyro_dataset.pkl"
            try:
                with open(pickle_file_path, 'wb') as f:
                    pickle.dump(gyro_dataset, f)
                    logger.info("Processed GYRO data and saved to dataset.")
            except Exception as e:
                logger.error(f"Error saving dataset for GYRO data: {e}. Traceback: {traceback.format_exc()}")
        if len(bvp_dataset) != 0:
            pickle_file_path = output_folder / "bvp_dataset.pkl"
            try:
                with open(pickle_file_path, 'wb') as f:
                    pickle.dump(bvp_dataset, f)
                    logger.info("Processed BVP data and saved to dataset.")
            except Exception as e:
                logger.error(f"Error saving dataset for BVP data: {e}. Traceback: {traceback.format_exc()}")
        if len(ppg_dataset) != 0:
            pickle_file_path = output_folder / "ppg_dataset.pkl"
            try:
                with open(pickle_file_path, 'wb') as f:
                    pickle.dump(ppg_dataset, f)
                    logger.info("Processed PPG data and saved to dataset.")
            except Exception as e:
                logger.error(f"Error saving dataset for PPG data: {e}. Traceback: {traceback.format_exc()}")
        if len(gsr_dataset) != 0:
            pickle_file_path = output_folder / "gsr_dataset.pkl"
            try:
                with open(pickle_file_path, 'wb') as f:
                    pickle.dump(gsr_dataset, f)
                    logger.info("Processed GSR data and saved to dataset.")
            except Exception as e:
                logger.error(f"Error saving dataset for GSR data: {e}. Traceback: {traceback.format_exc()}")
        if len(temp_dataset) != 0:
            pickle_file_path = output_folder / "temp_dataset.pkl"
            try:
                with open(pickle_file_path, 'wb') as f:
                    pickle.dump(temp_dataset, f)
                    logger.info("Processed TEMP data and saved to dataset.")
            except Exception as e:
                logger.error(f"Error saving dataset for TEMP data: {e}. Traceback: {traceback.format_exc()}")
        if len(tag_dataset) != 0:
            pickle_file_path = output_folder / "tag_dataset.pkl"
            try:
                with open(pickle_file_path, 'wb') as f:
                    pickle.dump(tag_dataset, f)
                    logger.info("Processed TAG data and saved to dataset.")
            except Exception as e:
                logger.error(f"Error saving dataset for TAG data: {e}. Traceback: {traceback.format_exc()}")
        if len(ibi_dataset) != 0:
            pickle_file_path = output_folder / "ibi_dataset.pkl"
            try:
                with open(pickle_file_path, 'wb') as f:
                    pickle.dump(ibi_dataset, f)
                    logger.info("Processed IBI data and saved to dataset.")
            except Exception as e:
                logger.error(f"Error saving dataset for IBI data: {e}. Traceback: {traceback.format_exc()}")

        logger.info("Finished saving datasets.")

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

    def active_streams(self):
        streams = {}
        for stream in self.streams:
            if self.is_stream_active(stream):
                streams[stream.name] = stream

        # Update the current streams dictionary
        self.streams.update(streams)

        return self.streams

    def is_stream_active(self, stream):
        """Helper function to check if a stream is active."""
        try:
            inlet = StreamInlet(stream)
            inlet.pull_sample(timeout=0.1)
            return True
        except:
            return False

    def update_streams(self):
        """Keep updating the available streams."""
        while not self.stop_signal:  # Adjusted to use stop_signal
            self.streams = self.active_streams()
            time.sleep(self.stream_update_interval)

    def handle_disconnected_streams_thread(self):
        """Separate thread to reconnect to the streams that were previously disconnected with a backoff strategy."""
        while not self.stop_signal:
            for stream_id in list(self.disconnected_streams):
                if stream_id not in self.backoff_times:
                    self.backoff_times[stream_id] = self.backoff_start
                else:
                    self.backoff_times[stream_id] *= self.backoff_factor
                    if self.backoff_times[stream_id] > self.backoff_limit:
                        self.backoff_times[stream_id] = self.backoff_limit

                time.sleep(self.backoff_times[stream_id])

                # Try to reconnect
                if stream_id in self.check_streams():
                    self.stream_inlets[stream_id] = StreamInlet(self.streams[stream_id])
                    self.connected[stream_id] = True
                    self.backoff_times[stream_id] = self.backoff_start
                    self.disconnected_streams.remove(stream_id)  # Remove stream_id from disconnected_streams set
                    logger.info(f"Stream reconnected: {stream_id}")


    def check_disconnect(self, stream_id, miss_count):
        """Check if the stream is disconnected based on the miss count and log it."""
        thresholds = {
            "EEG": 2,
            "BVP": 10,
            "GYRO": 15,
            "ACC": 15,
            "GSR": 40,
            "TEMP": 40,
            "TAG": 2000
        }
        threshold = thresholds.get(stream_id.split('_')[-1], 20)  # default to 20
        if miss_count.get(stream_id) > threshold:
            if self.connected[stream_id]:
                logger.info(f"Stream disconnected: {stream_id}")
            self.connected[stream_id] = False

    def handle_disconnection(self, stream_id, disconnection_duration):
        max_interpolation_duration = 120.0  # 2 minutes

        # Extract the stream type from the stream_id
        stream_type = stream_id.split('_')[-1]

        if disconnection_duration <= max_interpolation_duration:
            interval = 1.0 / self.nominal_srates[stream_id]
            missing_sample_count = int(disconnection_duration / interval)
            last_valid_timestamp = self.last_received_timestamps[stream_id]
            interpolated_timestamps = [last_valid_timestamp + i * interval for i in range(1, missing_sample_count + 1)]

            # Use the buffer to get historical data points for interpolation
            buffer_data = self.buffers[stream_type]
            if buffer_data:
                if stream_type in ["EEG", "BVP", "PPG", "GYRO", "ACC"]:
                    # Spline interpolation for complex signals
                    # Use as many historical data points as available for better accuracy
                    historical_data = np.array(buffer_data[-missing_sample_count:])
                    time_indices = np.linspace(0, 1, len(historical_data))
                    spline = interp1d(time_indices, historical_data, kind='cubic', axis=0, fill_value="extrapolate")
                    interpolated_data = spline(np.linspace(0, 1, missing_sample_count))

                elif stream_type in ["GSR", "TEMP"]:
                    # Linear interpolation for more stable signals
                    historical_data = np.array(buffer_data[-missing_sample_count:])
                    interpolated_data = np.linspace(historical_data[0], historical_data[-1], missing_sample_count)

                elif stream_type == "TAG":
                    # Repeat the last known value for TAG data
                    last_valid_data = buffer_data[-1]
                    interpolated_data = [last_valid_data for _ in range(missing_sample_count)]

                else:
                    # Default handling for unrecognized stream types
                    last_valid_data = buffer_data[-1]
                    interpolated_data = [last_valid_data for _ in range(missing_sample_count)]

                self.save_to_h5(stream_id, interpolated_data, interpolated_timestamps)

    def estimate_timestamps(self, last_timestamp, stream_id, count):
        interval = 1.0 / self.nominal_srates[stream_id]
        return [last_timestamp + i * interval for i in range(1, count + 1)]

    def record_streams(self):
        self.streams = self.check_streams()
        self.stream_update_thread.start()
        self.disconnect_handler_thread.start()  # Start the disconnection handler thread

        miss_count = {}
        self.last_received_timestamps = {}
        self.nominal_srates = {stream_id: stream.nominal_srate() for stream_id, stream in self.streams.items()}
        disconnection_times = {}

        for stream_id, stream in self.streams.items():
            self.last_received_timestamps[stream_id] = None
            inlet = StreamInlet(stream)
            self.stream_inlets[stream_id] = inlet
            self.connected[stream_id] = True
            logger.info(f"Stream Connected: {stream_id}")
            output_file = f"{self.output_folder}/{stream_id}.h5"
            self.output_files[stream_id] = output_file
            miss_count[stream_id] = 0

        while not self.stop_signal:
            for stream_id, inlet in self.stream_inlets.items():
                sample = inlet.pull_sample(timeout=0.0)
                if sample[0] is None:
                    miss_count[stream_id] = miss_count.get(stream_id) + 1
                else:
                    miss_count[stream_id] = 0

                # Check for disconnect using the provided method
                self.check_disconnect(stream_id, miss_count)

                # If a stream is disconnected, add it to the set of disconnected streams
                if not self.connected[stream_id]:
                    self.disconnected_streams.add(stream_id)

            for stream_id, inlet in self.stream_inlets.items():
                if self.connected[stream_id]:
                    samples, sample_timestamps = inlet.pull_chunk(timeout=0.1)
                    if samples:
                        self.save_to_h5(stream_id, samples, sample_timestamps)
                        self.last_received_timestamps[stream_id] = sample_timestamps[-1]
                    elif stream_id in disconnection_times:
                        # Handle disconnection upon reconnection
                        disconnection_duration = local_clock() - disconnection_times[stream_id]
                        self.handle_disconnection(stream_id, disconnection_duration)
                        del disconnection_times[stream_id]
                else:
                    disconnection_times[stream_id] = local_clock()

            time.sleep(0.1)

    def stop(self):
        self.stop_signal = True

        # Signal threads to stop
        self.stream_update_thread.join()
        self.disconnect_handler_thread.join()

        # Save datasets
        self.save_datasets()




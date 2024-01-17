import datetime
import logging
import os
import pickle
import re
import traceback
from pathlib import Path

import scipy
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d

import h5py
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure a basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, folder_path, NA_0=False):
        self.folder_path = folder_path
        self.NA_0 = NA_0
        self.gaps = {}
        self.datasets = {}
        self.sample_intervals = {}
        self.stream_sample_rates = {
            "EEG": 256, "BVP": 64, "PPG": 64, "GYRO": 50, "ACC": 32, "GSR": 4, "TEMP": 4, "TAG": 0.1
        }
        self.channel_mapping = {
            'ACC': ['X', 'Y', 'Z'],
            'GYRO': ['X', 'Y', 'Z'],
            'EEG': ['AF7', 'AF8', 'TP9', 'TP10', 'R_AUX'],
            'TAG': ['TAG'],
            'GSR': ['GSR'],
            'BVP': ['BVP'],
            'TEMP': ['TEMP'],
            'PPG': ['Ambient', 'IR', 'Red']
        }
        self.nominal_srates = {'GSR': 4}

    def process_files(self):
        files = [f for f in os.listdir(self.folder_path) if f.endswith('.h5')]
        for file in files:
            self.process_file(file)

    def process_file(self, file):
        dataset_name = file.split('.')[0]
        stream_type = dataset_name.split('_')[-1]
        filepath = os.path.join(self.folder_path, file)

        with h5py.File(filepath, 'r') as hf:
            if dataset_name in hf:
                logger.info(f"Processing dataset: {dataset_name} in file: {file}")
                data = np.array(hf[dataset_name][:], dtype=float)
                elapsed_seconds = np.array(hf[f"{dataset_name}_timestamps"][:], dtype=np.float64)
                relative_timestamps = elapsed_seconds - elapsed_seconds[0]

                # Get the reference time from the folder path
                reference_time = self.extract_unix_timestamp(self.folder_path)

                # Convert elapsed seconds to actual timestamps
                # Ensure that elapsed_seconds are reasonable values
                if np.any(relative_timestamps > 1e10):  # Arbitrary large number, adjust as needed
                    logger.error("Elapsed seconds are too large, check the dataset.")
                    return

                timestamps = [reference_time + ts for ts in relative_timestamps]

                # Convert timestamps to datetime and ensure they are within a reasonable range
                timestamps = pd.to_datetime(timestamps, unit='s')
                if timestamps.min().year < 1970 or timestamps.max().year > 2100:  # Adjust years as needed
                    logger.error("Timestamps are outside of a reasonable range.")
                    return

                # Truncate data and timestamps to the length of the shorter array
                min_length = min(len(data), len(timestamps))
                data = data[:min_length]
                timestamps = timestamps[:min_length]

                if stream_type == 'GSR':
                    data, timestamps = self.smooth_and_realign_data(data, timestamps, stream_type)

                # Convert timestamps to datetime
                self.start_time = datetime.datetime.utcfromtimestamp(reference_time)
                df = pd.DataFrame(data, index=timestamps)

                # Replace zeros with NaN if NA_0 is True
                if self.NA_0:
                    df.replace(0, np.nan, inplace=True)

                self.gaps[dataset_name] = self.detect_gaps(df, gap_threshold=1)

                # Filter out unrealistic temperature values if this is temperature data
                if 'TEMP' in dataset_name:
                    df = df[df <= 40]

                # Store the DataFrame in the datasets dictionary
                self.datasets[dataset_name] = df
            else:
                logger.warning(f"Dataset '{dataset_name}' not found in file: {file}.")


    def extract_unix_timestamp(self, folder_path):
        match = re.search(r'(\d+)_\d+/RawData$', folder_path)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Expected Unix timestamp not found in the folder path: {folder_path}")

    def smooth_and_realign_data(self, data, timestamps, stream_type):
        # Convert datetime to Unix timestamp (numerical values)
        # This conversion is necessary for numerical operations
        unix_timestamps = timestamps.view('int64') / 1e9  # Convert to seconds

        # Handle NaN values (gaps) in the data
        data_series = pd.Series(data[:, 0])
        original_nans = data_series.isna()

        # Interpolate over short gaps before applying the filter
        data_series.interpolate(limit_direction='both', inplace=True)

        # Apply a Savitzky-Golay filter for smoothing
        window_length, poly_order = 51, 3  # Ensure window_length is odd
        smoothed_data = savgol_filter(data_series, window_length, poly_order)

        # Restore original NaN values to maintain gap positions
        smoothed_data[original_nans] = np.nan

        # Re-align timestamps for GSR data
        if stream_type in self.nominal_srates:
            nominal_srate = self.nominal_srates[stream_type]
            total_duration = unix_timestamps[-1] - unix_timestamps[0]
            num_samples = int(total_duration * nominal_srate)
            new_unix_timestamps = np.linspace(unix_timestamps[0], unix_timestamps[-1], num_samples)

            # Interpolation function for realignment, using Unix timestamps
            valid_indices = ~original_nans
            interpolation_function = interp1d(unix_timestamps[valid_indices], smoothed_data[valid_indices],
                                              kind='linear',
                                              bounds_error=False, fill_value="extrapolate")
            aligned_data = interpolation_function(new_unix_timestamps)

            # Convert Unix timestamps back to datetime for returning
            new_timestamps = pd.to_datetime(new_unix_timestamps, unit='s')
            return aligned_data, new_timestamps
        else:
            return smoothed_data, timestamps.view('int64') / 1e9

    def highlight_gaps(self, ax, df, gap_threshold=1):
        # Convert the index to seconds
        if isinstance(df.index, pd.DatetimeIndex):
            time_seconds = (df.index - df.index[0]).total_seconds().to_numpy()
        else:
            time_seconds = df.index.to_numpy()

        # Iterate through each column to find and highlight gaps
        for column in df.columns:
            is_gap = df[column].isna()
            if is_gap.any():
                start_idx = None
                for idx, gap in enumerate(is_gap):
                    if gap and start_idx is None:
                        start_idx = idx  # Start of a gap
                    elif not gap and start_idx is not None:
                        # End of a gap, highlight the region
                        end_idx = idx
                        if time_seconds[end_idx] - time_seconds[start_idx] > gap_threshold:
                            ax.axvspan(time_seconds[start_idx], time_seconds[end_idx], color='red', alpha=0.5)
                        start_idx = None
                # Check if there is a gap at the end
                if start_idx is not None:
                    ax.axvspan(time_seconds[start_idx], time_seconds[-1], color='red', alpha=0.5)

    def detect_gaps(self, df, gap_threshold=2, nan_sequence_threshold=2):
        # Initialize the gap array with False
        gaps = np.zeros(len(df), dtype=bool)

        # Detect gaps based on time difference
        if isinstance(df.index, pd.DatetimeIndex):
            time_seconds = (df.index - df.index[0]).total_seconds()
            time_diffs = np.diff(time_seconds)
            time_gap_indices = np.where(time_diffs > gap_threshold)[0]
            # Insert NaNs into DataFrame at detected time gaps
            for idx in time_gap_indices:
                # Insert NaNs at the next index after the gap
                next_idx = idx + 1
                if next_idx < len(df):
                    df.iloc[next_idx] = np.nan
                    gaps[next_idx] = True

        for column_name in df.columns:
            # Replace contiguous zero values with NaN
            df[column_name].replace(to_replace=0, method='ffill', limit=nan_sequence_threshold - 1, inplace=True)
            df[column_name].replace(to_replace=0, value=np.nan, inplace=True)

            # Detect contiguous NaN values in the DataFrame
            is_nan = df[column_name].isna()
            nan_sequences = is_nan.ne(is_nan.shift()).cumsum()
            nan_sequences[~is_nan] = 0
            nan_sequence_counts = nan_sequences.value_counts()
            long_nan_sequences = nan_sequence_counts[nan_sequence_counts >= nan_sequence_threshold].index

            for seq in long_nan_sequences:
                # Mark the entire sequence as gaps
                gaps[nan_sequences == seq] = True

        return gaps

    def plot_data_quality(self):
        quality_folder = Path(self.folder_path) / "DataQuality"
        quality_folder.mkdir(parents=True, exist_ok=True)

        for dataset_name, df in self.datasets.items():
            if df.empty:
                logger.info(f"No data found for {dataset_name}.")
                continue

            stream_type = dataset_name.split('_')[-1]
            channel_labels = self.channel_mapping.get(stream_type, df.columns)
            num_channels = len(df.columns)

            fig, axs = plt.subplots(num_channels, 1, figsize=(15, 5 * num_channels), sharex=True)
            fig.suptitle(f'Data Quality for {dataset_name}')

            seconds_since_start = (df.index - df.index[0]).total_seconds()

            if num_channels == 1:
                axs = [axs]  # Make axs a list if only one subplot

            for i, (column, ax) in enumerate(zip(df.columns, axs)):
                label = channel_labels[i] if i < len(channel_labels) else f"Channel {i}"
                # Plot non-gap data
                non_gap_data = df[column].dropna()
                non_gap_seconds = seconds_since_start[~df[column].isna()]
                ax.plot(non_gap_seconds, non_gap_data, label=label)

                # Highlight gaps
                self.highlight_gaps(ax, df[[column]], gap_threshold=2)

                # Calculate and display percentage of intact data
                intact_data_count = non_gap_data.count()
                total_data_count = len(df[column])
                intact_percentage = 100 * intact_data_count / total_data_count
                ax.set_title(f'Channel: {label} - {intact_percentage:.2f}% Data Intact')

                ax.set_ylabel('Value')
                if i == num_channels - 1:
                    ax.set_xlabel('Time (seconds)')

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])

            # Save plot to the DataQuality folder
            plot_filename = quality_folder / f"{dataset_name}_quality.png"
            plt.savefig(plot_filename)
            plt.close()

    def generate_spike_indices_dict(self):
        """
        Generates a dictionary of spike indices for each stream type.

        Returns:
        dict: A dictionary with stream types as keys and lists of arrays of spike indices as values.
        """
        spike_indices_dict = {stream_type: [] for stream_type in self.stream_sample_rates.keys()}

        for dataset_name, df in self.datasets.items():
            # Determine the stream type from the dataset name
            stream_type = dataset_name.split('_')[-1]

            # Check if the stream type is one for which we detect spikes
            if stream_type in spike_indices_dict:
                # Detect spikes for each channel of the dataset
                # Assuming data is in a DataFrame where each column is a channel
                for column in df.columns:
                    data = df[column].dropna().values  # Remove NaNs which can affect peak detection
                    spikes = self.detect_spikes(data, stream_type)
                    spike_indices_dict[stream_type].append(spikes)

        return spike_indices_dict

    def detect_spikes(self, data, stream_type, width=None, prominence=None):
        """
        Detect spikes in the data using peak detection.

        Parameters:
        data (np.array): The data in which to detect spikes.
        stream_type (str): The type of the data stream (e.g., 'EEG', 'ACC').
        width (float or None): Required width of peaks in samples. If None, not used.
        prominence (float or None): Required prominence of peaks. If None, not used.

        Returns:
        np.array: Indices of the detected spikes.
        """
        # Default height values for peak detection based on stream type
        default_heights = {
            'EEG': 500,  # EEG signals may have spikes of significant amplitude
            'ACC': 30,  # Accelerometer data might have moderate spikes
            'GYRO': 30,  # Gyroscope data can also have moderate spikes
            'BVP': 200,  # Blood Volume Pulse might have smaller spikes
            'PPG': 50,  # Photoplethysmogram signals might have smaller spikes
            'GSR': 0.1,  # Galvanic Skin Response spikes are usually subtle
            'TEMP': 30,  # Temperature changes are typically gradual
            # Add or adjust for other stream types as necessary
        }

        height = default_heights.get(stream_type, 1)  # Default height if stream type not in the list

        # Detect peaks
        peaks, _ = find_peaks(data, height=height, width=width, prominence=prominence)

        return peaks

    def compare_spikes_within_type(self, spike_indices_dict, tolerance=0.05, early_spike_window=5.0):
        """
        Compare spikes within similar stream types to find common spikes, prioritizing early spikes.

        Parameters:
        spike_indices_dict (dict): Dictionary with stream names as keys and arrays of spike indices as values.
        tolerance (float): Tolerance in seconds for considering spikes as common.
        early_spike_window (float): Time window in seconds to prioritize early spikes.

        Returns:
        dict: Dictionary with common spike times for each stream type.
        """
        common_spikes = {}
        for stream_type, spikes_list in spike_indices_dict.items():
            if not spikes_list:
                continue

            # Convert lists of spike indices to sets for efficient intersection
            spike_sets = [set(spikes) for spikes in spikes_list]

            # Prioritize early spikes within the specified time window
            early_spike_sets = [
                set(filter(lambda x: x / self.stream_sample_rates[stream_type] <= early_spike_window, spikes)) for
                spikes in spikes_list]
            early_common_spikes_set = set.intersection(*early_spike_sets)

            # If early common spikes are found, use them
            if early_common_spikes_set:
                common_spikes_set = early_common_spikes_set
            else:
                # Otherwise, find intersection of spikes across all datasets of this stream type
                common_spikes_set = set.intersection(*spike_sets)

            # Convert common spike indices back to times
            if stream_type in self.stream_sample_rates:
                sampling_rate = self.stream_sample_rates[stream_type]
                common_spikes[stream_type] = [index / sampling_rate for index in common_spikes_set]
            else:
                logger.warning(f"Sampling rate for stream type '{stream_type}' not found.")

        return common_spikes

    def compare_spikes_across_types(self, spike_indices_dict, tolerance=0.05):
        """
        Compare spikes across different stream types to refine common spikes.

        Parameters:
        spike_indices_dict (dict): Dictionary with stream types as keys and lists of arrays of spike indices as values.
        tolerance (float): Time tolerance in seconds for considering spikes as common across types.

        Returns:
        dict: Dictionary with refined common spike times across stream types.
        """
        # Initialize a dictionary to store the refined common spikes
        refined_common_spikes = {}

        # List of stream types to be compared
        stream_types = list(spike_indices_dict.keys())

        # Iterate over each stream type
        for i, stream_type1 in enumerate(stream_types):
            for stream_type2 in stream_types[i+1:]:
                # Compare spikes between two stream types
                common_spikes = self.find_common_spikes(spike_indices_dict[stream_type1],
                                                        spike_indices_dict[stream_type2],
                                                        tolerance)

                # Store the common spikes
                refined_common_spikes[(stream_type1, stream_type2)] = common_spikes

        return refined_common_spikes

    def find_common_spikes(self, spikes_list1, spikes_list2, tolerance):
        """
        Find common spikes between two lists of spike arrays with a given tolerance.

        Parameters:
        spikes_list1 (list of np.array): First list of arrays containing spike indices.
        spikes_list2 (list of np.array): Second list of arrays containing spike indices.
        tolerance (float): Tolerance for considering spikes as common.

        Returns:
        list: List of common spikes.
        """
        common_spikes = []
        for spikes1 in spikes_list1:
            for spikes2 in spikes_list2:
                # Check if spikes1 or spikes2 are not suitable for iteration
                if not isinstance(spikes1, np.ndarray) or not isinstance(spikes2, np.ndarray) or \
                   spikes1.ndim == 0 or spikes2.ndim == 0:
                    continue

                # Check if spikes1 or spikes2 are empty
                if spikes1.size == 0 or spikes2.size == 0:
                    continue

                # Find common spikes within the tolerance window
                for spike1 in spikes1:
                    for spike2 in spikes2:
                        if abs(spike1 - spike2) <= tolerance:
                            common_spikes.append((spike1, spike2))
        return common_spikes

    def synchronize_timestamps(self, common_spikes, datasets):
        synchronized_datasets = {}

        for dataset_name, df in datasets.items():
            stream_type = dataset_name.split('_')[-1]

            for (stream_type1, stream_type2), spikes in common_spikes.items():
                if stream_type in [stream_type1, stream_type2] and spikes:
                    reference_spike = spikes[0]

                    # Find the corresponding timestamp in the dataset, skipping gaps
                    non_gap_data = df.dropna()
                    if not non_gap_data.empty:
                        closest_timestamp = non_gap_data.index[np.abs(non_gap_data.index - reference_spike).argmin()]
                        time_offset = closest_timestamp - reference_spike
                        df.index = pd.to_datetime(df.index) + pd.Timedelta(seconds=time_offset)
                        break

            synchronized_datasets[dataset_name] = df

        return synchronized_datasets

    def calculate_sfreq(self, stream_id, df):
        try:
            # Convert timestamps to seconds, excluding gaps
            non_gap_data = df.dropna()
            if non_gap_data.empty:
                logger.error(f"No valid data found for dataset: {stream_id}")
                return None

            time_seconds = (non_gap_data.index - non_gap_data.index[0]).total_seconds()

            # Calculate average time difference
            time_diffs = np.diff(time_seconds)
            avg_time_diff = np.mean(time_diffs)
            sfreq = 1 / avg_time_diff  # Sampling frequency is the reciprocal of the average time difference

            return sfreq
        except Exception as e:
            logger.error(f"Unexpected error in calculate_sfreq for dataset {stream_id}: {e}")
            return None

    def save_datasets(self):
        output_folder = Path(self.folder_path, "Datasets")
        output_folder.mkdir(parents=True, exist_ok=True)

        for dataset_type in self.stream_sample_rates.keys():
            dataset_dict = {}

            for stream_id, df in self.datasets.items():
                if dataset_type in stream_id:
                    sfreq = self.calculate_sfreq(stream_id, df)
                    gaps = self.gaps[stream_id]

                    # Map channel names for each dataset type
                    channel_names = self.channel_mapping.get(dataset_type, df.columns.tolist())
                    df.columns = channel_names

                    dataset_dict[stream_id] = {'data': df, 'sfreq': sfreq, 'gaps': gaps}

            if dataset_dict:
                pickle_file_path = output_folder / f"{dataset_type.lower()}_dataset.pkl"
                try:
                    with open(pickle_file_path, 'wb') as f:
                        pickle.dump(dataset_dict, f, protocol=4)
                        logger.info(f"Processed {dataset_type} data and saved to dataset.")
                except Exception as e:
                    logger.error(f"Error saving dataset for {dataset_type} data: {e}. Traceback: {traceback.format_exc()}")

    def plot_synchronized_data(self):
        data_quality_folder = Path(self.folder_path + "/Datasets/DataQuality")
        data_quality_folder.mkdir(parents=True, exist_ok=True)

        for dataset_type, channel_labels in self.channel_mapping.items():
            relevant_datasets = {k: v for k, v in self.datasets.items() if dataset_type in k}

            if relevant_datasets:
                fig, axs = plt.subplots(len(relevant_datasets), 1, figsize=(15, 5 * len(relevant_datasets)),
                                        sharex=True)
                fig.suptitle(f'Synchronized Data for {dataset_type}')

                if len(relevant_datasets) == 1:
                    axs = [axs]

                for i, (stream_id, df) in enumerate(relevant_datasets.items()):
                    seconds_since_start = (df.index - df.index[0]).total_seconds()

                    # Assuming gaps are represented by NaNs
                    gaps = df.isna().any(axis=1)

                    # Calculate intactness
                    total_data_count = len(df)
                    non_gap_count = total_data_count - gaps.sum()
                    intact_percentage = 100 * non_gap_count / total_data_count

                    for col_index, column in enumerate(df.columns):
                        label = channel_labels[col_index] if col_index < len(channel_labels) else f'Channel {col_index}'
                        axs[i].plot(seconds_since_start, df[column], label=label)

                        self.highlight_gaps(axs[i], df[[column]], gap_threshold=2)

                    axs[i].legend()
                    axs[i].set_ylabel('Value')
                    axs[i].set_title(f'Dataset: {stream_id} - {intact_percentage:.2f}% Data Intact')

                axs[-1].set_xlabel('Time (seconds)')
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])

                plot_filename = data_quality_folder / f"{dataset_type}_synchronized.png"
                plt.savefig(plot_filename)
                plt.close(fig)

    def standardize_timestamps(self):
        """
        Standardizes timestamps across all datasets based on their individual sampling frequencies and a common starting timestamp.
        """
        # Determine the common starting timestamp
        common_start_time = self.start_time

        # Generate and resample datasets to new timestamps
        for stream_id, df in self.datasets.items():
            sfreq = self.calculate_sfreq(stream_id, df)
            if sfreq is None:
                continue  # Skip if sfreq couldn't be calculated

            # Sort DataFrame index to ensure monotonicity
            df.sort_index(inplace=True)

            # Remove any duplicate timestamps
            df = df[~df.index.duplicated(keep='first')]

            new_timestamps = self.generate_timestamps_from_start(common_start_time, sfreq, df)

            # Reindex using 'nearest' method with limit of 1
            self.datasets[stream_id] = df.reindex(new_timestamps, method='nearest', limit=1)

    def generate_timestamps_from_start(self, start_time, sfreq, df):
        """
        Generates a series of timestamps for a given stream from start_time based on the sampling frequency sfreq,
        taking into account the actual data points to handle gaps.
        """
        # Initialize the list of timestamps with the start time
        timestamps = [start_time]

        # Loop over the dataframe to fill in the timestamps considering gaps
        for current_time in df.index[1:]:
            # Calculate the expected next time based on the sampling frequency
            expected_next_time = timestamps[-1] + pd.Timedelta(seconds=1 / sfreq)

            while expected_next_time < current_time:
                # If there's a gap, fill it with expected timestamps based on sfreq
                timestamps.append(expected_next_time)
                expected_next_time += pd.Timedelta(seconds=1 / sfreq)

            # Add the current time from the dataframe
            timestamps.append(current_time)

        return pd.to_datetime(timestamps)

    def process_and_synchronize_data(self):
        # Process files and plot data quality
        self.process_files()
        self.plot_data_quality()

        # self.standardize_timestamps()
        # self.plot_data_quality()

        # Generate spike indices for each stream type
        spike_indices_dict = self.generate_spike_indices_dict()

        # Compare spikes within similar stream types
        common_spikes_within_type = self.compare_spikes_within_type(spike_indices_dict)

        # Refine common spikes across different stream types
        refined_common_spikes = self.compare_spikes_across_types(common_spikes_within_type)

        # Synchronize all datasets based on refined common spikes
        synchronized_datasets = self.synchronize_timestamps(refined_common_spikes, self.datasets)

        # Update self.datasets with synchronized datasets
        self.datasets = synchronized_datasets

        # Save the synchronized datasets
        self.save_datasets()

# Usage example
data_processor = DataProcessor(folder_path='D:/Study Data/tv_gi/session_5/1704828739_10/RawData', NA_0=True)
data_processor.process_and_synchronize_data()
data_processor.plot_synchronized_data()
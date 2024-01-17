import os
import h5py
import pandas as pd
import numpy as np


def inspect_hdf5_file(file_path):
    with h5py.File(file_path, 'r') as file:
        print("Datasets in the file:")
        for name in file.keys():
            print(name)


def clean_align_and_save_bad_data(bad_data_path, good_data_path, aligned_data_path):
    """
    Cleans the bad data by addressing negative timestamps, aligns it with the duration
    and sampling rate of the good file, and writes the cleaned and aligned data to a new file.

    Parameters:
    bad_data_path (str): Path to the HDF5 file with bad data.
    good_data_path (str): Path to the HDF5 file with good data (reference for alignment).
    aligned_data_path (str): Path for the output HDF5 file with cleaned and aligned data.
    """
    # Load data and timestamps from the good file
    with h5py.File(good_data_path, 'r') as good_file:
        good_dataset_name = list(good_file.keys())[0].replace("_timestamps", "")
        good_timestamps = good_file[f'{good_dataset_name}_timestamps'][:]

    # Calculate the duration and sampling rate of the good file
    good_duration = good_timestamps[-1] - good_timestamps[0]
    good_sampling_rate = len(good_timestamps) / good_duration

    # Load data and timestamps from the bad file
    with h5py.File(bad_data_path, 'r') as bad_file:
        bad_dataset_name = list(bad_file.keys())[0].replace("_timestamps", "")
        bad_data = bad_file[bad_dataset_name][:]
        bad_timestamps = bad_file[f'{bad_dataset_name}_timestamps'][:]

    # Remove negative timestamps and corresponding data points from the bad file
    valid_indices = bad_timestamps > 0
    cleaned_bad_data = bad_data[valid_indices]
    cleaned_bad_timestamps = bad_timestamps[valid_indices]

    # Recalculate timestamps for the bad file based on good file's duration and sampling rate
    recalculated_timestamps = np.linspace(good_timestamps[0], good_timestamps[0] + good_duration, len(cleaned_bad_data))

    # Write the cleaned and recalculated data to a new HDF5 file
    with h5py.File(aligned_data_path, 'w') as aligned_file:
        aligned_file.create_dataset(bad_dataset_name, data=cleaned_bad_data)
        aligned_file.create_dataset(f'{bad_dataset_name}_timestamps', data=recalculated_timestamps)


def sanitize_negative_timestamps(bad_data_path, repaired_data_path):
    """
    Sanitizes the bad data by addressing negative timestamps and writes the cleaned data to a new file.

    Parameters:
    bad_data_path (str): Path to the HDF5 file with bad data.
    repaired_data_path (str): Path for the output HDF5 file with sanitized data.
    """
    with h5py.File(bad_data_path, 'r') as bad_file:
        # Determine valid indices from timestamp datasets
        valid_indices = None
        for dataset_name in bad_file.keys():
            if dataset_name.endswith('_timestamps'):
                timestamps = bad_file[dataset_name][:]
                current_valid_indices = timestamps > 0
                if valid_indices is None:
                    valid_indices = current_valid_indices
                else:
                    # Combine valid indices from all timestamp datasets
                    valid_indices = valid_indices & current_valid_indices

        if valid_indices is None:
            raise ValueError("No timestamp datasets found in the file.")

        with h5py.File(repaired_data_path, 'w') as repaired_file:
            for dataset_name in bad_file.keys():
                data = bad_file[dataset_name][:]
                # Check if the data is multi-dimensional and matches the length of valid_indices
                if data.ndim > 1 and data.shape[0] == len(valid_indices):
                    sanitized_data = data[valid_indices, ...]
                elif data.ndim == 1 and len(data) == len(valid_indices):
                    sanitized_data = data[valid_indices]
                else:
                    sanitized_data = data  # Keep data as is if dimensions do not match
                repaired_file.create_dataset(dataset_name, data=sanitized_data)


def rename_h5_datasets(h5_filepath, base_filename, stream_type):
    try:
        with h5py.File(h5_filepath, 'a') as h5_file:
            # Check if the dataset exists and then rename it
            if stream_type in h5_file:
                h5_file.move(stream_type, f"{base_filename}")
            if f"{stream_type}_timestamps" in h5_file:
                h5_file.move(f"{stream_type}", f"{base_filename}_timestamps")
            if f"EDA_timestamps" in h5_file:
                h5_file.move(f"EDA_timestamps", f"{base_filename}_timestamps")
        print(f"Datasets renamed in file: {h5_filepath}")
    except Exception as e:
        print(f"Error processing file {h5_filepath}: {e}")


def process_h5_files_in_folder(folder_path):
    # Iterate over all .h5 files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.h5'):
            # Extract device name and sensor type from the filename
            base_filename = os.path.splitext(filename)[0]
            device_name, stream_type = base_filename.split('_')

            # Full path of the .h5 file
            h5_filepath = os.path.join(folder_path, filename)

            # Rename the datasets within the .h5 file
            rename_h5_datasets(h5_filepath, base_filename, stream_type)



def convert_csv_to_hdf5(csv_folder_path, h5_output_dir):

    for filename in os.listdir(csv_folder_path):
        if filename.endswith('.csv'):
            csv_filepath = os.path.join(csv_folder_path, filename)

            # Read the CSV file
            csv_data = pd.read_csv(csv_filepath, header=None)

            # Extract the starting Unix time and sampling frequency
            start_time = csv_data.iloc[0, 0]
            sampling_frequency = csv_data.iloc[1, 0]
            data = csv_data.iloc[2:].to_numpy()

            # Generate timestamps
            timestamps = start_time + np.arange(data.shape[0]) / sampling_frequency

            # Prepare the output file name based on the CSV file name
            base_filename = os.path.splitext(os.path.basename(csv_filepath))[0]
            h5_filename = f"{base_filename}.h5"
            h5_filepath = os.path.join(h5_output_dir, h5_filename)


            # Create the .h5 file
            with h5py.File(h5_filepath, 'w') as h5_file:
                # Create datasets for data and timestamps
                h5_file.create_dataset(f"{base_filename}", data=data)
                h5_file.create_dataset(f"{base_filename}_timestamps", data=timestamps)


def convert_hdf5_to_csv(file_path):
    try:
        with h5py.File(file_path, 'r') as hf:
            # Iterate over groups/datasets in the file
            for key in hf.keys():
                # Assuming each key in the HDF5 file is a separate dataset
                data = hf[key][:]
                # Convert the data to a pandas DataFrame
                df = pd.DataFrame(data)

                # Derive CSV file name from the HDF5 file name and key
                csv_file_name = os.path.splitext(file_path)[0] + f"_{key}.csv"
                # Save the DataFrame to a CSV file
                df.to_csv(csv_file_name, index=False)
                print(f"Saved {key} to {csv_file_name}")
        return "Conversion successful"
    except Exception as e:
        return f"Error during conversion: {e}"


def analyze_h5_file(file_path):
    results = {}
    with h5py.File(file_path, 'r') as file:
        for dataset_name in file:
            dataset = file[dataset_name][:]
            results[dataset_name] = {
                "shape": dataset.shape,
                "size": dataset.size
            }

            # Check if the dataset is a timestamp dataset and correct it
            if "timestamps" in dataset_name:
                if len(dataset) > 1:
                    # Correct only the timestamps
                    corrected_timestamps = correct_timestamps(dataset)

                    # Calculate the time differences and average sampling frequency
                    time_diffs = np.diff(corrected_timestamps)
                    time_diffs = time_diffs[time_diffs > 0]  # Ignore zero and negative differences
                    avg_sampling_interval = np.mean(time_diffs) if len(time_diffs) > 0 else np.nan
                    avg_sampling_frequency = 1 / avg_sampling_interval if avg_sampling_interval > 0 else 0

                    results[dataset_name].update({
                        "avg_sampling_interval": avg_sampling_interval,
                        "avg_sampling_frequency": avg_sampling_frequency,
                        "corrected_timestamps_shape": corrected_timestamps.shape
                    })
                else:
                    # Handle case where dataset may not have enough data for correction
                    results[dataset_name].update({
                        "avg_sampling_interval": np.nan,
                        "avg_sampling_frequency": 0,
                        "corrected_timestamps_shape": dataset.shape
                    })

    return results


def load_timestamps(file_path, dataset_name):
    """Load timestamps from a specific dataset in an HDF5 file."""
    with h5py.File(file_path, 'r') as hf:
        if dataset_name in hf:
            return hf[f'{dataset_name}_timestamps'][:]
        else:
            print(f"Dataset {dataset_name} not found in {file_path}")
            return None

def compare_timestamps(*args):
    """Compare the first few timestamps from multiple arrays to check synchronization."""
    min_length = min(len(arr) for arr in args if arr is not None)
    for i in range(min(min_length, 10)):  # Compare the first 5 timestamps
        timestamps = [arr[i] for arr in args if arr is not None]
        print(f"Timestamps at index {i}: {timestamps}")

def correct_timestamps(timestamps):
    # Calculate the sampling frequency from the initial set of samples
    initial_sample_count = 256 * 60 * 2  # 2 minutes of data at 256 Hz
    initial_timestamps = timestamps[:initial_sample_count]
    sampling_frequency = calculate_sampling_frequency(initial_timestamps)
    interval = 1.0 / sampling_frequency
    corrected_timestamps = np.array(timestamps, copy=True)
    start_time = corrected_timestamps[0] if not np.isnan(corrected_timestamps[0]) else 0

    return np.array([start_time + i * interval for i in range(len(corrected_timestamps))])


def calculate_sampling_frequency(timestamps):
    # Remove NaN values and calculate differences
    clean_timestamps = timestamps[~np.isnan(timestamps)]
    time_diffs = np.diff(clean_timestamps)

    # Calculate average time difference and convert to frequency
    avg_time_diff = np.mean(time_diffs)
    return 1.0 / avg_time_diff if avg_time_diff != 0 else 0


file_paths = [
    'D:/Study Data/bz_fk/session_1/1700760024_10/RawData/8413C6_GSR.h5'
]


# Load timestamps from each file (replace 'timestamp_dataset_name' with the actual name of the timestamp dataset)
timestamps_data = [load_timestamps(file_path, os.path.splitext(os.path.basename(file_path))[0]) for file_path in file_paths]

# for i,timestamps in enumerate(timestamps_data):
#     timestamps_data[i] = correct_timestamps(timestamps)
#
#
#
# for file_path in file_paths:
#     convert_hdf5_to_csv(file_path)
# #
# Compare timestamps
compare_timestamps(*timestamps_data)
#
# Analyzing each file
for file_path in file_paths:
    print(f"Analyzing {file_path}")
    analysis = analyze_h5_file(file_path)
    print(analysis, '\n')



# convert_hdf5_to_csv('C:/Users/siddh/Documents/StreamSense/1701373662_127079/RawData/MuseS-4646_EEG.h5')
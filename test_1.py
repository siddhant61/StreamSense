import os
import h5py
import pandas as pd
import numpy as np




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

            # Calculate average sampling frequency for timestamp datasets
            if "timestamps" in dataset_name and len(dataset) > 1:
                time_diffs = np.diff(dataset)
                avg_sampling_interval = np.mean(time_diffs)
                avg_sampling_frequency = 1 / avg_sampling_interval if avg_sampling_interval > 0 else 0
                results[dataset_name]["avg_sampling_interval"] = avg_sampling_interval
                results[dataset_name]["avg_sampling_frequency"] = avg_sampling_frequency

    return results

def load_timestamps(file_path, dataset_name):
    """Load timestamps from a specific dataset in an HDF5 file."""
    with h5py.File(file_path, 'r') as hf:
        if dataset_name in hf:
            return hf[dataset_name][:]
        else:
            print(f"Dataset {dataset_name} not found in {file_path}")
            return None

def compare_timestamps(*args):
    """Compare the first few timestamps from multiple arrays to check synchronization."""
    min_length = min(len(arr) for arr in args if arr is not None)
    for i in range(min(min_length, 10)):  # Compare the first 5 timestamps
        timestamps = [arr[i] for arr in args if arr is not None]
        print(f"Timestamps at index {i}: {timestamps}")


file_paths = [
    'C:/Users/siddh/Documents/StreamSense/1700874079_762413/RawData/8413C6_ACC.h5',
    'C:/Users/siddh/Documents/StreamSense/1700874079_762413/RawData/8413C6_BVP.h5',
    'C:/Users/siddh/Documents/StreamSense/1700874079_762413/RawData/8413C6_GSR.h5',
    'C:/Users/siddh/Documents/StreamSense/1700874079_762413/RawData/8413C6_TEMP.h5',
    'C:/Users/siddh/Documents/StreamSense/1700874079_762413/RawData/A4880B_ACC.h5',
    'C:/Users/siddh/Documents/StreamSense/1700874079_762413/RawData/A4880B_BVP.h5',
    'C:/Users/siddh/Documents/StreamSense/1700874079_762413/RawData/A4880B_GSR.h5',
    'C:/Users/siddh/Documents/StreamSense/1700874079_762413/RawData/A4880B_TEMP.h5'
]



# Load timestamps from each file (replace 'timestamp_dataset_name' with the actual name of the timestamp dataset)
timestamps_data = [load_timestamps(file_path, os.path.splitext(os.path.basename(file_path))[0]+'_timestamps') for file_path in file_paths]

# for file_path in file_paths:
#     convert_hdf5_to_csv(file_path)
#
# # Compare timestamps
# compare_timestamps(*timestamps_data)
#
# Analyzing each file
for file_path in file_paths:
    print(f"Analyzing {file_path}")
    analysis = analyze_h5_file(file_path)
    print(analysis, '\n')



# convert_hdf5_to_csv('C:/Users/siddh/Documents/StreamSense/1700869713_17715/RawData/A4880B_TEMP.h5')
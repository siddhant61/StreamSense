import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_sfreq(timestamps):
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


def save_data_to_pickle(files):
    output_folder = Path("C:/Users/siddh/Documents/StreamSense/1692645064_14411/Dataset")
    output_folder.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for file, path in files.items():
        try:
            df = pd.read_csv(path, header=0, dtype={'TimeStamp': str})
            stream_type = file.split("_")[-1].split(".")[0]  # Assuming the file name ends with the stream type

            if stream_type not in datasets:
                datasets[stream_type] = {}

            if stream_type in ["EEG", "ACC", "GYRO", "TEMP", "BVP", "GSR"]:
                sfreq = calculate_sfreq(df['TimeStamp'])
                if stream_type == "EEG":
                    df = df.drop(['TimeStamp', 'AUX'], axis = 1)
                datasets[stream_type][file] = {'data': df, 'sfreq': sfreq}
            else:
                datasets[stream_type][file] = {'data': df}
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Saving the dataset as a pickle file for each stream type
    for stream_type, dataset in datasets.items():
        pickle_file_path = f"{output_folder}/{stream_type}_dataset.pkl"
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(dataset, f)


files  = {}
path = "C:/Users/siddh/Documents/StreamSense/1692645064_14411/RAW_Data/"
file_names = os.listdir(path)
for file in file_names:
    files[file.replace('.csv', '')] = path+file

save_data_to_pickle(files)
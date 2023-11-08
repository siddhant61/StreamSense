import pylsl
import time
import threading
import matplotlib.pyplot as plt

DURATION = 10  # Duration for which each thread collects data

def get_sampling_frequency(stream, sampling_rates):
    inlet = pylsl.StreamInlet(stream)
    start_time = pylsl.local_clock()
    num_samples = 0
    while pylsl.local_clock() - start_time < DURATION:
        sample, timestamp = inlet.pull_sample(0.0)
        if sample is not None:
            num_samples += 1
    sampling_frequency = num_samples / DURATION
    sampling_rates[stream.name()] = sampling_frequency
    print(f"Stream: {stream.name()}, Frequency: {sampling_frequency}")


streams = pylsl.resolve_streams()
if not streams:
    print("No streams found!")
else:
    sampling_rates = {}
    threads = []

    for stream in streams:
        print(f"Found stream: {stream.name()}")
        thread = threading.Thread(target=get_sampling_frequency, args=(stream, sampling_rates))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()  # Wait for all threads to finish

    # Print the final sampling rates
    print("Final Sampling Rates:", sampling_rates)

    # Plotting
    keys = list(sampling_rates.keys())
    values = list(sampling_rates.values())

    if values:
        plt.bar(keys, values)
        plt.ylabel('Sampling Frequency')
        plt.title('LSL Stream Sampling Frequencies')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No values to plot!")
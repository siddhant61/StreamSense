import pylsl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import time
import math
from collections import defaultdict

# Configuration
DURATION = 2
UPDATE_INTERVAL = 2
EXPECTED_RATES = {"EEG": 256, "BVP": 64, "PPG": 64, "GYRO": 50, "ACC": 32, "GSR": 4, "TEMP": 4, "TAG": 0.1}

def calculate_sampling_frequency(stream, sampling_rates):
    inlet = pylsl.StreamInlet(stream)
    while True:
        start_time = pylsl.local_clock()
        num_samples = collect_samples(inlet, start_time)
        update_sampling_rate(sampling_rates, stream, num_samples / DURATION)
        time.sleep(DURATION)

def collect_samples(inlet, start_time):
    num_samples = 0
    while pylsl.local_clock() - start_time < DURATION:
        if inlet.pull_sample(0.0)[0]:
            num_samples += 1
    return num_samples

def update_sampling_rate(sampling_rates, stream, freq):
    sampling_rates[stream.type()][stream.name()] = freq/DURATION


def update_plot(i, ax, sampling_rates, figsize=(8, 4)):
    try:
        ax.clear()
        ax.figure.set_size_inches(*figsize)
        ax.set_title("Active LSL Streams", fontsize=20, fontweight='bold', loc='left')

        # Turn off the axis lines and labels
        ax.axis('off')

        # Define the number of streams per row and calculate the number of rows needed
        streams_per_row = 3
        num_rows = math.ceil(len(sampling_rates) / streams_per_row)
        num_columns = min(len(sampling_rates), streams_per_row)

        # Calculate the column width and row height based on the number of columns and rows
        column_width = 1 / num_columns
        row_height = 1 / num_rows

        # Set an initial font size
        header_font_size = 12  # Adjust as needed
        stream_font_size = 10  # Adjust as needed
        vertical_padding = 0.02  # Space between rows

        # Initialize the starting y position (top of the plot)
        current_y = 1 - vertical_padding

        # Iterate over the stream types and place them in the plot
        for row in range(num_rows):
            for col in range(num_columns):
                index = row * streams_per_row + col
                if index >= len(sampling_rates):  # Check if we've placed all stream types
                    break
                stream_type = sorted(sampling_rates.keys())[index]
                streams = sampling_rates[stream_type]

                # Position the stream type header
                x_position = col * column_width
                ax.text(x_position, current_y, f"{stream_type} Streams",
                        fontsize=header_font_size, fontweight='bold', transform=ax.transAxes, ha='left', va='top')

                # Position each stream detail below the header
                for stream_index, (name, freq) in enumerate(sorted(streams.items()), start=1):
                    expected_freq = EXPECTED_RATES.get(stream_type, 0)/2
                    color = 'green' if freq >= expected_freq else 'red'
                    vertical_padding = 0.05
                    stream_y_position = current_y - (stream_index * vertical_padding)
                    ax.text(x_position, stream_y_position, f"{name}: {freq:.2f} Hz",
                            fontsize=stream_font_size, color=color, transform=ax.transAxes, ha='left', va='top')

            # Update the y position for the next row
            current_y -= row_height

        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
        plt.tight_layout()
    except Exception as e:
        print("An error occurred while updating the plot:", str(e))

def initialize_streams():
    streams = pylsl.resolve_streams()
    if not streams:
        print("No streams found!")
        return None
    sampling_rates = defaultdict(dict)
    return streams, sampling_rates

def main():
    streams, sampling_rates = initialize_streams()
    if not streams:
        return

    fig, ax = plt.subplots(figsize=(4, 4))

    for stream in streams:
        threading.Thread(target=calculate_sampling_frequency, args=(stream, sampling_rates), daemon=True).start()

    ani = animation.FuncAnimation(fig, update_plot, fargs=(ax, sampling_rates), interval=1000 * UPDATE_INTERVAL)
    plt.show()

if __name__ == "__main__":
    main()

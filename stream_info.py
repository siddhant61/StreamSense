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
        ax.figure.set_size_inches(*figsize)  # Set the figure size
        ax.set_title("Active LSL Streams", fontsize=16)

        # Determine the grid size for the matrix layout
        num_stream_types = len(sampling_rates)
        num_columns = math.ceil(math.sqrt(num_stream_types))  # Square root rounded up for columns
        num_rows = math.ceil(num_stream_types / num_columns)  # Number of rows needed

        # Set the aspect of the plot box to auto to fill the figure
        ax.set_aspect('auto')

        # Set the plot limits
        ax.set_xlim(0, num_columns)
        ax.set_ylim(0, num_rows)
        ax.invert_yaxis()  # Invert the y-axis so the top-left is (0,0)
        ax.axis('off')

        # Calculate the height and width of each block in the matrix
        block_width = 1 / num_columns
        block_height = 1 / num_rows

        # Padding term for the gaps between columns and rows (as a fraction of block size)
        column_padding = block_width * 0.1
        row_padding = block_height * 0.1

        # Loop through each stream type and plot it in the matrix
        for index, stream_type in enumerate(sampling_rates.keys()):
            streams_info = sampling_rates[stream_type]  # This should be a dictionary {stream_name: frequency}

            # Calculate the row and column for the current stream type
            col = index % num_columns
            row = index // num_columns

            # Calculate the position for the stream type title
            x_position = col + column_padding  # Adjust for padding
            y_position = row + row_padding  # Adjust for padding

            # Dynamically adjust the font size based on the block size
            header_font_size = max(min((block_width - 2 * column_padding) * figsize[0], (block_height - 2 * row_padding) * figsize[1]), 10)
            stream_font_size = max(header_font_size * 0.5, 8)  # Ensure stream font is smaller than header

            # Stream type title
            ax.text(x_position, y_position, f"{stream_type}:", ha='left', va='top', fontsize=header_font_size, fontweight='bold')

            # Adjust the vertical space per stream within the padded area
            stream_space = (block_height - 2 * row_padding) * 0.99 / max(len(streams_info), 1)  # Avoid division by zero
            # List each stream below the stream type heading
            for stream_index, (name, freq) in enumerate(streams_info.items(), start=1):
                expected_freq = EXPECTED_RATES.get(stream_type, 0)
                color = 'green' if freq >= expected_freq / 2 else 'red'
                # Update y_position for each stream
                stream_y_position = y_position + (stream_index * stream_space)
                ax.text(x_position, stream_y_position, f"{stream_index}. {name}: {freq:.2f} Hz", ha='left', va='top', fontsize=stream_font_size, color=color)

        # Adjust the layout to ensure no text is cut off and the matrix is well-distributed
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0, wspace=0, hspace=0)
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

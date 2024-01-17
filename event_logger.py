import pylsl
import argparse
import os

def log_event(event, start_time, file_name):
    current_time = pylsl.local_clock()
    elapsed_time = current_time - start_time
    log_entry = f"{elapsed_time:.6f}: {event}\n"

    with open(file_name, "a") as file:
        file.write(log_entry)

    print(f"Logged event: {log_entry.strip()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event Logger")
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder path for logging events')
    parser.add_argument('--start_time', type=str, required=True, help='Synchronized start time')
    args = parser.parse_args()

    start_time = float(args.start_time)
    file_name = os.path.join(args.output_folder, "event_log.txt")

    print("Event Logger started. Type your events here:")

    while True:
        event = input("> ")
        if event.lower() == 'exit':
            break
        log_event(event, start_time, file_name)

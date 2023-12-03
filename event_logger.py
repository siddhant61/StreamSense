import pylsl
import time


# Function to log an event with a timestamp
def log_event(event, start_time, file_name="event_log.txt"):
    # Get the current time from pylsl's clock
    current_time = pylsl.local_clock()

    # Calculate elapsed time since the experiment started
    elapsed_time = current_time - start_time

    # Create the log entry
    log_entry = f"{elapsed_time:.6f}: {event}\n"

    # Write the log entry to the file
    with open(file_name, "a") as file:
        file.write(log_entry)

    print(f"Logged event: {log_entry.strip()}")


# Record the start time of the experiment
start_time = pylsl.local_clock()

# Example of how to use the log_event function
while True:
    event = input("Enter event (or type 'exit' to stop): ")
    if event.lower() == 'exit':
        break
    log_event(event, start_time)

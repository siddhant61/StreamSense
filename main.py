import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import multiprocessing
import argparse
from datetime import datetime
from pathlib import Path
import threading
import time
import userpaths
import wmi
import logging
from pylsl import local_clock

synchronized_start_time = local_clock()

# Setup logging
logger = logging.getLogger("main.py")
logger.setLevel(logging.CRITICAL)
fh = logging.FileHandler("Logs/main.log")
fh.setLevel(logging.CRITICAL)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

from recorder.stream_recorder import StreamRecorder
from helper.find_devices import FindDevices
from streamer.stream_muse import StreamMuse
from streamer.stream_e4 import StreamE4
from viewer.view_streams import ViewStreams
from experiments.visual_oddball import VisualOddball


@dataclass
class AppState:
    """Container for mutable application state managed by the CLI."""

    root_output_folder: Optional[str] = None
    root_output_path: Optional[Path] = None
    recorder_thread: Optional[threading.Thread] = None
    recorder: Optional[StreamRecorder] = None
    # Note: Removed muse_threads and e4_threads - streamers already use multiprocessing.Process internally
    # Using Thread to wrap Process was an anti-pattern identified in Phase 3 concurrency analysis
    muse_streamers: Dict[str, StreamMuse] = field(default_factory=dict)
    e4_streamers: Dict[str, StreamE4] = field(default_factory=dict)

    def ensure_output_folder(self) -> str:
        """Ensure that the root output directory exists and return its string path."""

        if not self.root_output_folder:
            documents = userpaths.get_my_documents().replace("\\", "/")
            folder = f"{documents}/StreamSense/{str(datetime.today().timestamp()).replace('.', '_')}"
            path = Path(folder)
            path.mkdir(parents=True, exist_ok=True)
            self.root_output_folder = folder
            self.root_output_path = path
        return self.root_output_folder

    def reset_output_folder(self) -> None:
        """Clear cached output folder metadata so a new session can be created."""

        self.root_output_folder = None
        self.root_output_path = None


def connect_muse_devices(state: AppState):
    """Discover Muse devices and launch streaming threads, updating the shared state."""

    muse_reg: Dict[str, str] = {}
    devices = FindDevices()
    muses, com_ports = devices.find_muses_with_ports()

    print(f"{len(com_ports)} serial port(s) available: {com_ports}\n")
    logger.info(f"{len(com_ports)} free serial port(s) detected.\n")

    if len(com_ports) != 0:
        if len(com_ports) > len(muses):
            n = len(muses)
        else:
            n = len(com_ports)
        if len(muses) != 0:
            for i in range(n):
                key = muses[i][0]
                value = muses[i][1]
                muse_reg[key] = value
            print(f"{len(muse_reg)} Muse device(s) registered.\n")
            logger.info(f"{len(muse_reg)} Muse device(s) registered.\n")
        else:
            print("No Muse devices found.\n")
            logger.info("No Muse devices found.\n")

        if len(muse_reg) != 0:
            state.muse_streamers.clear()
            for i in range(len(muse_reg)):
                streamer_key = f"muse_streamer_{i + 1}"
                streamer = StreamMuse(
                    list(muse_reg.keys())[i],
                    list(muse_reg.values())[i],
                    com_ports[n - i - 1],
                    state.ensure_output_folder(),
                    synchronized_start_time,
                )
                state.muse_streamers[streamer_key] = streamer

            if len(state.muse_streamers) != 0:
                streamers = list(state.muse_streamers.values())
                # Call start_streaming directly - no need for Thread wrapper
                # Each StreamMuse creates its own Process internally
                for i, streamer in enumerate(streamers):
                    streamer.start_streaming()
                    streamer.connected_event.wait()
                    if i == 0:
                        time.sleep(5)  # Delay for 5 seconds after the first device

                print(f"{len(state.muse_streamers)} Muse streaming process(es) running.\n")
                logger.info(f"{len(state.muse_streamers)} Muse streaming process(es) running.\n")
            else:
                print("No Muse streaming processes running.\n")
                logger.info("No Muse streaming processes running.\n")
    return muse_reg, state.muse_streamers

def connect_e4_devices(state: AppState):
    """Discover E4 devices and start streaming threads, updating the shared state."""

    e4_reg: Dict[str, str] = {}
    devices = FindDevices()

    e4s = devices.find_empatica()

    e4_server = False

    while not (e4_server):
        print('Checking for E4 Server Process.\n')
        logger.info('Checking for E4 Server Process.\n')
        f = wmi.WMI()

        flag = 0

        # Iterating through all the running processes
        for process in f.Win32_Process():
            if "EmpaticaBLEServer.exe" == process.Name:
                print("E4 Server is running. Finding E4 devices.\n")
                logger.info("E4 Server is running. Finding E4 devices.\n")
                e4_server = True
                flag = 1
                break

        if flag == 0:
            e4_server = False
            print("E4 Server is not running. Please start the server first.\n")
            logger.info("E4 Server is not running. Please start the server first.\n")
            time.sleep(10)

    if len(e4s) != 0:
        for i in range(len(e4s)):
            key = str(e4s[i])
            value = str(e4s[i])
            e4_reg[key] = value
    else:
        print("No E4 devices found.")
        logger.info("No E4 devices found.")

    if len(e4_reg) != 0:
        state.e4_streamers.clear()
        output_path = state.root_output_path or Path(state.ensure_output_folder())
        for i in range(len(e4_reg)):
            streamer_key = f"e4_streamer_{i + 1}"
            streamer = StreamE4(list(e4_reg.values())[i], output_path, synchronized_start_time)
            state.e4_streamers[streamer_key] = streamer

        print(f"{len(e4_reg)} E4 device(s) registered.\n")
        logger.info(f"{len(e4_reg)} E4 device(s) registered.\n")

        if len(state.e4_streamers) != 0:
            streamers = list(state.e4_streamers.values())
            # Call start_streaming directly - no need for Thread wrapper
            # Each StreamE4 creates its own Process internally
            for i, streamer in enumerate(streamers):
                streamer.start_streaming()
                streamer.connected_event.wait()

        print(f"{len(state.e4_streamers)} E4 streaming process(es) running.\n")
        logger.info(f"{len(state.e4_streamers)} E4 streaming process(es) running.\n")
    else:
        print("No E4 devices registered.\n")
        print("No E4 streaming processes running.\n")
        logger.info("No E4 streaming processes running.\n")
    return e4_reg, state.e4_streamers


def log_and_print(message, logger):
    """Log and print the given message."""
    print(message)
    logger.info(message)


def start_recording(state: AppState) -> None:
    """Launch the recording thread and remember it in the application state."""

    state.ensure_output_folder()
    recorder = StreamRecorder(state.root_output_folder)
    thread = threading.Thread(target=recorder.record_streams)
    thread.start()
    if not recorder.started_event.wait(timeout=10):
        warning = "Recorder failed to report readiness within 10 seconds."
        print(f"{warning}\n")
        logger.critical(warning)
    state.recorder = recorder
    state.recorder_thread = thread


def start_visual_oddball(state: AppState) -> None:
    """Kick off the visual oddball experiment using the configured output folder."""

    state.ensure_output_folder()
    exp = VisualOddball(state.root_output_folder)
    sequence = (10, 3)
    exp.start_oddball(sequence)


def start_event_logger(state: AppState) -> None:
    """Open the event logger console in a new process."""

    state.ensure_output_folder()
    if state.root_output_path:
        start_event_logger_process(str(state.root_output_path))


def stop_all_streams(state: AppState) -> None:
    """Stop all active streamers and the recorder, clearing the stored state."""

    if state.recorder and state.recorder_thread:
        logger.info("Current state saved.")
        state.recorder.stop()
        state.recorder_thread.join()
        state.recorder = None
        state.recorder_thread = None

    # Stop E4 streamers - no need to join threads as stop_streaming() handles process cleanup
    if state.e4_streamers:
        for e4_instance in state.e4_streamers.values():
            e4_instance.stop_streaming()
        state.e4_streamers.clear()

    # Stop Muse streamers - no need to join threads as stop_streaming() handles process cleanup
    if state.muse_streamers:
        for muse_instance in state.muse_streamers.values():
            muse_instance.stop_streaming()
        state.muse_streamers.clear()

    state.reset_output_folder()

def display_menu():
    """Display the main menu options."""
    menu_options = \
        """
        The following options are available:
        (1) Connect and stream Muse devices.
        (2) View all the active LSL Streams.
        (3) Connect and stream E4 devices.
        (4) Start recording all the streams.
        (5) Run the visual oddball paradigm.
        (6) Start the event logger console.
        (7) Stop all the active LSL streams.
        """
    print(menu_options)

def display_streams_menu():
    """Display the streams menu options."""
    menu_options = \
        """
        The following options are available:
        (1) View all the active EEG Streams.
        (2) View all the active ACC Streams.
        (3) View all the active BVP Streams.
        (4) View all the active GSR Streams.
        (5) View all the active PPG Streams.
        (6) Go back to the main menu.
        """
    print(menu_options)

def start_event_logger_process(output_folder):
    try:
        script_path = os.path.join(os.path.dirname(sys.argv[0]), 'event_logger.py')
        command = [sys.executable, script_path, '--output_folder', output_folder, '--start_time', str(synchronized_start_time)]

        # Open a new console window and run the event_logger script
        subprocess.Popen(["start", "cmd", "/k"] + command, shell=True)
    except Exception as e:
        print(e)


def run_view_menu():
    """Interactive loop for viewing different categories of streams."""

    while True:
        display_streams_menu()
        view_choice = input("> ").strip()
        if view_choice == "6":
            break
        try:
            view_choice_int = int(view_choice)
        except ValueError:
            print("Invalid Input. Please choose within (1-6)\n")
            continue

        if 1 <= view_choice_int <= 5:
            viewer = ViewStreams()
            viewer.start_viewing(view_choice_int)
        else:
            print("Invalid Input. Please choose within (1-6)\n")


def run_menu_loop(state: AppState):
    """Interactive loop for the main command menu."""

    while True:
        display_menu()
        user_input = input("> ").strip()
        try:
            choice = int(user_input)
        except ValueError:
            print("Invalid Input. Please choose within (1-7)\n")
            continue

        if choice == 1:
            connect_muse_devices(state)
        elif choice == 2:
            run_view_menu()
        elif choice == 3:
            connect_e4_devices(state)
        elif choice == 4:
            start_recording(state)
        elif choice == 5:
            start_visual_oddball(state)
        elif choice == 6:
            start_event_logger(state)
        elif choice == 7:
            stop_all_streams(state)
        else:
            print("Invalid Input. Please choose within (1-7)\n")

# def get_user_choice():
#     """Get a valid user choice from the menu."""
#     while True:
#         try:
#             choice = int(input("Enter your choice: "))
#             if 1 <= choice <= 5:
#                 return choice
#             else:
#                 print("Invalid Input. Please choose within (1-5)\n")
#         except ValueError:
#             print("Please enter a valid number.\n")

if __name__ == '__main__':

    multiprocessing.freeze_support()
    state = AppState()
    parser = argparse.ArgumentParser(description='Command-line options for the script.')
    parser.add_argument('--command', choices=['menu', 'stream', 'view', 'record', 'oddball', 'logger', 'stop'],
                        default='menu',
                        help='The command to execute. If no command is provided, the default is "menu".')
    parser.add_argument('--dev', choices=['muse', 'e4'],
                        help='The device to stream. This option is used with the "stream" command.')
    parser.add_argument('--data', choices=['eeg', 'bvp', 'acc', 'gsr', 'ppg'],
                        help='The data stream to view. This option is used with the "view" command.')

    print("Type 'help' to display the command options or 'exit' to quit.")

    while True:
        try:
            user_input = input("> ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered == "exit":
            break
        if lowered == "help":
            parser.print_help()
            continue

        try:
            args = parser.parse_args(user_input.split())
        except SystemExit:
            print("usage: [-h] {menu,stream --dev {muse,e4}, view --data {eeg,bvp,acc,gsr,ppg}, record, oddball, logger, stop}")
            continue

        if args.command == 'menu':
            run_menu_loop(state)
        elif args.command == 'stream':
            if args.dev == 'muse':
                connect_muse_devices(state)
            elif args.dev == 'e4':
                connect_e4_devices(state)
            else:
                print("Please specify --dev {muse,e4} for the stream command.")
        elif args.command == 'view':
            if args.data:
                data_map = {'eeg': 1, 'acc': 2, 'bvp': 3, 'gsr': 4, 'ppg': 5}
                viewer = ViewStreams()
                viewer.start_viewing(data_map[args.data])
            else:
                run_view_menu()
        elif args.command == 'record':
            start_recording(state)
        elif args.command == 'oddball':
            start_visual_oddball(state)
        elif args.command == 'logger':
            start_event_logger(state)
        elif args.command == 'stop':
            stop_all_streams(state)


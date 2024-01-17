import asyncio
import os
import subprocess
import sys
from queue import Queue

active_record = None
active_e4 = []
active_muse = []
muse_threads = {}
e4_threads = {}
muse_streamers = {}
e4_streamers = {}
global save_data_thread
save_data_thread = None
root_output_folder = None
root_output_folder_path= None
global stop_signal
stop_signal = False
import multiprocessing
import argparse
from datetime import datetime
from pathlib import Path
import threading
import time
import userpaths
import serial
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


def connect_muse_devices(root_output_folder):
    muse_reg = {}
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
            for i in range(len(muse_reg)):
                key = f"muse_streamer_{i + 1}"
                value = StreamMuse(list(muse_reg.keys())[i], list(muse_reg.values())[i], com_ports[n-i-1], root_output_folder, synchronized_start_time)
                muse_streamers[key] = value
                key = f"thread_{i + 1}"
                value = threading.Thread(target=list(muse_streamers.values())[i].start_streaming)
                muse_threads[key] = value

            if len(muse_threads) != 0:
                for i in range(len(muse_threads)):
                    list(muse_threads.values())[i].start()
                    list(muse_streamers.values())[i].connected_event.wait()
                    if i == 0:
                        time.sleep(5)  # Delay for 5 seconds after the first device

                print(f"{len(muse_threads)} Muse streaming thread(s) running.\n")
                logger.info(f"{len(muse_threads)} Muse streaming thread(s) running.\n")
            else:
                print("No Muse streaming threads running.\n")
                logger.info("No Muse streaming threads running.\n")
    return muse_reg, muse_streamers, muse_threads

def connect_e4_devices(root_output_folder):
    e4_reg = {}
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
        for i in range(len(e4_reg)):
            key = f"e4_streamer_{i + 1}"
            value = StreamE4(list(e4_reg.values())[i], root_output_folder, synchronized_start_time)
            e4_streamers[key] = value

            key = f"thread_{i + 1}"
            value = threading.Thread(target=list(e4_streamers.values())[i].start_streaming)
            e4_threads[key] = value
        print(f"{len(e4_reg)} E4 device(s) registered.\n")
        logger.info(f"{len(e4_reg)} E4 device(s) registered.\n")

        if len(e4_threads) != 0:
            for i in range(len(e4_threads)):
                list(e4_threads.values())[i].start()
                list(e4_streamers.values())[i].connected_event.wait()

        print(f"{len(e4_threads)} E4 streaming thread(s) running.\n")
        logger.info(f"{len(e4_threads)} E4 streaming thread(s) running.\n")
    else:
        print("No E4 devices registered.\n")
        print("No E4 streaming threads running.\n")
        logger.info("No E4 streaming threads running.\n")
    return e4_reg, e4_streamers, e4_threads

def log_and_print(message, logger):
    """Log and print the given message."""
    print(message)
    logger.info(message)

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
    flag = 1
    parser = argparse.ArgumentParser(description='Command-line options for the script.')
    parser.add_argument('--command', choices=['menu', 'stream', 'view', 'record', 'oddball', 'logger', 'stop'],
                        default='menu',
                        help='The command to execute. If no command is provided, the default is "menu".')
    parser.add_argument('--dev', choices=['muse', 'e4'],
                        help='The device to stream. This option is used with the "stream" command.')
    parser.add_argument('--data', choices=['eeg', 'bvp', 'acc', 'gsr', 'ppg'],
                        help='The data stream to view. This option is used with the "view" command.')

    print("Type 'help' to display the command options or 'exit' to quit.")
    args = parser.parse_args()

    while True:
        try:
            user_input = input("> ").strip()

            if user_input.lower() == "exit":
                break

            elif user_input.lower() == "help":
                # Display the menu
                parser.print_help()

            else:
                # Execute the command
                args = parser.parse_args(user_input.split())
                if args.command == 'menu':
                    while True:
                        display_menu()
                        user_input = input("> ").strip()

                        if int(user_input) == 1:
                            # Create a root directory for saving data and oddball results only if not already created
                            if not root_output_folder:
                                root_output_folder = userpaths.get_my_documents().replace("\\",
                                                                                          "/") + f"/StreamSense/{str(datetime.today().timestamp()).replace('.', '_')}"
                                root_output_folder_path = Path(root_output_folder)
                                root_output_folder_path.mkdir(parents=True, exist_ok=True)
                                if flag == 1:
                                    flag = 0

                            muse_reg, muse_streamers, muse_threads = connect_muse_devices(root_output_folder)
                            for thread in muse_threads.values():
                                active_muse.append(thread)
                            pass

                        elif int(user_input) == 2:
                            while True:
                                display_streams_menu()
                                view_choice = input("> ").strip()
                                view_choice = int(view_choice)

                                if view_choice < 6 and view_choice > 0:
                                    viewer = ViewStreams()
                                    viewer.start_viewing(view_choice, )
                                    pass
                                elif view_choice == 6:
                                    break
                                else:
                                    print("Invalid Input. Please choose within (1-5)\n")
                                    pass

                        elif int(user_input) == 3:
                            # Create a root directory for saving data and oddball results only if not already created
                            if not root_output_folder:
                                root_output_folder = userpaths.get_my_documents().replace("\\",
                                                                                          "/") + f"/StreamSense/{str(datetime.today().timestamp()).replace('.', '_')}"
                                root_output_folder_path = Path(root_output_folder)
                                root_output_folder_path.mkdir(parents=True, exist_ok=True)
                                if flag == 1:
                                    flag = 0

                            e4_reg, e4_streamers, e4_threads = connect_e4_devices(root_output_folder_path)
                            for thread in e4_threads.values():
                                active_e4.append(thread)
                            pass

                        elif int(user_input) == 4:
                            # Create a root directory for saving data and oddball results only if not already created
                            if not root_output_folder:
                                root_output_folder = userpaths.get_my_documents().replace("\\\\",
                                                                                          "/") + f"/StreamSense/{str(datetime.today().timestamp()).replace('.', '_')}"
                                root_output_folder_path = Path(root_output_folder)
                                root_output_folder_path.mkdir(parents=True, exist_ok=True)
                                if flag == 1:
                                    flag = 0

                            data_recorder = StreamRecorder(root_output_folder)
                            save_data_thread = threading.Thread(
                                target=data_recorder.record_streams)
                            save_data_thread.start()
                            time.sleep(5)
                            active_record = save_data_thread

                            pass

                        elif int(user_input) == 5:
                            # Create a root directory for saving data and oddball results only if not already created
                            if not root_output_folder:
                                root_output_folder = userpaths.get_my_documents().replace("\\",
                                                                                          "/") + f"/StreamSense/{str(datetime.today().timestamp()).replace('.', '_')}"
                                root_output_folder_path = Path(root_output_folder)
                                root_output_folder_path.mkdir(parents=True, exist_ok=True)
                                if flag == 1:
                                    flag = 0
                            exp = VisualOddball(root_output_folder)
                            sequence = (10, 3)
                            exp.start_oddball(sequence)
                            pass

                        elif int(user_input) == 6:
                            # Create a root directory for saving data and oddball results only if not already created
                            if not root_output_folder:
                                root_output_folder = userpaths.get_my_documents().replace("\\",
                                                                                          "/") + f"/StreamSense/{str(datetime.today().timestamp()).replace('.', '_')}"
                                root_output_folder_path = Path(root_output_folder)
                                root_output_folder_path.mkdir(parents=True, exist_ok=True)
                                if flag == 1:
                                    flag = 0
                            start_event_logger_process(root_output_folder_path)

                        elif int(user_input) == 7:
                            stop_signal = True
                            if active_record:
                                logger.info("Current state saved.")
                                data_recorder.stop()
                                active_record.join()

                            if active_e4:
                                # Stop the streaming in the stream_e4 script
                                for e4_instance in e4_streamers.values():
                                    e4_instance.stop_streaming()
                                # Join the E4 threads to ensure they stop gracefully
                                for thread in active_e4:
                                    thread.join()

                            if active_muse:
                                # Stop the streaming in the stream_muse script
                                for muse_instance in muse_streamers.values():
                                    muse_instance.stop_streaming()
                                # Join the Muse threads to ensure they stop gracefully
                                for thread in active_muse:
                                    thread.join()

                            root_output_folder = None
                            pass

                elif args.command == 'stream':
                    if args.dev == 'muse':
                        # Create a root directory for saving data and oddball results only if not already created
                        if not root_output_folder:
                            root_output_folder = userpaths.get_my_documents().replace("\\",
                                                                                      "/") + f"/StreamSense/{str(datetime.today().timestamp()).replace('.', '_')}"
                            root_output_folder_path = Path(root_output_folder)
                            root_output_folder_path.mkdir(parents=True, exist_ok=True)
                            if flag == 1:
                                flag = 0

                        muse_reg, muse_streamers, muse_threads = connect_muse_devices(root_output_folder)
                        for thread in muse_threads.values():
                            active_muse.append(thread)
                        pass

                    elif args.dev == 'e4':
                        # Create a root directory for saving data and oddball results only if not already created
                        if not root_output_folder:
                            root_output_folder = userpaths.get_my_documents().replace("\\",
                                                                                      "/") + f"/StreamSense/{str(datetime.today().timestamp()).replace('.', '_')}"
                            root_output_folder_path = Path(root_output_folder)
                            root_output_folder_path.mkdir(parents=True, exist_ok=True)
                            if flag == 1:
                                flag = 0

                        e4_reg, e4_streamers, e4_threads = connect_e4_devices(root_output_folder_path)
                        for thread in e4_threads.values():
                            active_e4.append(thread)
                        pass

                elif args.command == 'view':
                    if args.data == 'eeg':
                        viewer = ViewStreams()
                        viewer.start_viewing(1, )
                        pass

                    elif args.data == 'acc':
                        viewer = ViewStreams()
                        viewer.start_viewing(2, )
                        pass

                    elif args.data == 'bvp':
                        viewer = ViewStreams()
                        viewer.start_viewing(3, )
                        pass

                    elif args.data == 'gsr':
                        viewer = ViewStreams()
                        viewer.start_viewing(4, )
                        pass
                    elif args.data == 'ppg':
                        viewer = ViewStreams()
                        viewer.start_viewing(5, )
                        pass

                elif args.command == 'record':
                    # Create a root directory for saving data and oddball results only if not already created
                    if not root_output_folder:
                        root_output_folder = userpaths.get_my_documents().replace("\\\\",
                                                                                  "/") + f"/StreamSense/{str(datetime.today().timestamp()).replace('.', '_')}"
                        root_output_folder_path = Path(root_output_folder)
                        root_output_folder_path.mkdir(parents=True, exist_ok=True)
                        if flag == 1:
                            flag = 0
                    data_recorder = StreamRecorder(root_output_folder)
                    save_data_thread = threading.Thread(target=data_recorder.record_streams)
                    save_data_thread.start()
                    time.sleep(5)
                    active_record = save_data_thread

                    pass

                elif args.command == 'oddball':
                    # Create a root directory for saving data and oddball results only if not already created
                    if not root_output_folder:
                        root_output_folder = userpaths.get_my_documents().replace("\\",
                                                                                  "/") + f"/StreamSense/{str(datetime.today().timestamp()).replace('.', '_')}"
                        root_output_folder_path = Path(root_output_folder)
                        root_output_folder_path.mkdir(parents=True, exist_ok=True)
                        if flag == 1:
                            flag = 0
                    exp = VisualOddball(root_output_folder)
                    sequence = (10, 3)
                    exp.start_oddball(sequence)
                    print(time.time())
                    pass

                elif args.command == 'stop':
                    stop_signal = True
                    if active_record:
                        logger.info("Current state saved.")
                        data_recorder.stop()
                        active_record.join()

                    if active_e4:
                        # Stop the streaming in the stream_e4 script
                        for e4_instance in e4_streamers.values():
                            e4_instance.stop_streaming()
                        # Join the E4 threads to ensure they stop gracefully
                        for thread in active_e4:
                            thread.join()

                    if active_muse:
                        # Stop the streaming in the stream_muse script
                        for muse_instance in muse_streamers.values():
                            muse_instance.stop_streaming()
                        # Join the Muse threads to ensure they stop gracefully
                        for thread in active_muse:
                            thread.join()

                    root_output_folder = None
                    pass

                elif args.command == 'logger':
                    # Create a root directory for saving data and oddball results only if not already created
                    if not root_output_folder:
                        root_output_folder = userpaths.get_my_documents().replace("\\",
                                                                                  "/") + f"/StreamSense/{str(datetime.today().timestamp()).replace('.', '_')}"
                        root_output_folder_path = Path(root_output_folder)
                        root_output_folder_path.mkdir(parents=True, exist_ok=True)
                        if flag == 1:
                            flag = 0
                    start_event_logger_process(root_output_folder_path)

                elif args.command == 'exit':
                    exit()
        except:
            print("usage:[-h] {menu,stream[--dev {muse,e4}],view[--data {eeg,bvp,acc,gsr}],record,oddball, logger, stop}")
            pass
from queue import Queue

from helper.e4_helper import EmpaticaE4

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
import multiprocessing as mp
import argparse
from datetime import datetime
from pathlib import Path
import threading
import time
import userpaths
import serial
import wmi
import logging
from recorder.stream_recorder import StreamRecorder
from helper.find_devices import FindDevices
from streamer.stream_muse import StreamMuse
from streamer.stream_e4 import StreamE4
from viewer.view_streams import ViewStreams
from experiments.visual_oddball import VisualOddball
import warnings
warnings.filterwarnings("ignore")

main_logger = logging.getLogger(__name__)
synchronized_start_time = time.time()

def _setup_logger(root_output_folder):
    """Set up logger to write logs to a file."""
    log_file_path = root_output_folder / "Logs" / "main_logger.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    main_logger = logging.getLogger("main_logger")
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('[%(levelname)s]: %(message)s')
    file_handler.setFormatter(formatter)
    main_logger.addHandler(file_handler)
    main_logger.setLevel(logging.DEBUG)

def connect_muse_devices(root_output_folder):
    muse_reg = {}
    devices = FindDevices()
    ports = devices.serial_ports()
    com_ports = []

    for port in ports:
        try:
            ser = serial.Serial(port)
            if ser.isOpen():
                com_ports.append(port)
            ser.close()
        except serial.serialutil.SerialException:
            print(f"{port} is not available.\n")
            main_logger.info(f"{port} is not available.\n")

    print(f"{len(com_ports)} free serial port(s) detected.\n")
    main_logger.info(f"{len(com_ports)} free serial port(s) detected.\n")

    if len(com_ports) != 0:
        q1 = Queue()
        t1 = threading.Thread(target=devices.find_muse, args=(q1,))
        t1.start()
        muses = q1.get()
        t1.join()
        if len(com_ports) > len(muses):
            n = len(muses)
        else:
            n = len(com_ports)
        if len(muses) != 0:
            for i in range(n):
                key = muses[i]['name']
                value = muses[i]['address']
                muse_reg[key] = value
            print(f"{len(muse_reg)} Muse device(s) registered.\n")
            main_logger.info(f"{len(muse_reg)} Muse device(s) registered.\n")
        else:
            print("No Muse devices found.\n")
            main_logger.info("No Muse devices found.\n")

        if len(muse_reg) != 0:
            for i in range(len(muse_reg)):
                key = f"muse_streamer_{i + 1}"
                value = StreamMuse(list(muse_reg.keys())[i], list(muse_reg.values())[i], com_ports[i], root_output_folder,synchronized_start_time)
                muse_streamers[key] = value
                key = f"thread_{i + 1}"
                value = threading.Thread(target=list(muse_streamers.values())[i].start_streaming)
                muse_threads[key] = value

            if len(muse_threads) != 0:
                for i in range(len(muse_threads)):
                    list(muse_threads.values())[i].start()
                    list(muse_streamers.values())[i].connected_event.wait()

                print(f"{len(muse_threads)} Muse streaming thread(s) running.\n")
                main_logger.info(f"{len(muse_threads)} Muse streaming thread(s) running.\n")
            else:
                print("No Muse streaming threads running.\n")
                main_logger.info("No Muse streaming threads running.\n")
    return muse_reg, muse_streamers, muse_threads

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
        (6) Stop all the active LSL streams.
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


def start_streaming_process(device_name, root_output_folder, synchronized_start_time, shared_queues):
    # Create instances of EmpaticaE4 and StreamE4 within the child process
    e4 = EmpaticaE4(device_name)
    stream_e4 = StreamE4(e4, root_output_folder, synchronized_start_time, shared_queues)
    stream_e4.start_streaming()


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

    mp.freeze_support()
    flag = 1
    parser = argparse.ArgumentParser(description='Command-line options for the script.')
    parser.add_argument('command', choices=['menu', 'stream', 'view', 'record', 'oddball', 'stop'],
                        help='The command to execute.')
    parser.add_argument('--dev', choices=['muse', 'e4'], help='The device to stream. Used with the stream command.')
    parser.add_argument('--data', choices=['eeg', 'bvp', 'acc', 'gsr', 'ppg'],
                        help='The data stream to view. Used with the view command.')
    data_recorder = StreamRecorder(root_output_folder)
    print("Type 'help' to display the command options or 'exit' to quit.")



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
                                    _setup_logger(root_output_folder_path)
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
                                    _setup_logger(root_output_folder_path)
                                    flag = 0

                            e4_reg = {}
                            devices = FindDevices()
                            e4_processes = []
                            e4_server = False

                            while not (e4_server):
                                print('Checking for E4 Server Process.\n')
                                main_logger.info('Checking for E4 Server Process.\n')
                                f = wmi.WMI()

                                flag = 0

                                # Iterating through all the running processes
                                for process in f.Win32_Process():
                                    if "EmpaticaBLEServer.exe" == process.Name:
                                        print("E4 Server is running. Finding E4 devices.\n")
                                        main_logger.info("E4 Server is running. Finding E4 devices.\n")
                                        e4_server = True
                                        flag = 1
                                        break

                                if flag == 0:
                                    e4_server = False
                                    print("E4 Server is not running. Please start the server first.\n")
                                    main_logger.info("E4 Server is not running. Please start the server first.\n")
                                    time.sleep(10)

                            q2 = Queue()
                            t2 = threading.Thread(target=devices.find_empatica, args=(q2,))
                            t2.start()
                            e4s = q2.get()
                            t2.join()

                            print(e4s)

                            if len(e4s) != 0:
                                shared_queues_dict = {
                                    e4: {
                                        'acc': mp.Queue(),
                                        'bvp': mp.Queue(),
                                        'gsr': mp.Queue(),
                                        'tmp': mp.Queue(),
                                        'ibi': mp.Queue(),
                                        'hr': mp.Queue(),
                                        'tag': mp.Queue()
                                    }
                                    for e4 in e4s
                                }
                                for i in range(len(e4s)):

                                    print(e4s[i], shared_queues_dict[e4s[i]])
                                    p = mp.Process(target=start_streaming_process, args=(
                                    e4s[i], root_output_folder, synchronized_start_time, shared_queues_dict[e4s[i]]))
                                    p.start()
                                    e4_processes.append(p)

                            else:
                                print("No E4 devices found.")
                                main_logger.info("No E4 devices found.")


                        elif int(user_input) == 4:
                            # Create a root directory for saving data and oddball results only if not already created
                            if not root_output_folder:
                                root_output_folder = userpaths.get_my_documents().replace("\\\\",
                                                                                          "/") + f"/StreamSense/{str(datetime.today().timestamp()).replace('.', '_')}"
                                root_output_folder_path = Path(root_output_folder)
                                root_output_folder_path.mkdir(parents=True, exist_ok=True)
                                if flag == 1:
                                    _setup_logger(root_output_folder_path)
                                    flag = 0

                            # Pass the data_queue to StreamRecorder
                            data_recorder.root_output_folder = root_output_folder
                            save_data_thread = threading.Thread(
                                target=data_recorder.start_data_collection_and_writing)
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
                                    _setup_logger(root_output_folder_path)
                                    flag = 0
                            exp = VisualOddball(root_output_folder)
                            sequence = (10, 3)
                            exp.start_oddball(sequence)
                            pass

                        elif int(user_input) == 6:
                            stop_signal = True
                            if active_record:
                                main_logger.info("Current state saved.")
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
                                _setup_logger(root_output_folder_path)
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
                                _setup_logger(root_output_folder_path)
                                flag = 0

                        e4_reg = {}
                        devices = FindDevices()
                        e4_processes = []
                        e4_server = False

                        while not (e4_server):
                            print('Checking for E4 Server Process.\n')
                            main_logger.info('Checking for E4 Server Process.\n')
                            f = wmi.WMI()

                            flag = 0

                            # Iterating through all the running processes
                            for process in f.Win32_Process():
                                if "EmpaticaBLEServer.exe" == process.Name:
                                    print("E4 Server is running. Finding E4 devices.\n")
                                    main_logger.info("E4 Server is running. Finding E4 devices.\n")
                                    e4_server = True
                                    flag = 1
                                    break

                            if flag == 0:
                                e4_server = False
                                print("E4 Server is not running. Please start the server first.\n")
                                main_logger.info("E4 Server is not running. Please start the server first.\n")
                                time.sleep(10)

                        q2 = Queue()
                        t2 = threading.Thread(target=devices.find_empatica, args=(q2,))
                        t2.start()
                        e4s = q2.get()
                        t2.join()

                        if len(e4s) != 0:
                            shared_queues_dict = {
                                e4: {
                                    'acc': mp.Queue(),
                                    'bvp': mp.Queue(),
                                    'gsr': mp.Queue(),
                                    'tmp': mp.Queue(),
                                    'ibi': mp.Queue(),
                                    'hr': mp.Queue(),
                                    'tag': mp.Queue()
                                }
                                for e4 in e4s
                            }
                            for i in range(len(e4s)):
                                p = mp.Process(target=start_streaming_process, args=(
                                    e4s[i], root_output_folder, synchronized_start_time, shared_queues_dict[e4s[i]]))
                                p.start()
                                e4_processes.append(p)

                        else:
                            print("No E4 devices found.")
                            main_logger.info("No E4 devices found.")

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
                            _setup_logger(root_output_folder_path)
                            flag = 0
                    data_recorder.root_output_folder = root_output_folder
                    save_data_thread = threading.Thread(target=data_recorder.start_data_collection_and_writing)
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
                            _setup_logger(root_output_folder_path)
                            flag = 0
                    exp = VisualOddball(root_output_folder)
                    sequence = (10, 3)
                    exp.start_oddball(sequence)
                    pass

                elif args.command == 'stop':
                    stop_signal = True
                    if active_record:
                        main_logger.info("Current state saved.")
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

                elif args.command == 'exit':
                    exit()
        except:
            print("usage:[-h] {menu,stream[--dev {muse,e4}],view[--data {eeg,bvp,acc,gsr}],record,oddball,stop}")
            pass
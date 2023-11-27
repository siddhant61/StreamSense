
from collections import deque
import os
import threading
import time
from functools import partial
from multiprocessing import Process, Event, Queue

from pygatt.exceptions import NotConnectedError
from helper.muse_helper import Muse
from pylsl import StreamInfo, StreamOutlet, local_clock
from muselsl.constants import *


from muselsl.constants import (
    AUTO_DISCONNECT_DELAY,
    MUSE_SAMPLING_EEG_RATE, LSL_EEG_CHUNK,
    MUSE_SAMPLING_PPG_RATE, LSL_PPG_CHUNK,
    MUSE_SAMPLING_ACC_RATE, LSL_ACC_CHUNK,
    MUSE_SAMPLING_GYRO_RATE, LSL_GYRO_CHUNK
)
import warnings
from queue import Queue as ThreadSafeQueue
from threading import Thread
import logging

# Setup logging
logger = logging.getLogger("stream_muse.py")
logger.setLevel(logging.CRITICAL)
fh = logging.FileHandler("Logs/stream_muse.log")
fh.setLevel(logging.CRITICAL)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
warnings.filterwarnings("ignore")

class ThreadPool:
    def __init__(self, num_threads):
        self.tasks = ThreadSafeQueue()
        self.threads = []
        self.stop_signal = Event()

        for _ in range(num_threads):
            thread = Thread(target=self._worker)
            thread.start()
            self.threads.append(thread)

    def _worker(self):
        while not self.stop_signal.is_set():
            func, args, kwargs = self.tasks.get()
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"Error in thread pool worker: {e}")
            finally:
                self.tasks.task_done()

    def submit(self, func, *args, **kwargs):
        self.tasks.put((func, args, kwargs))

    def join(self):
        self.tasks.join()

    def stop(self):
        self.stop_signal.set()
        for thread in self.threads:
            thread.join()


# Constants for robust reconnection
CONNECTION_CHECK_INTERVAL = 5  # Time (in seconds) to check for lost connection
MAX_RETRIES_MUSE = 5  # Maximum number of reconnection attempts
INITIAL_RETRY_DELAY_MUSE = 2  # Initial delay (in seconds) before retrying a connection


class StreamMuse:
    def __init__(self, name, address, interface, root_output_folder,synchronized_start_time):
        logger.info(f"Initializing StreamMuse for device: {name} at address: {address}")
        self.root_output_folder = root_output_folder
        self.interface = interface
        self.address = address
        self.name = name
        self.stop_signal = Event()
        self.connected_event = Event()
        self.queue = Queue()
        self.process = None
        self.eeg_outlet = None
        self.ppg_outlet = None
        self.acc_outlet = None
        self.gyro_outlet = None
        self.shared_eeg = Queue()
        self.shared_acc = Queue()
        self.shared_ppg = Queue()
        self.shared_gyro = Queue()
        self.shared_tel = Queue()
        self.shared_con = Queue()
        self.synchronized_start_time = synchronized_start_time
        self.stream_config = {
            'EEG': {
                'channels': ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10', 'R_AUX'],
                'sampling_rate': MUSE_SAMPLING_EEG_RATE,
                'type': 'EEG',
                'unit': 'microvolts',
                'chunk_size': LSL_EEG_CHUNK,
                'enabled': True
            },
            'PPG': {
                'channels': ['PPG1', 'PPG2', 'PPG3'],
                'sampling_rate': MUSE_SAMPLING_PPG_RATE,
                'type': 'PPG',
                'unit': 'mmHg',
                'chunk_size': LSL_PPG_CHUNK,
                'enabled': True
            },
            'GYRO': {
                'channels': ['GYR_X', 'GYR_Y', 'GYR_Z'],
                'sampling_rate': MUSE_SAMPLING_GYRO_RATE,
                'type': 'gyroscope',
                'unit': 'dps',
                'chunk_size': LSL_GYRO_CHUNK,
                'enabled': True
            },
            'ACC': {
                'channels': ['ACC_X', 'ACC_Y', 'ACC_Z'],
                'sampling_rate': MUSE_SAMPLING_ACC_RATE,
                'type': 'accelerometer',
                'unit': 'g',
                'chunk_size': LSL_ACC_CHUNK,
                'enabled': True
            }
        }


    def start_streaming(self):
        try:
            #local variables for frequent access
            queue = self.queue
            self.process = Process(target=self.stream)
            self.process.start()
            logger.info("Process for start_adapter started.")
            result = queue.get()  # Wait until we get a result from the process
            logger.info(f"Received result from queue: {result}")
            if result == 'connected':
                logger.info(f"Starting streaming for device: {self.name}")
                print(f"Starting streaming for device: {self.name}")
                self.connected_event.set()
                logger.info("Connected event set.")
            else:
                logger.warning(f"Unexpected result from queue: {result}")
        except Exception as e:
            logger.error(f"Error in start_streaming: {e}")

    def stop_streaming(self):
            logger.info(f"Stopping streaming for device: {self.name}")
            self.stop_signal.set()
            if self.process:
                self.process.terminate()
                self.process.join()

    def stream(self):
        try:
            logger.info(f"Starting stream method for device: {self.name}")
            self.eeg_outlet, self.ppg_outlet, self.acc_outlet, self.gyro_outlet = self._setup_lsl_streams()

            # Define recent_data_cache for each data type
            recent_data_cache = {
                'EEG': deque(maxlen=10),
                'PPG': deque(maxlen=10),
                'ACC': deque(maxlen=10),
                'GYRO': deque(maxlen=10)
            }

            lock = threading.Lock()

            def cleanup_cache():
                current_time = local_clock()  # Use LSL clock
                for data_type, cache in recent_data_cache.items():
                    # Determine the appropriate interval based on data type
                    if data_type == 'EEG':
                        interval = 1 / MUSE_SAMPLING_EEG_RATE
                    elif data_type == 'PPG':
                        interval = 1 / MUSE_SAMPLING_PPG_RATE
                    elif data_type == 'ACC':
                        interval = 1 / MUSE_SAMPLING_ACC_RATE
                    elif data_type == 'GYRO':
                        interval = 1 / MUSE_SAMPLING_GYRO_RATE
                    else:
                        continue

                    # Remove timestamps older than the expected interval
                    with lock:
                        recent_data_cache[data_type] = [timestamp for timestamp in cache if
                                                        current_time - timestamp < interval]

            # Schedule the cleanup function to run periodically (e.g., every second)
            cleanup_timer = threading.Timer(1, cleanup_cache)
            cleanup_timer.start()

            def data_processor(stream_id, muse, outlet):
                logger.info(f"Starting data processing for {stream_id} on device: {self.name}")
                # Initialize the initial timestamps
                initial_lsl_timestamp = None
                previous_sample_timestamp = None

                while not self.stop_signal.is_set():
                    data = None
                    if stream_id == f"{muse.name}_EEG":
                        logger.debug(f"Attempting to retrieve EEG data for {muse.name}")
                        data = self.shared_eeg.get()
                        sampling_rate = MUSE_SAMPLING_EEG_RATE
                        logger.debug(f"Retrieved EEG data for {muse.name}: {len(data)}")
                    elif stream_id == f"{muse.name}_PPG":
                        data = self.shared_ppg.get()
                        sampling_rate = MUSE_SAMPLING_PPG_RATE
                    elif stream_id == f"{muse.name}_ACC":
                        data = self.shared_acc.get()
                        sampling_rate = MUSE_SAMPLING_ACC_RATE
                    elif stream_id == f"{muse.name}_GYRO":
                        data = self.shared_gyro.get()
                        sampling_rate = MUSE_SAMPLING_GYRO_RATE
                    else:
                        logger.error(f"Unknown data type: {stream_id}")
                        return

                    if data is None:
                        logger.warning(f"No data received for {stream_id}. Retrying...")
                        continue

                    sample_data_chunk, sample_timestamps  = data
                    data_type = stream_id.split('_')[-1]

                    for i in range(sample_data_chunk.shape[1]):
                        sample_data = sample_data_chunk[:, i]
                        sample_timestamp = sample_timestamps[i]

                        # Generate new timestamp using LSL local clock
                        lsl_timestamp = local_clock()

                        # Check if it's the first sample to set the initial LSL timestamp
                        if initial_lsl_timestamp is None:
                            initial_lsl_timestamp = lsl_timestamp

                        if previous_sample_timestamp is not None:
                            logger.debug(
                                f"Interval between samples for {stream_id}: {previous_sample_timestamp}")

                        BUFFER = 1e-4  # adjust as needed

                        # Check for duplicates using the lock
                        with lock:
                            is_duplicate = any(
                                abs(lsl_timestamp - ts) < BUFFER for ts in recent_data_cache[data_type])
                            if is_duplicate:
                                logger.warning(f"Duplicate timestamp detected for {stream_id}: {lsl_timestamp}")
                                continue

                            # Append the timestamp to the cache
                            recent_data_cache[data_type].append(lsl_timestamp)

                        # Schedule the removal of the timestamp from the cache after the interval
                        removal_delay = 1 / sampling_rate
                        removal_timer = threading.Timer(removal_delay, recent_data_cache[data_type].remove,
                                                        args=[lsl_timestamp])
                        removal_timer.start()
                        # Calculate the corrected timestamp
                        corrected_timestamp = sample_timestamp
                        outlet.push_sample(sample_data.tolist(), corrected_timestamp)
                        self.last_data_received_timestamp = corrected_timestamp
                        logger.debug(
                            f"Sample pushed to LSL outlet for {stream_id}: Sample Size: {len(sample_data)}, Timestamp: {corrected_timestamp}")

                        previous_sample_timestamp = lsl_timestamp

            logger.info(f"Setting up Muse object for device: {self.name}")
            muse = Muse(
                name=self.name,
                address=self.address,
                interface=self.interface,
                synchronized_start_time=self.synchronized_start_time,
                preset=50,
                shared_eeg=self.shared_eeg,
                shared_ppg=self.shared_ppg,
                shared_acc=self.shared_acc,
                shared_gyro=self.shared_gyro,
                shared_tel=self.shared_tel,
                shared_con=self.shared_con
            )

            thread_pool = ThreadPool(num_threads=4)

            # Start the data_processor in a separate thread
            if self.eeg_outlet is not None:
                eeg_processor_thread = thread_pool.submit(data_processor, f"{self.name}_EEG", muse, self.eeg_outlet)
            if self.ppg_outlet is not None:
                ppg_processor_thread = thread_pool.submit(data_processor, f"{self.name}_PPG", muse, self.ppg_outlet)
            if self.acc_outlet is not None:
                acc_processor_thread = thread_pool.submit(data_processor, f"{self.name}_ACC", muse, self.acc_outlet)
            if self.gyro_outlet is not None:
                gyro_processor_thread = thread_pool.submit(data_processor, f"{self.name}_GYRO", muse, self.gyro_outlet)

            try:
                if muse.connect():
                    logger.info("Connected.")
                    self.queue.put('connected')

                    CHECK_FREQUENCY = 0.1
                    threshold = 30 / CHECK_FREQUENCY

                    miss_count = 0

                    logger.info(f"Starting Muse streaming for device: {self.name}")
                    muse.start()
                    logger.info("Muse streaming started. Waiting for data...")

                    muse.start_keep_alive()

                    time.sleep(1)



                    # Starting the connection monitor in a separate thread
                    connection_monitor_thread = threading.Thread(target=self._monitor_connection(muse, CHECK_FREQUENCY))
                    connection_monitor_thread.start()

                    while not self.stop_signal.is_set():
                        time.sleep(CHECK_FREQUENCY)
                        current_time = local_clock()

                        # if current_time - muse.last_timestamp >= CHECK_FREQUENCY:
                        #     miss_count += 1
                        # else:
                        #     miss_count = 0
                        #
                        # if miss_count > threshold:
                        #     logger.warning("Data loss detected.")
                        #     print("Data loss detected.")
                        #     self._reconnect_muse(muse)
                        #     miss_count = 0
                else:
                    print("Connection failed.")

            except KeyboardInterrupt:
                muse.stop()
                muse.disconnect()
                return
            except Exception as e:
                logger.error(f"Exception during streaming: {e}")
                self._reconnect_muse(muse)

            if connection_monitor_thread.is_alive():
                connection_monitor_thread.join()
            thread_pool.join()
            thread_pool.stop()
        except Exception as e:
            logger.error(f"Exception in stream method: {e}")


    def _setup_stream_info_outlet(self, stream_type):
        config = self.stream_config[stream_type]
        info = StreamInfo(f'{self.name}_{stream_type}', stream_type, len(config['channels']), config['sampling_rate'],
                          'float32', f'Muse{self.address}')
        info.desc().append_child_value("manufacturer", "Muse")
        channels = info.desc().append_child("channels")

        for c in config['channels']:
            channels.append_child("channel") \
                .append_child_value("label", c) \
                .append_child_value("unit", config['unit']) \
                .append_child_value("type", config['type'])
        outlet = StreamOutlet(info, self.stream_config[stream_type]['chunk_size'])
        return outlet

    def _reconnect_muse(self, muse):
        """
        Enhanced reconnection strategy with exponential backoff.
        """
        retry_count = 0
        delay = INITIAL_RETRY_DELAY_MUSE
        # print(f"Attempting to reconnect to device: {self.name}. Retry count: {retry_count + 1}")
        logger.warning(f"Attempting to reconnect to device: {self.name}. Retry count: {retry_count + 1}")
        while retry_count < MAX_RETRIES_MUSE:
            try:
                if muse.connect(reconnect=True):
                    muse.start()
                    # print("Reconnected successfully!")
                    logger.warning("Reconnected successfully!")
                    return
            except Exception as e:
                retry_count += 1
                delay *= 2  # Exponential backoff
                logger.warning(
                    f"Attempt {retry_count + 1} to reconnection failed: {e}. Retrying in {delay} seconds..."
                )
                time.sleep(delay)

    def _setup_lsl_streams(self):
        # set up LSL outlets
        if self.stream_config['EEG']['enabled']:
            logger.info(f"Setting up LSL stream for EEG on device: {self.name}")
            self.eeg_outlet = self._setup_stream_info_outlet('EEG')
            logger.info("EEG outlet set up.")
        if self.stream_config['PPG']['enabled']:
            logger.info(f"Setting up LSL stream for PPG on device: {self.name}")
            self.ppg_outlet = self._setup_stream_info_outlet('PPG')
            logger.info("PPG outlet set up.")
        if self.stream_config['ACC']['enabled']:
            logger.info(f"Setting up LSL stream for ACC on device: {self.name}")
            self.acc_outlet = self._setup_stream_info_outlet('ACC')
            logger.info("ACC outlet set up.")
        if self.stream_config['GYRO']['enabled']:
            logger.info(f"Setting up LSL stream for GYRO on device: {self.name}")
            self.gyro_outlet = self._setup_stream_info_outlet('GYRO')
            logger.info("GYRO outlet set up.")

        return self.eeg_outlet, self.ppg_outlet, self.acc_outlet, self.gyro_outlet

    def _monitor_connection(self, muse, MONITORING_INTERVAL):
        """
        Monitors the connection status of the device.
        """
        while not self.stop_signal.is_set():
            current_time = local_clock()
            if current_time - muse.last_timestamp > AUTO_DISCONNECT_DELAY:
                logger.warning("Connection timeout detected.")
                # print("Connection timeout detected:", muse.name)
                self._reconnect_muse(muse)
                self.last_data_received_timestamp = current_time
                # print("Reconnected:", muse.name)
            time.sleep(MONITORING_INTERVAL)







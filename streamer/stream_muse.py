import logging
import threading
from time import time
from threading import Thread

import pygatt
import serial
from muselsl import backends
from functools import partial
from multiprocessing import Process, Event, Queue
from helper.muse_helper import Muse
from pylsl import StreamInfo, StreamOutlet
from helper.serial_helper import BGAPIBackend, NotConnectedError, ExpectedResponseTimeout
from muselsl.constants import (
    AUTO_DISCONNECT_DELAY, MUSE_NB_EEG_CHANNELS,
    MUSE_SAMPLING_EEG_RATE, LSL_EEG_CHUNK, MUSE_NB_PPG_CHANNELS,
    MUSE_SAMPLING_PPG_RATE, LSL_PPG_CHUNK, MUSE_NB_ACC_CHANNELS,
    MUSE_SAMPLING_ACC_RATE, LSL_ACC_CHUNK, MUSE_NB_GYRO_CHANNELS,
    MUSE_SAMPLING_GYRO_RATE, LSL_GYRO_CHUNK
)

acc_enabled = True
eeg_disabled = False
ppg_enabled = False
gyro_enabled = True
MAX_RETRIES_MUSE = 3
RETRY_DELAY_MUSE = 1
muse_logger = logging.getLogger(__name__)

class StreamMuse:

    def __init__(self, name, address, interface):
        self.interface = interface
        self.address = address
        self.name = name
        self.stop_signal = False
        self.connected_event = Event()
        self.queue = Queue()

    def start_streaming(self):
        process = Process(target=self.start_adapter)
        process.start()
        result = self.queue.get()  # Wait until we get a result from the process
        if result == 'connected':
            self.connected_event.set()
        process.join()

    def stop_streaming(self):
        self.stop_signal = True

    def start_adapter(self):
        adapter = BGAPIBackend(serial_port=self.interface)
        print(self.interface)
        # Since start is a blocking call, we can call it directly
        adapter.start()
        muse_logger.info(f"{self.interface} connected")
        print(f"{self.interface} connected")
        self.queue.put('connected')
        self.stream(adapter)


    def stream(self, adapter):
        while not self.stop_signal:

            timeout = AUTO_DISCONNECT_DELAY

            # If no data types are enabled, we warn the user and return immediately.
            if eeg_disabled and not ppg_enabled and not acc_enabled and not gyro_enabled:
                muse_logger.info('Stream initiation failed: At least one data source must be enabled.')
                return

            # For any backend except bluemuse, we will start LSL streams hooked up to the muse callbacks.
            if not eeg_disabled:
                eeg_info = StreamInfo(f'{self.name}_EEG', 'EEG', MUSE_NB_EEG_CHANNELS, MUSE_SAMPLING_EEG_RATE, 'float32',
                                      'Muse%s' % self.address)
                eeg_info.desc().append_child_value("manufacturer", "Muse")
                eeg_channels = eeg_info.desc().append_child("channels")

                for c in ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']:
                    eeg_channels.append_child("channel") \
                        .append_child_value("label", c) \
                        .append_child_value("unit", "microvolts") \
                        .append_child_value("type", "EEG")

                eeg_outlet = StreamOutlet(eeg_info, LSL_EEG_CHUNK)

            if ppg_enabled:
                ppg_info = StreamInfo(f'{self.name}_PPG', 'PPG', MUSE_NB_PPG_CHANNELS, MUSE_SAMPLING_PPG_RATE,
                                      'float32', 'Muse%s' % self.address)
                ppg_info.desc().append_child_value("manufacturer", "Muse")
                ppg_channels = ppg_info.desc().append_child("channels")

                for c in ['PPG1', 'PPG2', 'PPG3']:
                    ppg_channels.append_child("channel") \
                        .append_child_value("label", c) \
                        .append_child_value("unit", "mmHg") \
                        .append_child_value("type", "PPG")

                ppg_outlet = StreamOutlet(ppg_info, LSL_PPG_CHUNK)

            if acc_enabled:
                acc_info = StreamInfo(f'{self.name}_ACC', 'ACC', MUSE_NB_ACC_CHANNELS, MUSE_SAMPLING_ACC_RATE,
                                      'float32', 'Muse%s' % self.address)
                acc_info.desc().append_child_value("manufacturer", "Muse")
                acc_channels = acc_info.desc().append_child("channels")

                for c in ['X', 'Y', 'Z']:
                    acc_channels.append_child("channel") \
                        .append_child_value("label", c) \
                        .append_child_value("unit", "g") \
                        .append_child_value("type", "accelerometer")

                acc_outlet = StreamOutlet(acc_info, LSL_ACC_CHUNK)

            if gyro_enabled:
                gyro_info = StreamInfo(f'{self.name}_GYRO', 'GYRO', MUSE_NB_GYRO_CHANNELS, MUSE_SAMPLING_GYRO_RATE,
                                       'float32', 'Muse%s' % self.address)
                gyro_info.desc().append_child_value("manufacturer", "Muse")
                gyro_channels = gyro_info.desc().append_child("channels")

                for c in ['X', 'Y', 'Z']:
                    gyro_channels.append_child("channel") \
                        .append_child_value("label", c) \
                        .append_child_value("unit", "dps") \
                        .append_child_value("type", "gyroscope")

                gyro_outlet = StreamOutlet(gyro_info, LSL_GYRO_CHUNK)

            def push(data, timestamps, outlet):
                for ii in range(data.shape[1]):
                    outlet.push_sample(data[:, ii], timestamps[ii])

            push_eeg = partial(push, outlet=eeg_outlet) if not eeg_disabled else None
            push_ppg = partial(push, outlet=ppg_outlet) if ppg_enabled else None
            push_acc = partial(push, outlet=acc_outlet) if acc_enabled else None
            push_gyro = partial(push, outlet=gyro_outlet) if gyro_enabled else None

            # Create the Muse object with the specified settings
            muse = Muse(
                address=self.address, adapter= adapter,
                callback_eeg=push_eeg, callback_acc=push_acc,
                callback_ppg=push_ppg, callback_gyro=push_gyro)

            didConnect = muse.connect()

            if didConnect:
                muse_logger.info('Connected.')
                muse.start()
                eeg_string = " EEG" if not eeg_disabled else ""
                ppg_string = " PPG" if ppg_enabled else ""
                acc_string = " ACC" if acc_enabled else ""
                gyro_string = " GYRO" if gyro_enabled else ""

                muse_logger.info("Streaming%s%s%s%s..." %
                      (eeg_string, ppg_string, acc_string, gyro_string))

                while time() - muse.last_timestamp < timeout and not self.stop_signal:
                    try:
                        backends.sleep(1)
                    except KeyboardInterrupt:
                        muse.stop()
                        muse.disconnect()
                        break
                    except Exception as e:
                        logging.error(f"Exception during streaming: {e}")
                        logging.info("Attempting to reconnect in 5 seconds...")
                        backends.sleep(5)
                        continue

                muse_logger.info('Disconnected.')

            self.reconnect(adapter)

    def reconnect(self, adapter):
        muse_logger.info('Reconnecting')
        self.stream(adapter)



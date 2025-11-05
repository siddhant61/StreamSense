import time
from queue import Empty

import pylsl
import logging
from multiprocessing import Queue

from pylsl import local_clock

from helper.e4_helper import EmpaticaE4
from streamer.base_streamer import BaseStreamer

# SELECT DATA TO STREAM
acc = True  # 3-axis acceleration
bvp = True  # Blood Volume Pulse
gsr = True  # Galvanic Skin Response (Electrodermal Activity)
tmp = True  # Temperature
ibi = False # Inter Beat Interval, not available
hr = False # Heart Rate, not available
tag = True # Marker from watch
MAX_RETRIES_E4 = 3
INITIAL_RETRY_DELAY_E4 = 1


logger = logging.getLogger("stream_e4.py")
logger.setLevel(logging.CRITICAL)
file_handler = logging.FileHandler('Logs/stream_e4.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class StreamE4(BaseStreamer):
    """
    Empatica E4 wearable streamer inheriting from BaseStreamer.

    Streams ACC, BVP, GSR, TEMP, TAG data from E4 devices to LSL.
    """

    def __init__(self, e4, root_output_folder, synchronized_start_time):
        # Initialize BaseStreamer
        super().__init__(
            device_name=e4,
            synchronized_start_time=synchronized_start_time,
            root_output_folder=root_output_folder
        )

        # Device-specific attributes
        # Note: self.device_name, self.stop_signal, self.connected_event, self.queue, self.process
        # are provided by BaseStreamer
        self.current_e4 = e4  # Keep for backwards compatibility
        self.empatica_e4 = None
        self.streaming = False  # Local tracking (BaseStreamer also tracks this)
        self.connected = False  # Local tracking
        self.subscribed_streams = {
            "acc": False,
            "bvp": False,
            "gsr": False,
            "tmp": False,
            "tag": False,
            "ibi": False
        }
        self.outletACC = None
        self.outletBVP = None
        self.outletGSR = None
        self.outletTEMP = None
        self.outletIBI = None
        self.outletHR = None
        self.outletTAG = None
        self.synchronized_start_time = synchronized_start_time

    def connect(self):
        try:
            self.empatica_e4 = EmpaticaE4(self.current_e4)
            self.connected = True
        except Exception as e:
            logger.error(f"Connection failed: {e}")

    def subscribe_to_data(self):
        self.empatica_e4.suspend_streaming()
        try:
            if acc:
                self.empatica_e4.subscribe_to_stream('acc')
                time.sleep(0.2)
            if bvp:
                self.empatica_e4.subscribe_to_stream('bvp')
                time.sleep(0.2)
            if gsr:
                self.empatica_e4.subscribe_to_stream('gsr')
                time.sleep(0.2)
            if tmp:
                self.empatica_e4.subscribe_to_stream('tmp')
                time.sleep(0.2)
            if ibi:
                self.empatica_e4.subscribe_to_stream('ibi')
                time.sleep(0.2)
            if hr:
                self.empatica_e4.subscribe_to_stream('hr')
                time.sleep(0.2)
            if tag:
                self.empatica_e4.subscribe_to_stream('tag')
                time.sleep(0.2)
        except Exception as e:
            logger.error(f"Subscription failed: {e}")

    def _setup_lsl_outlets(self):
        """Set up LSL outlets for the device (required by BaseStreamer)."""
        self.empatica_e4.start_streaming()
        logger.info("Starting LSL streaming")
        if acc:
            infoACC = pylsl.StreamInfo(f'{self.current_e4}_ACC', 'ACC', 3, 32, 'int32',
                                       f'ACC-empatica_e4_{self.current_e4}');
            self.outletACC = pylsl.StreamOutlet(infoACC)
        if bvp:
            infoBVP = pylsl.StreamInfo(f'{self.current_e4}_BVP', 'BVP', 1, 64, 'float32',
                                       f'BVP-empatica_e4_{self.current_e4}');
            self.outletBVP = pylsl.StreamOutlet(infoBVP)
        if gsr:
            infoGSR = pylsl.StreamInfo(f'{self.current_e4}_GSR', 'GSR', 1, 4, 'float32',
                                       f'GSR-empatica_e4_{self.current_e4}');
            self.outletGSR = pylsl.StreamOutlet(infoGSR)
        if tmp:
            infoTEMP = pylsl.StreamInfo(f'{self.current_e4}_TEMP', 'TEMP', 1, 4, 'float32',
                                        f'TEMP-empatica_e4_{self.current_e4}');
            self.outletTEMP = pylsl.StreamOutlet(infoTEMP)
        if ibi:
            infoIBI = pylsl.StreamInfo(f'{self.current_e4}_IBI', 'IBI', 1, 64, 'float32',
                                       f'IBI-empatica_e4_{self.current_e4}');
            self.outletIBI = pylsl.StreamOutlet(infoIBI)
        if hr:
            infoHR = pylsl.StreamInfo(f'{self.current_e4}_HR', 'HR', 1, 64, 'float32',
                                       f'HR-empatica_e4_{self.current_e4}');
            self.outletHR = pylsl.StreamOutlet(infoHR)
        if tag:
            infoTAG = pylsl.StreamInfo(f'{self.current_e4}_TAG', 'TAG', 1, 32, 'float32',
                                        f'TAG-empatica_e4_{self.current_e4}');
            self.outletTAG = pylsl.StreamOutlet(infoTAG)


    def reconnect(self):
        logger.info("Reconnecting...")
        retry_count = 0
        reconnect_delay = 2  # Constant reconnect delay of 2 seconds

        while not self.connected and retry_count < MAX_RETRIES_E4 and not self.stop_signal.is_set():
            try:
                self.connect()
                retry_count += 1
                if not self.connected:
                    logger.info(
                        f"Reconnection attempt {retry_count} failed. Retrying in {reconnect_delay} seconds...")
                    time.sleep(reconnect_delay)
            except Exception as e:  # Catch specific exceptions
                logger.error(f"Error during reconnection attempt {retry_count}: {str(e)}")
                self.connected = False

        if self.connected:
            logger.info("Reconnected successfully!")
            time.sleep(1)
            self.subscribe_to_data()
            time.sleep(1)
            self.stream()
        else:
            logger.error("Failed to reconnect after multiple attempts.")

    def stream(self):
        try:
            self.streaming = True
            logger.info("Streaming ACC BVP GSR TEMP IBI HR TAG...")
            print("Streaming ACC BVP GSR TEMP TAG...\n")
            self.queue.put('connected')

            # Initialize initial timestamps
            initial_device_timestamp = None

            while not self.stop_signal.is_set():
                try:
                    response = self.empatica_e4.lsl_data_queue.get()
                    if "connection lost to device" in response:
                        logger.info(response)
                        self.connected = False
                        # Reset initial timestamps due to disconnection
                        initial_device_timestamp = None
                        self.reconnect()
                        continue
                    samples = response.split("\n")
                    logger.info(f"Data: {samples}")
                    for i in range(len(samples) - 1):
                        stream_type = samples[i].split()[0]
                        logger.info(f"Stream Type: {stream_type}")

                        device_timestamp = float(samples[i].split()[1].replace(',', '.'))

                        # Set initial timestamps upon reconnection or start of streaming
                        if initial_device_timestamp is None:
                            initial_device_timestamp = device_timestamp

                        # Calculate the elapsed time since the initial device timestamp
                        elapsed_time = device_timestamp - initial_device_timestamp

                        # Calculate the corrected timestamp
                        corrected_timestamp = elapsed_time

                        if stream_type == "E4_Acc":
                            data = [
                                int(samples[i].split()[2].replace(',', '.')),
                                int(samples[i].split()[3].replace(',', '.')),
                                int(samples[i].split()[4].replace(',', '.'))
                            ]
                            self.outletACC.push_sample(data, timestamp=corrected_timestamp)
                        elif stream_type == "E4_Bvp":
                            data = [float(samples[i].split()[2].replace(',', '.'))]
                            self.outletBVP.push_sample(data, timestamp=corrected_timestamp)
                        elif stream_type == "E4_Gsr":
                            data = [float(samples[i].split()[2].replace(',', '.'))]
                            self.outletGSR.push_sample(data, timestamp=corrected_timestamp)
                        elif stream_type == "E4_Temperature":
                            data = [float(samples[i].split()[2].replace(',', '.'))]
                            self.outletTEMP.push_sample(data, timestamp=corrected_timestamp)
                        elif stream_type == "E4_Ibi":
                            data = [float(samples[i].split()[2].replace(',', '.'))]
                            self.outletIBI.push_sample(data, timestamp=corrected_timestamp)
                        elif stream_type == "E4_Hr":
                            data = [float(samples[i].split()[2].replace(',', '.'))]
                            self.outletHR.push_sample(data, timestamp=corrected_timestamp)
                        elif stream_type == "E4_Tag":
                            data = [float(samples[i].split()[2].replace(',', '.'))]
                            self.outletTAG.push_sample(data, timestamp=corrected_timestamp)

                except Empty:
                    logger.warning("No data received for 10 seconds.")
        except Exception as e:
            logger.error(f"Error in stream method: {e}")
        except KeyboardInterrupt:
            logger.info("Disconnecting from device")
            self.empatica_e4.disconnect()
            self.connected = False

    def _stream_wrapper(self):
        """Main streaming logic that runs in the process (required by BaseStreamer)."""
        self.connect()
        time.sleep(2)
        self.subscribe_to_data()
        time.sleep(1)
        self._setup_lsl_outlets()
        time.sleep(1)
        self.stream()

    # Note: start_streaming() and stop_streaming() are provided by BaseStreamer
    # They provide more robust error handling and cleanup



import logging
import multiprocessing
import socket
import threading
import subprocess
import time
import pickle
from datetime import datetime, timezone
from queue import Queue

# Setup logging
from pylsl import local_clock

logger = logging.getLogger("e4_helper.py")
logger.setLevel(logging.CRITICAL)
fh = logging.FileHandler("Logs/e4_helper.log")
fh.setLevel(logging.CRITICAL)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

MONITORING_INTERVAL=10
MAX_RETRIES = 10
RETRY_WAIT = 5  # wait 5 seconds before retrying
EXE_PATH = "D:/E4StreamingServer1.0.4.5400/EmpaticaBLEServer.exe"
API_KEY = "7abb651d308e498fa558642f5c2b7a66"
SERVER_ADDRESS = '127.0.0.1'
SERVER_PORT = 28000
BUFFER_SIZE = 4096


class EmpaticaServerConnectError(Exception):
    """
    Custom exception for when the socket fails to connect to the Empatica Server.
    """
    pass


class EmpaticaCommandError(Exception):
    """
    Custom exception for when an Empatica response is an error message.
    """
    pass


class EmpaticaDataError(Exception):
    """
    Custom exception for when there is an error when parsing a data message.
    """
    pass


class EmpaticaDataStreams:
    """
    Applicable data streams that can be received from the Empatica server.
    """
    ACC = b'acc'
    BAT = b'bat'
    BVP = b'bvp'
    GSR = b'gsr'
    IBI = b'ibi'
    TAG = b'tag'
    TMP = b'tmp'
    ALL_STREAMS = [b'acc', b'bat', b'bvp', b'gsr', b'ibi', b'tag', b'tmp']


def start_e4_server(exe_path):
    """
    Starts the Empatica Streaming Server.
    :param exe_path: str: full path to Empatica Streaming Server executable
    :return: None.
    """
    subprocess.Popen(exe_path)

class EmpaticaServer():
    def __init__(self):
        self.connected_event = multiprocessing.Event()  # Event to indicate successful connection
        self.stop_signal = multiprocessing.Event()
        self.start_e4_server(EXE_PATH, API_KEY)
        time.sleep(5)

    def start_e4_server(self, exe_path, api_key):
        command = f"{exe_path} {api_key} {SERVER_ADDRESS} {SERVER_PORT}"
        subprocess.Popen(command.split())

    def find_e4s(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((SERVER_ADDRESS, SERVER_PORT))
            device_list_response = self.send_command(sock, "device_discover_list")
            parts = device_list_response.split('|')
            device_names = [part.strip().split()[0] for part in parts[1:]]
            return device_names

    def send_command(self, sock, command):
        try:
            sock.sendall(command.encode() + b'\r\n')
            response = sock.recv(BUFFER_SIZE)
            return response.decode().strip()
        except socket.error as e:
            logging.error(f"Socket error: {e}")
            return None

    def connect_and_monitor_e4(self, device_name):
        # Create and start a new process for connecting and monitoring the device
        process = multiprocessing.Process(target=self._connect_and_monitor_process, args=(device_name,))
        process.start()
        return process

    def _connect_and_monitor_process(self, device_name):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((SERVER_ADDRESS, SERVER_PORT))
                for attempt in range(MAX_RETRIES):
                    logging.warning(f"Attempt {attempt + 1} to connect to {device_name}")
                    self.send_command(sock, f"device_discover_list {device_name}")
                    time.sleep(RETRY_WAIT)
                    response = self.send_command(sock, f"device_connect_btle {device_name}")

                    if "OK" in response or "connected" in response:
                        logging.warning(f"Connected to {device_name}")
                        self.connected_event.set()
                        time.sleep(5)
                        # Start monitoring in a new thread
                        monitoring_thread = threading.Thread(target=self.monitor_e4, args=(device_name,))
                        monitoring_thread.start()
                        break
                    elif "ERR" in response:
                        logging.warning(f"Error connecting to {device_name}: {response}")
                    time.sleep(RETRY_WAIT)

                if attempt == MAX_RETRIES - 1:
                    logging.warning(f"Failed to connect to {device_name} after {MAX_RETRIES} attempts")
        except Exception as e:
            logging.error(f"Error with {device_name}: {e}")

    def monitor_e4(self, device_name):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((SERVER_ADDRESS, SERVER_PORT))
            logging.info(f"Monitoring device {device_name}")

            while not self.stop_signal.is_set():
                if not self.is_device_connected(sock, device_name):
                    logging.warning(f"Device {device_name} disconnected. Attempting to reconnect.")
                    # Graceful reconnection attempt with delay
                    for _ in range(MAX_RETRIES):
                        response = self.send_command(sock, f"device_discover_list {device_name}")
                        if device_name in response:
                            if "OK" in self.send_command(sock, f"device_connect_btle {device_name}"):
                                break
                        time.sleep(RETRY_WAIT)
                else:
                    logging.info(f"{device_name} is connected and being monitored.")
                time.sleep(MONITORING_INTERVAL)

    def is_device_connected(self, sock, device_name):
        response = self.send_command(sock, "device_list")
        if response is None:
            logging.error("Received None response in is_device_connected")
            return False
        return device_name in response

    def stop_monitoring(self):
        self.stop_signal.set()

class EmpaticaClient:

    def __init__(self, serverAddress='127.0.0.1', serverPort=28000, bufferSize=4096, device = None):
        self.serverAddress = serverAddress
        self.serverPort = serverPort
        self.bufferSize = bufferSize
        self.command_queue = Queue()
        self.data_queue = Queue()  # This is for data samples
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(3)
        self.device = device
        self.errors = {"EmpaticaDataError": [], "EmpaticaCommandError": []}
        self.readings = 0
        self.last_error = ""
        self.running = True
        self.thread = None

    def connect(self):
        self.s.connect((self.serverAddress, self.serverPort))
        logger.debug("Connected to server")
        self.start_handling_thread()

    def disconnect(self):
        self.stop_handling_thread()
        self.s.close()

    def recv(self):
        return self.data_queue.get()

    def send(self, command):
        self.command_queue.put(command)

    def start_handling_thread(self):
        self.thread = threading.Thread(target=self.handle_communication)
        self.thread.daemon = True
        self.thread.start()

    def stop_handling_thread(self):
        if self.thread:
            self.running = False  # Signal the thread to stop
            self.thread.join()  # Wait for the thread to actually finish
            self.thread = None  # Clear the reference

    def handle_communication(self):
        while self.running:
            self.handle_commands()
            self.handle_reading_receive()

    def handle_commands(self):
        if not self.command_queue.empty():
            command = self.command_queue.get()
            self.s.sendall(command.encode())
            logger.debug(f"Sent: {command.strip()}")

    def handle_reading_receive(self):
        try:
            response = self.s.recv(self.bufferSize)
            logger.debug(f"Received: {response}")
            if response:
                response_bytes = response.split()
                logger.debug(f"Header:{response_bytes[0][0:2]}")
                if response_bytes:
                    if response_bytes[0] == b'R':
                        if b'ERR' in response_bytes:
                            self.handle_error_code(response_bytes)
                        elif b'connection' in response_bytes:
                            self.handle_error_code(response_bytes)
                        elif b'device' in response_bytes:
                            self.handle_error_code(response_bytes)
                        elif b'device_list' in response_bytes:
                            self.device_list = []
                            for i in range(4, 4 * int(response_bytes[2]) + 1, 3):
                                if response_bytes[i + 1] == b'Empatica_E4':
                                    self.device_list.append(response_bytes[i])
                            self.data_queue.put(self.device_list)
                        elif b'device_connect' in response_bytes:
                            self.device.connected = True
                            self.device.start_window_timer()
                        elif b'device_disconnect' in response_bytes:
                            self.device.connected = False
                        elif b'device_subscribe' in response_bytes:
                            stream_name = response_bytes[2].decode("utf-8")
                            stream_status = response_bytes[3].decode("utf-8")
                            if "OK" == stream_status:
                                self.device.subscribed_streams[stream_name] = True
                            else:
                                self.device.subscribed_streams[stream_name] = False
                    elif response_bytes[0][0:2] == b'E4':
                        if self.device:
                            logger.debug(f"Data: {response_bytes}")
                            self.device.raw_data_queue.put(response_bytes)
        except socket.timeout:
            logger.debug("Socket read timeout.")
        except Exception as e:
            logger.error(f"Error in handle_reading_receive: {e}")


    def stop_reading_thread(self):
        """
        Sets the reading thread variable to False to stop the reading thread.
        :return: None.
        """
        self.reading = False

    def handle_error_code(self, error):
        """
        Parses error code for formatting in Exception message.
        :param error: bytes-like error message.
        :return: None.
        """
        message = ""
        for err in error:
            message = message + err.decode("utf-8") + " "
        self.last_error = "EmpaticaCommandError - " + message
        self.errors["EmpaticaCommandError"].append(message)

    def __enter__(self):
        # Connect when entering the context
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Disconnect when exiting the context
        self.disconnect()


class EmpaticaE4:
    """
    Class to wrap the client socket connection and configure the data streams.
    """
    def __init__(self, device_name, window_size=None, wrist_sensitivity=1):
        """
        Initializes the socket connection and connects the Empatica E4 specified.
        :param device_name: str: The Empatica E4 to connect to
        :param window_size: int: The size of windows in seconds, default None
        :param wrist_sensitivity: int: The number of samples to determine if E4 is on wrist, default is one
        """
        self.errors = {"EmpaticaDataError": [], "EmpaticaCommandError": []}
        self.readings = 0
        self.last_error = ""
        self.subscribed_streams = {
            "acc": False,
            "bvp": False,
            "gsr": False,
            "tmp": False,
            "tag": False,
            "ibi": False,
            "hr": False,
            "bat": False
        }
        self.wrist_sensitivity = wrist_sensitivity
        self.window_size = window_size
        self.on_wrist = False
        self.connected = False
        self.raw_data_queue = Queue()
        self.lsl_data_queue = Queue()
        self.window_thread = threading.Thread(target=self.timer_thread)
        self.client = EmpaticaClient(device=self)
        self.client.connect()
        time.sleep(1)
        self.connect(device_name)
        self.start_data_handling_thread()
        while not self.connected:
            pass
        self.acc_3d, self.acc_x, self.acc_y, self.acc_z, self.acc_timestamps = [], [], [], [], []
        self.bvp, self.bvp_timestamps = [], []
        self.gsr, self.gsr_timestamps = [], []
        self.tmp, self.tmp_timestamps = [], []
        self.tag, self.tag_timestamps = [], []
        self.ibi, self.ibi_timestamps = [], []
        self.bat, self.bat_timestamps = [], []
        self.hr, self.hr_timestamps = [], []
        self.windowed_readings = []

    def start_data_handling_thread(self):
        t = threading.Thread(target=self.handle_incoming_data)
        t.daemon = True
        t.start()

    def handle_incoming_data(self):
        while self.connected:
            if not self.raw_data_queue.empty():
                data = self.raw_data_queue.get()
                self.handle_data_stream(data)
            else:
                logger.debug("No new data in the queue. Waiting...")
                time.sleep(0.1)

    def format_and_queue_data(self, data):
        data_type = data[0][3:].decode('utf-8')
        formatted_data = f"E4_{data_type} {data[1].decode('utf-8')} "
        for val in data[2:]:
            formatted_data += f"{val.decode('utf-8')} "
        formatted_data += "\n"
        logger.debug(f"Data Pushed: {formatted_data}")
        self.lsl_data_queue.put(formatted_data)

    def handle_data_stream(self, data):
        """
        Parses and saves the data received from the Empatica Server.
        :param data: bytes-like packet.
        :return: None.
        """
        try:
            self.readings += 1
            data_type = data[0][3:]
            if data_type == b'Acc':
                self.acc_3d.extend([float(data[2]), float(data[3]), float(data[4])])
                self.acc_x.append(float(data[2]))
                self.acc_y.append(float(data[3]))
                self.acc_z.append(float(data[4]))
                self.acc_timestamps.append(float(data[1]))
                self.format_and_queue_data(data)
            elif data_type == b'Bvp':
                self.bvp.append(float(data[2]))
                self.bvp_timestamps.append(float(data[1]))
                self.format_and_queue_data(data)
            elif data_type == b'Gsr':
                self.gsr.append(float(data[2]))
                self.gsr_timestamps.append(float(data[1]))
                self.format_and_queue_data(data)
                if all(ele == 0 for ele in self.gsr[-self.wrist_sensitivity:]):
                    self.on_wrist = False
                else:
                    self.on_wrist = True
            elif data_type == b'Temperature':
                self.tmp.append(float(data[2]))
                self.tmp_timestamps.append(local_clock())
                self.format_and_queue_data(data)
            elif data_type == b'Ibi':
                self.ibi.append(float(data[2]))
                self.ibi_timestamps.append(local_clock())
                self.format_and_queue_data(data)
            elif data_type == b'Hr':
                self.hr.append(float(data[2]))
                self.hr_timestamps.append(local_clock())
                self.format_and_queue_data(data)
            elif data_type == b'Battery':
                self.bat.append(float(data[2]))
                self.bat_timestamps.append(local_clock())
                self.format_and_queue_data(data)
            elif data_type == b'Tag':
                self.tag.append(float(data[2]))
                self.tag_timestamps.append(local_clock())
                self.format_and_queue_data(data)
            else:
                self.last_error = "EmpaticaDataError - " + str(data)
                self.errors["EmpaticaDataError"].append(data)

        except Exception as e:
            self.last_error = "EmpaticaDataError - " + str(data) + str(e)
            self.errors["EmpaticaDataError"].append(str(data) + str(e))

    @staticmethod
    def get_unix_timestamp(current_time=None):
        if current_time:
            dt = current_time
        else:
            dt = datetime.now(timezone.utc)
        utc_time = dt.replace(tzinfo=timezone.utc)
        return utc_time.timestamp()

    def start_window_timer(self):
        """
        Starts the window timer thread.
        :return:
        """
        if self.window_size:
            self.window_thread.start()

    def timer_thread(self):
        """
        Thread that will split window after window elapses.
        :return:
        """
        if self.window_size:
            while self.connected:
                time.sleep(self.window_size - time.monotonic() % self.window_size)
                self.split_window()

    def split_window(self):
        """
        Splits the current dataset into window and saves it to windowed_readings.
        :return:
        """
        # Save all the readings to our window
        self.windowed_readings.append(
            (self.acc_3d[-(32*3)*self.window_size:],
             self.acc_x[-32*self.window_size:],
             self.acc_y[-32*self.window_size:],
             self.acc_z[-32*self.window_size:],
             self.acc_timestamps[-32*self.window_size:],
             self.bvp[-64*self.window_size:], self.bvp_timestamps[-64*self.window_size:],
             self.gsr[-4*self.window_size:], self.gsr_timestamps[-4*self.window_size:],
             self.tmp[-4*self.window_size:], self.tmp_timestamps[-4*self.window_size:],
             self.tag, self.tag_timestamps,
             self.ibi, self.ibi_timestamps,
             self.bat, self.bat_timestamps,
             self.hr, self.hr_timestamps)
        )
        # Clear all readings collected so far
        self.tag, self.tag_timestamps = [], []
        self.ibi, self.ibi_timestamps = [], []
        self.bat, self.bat_timestamps = [], []
        self.hr, self.hr_timestamps = [], []

    def close(self):
        """
        Closes the socket connection.
        :return: None.
        """
        self.connected = False
        self.client.disconnect()

    def send(self, command):
        """
        Blocking method to send data to Empatica Server.
        :param command: bytes-like: data to send
        :return: None.
        """
        self.client.send(command)

    def receive(self):
        """
        Blocking method to receive data from Empatica Server.
        :return: bytes-like: packet received.
        """
        return self.client.recv()

    def connect(self, device_name, timeout=3):
        """
        Sends the connect command packet to the Empatica Server.
        :param timeout: int: seconds before EmpaticaServerConnectError raised
        :param device_name: bytes-like: Empatica E4 to connect to
        :return: None.
        """
        command = "device_connect " + device_name + "\r\n"
        # self.client.device = self
        self.send(command)

        start_time = local_clock()
        while not self.connected:
            elapsed_time = local_clock() - start_time
            if elapsed_time > timeout:
                print(f"Connection timed out for device: {device_name} after {elapsed_time} seconds")
                raise EmpaticaServerConnectError(f"Could not connect to {device_name}!")
            # Sleep for a brief moment before checking the connection status again
            time.sleep(0.1)

        print(f"Connected to device: {device_name}")

    def disconnect(self, timeout=3):
        """
        Sends the disconnect command packet to the Empatica Server.
        :param timeout: int: seconds before EmpaticaServerConnectError raised
        :return: None.
        """
        command = "device_disconnect\r\n"
        self.send(command)
        start_time = local_clock()
        while self.connected:
            if local_clock() - start_time > timeout:
                raise EmpaticaServerConnectError(f"Could not disconnect from device!")
            pass
        self.client.stop_reading_thread()
        self.client.disconnect()
        self.connected = False

    def save_readings(self, filename):
        """
        Saves the readings currently collected to the specified filepath.
        :param filename: str: full path to file to save to
        :return: None.
        """
        if self.windowed_readings:
            with open(filename, "wb") as file:
                pickle.dump(self.windowed_readings, file)
        else:
            with open(filename, "w") as file:
                for reading in self.acc_3d:
                    file.write(str(reading) + ",")
                file.write("\n")
                for reading in self.acc_x:
                    file.write(str(reading) + ",")
                file.write("\n")
                for reading in self.acc_y:
                    file.write(str(reading) + ",")
                file.write("\n")
                for reading in self.acc_z:
                    file.write(str(reading) + ",")
                file.write("\n")
                for reading in self.acc_timestamps:
                    file.write(str(reading) + ",")
                file.write("\n")
                for reading in self.gsr:
                    file.write(str(reading) + ",")
                file.write("\n")
                for reading in self.gsr_timestamps:
                    file.write(str(reading) + ",")
                file.write("\n")
                for reading in self.bvp:
                    file.write(str(reading) + ",")
                file.write("\n")
                for reading in self.bvp_timestamps:
                    file.write(str(reading) + ",")
                file.write("\n")
                for reading in self.tmp:
                    file.write(str(reading) + ",")
                file.write("\n")
                for reading in self.tmp_timestamps:
                    file.write(str(reading) + ",")
                file.write("\n")
                for reading in self.hr:
                    file.write(str(reading) + ",")
                file.write("\n")
                for reading in self.hr_timestamps:
                    file.write(str(reading) + ",")
                file.write("\n")
                for reading in self.ibi:
                    file.write(str(reading) + ",")
                file.write("\n")
                for reading in self.ibi_timestamps:
                    file.write(str(reading) + ",")
                file.write("\n")
                for reading in self.bat:
                    file.write(str(reading) + ",")
                file.write("\n")
                for reading in self.bat_timestamps:
                    file.write(str(reading) + ",")
                file.write("\n")

    def clear_readings(self):
        """
        Clears the readings collected.
        :return: None.
        """
        self.acc_3d[:], self.acc_x[:], self.acc_y[:], self.acc_z[:], self.acc_timestamps[:] = [], [], [], [], []
        self.bvp[:], self.bvp_timestamps[:] = [], []
        self.gsr[:], self.gsr_timestamps[:] = [], []
        self.tmp[:], self.tmp_timestamps[:] = [], []
        self.tag[:], self.tag_timestamps[:] = [], []
        self.ibi[:], self.ibi_timestamps[:] = [], []
        self.bat[:], self.bat_timestamps[:] = [], []
        self.hr[:], self.hr_timestamps[:] = [], []

    def subscribe_to_stream(self, stream, timeout=3):
        """
        Subscribes the socket connection to a data stream, blocks until the Empatica Server responds.
        :param timeout: int: seconds before EmpaticaServerConnectError raised
        :param stream: bytes-like: data to stream.
        :return: None.
        """
        command = "device_subscribe " + stream + " ON\r\n"
        self.send(command)
        start_time = local_clock()
        while not self.subscribed_streams.get(stream):
            if local_clock() - start_time > timeout:
                raise EmpaticaServerConnectError(f"Could not subscribe to {stream}!")
            pass

    def unsubscribe_from_stream(self, stream, timeout=3):
        """
        Unsubscribes the socket connection from a data stream, blocks until the Empatica Server responds.
        :param timeout: int: seconds before EmpaticaServerConnectError raised
        :param stream: bytes-like: data to stop streaming.
        :return: None.
        """
        command = "device_subscribe " + stream + " OFF\r\n"
        self.send(command)
        start_time = local_clock()
        while self.subscribed_streams.get(stream):
            if local_clock() - start_time > timeout:
                raise EmpaticaServerConnectError(f"Could not unsubscribe to {stream}!")
            pass

    def suspend_streaming(self):
        """
        Stops the data streaming from the Empatica Server for the Empatica E4.
        :return: None.
        """
        command = "pause ON\r\n"
        self.send(command)

    def start_streaming(self):
        """
        Starts the data streaming from the Empatica Server for the Empatica E4.
        :return: None.
        """
        command = "pause OFF\r\n"
        self.send(command)

    def get_max_sample_rate(self):
        stream_rates = {
            "acc": 32,
            "bvp": 64,
            "gsr": 4,
            "tmp": 4,
            "bat": 4,
            "ibi": 64,
            "hr": 64,
            "tag":32
        }
        max_rate = 0
        for stream, is_subscribed in self.subscribed_streams.items():
            max_rate = max(max_rate, stream_rates[stream])
        return max_rate

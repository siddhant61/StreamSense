import logging
import socket
import subprocess
import multiprocessing
import time

EXE_PATH = "D:/E4StreamingServer1.0.4.5400/EmpaticaBLEServer.exe"
API_KEY = "7abb651d308e498fa558642f5c2b7a66"
SERVER_ADDRESS = '127.0.0.1'
SERVER_PORT = 28000
BUFFER_SIZE = 4096
MAX_RETRIES = 3
RETRY_DELAY = 10

logging.basicConfig(level=logging.INFO)

class EmpaticaServer():
    def __init__(self):
        self.start_e4_server(EXE_PATH, API_KEY)
        print("Server started")
        time.sleep(5)

    def start_e4_server(self, exe_path, api_key):
        command = f"{exe_path} {api_key} {SERVER_ADDRESS} {SERVER_PORT}"
        subprocess.Popen(command.split())

    def find_e4s(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((SERVER_ADDRESS, SERVER_PORT))
            device_list_response = self.send_command(sock, "device_discover_list")
            print(device_list_response)
            parts = device_list_response.split('|')
            device_names = [part.strip().split()[0] for part in parts[1:]]
            print("Devices:", device_names)
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
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((SERVER_ADDRESS, SERVER_PORT))
                retries = 0
                while retries < MAX_RETRIES:
                    print(f"Discovering {device_name}, attempt {retries + 1}")
                    # Send a discovery command for the specific device (if such a command exists)
                    response = self.send_command(sock, f"device_discover_list {device_name}")
                    print(f"Device list: {response}")
                    time.sleep(RETRY_DELAY)  # Wait for the discovery process to complete

                    print(f"Attempting to connect to {device_name}, attempt {retries + 1}")
                    response = self.send_command(sock, f"device_connect_btle {device_name}")
                    print(f"Connection Response: {response}")
                    if "OK" in response:
                        self.monitor_e4(sock, device_name)
                        break
                    elif "ERR" in response:
                        print(f"Error connecting to {device_name}: {response}")
                        time.sleep(RETRY_DELAY)
                        retries += 1
        except Exception as e:
            logging.error(f"Error with {device_name}: {e}")

    def monitor_e4(self, sock, device_name):
        logging.info(f"Monitoring device {device_name}")
        while True:
            if not self.is_device_connected(sock, device_name):
                logging.warning(f"Device {device_name} disconnected. Attempting to reconnect.")
                self.send_command(sock, f"device_discover_list {device_name}")
                self.send_command(sock, f"device_connect_btle {device_name}")
            else:
                logging.info(f"{device_name} is all good, no worries fam!")
            time.sleep(5)

    def is_device_connected(self, sock, device_name):
        response = self.send_command(sock, "device_list")
        return device_name in response

if __name__ == '__main__':
    server = EmpaticaServer()
    devices = server.find_e4s()

    for i in range(len(devices)):
        process = multiprocessing.Process(target=server.connect_and_monitor_e4, args=(devices[len(devices)-1-i],))
        process.start()
        time.sleep(10)  # Add a delay between connection attempts

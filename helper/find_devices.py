import logging
import asyncio
import bluetooth
from muselsl import backends
from helper.e4_helper import EmpaticaServerConnectError
import pywifi
import time
import serial.tools.list_ports
import warnings

# Setup logging
from helper.serial_helper import BGAPIBackend

logger = logging.getLogger("find_devices.py")
logger.setLevel(logging.CRITICAL)
fh = logging.FileHandler("Logs/find_devices.log")
fh.setLevel(logging.CRITICAL)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

class FindDevices:

    @staticmethod
    def find_muse():
        logger.info("Starting find_muse function")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        adapter = backends.BleakBackend()
        adapter.start()
        print('Searching for Muses, this may take up to 10 seconds...')
        devices = adapter.scan(timeout=10)
        muses = [d for d in devices if d['name'] and 'Muse' in d['name']]
        adapter.stop()
        for m in muses:
            print(f'Found device {m["name"]}, MAC Address {m["address"]}')
            logger.info(f"Found Muse device: {m['name']}, MAC Address: {m['address']}")
        if not muses:
            print('No Muses found.')
            logger.warning("No Muses found.")

        logger.info("Finished find_muse function")
        return muses

    @staticmethod
    def find_empatica(q, client):
        logger.info("Starting find_empatica function")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        e4s = []
        try:
            client.send("device_list\r\n")
            response = client.data_queue.get()
            for device in response:
                e4s.append(device.decode())
            q.put(e4s)
        except Exception as e:
            print(f"Error: {e}")
            print("Could not connect to Empatica E4:", client.device_list)
            logger.error(f"Error finding Empatica: {e}")
        except EmpaticaServerConnectError:
            print("Failed to connect to server. Ensure E4 Streaming Server is open and connected to the BLE dongle.")
            logger.error("Failed to connect to server. Ensure E4 Streaming Server is open and connected to the BLE dongle.")
        if not e4s:
            print("No E4 devices found.")
            logger.warning("No E4 devices found.")
        logger.info("Finished find_empatica function")

    @staticmethod
    def scan_bluetooth():
        logger.info("Starting scan_bluetooth function")
        devices = []
        nearby_devices = bluetooth.discover_devices(lookup_names=True)
        for addr, name in nearby_devices:
            devices.append({'name': name, 'address': addr, 'type': 'Bluetooth'})
        logger.info(f"Found {len(devices)} Bluetooth devices.")
        logger.info("Finished scan_bluetooth function")
        return devices

    @staticmethod
    def scan_wifi():
        logger.info("Starting scan_wifi function")
        devices = []
        wifi = pywifi.PyWiFi()
        iface = wifi.interfaces()[0]
        iface.scan()
        time.sleep(1)
        results = iface.scan_results()
        for i in results:
            bssid = i.bssid
            ssid = i.ssid
            devices.append({'name': f'{ssid}', 'address': f'{bssid}', 'type': 'WiFi'})
        logger.info(f"Found {len(devices)} WiFi devices.")
        logger.info("Finished scan_wifi function")
        return devices

    @staticmethod
    def serial_ports():
        logger.info("Starting serial_ports function")
        bled_ports = []
        available_ports = []
        ports = serial.tools.list_ports.comports()

        for port, desc, hwid in sorted(ports):
            if "Bluegiga Bluetooth Low Energy" in desc:
                bled_ports.append(port)

        for port in bled_ports:
            try:
                adapter = BGAPIBackend(
                    serial_port=port)
                adapter.start()
                adapter.stop()
                available_ports.append(port)

            except Exception as e:
                # The port is not available
                logger.warning(f"{port} is not available.")

        logger.info(f"Found {len(available_ports)} available Bluegiga BLED112 ports.")
        logger.info("Finished serial_ports function")
        return available_ports

warnings.filterwarnings("ignore")

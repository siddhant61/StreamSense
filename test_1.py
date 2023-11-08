
import threading
import time
import cProfile
from queue import Queue

import serial
import logging

import userpaths

from helper.find_devices import FindDevices
from streamer.stream_muse import StreamMuse


# Logging setup
logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s] [%(levelname)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # pr = cProfile.Profile()
    # pr.enable()
    logger.info("Starting the main script...")

    devices = FindDevices()
    ports = devices.serial_ports()

    com_ports = []
    muse_reg = {}

    logger.info("Identifying available serial ports...")
    for port in ports:
        try:
            ser = serial.Serial(port)
            if ser.isOpen():
                com_ports.append(port)
                logger.debug(f"Port {port} is available and open.")
            ser.close()
        except serial.serialutil.SerialException as e:
            logger.warning(f"{port} is not available. Error: {e}")

    if len(com_ports) != 0:
        q1 = Queue()
        t1 = threading.Thread(target=devices.find_muse, args=(q1, ))
        t1.start()
        muses = q1.get()
        t1.join()
        n = min(len(com_ports), len(muses))

        logger.info(f"Found {len(muses)} available Muses.")

        if len(muses) != 0:
            for i in range(n):
                key = muses[i]['name']
                value = muses[i]['address']
                muse_reg[key] = value
                logger.debug(f"Registered Muse {key} with address {value}.")

    if muse_reg:
        logger.info(f"Attempting to start streaming with Muse {list(muse_reg.keys())[0]}...")
        streamer = StreamMuse(list(muse_reg.keys())[0], list(muse_reg.values())[0], com_ports[0], userpaths.get_my_documents().replace("\\", "/") + \
                             f"/Data_Logs/")
        thread = threading.Thread(target=streamer.start_streaming)
        thread.start()
        time.sleep(120)
        streamer.stop_streaming()
        thread.join()
    else:
        logger.warning("No Muse devices were registered. Exiting.")
    # pr.disable()
    # with open('profile_output_1.txt', 'w') as f:
    #     pr.print_stats(f)


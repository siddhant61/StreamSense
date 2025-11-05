"""
StreamBioTalino: BITalino (r)evolution multi-sensor streamer

Streams ECG, EDA, EMG, EEG, and accelerometer data from BITalino devices to LSL.
BITalino is an open-source biosignal acquisition platform supporting multiple
physiological sensors simultaneously.

Author: StreamSense Team
Date: November 5, 2025
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional

import pylsl
from pylsl import StreamInfo, StreamOutlet, local_clock

from streamer.base_streamer import BaseStreamer

try:
    import bitalino
except ImportError:
    bitalino = None  # Will be caught in __init__

logger = logging.getLogger("stream_bitalino.py")
logger.setLevel(logging.INFO)
fh = logging.FileHandler('Logs/stream_bitalino.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


# BITalino channel configuration
# Users can customize this based on their sensor setup
DEFAULT_SENSOR_CONFIG = {
    0: {'name': 'ECG', 'type': 'ECG', 'unit': 'mV', 'enabled': True},
    1: {'name': 'EDA', 'type': 'EDA', 'unit': 'uS', 'enabled': True},
    2: {'name': 'EMG', 'type': 'EMG', 'unit': 'mV', 'enabled': True},
    3: {'name': 'EEG', 'type': 'EEG', 'unit': 'uV', 'enabled': False},
    4: {'name': 'ACC_X', 'type': 'ACC', 'unit': 'g', 'enabled': True},
    5: {'name': 'ACC_Y', 'type': 'ACC', 'unit': 'g', 'enabled': True},
}

# Valid BITalino sampling rates
VALID_SAMPLING_RATES = [1, 10, 100, 1000]
DEFAULT_SAMPLING_RATE = 1000  # Hz


class StreamBioTalino(BaseStreamer):
    """
    BITalino (r)evolution streamer inheriting from BaseStreamer.

    Streams multi-modal biosignals (ECG, EDA, EMG, EEG, ACC) to LSL.

    BITalino supports up to 6 analog channels with various sensors:
    - ECG: Electrocardiography (heart electrical activity)
    - EDA/GSR: Electrodermal Activity (skin conductance)
    - EMG: Electromyography (muscle activity)
    - EEG: Electroencephalography (brain activity)
    - ACC: Accelerometer (movement)
    - Custom: Any 0-5V analog signal

    Example:
        >>> with StreamBioTalino(
        ...     mac_address="00:13:01:05:12:34",
        ...     synchronized_start_time=time.time(),
        ...     root_output_folder="/data"
        ... ) as streamer:
        ...     streamer.start_streaming()
        ...     time.sleep(60)  # Record for 60 seconds
        ... # Automatic cleanup
    """

    def __init__(
        self,
        mac_address: str,
        synchronized_start_time: float,
        root_output_folder: str,
        sampling_rate: int = DEFAULT_SAMPLING_RATE,
        sensor_config: Optional[Dict] = None
    ):
        """
        Initialize BITalino streamer.

        Args:
            mac_address: BITalino MAC address, serial port, or IP:port
                Examples: "00:13:01:05:12:34", "COM3", "/dev/ttyUSB0", "192.168.4.1:8001"
            synchronized_start_time: Synchronized timestamp for multi-device coordination
            root_output_folder: Output directory for logs
            sampling_rate: Sampling frequency in Hz (1, 10, 100, or 1000)
            sensor_config: Optional sensor configuration dict. If None, uses DEFAULT_SENSOR_CONFIG
        """
        if bitalino is None:
            raise ImportError(
                "BITalino library not installed. "
                "Install with: pip install bitalino"
            )

        # Initialize BaseStreamer
        super().__init__(
            device_name=f"BITalino_{mac_address.replace(':', '')}",
            synchronized_start_time=synchronized_start_time,
            root_output_folder=root_output_folder
        )

        # BITalino-specific attributes
        self.mac_address = mac_address
        self.sensor_config = sensor_config or DEFAULT_SENSOR_CONFIG.copy()

        # Validate sampling rate
        if sampling_rate not in VALID_SAMPLING_RATES:
            logger.warning(
                f"Invalid sampling rate {sampling_rate}. Must be one of {VALID_SAMPLING_RATES}. "
                f"Using {DEFAULT_SAMPLING_RATE} Hz."
            )
            sampling_rate = DEFAULT_SAMPLING_RATE
        self.sampling_rate = sampling_rate

        # Get list of enabled channels
        self.active_channels = [
            ch for ch, config in self.sensor_config.items() if config['enabled']
        ]

        if not self.active_channels:
            raise ValueError("At least one sensor channel must be enabled")

        logger.info(
            f"Initialized StreamBioTalino: {mac_address}, "
            f"{len(self.active_channels)} channels @ {sampling_rate} Hz"
        )

        # Device connection (created in _stream_wrapper)
        self.device = None

        # LSL outlets (one per enabled sensor)
        self.outlets: Dict[int, StreamOutlet] = {}

    def _stream_wrapper(self):
        """Main streaming logic that runs in the process (required by BaseStreamer)."""
        try:
            logger.info(f"Connecting to BITalino at {self.mac_address}")

            # Connect to BITalino
            self.device = bitalino.BITalino(self.mac_address)

            # Get device version
            try:
                version = self.device.version()
                logger.info(f"BITalino version: {version}")
            except Exception as e:
                logger.warning(f"Could not retrieve version: {e}")

            # Setup LSL outlets
            self._setup_lsl_outlets()

            # Start acquisition
            logger.info(f"Starting acquisition at {self.sampling_rate} Hz")
            self.device.start(
                SamplingRate=self.sampling_rate,
                analogChannels=self.active_channels
            )

            # Signal successful connection
            self.queue.put('connected')
            logger.info("BITalino streaming started successfully")

            # Streaming loop
            samples_per_read = 100  # Read 100 samples at a time
            read_interval = samples_per_read / self.sampling_rate  # Time between reads

            while not self.stop_signal.is_set():
                try:
                    # Read data from BITalino
                    # Returns array shape (nSamples, 5+nChannels)
                    # Columns: [seq_num, digital_ch1-4, analog_ch1, analog_ch2, ...]
                    data = self.device.read(nSamples=samples_per_read)

                    # Get current timestamp
                    current_time = local_clock()

                    # Calculate timestamps for each sample (evenly spaced)
                    sample_interval = 1.0 / self.sampling_rate
                    timestamps = current_time - (samples_per_read - 1 - np.arange(samples_per_read)) * sample_interval

                    # Push each sample to appropriate LSL outlets
                    for i in range(samples_per_read):
                        sample = data[i, :]

                        # Columns 5+ contain analog channel data
                        for ch_idx, channel_num in enumerate(self.active_channels):
                            analog_value = sample[5 + ch_idx]

                            # Push to LSL outlet
                            outlet = self.outlets[channel_num]
                            outlet.push_sample([float(analog_value)], timestamps[i])

                    # Small sleep to avoid overwhelming the system
                    time.sleep(read_interval * 0.5)  # Sleep half the interval

                except Exception as e:
                    logger.error(f"Error reading BITalino data: {e}")
                    if not self.stop_signal.is_set():
                        time.sleep(1)  # Wait before retry

        except Exception as e:
            logger.error(f"BITalino streaming error: {e}")
            raise

        finally:
            # Cleanup
            if self.device:
                try:
                    logger.info("Stopping BITalino acquisition")
                    self.device.stop()
                    self.device.close()
                    logger.info("BITalino disconnected")
                except Exception as e:
                    logger.error(f"Error during BITalino cleanup: {e}")

    def _setup_lsl_outlets(self):
        """Set up LSL outlets for each enabled sensor (required by BaseStreamer)."""
        logger.info("Setting up LSL outlets")

        for channel_num in self.active_channels:
            config = self.sensor_config[channel_num]

            # Create LSL stream info
            stream_name = f"{self.device_name}_{config['name']}"
            stream_type = config['type']

            info = StreamInfo(
                stream_name,
                stream_type,
                1,  # One channel per outlet
                self.sampling_rate,
                'float32',
                f"BITalino_{self.mac_address}_CH{channel_num}"
            )

            # Add metadata
            info.desc().append_child_value("manufacturer", "PLUX Biosignals")
            info.desc().append_child_value("model", "BITalino (r)evolution")

            channels = info.desc().append_child("channels")
            ch = channels.append_child("channel")
            ch.append_child_value("label", config['name'])
            ch.append_child_value("unit", config['unit'])
            ch.append_child_value("type", config['type'])
            ch.append_child_value("channel_number", str(channel_num))

            # Create outlet
            outlet = StreamOutlet(info, chunk_size=32, max_buffered=360)
            self.outlets[channel_num] = outlet

            logger.info(f"Created LSL outlet: {stream_name} ({config['type']}) @ {self.sampling_rate} Hz")

    def get_battery_level(self) -> Optional[int]:
        """
        Get battery level from BITalino.

        Returns:
            Battery level integer, or None if not available
        """
        if self.device:
            try:
                state = self.device.state()
                return state.get('battery')
            except Exception as e:
                logger.warning(f"Could not get battery level: {e}")
        return None

    def __repr__(self) -> str:
        """String representation of the streamer."""
        return (
            f"StreamBioTalino(mac_address='{self.mac_address}', "
            f"sampling_rate={self.sampling_rate}, "
            f"channels={len(self.active_channels)}, "
            f"streaming={self.streaming})"
        )


# Example usage
if __name__ == "__main__":
    # Example: Stream ECG, EDA, and EMG from BITalino
    config = {
        0: {'name': 'ECG', 'type': 'ECG', 'unit': 'mV', 'enabled': True},
        1: {'name': 'EDA', 'type': 'EDA', 'unit': 'uS', 'enabled': True},
        2: {'name': 'EMG', 'type': 'EMG', 'unit': 'mV', 'enabled': True},
        3: {'name': 'EEG', 'type': 'EEG', 'unit': 'uV', 'enabled': False},
        4: {'name': 'ACC_X', 'type': 'ACC', 'unit': 'g', 'enabled': False},
        5: {'name': 'ACC_Y', 'type': 'ACC', 'unit': 'g', 'enabled': False},
    }

    with StreamBioTalino(
        mac_address="/dev/ttyUSB0",  # Or COM3 on Windows, or MAC address
        synchronized_start_time=time.time(),
        root_output_folder="/tmp/bitalino_test",
        sampling_rate=1000,
        sensor_config=config
    ) as streamer:
        print(f"Starting {streamer}")
        streamer.start_streaming()

        # Stream for 10 seconds
        print("Streaming... (10 seconds)")
        time.sleep(10)

        print("Stopping...")

    print("Done!")

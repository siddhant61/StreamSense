"""
Synthetic physiological data generators for hardware mocking.

Generates realistic synthetic data for:
- EEG: 5-channel electroencephalography (TP9, AF7, AF8, TP10, AUX)
- PPG: 3-channel photoplethysmography
- ACC: 3-axis accelerometer data
- GYRO: 3-axis gyroscope data
- BVP: Blood volume pulse
- GSR: Galvanic skin response
- TEMP: Skin temperature

All generators produce physiologically plausible signals with appropriate
noise characteristics for testing data processing pipelines.
"""

import numpy as np
from typing import List, Tuple


class PhysiologicalDataGenerator:
    """Base class for generating realistic physiological signals with controlled noise."""

    def __init__(self, sampling_rate: float, seed: int = 42):
        """
        Initialize data generator.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate in Hz
        seed : int
            Random seed for reproducible data generation
        """
        self.sampling_rate = sampling_rate
        self.rng = np.random.RandomState(seed)
        self.time = 0.0

    def advance_time(self, num_samples: int = 1) -> np.ndarray:
        """
        Advance internal time counter and return timestamps.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate timestamps for

        Returns
        -------
        np.ndarray
            Array of timestamps
        """
        timestamps = np.arange(num_samples) / self.sampling_rate + self.time
        self.time += num_samples / self.sampling_rate
        return timestamps

    def reset(self):
        """Reset generator to initial state."""
        self.time = 0.0


class EEGDataGenerator(PhysiologicalDataGenerator):
    """Generate synthetic EEG data with realistic frequency components."""

    def __init__(self, num_channels: int = 5, sampling_rate: float = 256.0, seed: int = 42):
        super().__init__(sampling_rate, seed)
        self.num_channels = num_channels
        # Base amplitude in microvolts
        self.base_amplitude = 50.0

    def generate(self, num_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate EEG samples with realistic frequency components.

        Simulates:
        - Delta (0.5-4 Hz): Deep sleep activity
        - Theta (4-8 Hz): Drowsiness
        - Alpha (8-13 Hz): Relaxed wakefulness
        - Beta (13-30 Hz): Active thinking
        - Gamma (30-100 Hz): Cognitive processing

        Parameters
        ----------
        num_samples : int
            Number of samples to generate

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (data, timestamps) where data has shape (num_channels, num_samples)
        """
        timestamps = self.advance_time(num_samples)
        t = np.linspace(0, num_samples / self.sampling_rate, num_samples)

        data = np.zeros((self.num_channels, num_samples))

        for ch in range(self.num_channels):
            # Alpha rhythm (8-13 Hz) - dominant in relaxed state
            alpha = 30 * np.sin(2 * np.pi * 10 * t + self.rng.uniform(0, 2 * np.pi))

            # Beta activity (13-30 Hz)
            beta = 15 * np.sin(2 * np.pi * 20 * t + self.rng.uniform(0, 2 * np.pi))

            # Theta activity (4-8 Hz)
            theta = 20 * np.sin(2 * np.pi * 6 * t + self.rng.uniform(0, 2 * np.pi))

            # Pink noise (1/f characteristic)
            noise = self.rng.normal(0, 5, num_samples)

            # Combine components
            data[ch] = alpha + beta + theta + noise

            # Add channel-specific variation
            data[ch] *= (1.0 + 0.1 * ch)

        return data, timestamps


class PPGDataGenerator(PhysiologicalDataGenerator):
    """Generate synthetic photoplethysmography (PPG) data."""

    def __init__(self, num_channels: int = 3, sampling_rate: float = 64.0,
                 heart_rate_bpm: float = 70.0, seed: int = 42):
        super().__init__(sampling_rate, seed)
        self.num_channels = num_channels
        self.heart_rate_bpm = heart_rate_bpm

    def generate(self, num_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate PPG samples with cardiac pulse waveform.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (data, timestamps) where data has shape (num_channels, num_samples)
        """
        timestamps = self.advance_time(num_samples)
        t = np.linspace(0, num_samples / self.sampling_rate, num_samples)

        data = np.zeros((self.num_channels, num_samples))
        heart_rate_hz = self.heart_rate_bpm / 60.0

        for ch in range(self.num_channels):
            # Fundamental cardiac frequency
            pulse = 100 * np.sin(2 * np.pi * heart_rate_hz * t)

            # Second harmonic (dicrotic notch)
            harmonic = 30 * np.sin(4 * np.pi * heart_rate_hz * t + np.pi / 4)

            # Baseline drift
            baseline = 50 + 10 * np.sin(2 * np.pi * 0.1 * t)

            # Noise
            noise = self.rng.normal(0, 2, num_samples)

            data[ch] = baseline + pulse + harmonic + noise

        return data, timestamps


class AccelerometerDataGenerator(PhysiologicalDataGenerator):
    """Generate synthetic 3-axis accelerometer data."""

    def __init__(self, sampling_rate: float = 52.0, seed: int = 42):
        super().__init__(sampling_rate, seed)

    def generate(self, num_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate accelerometer samples (gravity + motion).

        Parameters
        ----------
        num_samples : int
            Number of samples to generate

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (data, timestamps) where data has shape (3, num_samples) in units of g
        """
        timestamps = self.advance_time(num_samples)
        t = np.linspace(0, num_samples / self.sampling_rate, num_samples)

        # Gravity component (assuming ~vertical orientation with slight tilt)
        gravity = np.array([0.1, 0.05, 0.98])[:, np.newaxis]

        # Small periodic motion (head movements)
        motion_x = 0.05 * np.sin(2 * np.pi * 0.5 * t)  # Slow nodding
        motion_y = 0.03 * np.sin(2 * np.pi * 0.7 * t)  # Slight lateral movement
        motion_z = 0.02 * np.sin(2 * np.pi * 0.3 * t)  # Vertical bounce

        motion = np.vstack([motion_x, motion_y, motion_z])

        # Noise
        noise = self.rng.normal(0, 0.01, (3, num_samples))

        data = gravity + motion + noise

        return data, timestamps


class GyroscopeDataGenerator(PhysiologicalDataGenerator):
    """Generate synthetic 3-axis gyroscope data."""

    def __init__(self, sampling_rate: float = 52.0, seed: int = 42):
        super().__init__(sampling_rate, seed)

    def generate(self, num_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate gyroscope samples (angular velocity).

        Parameters
        ----------
        num_samples : int
            Number of samples to generate

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (data, timestamps) where data has shape (3, num_samples) in dps (degrees per second)
        """
        timestamps = self.advance_time(num_samples)
        t = np.linspace(0, num_samples / self.sampling_rate, num_samples)

        # Small rotational movements (head rotation)
        rotation_x = 5 * np.sin(2 * np.pi * 0.3 * t)  # Pitch
        rotation_y = 3 * np.sin(2 * np.pi * 0.4 * t)  # Yaw
        rotation_z = 2 * np.sin(2 * np.pi * 0.2 * t)  # Roll

        # Noise
        noise = self.rng.normal(0, 0.5, (3, num_samples))

        data = np.vstack([rotation_x, rotation_y, rotation_z]) + noise

        return data, timestamps


class BVPDataGenerator(PhysiologicalDataGenerator):
    """Generate synthetic Blood Volume Pulse data for E4."""

    def __init__(self, sampling_rate: float = 64.0, heart_rate_bpm: float = 70.0, seed: int = 42):
        super().__init__(sampling_rate, seed)
        self.heart_rate_bpm = heart_rate_bpm

    def generate(self, num_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate BVP samples."""
        timestamps = self.advance_time(num_samples)
        t = np.linspace(0, num_samples / self.sampling_rate, num_samples)

        heart_rate_hz = self.heart_rate_bpm / 60.0
        pulse = 1000 * np.sin(2 * np.pi * heart_rate_hz * t)
        noise = self.rng.normal(0, 10, num_samples)

        data = pulse + noise
        return data.reshape(1, -1), timestamps


class GSRDataGenerator(PhysiologicalDataGenerator):
    """Generate synthetic Galvanic Skin Response data for E4."""

    def __init__(self, sampling_rate: float = 4.0, seed: int = 42):
        super().__init__(sampling_rate, seed)

    def generate(self, num_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate GSR samples in microsiemens."""
        timestamps = self.advance_time(num_samples)
        t = np.linspace(0, num_samples / self.sampling_rate, num_samples)

        # Tonic level (slow changes)
        tonic = 2.0 + 0.5 * np.sin(2 * np.pi * 0.01 * t)

        # Phasic responses (event-related)
        phasic = 0.2 * np.sin(2 * np.pi * 0.05 * t)

        # Noise
        noise = self.rng.normal(0, 0.05, num_samples)

        data = tonic + phasic + noise
        return data.reshape(1, -1), timestamps


class TemperatureDataGenerator(PhysiologicalDataGenerator):
    """Generate synthetic skin temperature data for E4."""

    def __init__(self, sampling_rate: float = 4.0, base_temp: float = 32.0, seed: int = 42):
        super().__init__(sampling_rate, seed)
        self.base_temp = base_temp

    def generate(self, num_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate temperature samples in Celsius."""
        timestamps = self.advance_time(num_samples)
        t = np.linspace(0, num_samples / self.sampling_rate, num_samples)

        # Slow temperature drift
        drift = 0.5 * np.sin(2 * np.pi * 0.005 * t)

        # Noise
        noise = self.rng.normal(0, 0.02, num_samples)

        data = self.base_temp + drift + noise
        return data.reshape(1, -1), timestamps


# Convenience functions for single-sample generation
def generate_eeg_sample(num_channels: int = 5, num_samples: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single EEG chunk (default 12 samples at 256 Hz)."""
    gen = EEGDataGenerator(num_channels=num_channels)
    return gen.generate(num_samples)


def generate_ppg_sample(num_channels: int = 3, num_samples: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single PPG chunk (default 6 samples at 64 Hz)."""
    gen = PPGDataGenerator(num_channels=num_channels)
    return gen.generate(num_samples)


def generate_acc_sample(num_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single accelerometer chunk."""
    gen = AccelerometerDataGenerator()
    return gen.generate(num_samples)


def generate_gyro_sample(num_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single gyroscope chunk."""
    gen = GyroscopeDataGenerator()
    return gen.generate(num_samples)


def generate_bvp_sample(num_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single BVP chunk."""
    gen = BVPDataGenerator()
    return gen.generate(num_samples)


def generate_gsr_sample(num_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single GSR chunk."""
    gen = GSRDataGenerator()
    return gen.generate(num_samples)


def generate_temp_sample(num_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single temperature chunk."""
    gen = TemperatureDataGenerator()
    return gen.generate(num_samples)

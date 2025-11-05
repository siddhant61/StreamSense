"""
Hardware mocking infrastructure for StreamSense testing.

This package provides comprehensive mocking utilities for testing hardware-dependent
features without requiring physical devices:

- MockMuse: Simulated Muse headband with realistic EEG/PPG/ACC/GYRO data
- MockE4: Simulated Empatica E4 with realistic biometric data
- MockLSL: Lab Streaming Layer mock for testing stream publication
- DataGenerators: Synthetic physiological signal generators

Usage:
    from tests.mocks import MockMuse, MockE4, create_mock_lsl_outlet

    # Create mock Muse device
    muse = MockMuse(name="Muse-1234", address="00:11:22:33:44:55")
    muse.connect()
    eeg_data = muse.get_eeg_sample()

    # Create mock E4 device
    e4 = MockE4(device_id="A01234")
    e4.connect()
    bvp_data = e4.get_bvp_sample()
"""

from .mock_muse import MockMuse, MockMuseAdapter, MockBGAPIBackend
from .mock_e4 import MockE4, MockEmpaticaServer
from .mock_lsl import MockLSLOutlet, MockLSLStreamInfo, create_mock_lsl_outlet, mock_local_clock
from .data_generators import (
    generate_eeg_sample,
    generate_ppg_sample,
    generate_acc_sample,
    generate_gyro_sample,
    generate_bvp_sample,
    generate_gsr_sample,
    generate_temp_sample
)

__all__ = [
    # Muse mocks
    'MockMuse',
    'MockMuseAdapter',
    'MockBGAPIBackend',

    # E4 mocks
    'MockE4',
    'MockEmpaticaServer',

    # LSL mocks
    'MockLSLOutlet',
    'MockLSLStreamInfo',
    'create_mock_lsl_outlet',
    'mock_local_clock',

    # Data generators
    'generate_eeg_sample',
    'generate_ppg_sample',
    'generate_acc_sample',
    'generate_gyro_sample',
    'generate_bvp_sample',
    'generate_gsr_sample',
    'generate_temp_sample',
]

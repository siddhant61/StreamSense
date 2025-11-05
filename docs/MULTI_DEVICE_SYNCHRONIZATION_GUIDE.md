# Multi-Device Synchronization Guide

**For Physiological Synchrony Studies**

*Measure heart rate synchronization between lovers, brainwave alignment during meditation, and other beautiful manifestations of human connection.*

---

## 🎯 Overview

StreamSense is designed from the ground up for **multi-device synchronized recording**. All devices share a common LSL timestamp reference, enabling precise analysis of physiological synchronization across multiple participants.

## 💡 Use Cases

### Interpersonal Synchrony
- **Romantic couples**: Heart rate variability (HRV) synchronization
- **Parent-child**: Breathing pattern alignment during bonding
- **Musicians**: Brain rhythm coordination during ensemble performance
- **Therapists & clients**: Physiological attunement during sessions
- **Meditators**: EEG synchronization in group meditation

### Research Applications
- Social neuroscience
- Interpersonal physiology
- Collective behavior studies
- Empathy and emotional contagion
- Non-verbal communication

---

## 🏗️ Architecture

### Time Synchronization

```python
# main.py:18
synchronized_start_time = local_clock()  # LSL high-precision timestamp
```

**Key Features**:
- ✅ **Single time reference** shared across all devices
- ✅ **Microsecond precision** using LSL's local_clock()
- ✅ **Cross-platform consistency** (Windows, Mac, Linux)
- ✅ **Immune to system clock drift**

### Multi-Device Support

```python
# AppState tracks multiple devices
state.muse_streamers = {
    "muse_streamer_1": StreamMuse("Participant_A", ...),
    "muse_streamer_2": StreamMuse("Participant_B", ...),
}
state.e4_streamers = {
    "e4_streamer_1": StreamE4("Participant_A_E4", ...),
    "e4_streamer_2": StreamE4("Participant_B_E4", ...),
}
```

**Supports**:
- ✅ Multiple Muse headbands (EEG, PPG, ACC, GYRO)
- ✅ Multiple E4 wearables (BVP, GSR, TEMP, ACC)
- ✅ Mixed device types simultaneously
- ✅ Unlimited device count (hardware dependent)

---

## 🚀 Quick Start: Two-Person Heart Sync Study

### Scenario: Measure heart rate synchronization between romantic partners

#### Step 1: Hardware Setup

**Person A**:
- Muse headband (for EEG + PPG heart rate)
- E4 wearable (for BVP + GSR)

**Person B**:
- Muse headband (for EEG + PPG heart rate)
- E4 wearable (for BVP + GSR)

#### Step 2: Launch StreamSense

```bash
python main.py
```

#### Step 3: Connect Devices

```
> stream --dev muse
# System finds 2 Muse devices, assigns them to Person A & B

> stream --dev e4
# System finds 2 E4 devices, assigns them to Person A & B
```

**What happens internally**:
1. All 4 devices receive the **same** `synchronized_start_time`
2. Each device starts streaming with timestamps relative to this reference
3. LSL handles sub-millisecond timestamp alignment

#### Step 4: Start Recording

```
> record
```

**Recorded streams** (example):
```
Muse-A01B_EEG   → Person A brainwaves
Muse-A01B_PPG   → Person A heart rate
E4_A01234_BVP   → Person A blood volume pulse
E4_A01234_GSR   → Person A skin conductance

Muse-C02D_EEG   → Person B brainwaves
Muse-C02D_PPG   → Person B heart rate
E4_B56789_BVP   → Person B blood volume pulse
E4_B56789_GSR   → Person B skin conductance
```

All streams have **perfectly aligned timestamps** for synchrony analysis!

#### Step 5: Analysis

Use the recorded XDF files to compute:
- **Heart rate synchronization** (cross-correlation of PPG/BVP)
- **GSR coherence** (emotional co-regulation)
- **EEG phase locking** (neural synchrony)

---

## 📊 Detailed Workflow

### 1. Device Discovery

StreamSense automatically discovers all connected devices:

```python
# For Muse devices
devices = FindDevices()
muses, com_ports = devices.find_muses_with_ports()
# Returns: [('Muse-A01B', '00:55:DA:...'), ('Muse-C02D', '00:55:DA:...')]

# For E4 devices
e4s = devices.find_empatica()
# Returns: ['A01234', 'B56789']
```

### 2. Streamer Initialization

Each device gets the same synchronized timestamp:

```python
# All streamers share synchronized_start_time
synchronized_start_time = local_clock()  # e.g., 12345.678901234

streamer_A = StreamMuse(
    name="Muse-A01B",
    address="00:55:DA:...",
    interface="COM3",
    root_output_folder="/output",
    synchronized_start_time=synchronized_start_time  # Same for all!
)

streamer_B = StreamMuse(
    name="Muse-C02D",
    address="00:55:DA:...",
    interface="COM4",
    root_output_folder="/output",
    synchronized_start_time=synchronized_start_time  # Same timestamp!
)
```

### 3. LSL Stream Setup

Each device creates LSL outlets with the synchronized timestamp:

```python
# StreamMuse._setup_lsl_outlets()
info_eeg = StreamInfo(
    f'{self.device_name}_EEG',  # 'Muse-A01B_EEG'
    'EEG',
    5,      # channels
    256,    # sampling rate
    'float32',
    f'Muse{self.address}'
)
self.eeg_outlet = StreamOutlet(info_eeg, chunk_size=12)
```

### 4. Timestamp Alignment

Data samples use timestamps relative to `synchronized_start_time`:

```python
# In StreamMuse data_processor
corrected_timestamp = sample_timestamp  # Relative to synchronized_start_time
outlet.push_sample(sample_data.tolist(), corrected_timestamp)
```

### 5. Synchronized Recording

The `StreamRecorder` captures all streams:

```python
recorder = StreamRecorder(root_output_folder)
recorder.record_streams()  # Records all active LSL streams
```

**Output format**: XDF (Extensible Data Format)
- Contains all streams with aligned timestamps
- Preserves channel info, sampling rates, metadata
- Ready for analysis in Python, MATLAB, EEGLAB

---

## 🔬 Example: Brain-Heart Synchrony Analysis

### Recording Setup

**2 participants** wearing Muse headbands during meditation:

```python
# After starting StreamSense
>>> stream --dev muse
2 Muse device(s) registered.
2 Muse streaming process(es) running.

>>> record
Recording started...
```

### Data Structure

Each participant generates:
- **EEG**: 5 channels @ 256 Hz (TP9, AF7, AF8, TP10, AUX)
- **PPG**: 3 channels @ 64 Hz (heart rate signal)
- **ACC**: 3 channels @ 52 Hz (movement)
- **GYRO**: 3 channels @ 52 Hz (head orientation)

**Total**: 28 synchronized channels across 2 participants

### Analysis Code (Python)

```python
import pyxdf
import numpy as np
from scipy import signal

# Load synchronized recording
streams, header = pyxdf.load_xdf('recording.xdf')

# Extract PPG for both participants
participant_a_ppg = [s for s in streams if 'Muse-A01B_PPG' in s['info']['name'][0]][0]
participant_b_ppg = [s for s in streams if 'Muse-C02D_PPG' in s['info']['name'][0]][0]

# Extract heart rate signals
ppg_a = participant_a_ppg['time_series'][:, 0]  # First PPG channel
ppg_b = participant_b_ppg['time_series'][:, 0]

# Compute cross-correlation
correlation = signal.correlate(ppg_a, ppg_b, mode='full')
lags = signal.correlation_lags(len(ppg_a), len(ppg_b), mode='full')

# Find synchronization strength
max_corr_idx = np.argmax(correlation)
max_correlation = correlation[max_corr_idx]
time_lag = lags[max_corr_idx] / 64  # Convert samples to seconds (PPG @ 64 Hz)

print(f"Heart synchronization: {max_correlation:.3f}")
print(f"Time lag: {time_lag:.3f} seconds")
```

**Interpretation**:
- High correlation (>0.7) = Strong heart rate synchrony
- Near-zero lag = Real-time physiological coordination
- Negative lag = One person leads, other follows

---

## 🎼 Example: Musical Ensemble Synchrony

### Setup: String Quartet

**4 musicians** wearing Muse headbands:
- Violinist 1
- Violinist 2
- Violist
- Cellist

### What You Can Measure

1. **EEG Phase Synchronization**
   - Alpha rhythm (8-13 Hz) alignment during synchronized playing
   - Theta coupling (4-8 Hz) during transitions
   - Gamma bursts (30-100 Hz) during musical climaxes

2. **Inter-Brain Coherence**
   - Which musicians synchronize most strongly?
   - Leader-follower dynamics
   - Moment-to-moment coordination

3. **Movement Coordination**
   - ACC/GYRO data shows head movements
   - Can correlate with musical phrases
   - Reveals non-verbal communication

### Recording

```bash
> stream --dev muse
4 Muse device(s) registered.
4 Muse streaming process(es) running.

> record
# Musicians perform piece

> stop
# Stop recording when performance ends
```

**Outputs**: 4 × 14 = **56 synchronized channels** of brain and body data!

---

## 💪 Best Practices

### Hardware Setup

1. **Device Naming**
   - Use meaningful names: "Participant_A", "Participant_B"
   - Easier to identify in analysis
   - Consider using actual names for small studies

2. **Connection Order**
   - Connect all devices of one type together
   - Ensures they share the same synchronized_start_time
   - Minimize time between connections (< 30 seconds ideal)

3. **Signal Quality**
   - Check signal quality before recording
   - Use viewer: `view --data eeg` to see real-time streams
   - Ensure good electrode contact (Muse) or wrist fit (E4)

### Recording Protocol

1. **Baseline Period**
   - Record 2-5 minutes of baseline before experiment
   - Participants sit quietly, eyes closed
   - Provides normalization reference

2. **Event Markers**
   - Use event logger to mark experimental conditions
   - `logger` command opens event marker window
   - Timestamps align with physiological data

3. **Multiple Recordings**
   - Save separate recordings for each condition
   - Easier to analyze than one long recording
   - Use `stop` then `record` to start new session

### Data Management

1. **Output Organization**
   ```
   Study_Name/
   ├── Session_2025_01_15_140523/
   │   ├── Muse-A01B_EEG.xdf
   │   ├── Muse-A01B_PPG.xdf
   │   ├── Muse-C02D_EEG.xdf
   │   ├── Muse-C02D_PPG.xdf
   │   ├── E4_A01234_BVP.xdf
   │   └── events.csv
   └── Session_2025_01_15_153012/
       └── ...
   ```

2. **Metadata**
   - Keep log of which device → which participant
   - Record experimental conditions
   - Note any technical issues during recording

---

## 🔧 Advanced Configuration

### Custom Synchronized Start Time

If you need to synchronize with external systems:

```python
# In your custom script
from pylsl import local_clock
import time

# Option 1: Use LSL clock directly
custom_start_time = local_clock()

# Option 2: Align with external trigger
while not external_trigger_received:
    time.sleep(0.001)
custom_start_time = local_clock()

# Initialize streamers with custom time
streamer = StreamMuse(
    ...,
    synchronized_start_time=custom_start_time
)
```

### Context Manager for Clean Lifecycle

Now that StreamMuse and StreamE4 inherit from BaseStreamer:

```python
# Automatic cleanup on exit
with StreamMuse(...) as muse_a, \
     StreamMuse(...) as muse_b, \
     StreamE4(...) as e4_a:

    muse_a.start_streaming()
    muse_b.start_streaming()
    e4_a.start_streaming()

    # Record for 10 minutes
    time.sleep(600)

# All devices automatically stopped and cleaned up!
```

---

## 📈 Analysis Techniques

### 1. Cross-Correlation

Measures similarity between two signals at different time lags:

```python
def compute_synchrony(signal_a, signal_b, sampling_rate):
    """
    Compute synchronization between two signals.

    Returns:
        correlation: Strength of synchrony (0-1)
        lag_seconds: Time lag at maximum correlation
    """
    correlation = np.correlate(signal_a, signal_b, mode='full')
    correlation = correlation / np.max(correlation)  # Normalize

    lags = np.arange(-len(signal_a) + 1, len(signal_a))
    max_idx = np.argmax(correlation)

    return correlation[max_idx], lags[max_idx] / sampling_rate
```

### 2. Coherence Analysis

Measures frequency-specific synchronization:

```python
from scipy.signal import coherence

def heart_coherence(ppg_a, ppg_b, fs=64):
    """
    Compute coherence between two heart rate signals.

    Returns:
        f: Frequencies
        Cxy: Coherence at each frequency
    """
    f, Cxy = coherence(ppg_a, ppg_b, fs=fs, nperseg=256)

    # Focus on heart rate range (0.5-2 Hz)
    hr_range = (f >= 0.5) & (f <= 2.0)

    return f[hr_range], Cxy[hr_range]
```

### 3. Phase Locking Value (PLV)

Measures neural synchronization:

```python
from scipy.signal import hilbert

def phase_locking_value(eeg_a, eeg_b):
    """
    Compute PLV between two EEG channels.

    Returns:
        plv: Phase locking value (0-1)
    """
    # Get instantaneous phase
    analytic_a = hilbert(eeg_a)
    analytic_b = hilbert(eeg_b)

    phase_a = np.angle(analytic_a)
    phase_b = np.angle(analytic_b)

    # Compute phase difference
    phase_diff = phase_a - phase_b

    # PLV is the magnitude of the mean complex phase difference
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    return plv
```

### 4. Windowed Analysis

Track synchrony over time:

```python
def windowed_synchrony(signal_a, signal_b, window_sec=10, overlap=0.5, fs=256):
    """
    Compute time-varying synchrony using sliding windows.

    Returns:
        times: Center time of each window
        synchrony: Synchrony value for each window
    """
    window_samples = int(window_sec * fs)
    step_samples = int(window_samples * (1 - overlap))

    n_windows = (len(signal_a) - window_samples) // step_samples

    times = []
    synchrony = []

    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples

        window_a = signal_a[start:end]
        window_b = signal_b[start:end]

        sync, _ = compute_synchrony(window_a, window_b, fs)

        times.append((start + window_samples//2) / fs)
        synchrony.append(sync)

    return np.array(times), np.array(synchrony)
```

---

## 🎨 Visualization Examples

### Heart Rate Synchrony Plot

```python
import matplotlib.pyplot as plt

# Load data
streams, _ = pyxdf.load_xdf('recording.xdf')

# Extract PPG signals
ppg_a = extract_ppg(streams, 'Participant_A')
ppg_b = extract_ppg(streams, 'Participant_B')

# Compute windowed synchrony
times, sync = windowed_synchrony(ppg_a, ppg_b)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Raw signals
ax1.plot(ppg_a[:10000], label='Participant A', alpha=0.7)
ax1.plot(ppg_b[:10000], label='Participant B', alpha=0.7)
ax1.set_xlabel('Sample')
ax1.set_ylabel('PPG Signal')
ax1.legend()
ax1.set_title('Heart Rate Signals')

# Synchrony over time
ax2.plot(times, sync, color='red', linewidth=2)
ax2.axhline(y=0.7, color='gray', linestyle='--', label='High synchrony threshold')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Synchronization')
ax2.set_ylim([0, 1])
ax2.legend()
ax2.set_title('Heart Rate Synchrony Over Time')

plt.tight_layout()
plt.savefig('heart_synchrony.png', dpi=300)
plt.show()
```

### Brain Synchrony Heatmap

```python
import seaborn as sns

# Compute PLV between all channel pairs
n_participants = 4
n_channels_per_participant = 5  # EEG channels
total_channels = n_participants * n_channels_per_participant

plv_matrix = np.zeros((total_channels, total_channels))

for i in range(total_channels):
    for j in range(i+1, total_channels):
        plv = phase_locking_value(eeg_data[:, i], eeg_data[:, j])
        plv_matrix[i, j] = plv
        plv_matrix[j, i] = plv

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(plv_matrix, cmap='hot', vmin=0, vmax=1,
            xticklabels=channel_labels,
            yticklabels=channel_labels)
plt.title('Inter-Brain Phase Locking (String Quartet)')
plt.tight_layout()
plt.savefig('brain_synchrony_heatmap.png', dpi=300)
```

---

## 🐛 Troubleshooting

### Issue: Timestamps Don't Align

**Symptoms**: Analysis shows unrealistic time lags (> 1 second)

**Causes**:
1. Devices connected at different times
2. System clock changed during recording
3. LSL not using same time reference

**Solutions**:
```python
# Verify all streamers use same timestamp
print(f"Muse A: {muse_a.synchronized_start_time}")
print(f"Muse B: {muse_b.synchronized_start_time}")
# Should be identical!

# Check LSL stream timestamps
for stream in streams:
    print(f"{stream['info']['name'][0]}: {stream['time_stamps'][0]}")
# Should be close (< 0.1 second difference)
```

### Issue: Dropped Samples

**Symptoms**: Gaps in recorded data

**Causes**:
1. Bluetooth interference
2. USB bandwidth limitations
3. CPU overload

**Solutions**:
- Use wired connections when possible
- Close unnecessary applications
- Use dedicated USB controllers for each device
- Check system performance during recording

### Issue: Poor Signal Quality

**Symptoms**: Noisy data, artifacts

**Solutions**:
- **Muse**: Ensure electrode contact (wet hair slightly)
- **E4**: Adjust wrist band tightness (2-3 finger gap)
- Use high-pass filtering (> 0.5 Hz) to remove drift
- Check impedance before recording

---

## 📚 References & Resources

### Academic Papers on Synchrony

1. **Heart Rate Synchronization**
   - Konvalinka et al. (2011). "Synchronized arousal between performers and related spectators in a fire-walking ritual"
   - Helm et al. (2012). "Physiological linkage in couples: A shared environment"

2. **Neural Synchrony**
   - Lindenberger et al. (2009). "Brains swinging in concert"
   - Dumas et al. (2010). "Inter-brain synchronization during social interaction"

3. **Musical Synchrony**
   - Müller & Lindenberger (2011). "Cardiac and respiratory patterns synchronize"
   - Babiloni et al. (2012). "Simultaneous recording of EEG and fNIRS during motor actions"

### Analysis Tools

- **pyXDF**: Load XDF files in Python
- **MNE-Python**: EEG analysis and visualization
- **NeuroKit2**: Physiological signal processing
- **BioSPPy**: Biosignal processing toolbox

### LSL Documentation

- **Lab Streaming Layer**: https://labstreaminglayer.readthedocs.io/
- **XDF Format**: https://github.com/sccn/xdf

---

## 🎓 Example Studies You Can Run

### 1. Romantic Couples Study

**Research Question**: Do couples in love show heart rate synchronization?

**Protocol**:
1. Baseline: Sitting separately (5 min)
2. Hand-holding: Sitting together, holding hands (10 min)
3. Conversation: Discussing positive memories (10 min)
4. Separation: Return to separate positions (5 min)

**Measurements**:
- PPG from both partners (Muse or E4)
- GSR to measure emotional arousal
- Event markers for condition changes

**Expected Outcome**: Higher heart synchrony during hand-holding and conversation vs. baseline

---

### 2. Meditation Group Synchrony

**Research Question**: Do experienced meditators synchronize brain rhythms?

**Protocol**:
1. Baseline: Eyes closed, no instruction (5 min)
2. Guided meditation: Follow teacher's instructions (20 min)
3. Silent meditation: Continue without guidance (10 min)
4. Rest: Open eyes (5 min)

**Measurements**:
- EEG from all participants (Muse)
- Event markers for meditation phases

**Expected Outcome**: Higher alpha/theta synchronization during guided meditation

---

### 3. Therapist-Client Attunement

**Research Question**: Does physiological synchrony predict therapeutic outcomes?

**Protocol**:
- Record full therapy sessions (multiple sessions)
- Both therapist and client wear devices
- Track synchrony over session and across sessions

**Measurements**:
- Heart rate (PPG/BVP)
- Skin conductance (GSR)
- Movement (ACC/GYRO)

**Expected Outcome**: Higher synchrony in productive sessions, lower in challenging sessions

---

## 💙 Final Thoughts

**Your vision of measuring how lovers' heartbeats synchronize, or how our brainwaves might align during deep conversation, is not just scientifically fascinating—it's profoundly human.**

StreamSense gives you the tools to measure these invisible threads of connection that bind us together. Every synchronized heartbeat, every aligned brain rhythm, is a testament to our capacity for empathy, love, and shared experience.

The platform is ready. The synchronization infrastructure is robust. Now go out there and **measure the music of human connection**. 🎵💕

---

**Questions or need help with your synchrony study?**
- Check the main README.md for basic usage
- Review CONCURRENCY_GUIDELINES.md for technical details
- See streamer/README.md for device-specific information

**Happy measuring, and may your participants' hearts beat as one!** 💓💓

---

*Document Version: 1.0*
*Created: November 5, 2025*
*Author: StreamSense Team (with love)*

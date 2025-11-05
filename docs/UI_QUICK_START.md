# StreamSense UI - Quick Start Guide

**Professional Dashboard for Job Demonstrations**

---

## 🚀 Launch the UI

```bash
cd StreamSense
python ui/streamsense_ui.py
```

The professional dark-themed dashboard will open showing:
- **Left Panel**: Device discovery and recording controls
- **Right Panel**: Live LSL streams display

---

## 📱 Workflow for Job Demonstrations

### 1. Discover Devices

1. Click **"🔍 Discover Devices"** button
2. Wait while the system scans for:
   - Muse headbands (via BLED112 dongles)
   - Empatica E4 devices (via E4 Server)
   - BITalino devices (via Bluetooth)

3. Discovered devices appear as cards showing:
   - Device name
   - Device type
   - Connection status
   - Signal quality bar
   - Connect/Disconnect button

**Status messages** appear at the bottom showing discovery progress.

---

### 2. Connect to Devices

1. Click **"Connect"** on any discovered device card
2. Watch the status change:
   - ● Disconnected → ● Connected
   - Button changes to "Disconnect" (red)
   - Signal quality bar fills up
   - Status message confirms connection

3. Connect multiple devices for multi-device demonstrations!

**Tip**: Connected devices automatically start streaming to LSL.

---

### 3. Monitor Live Streams

The **right panel** shows all active LSL streams in real-time:
- 🧠 EEG streams
- ❤️ ECG/BVP streams
- 💧 EDA/GSR streams
- 💪 EMG streams
- 📍 Accelerometer streams
- And more...

Streams appear automatically as devices connect.

---

### 4. Start Recording

1. Click **"● Start Recording"** button
2. The button changes to **"■ Stop Recording"** (red)
3. Session ID appears: `Recording session: 20251105_143022`
4. Live duration timer updates every second: `Duration: 00:00:15`

All active LSL streams are now being recorded to:
```
Documents/StreamSense/[timestamp]/RawData/
```

**What's being recorded:**
- Raw sensor data from all connected devices
- High-precision LSL timestamps
- Automatically synchronized across devices

---

### 5. Stop Recording

1. Click **"■ Stop Recording"**
2. Recording stops and data is saved
3. Datasets are processed and saved to:
   ```
   Documents/StreamSense/[timestamp]/Dataset/
   ```

**Saved files:**
- `eeg_dataset.pkl` - EEG data in MNE format
- `ppg_dataset.pkl`, `bvp_dataset.pkl` - Heart rate data
- `acc_dataset.pkl`, `gyro_dataset.pkl` - Movement data
- `gsr_dataset.pkl`, `temp_dataset.pkl` - E4 data
- And more...

---

### 6. Disconnect Devices

1. Click **"Disconnect"** on any device card
2. Device stops streaming
3. Status returns to ● Disconnected

**Or close the window** - all devices disconnect automatically with graceful cleanup.

---

## 🎯 Job Interview Demonstration Script

### Scenario: Multi-Device Physiological Synchrony

**What to say:**
> "StreamSense is a multi-device physiological recording platform I built for synchrony research. Let me demonstrate the full workflow."

**1. Launch UI** (2 seconds)
```bash
python ui/streamsense_ui.py
```

**2. Discover devices** (10 seconds)
- Click "Discover Devices"
- "As you can see, it automatically discovers Muse headbands, Empatica E4 wristbands, and BITalino sensors across different connection types - Bluetooth LE, WiFi, and serial."

**3. Connect multiple devices** (5 seconds per device)
- Click "Connect" on 2-3 devices
- "Each device connects independently and starts streaming to Lab Streaming Layer, which provides microsecond-precision timestamps for perfect synchronization."

**4. Show live streams** (5 seconds)
- Point to right panel
- "Here you can see all the active streams - EEG from the headband, heart rate from the wristband, skin conductance, accelerometer data. All synchronized in real-time."

**5. Start recording** (2 seconds)
- Click "Start Recording"
- "Recording is as simple as one button click. The system captures all streams simultaneously with LSL timestamps."

**6. Brief pause** (10 seconds)
- Let recording run
- "The platform handles reconnection automatically if devices lose connection, and interpolates missing data intelligently based on signal type."

**7. Stop recording** (2 seconds)
- Click "Stop Recording"
- "When we stop, the data is immediately processed into analysis-ready formats - MNE for EEG, pandas DataFrames for other sensors."

**8. Explain use cases** (15 seconds)
- "This enables research on physiological synchrony - measuring how people's heartbeats or brain waves synchronize during interaction. Applications include couples therapy, team dynamics, meditation research, and more."

**Total demo time:** ~1 minute

---

## 🎨 Demo Mode (Without Hardware)

If you need to demonstrate without actual devices:

1. Edit `ui/streamsense_ui.py`
2. Find the `main()` function (line ~756)
3. Uncomment the demo device lines:

```python
# Uncomment for demo mode:
window.add_device("Muse-A01B", "Muse Headband")
window.update_device_status("Muse-A01B", True, 92)
window.add_device("E4-12345", "Empatica E4")
window.update_device_status("E4-12345", True, 87)
window.add_device("BITalino-001", "BITalino (r)evolution")
window.update_device_status("BITalino-001", False)
```

Now the UI shows **pre-populated devices** for presentation purposes.

**Note:** Demo devices won't actually stream or record, but the UI shows all functionality.

---

## ⚙️ Technical Architecture (For Technical Interviews)

**When asked about technical design:**

### Backend Controller Pattern
```python
StreamSenseUI (PyQt5 View)
    ↓ Qt Signals/Slots
StreamSenseController (Business Logic)
    ↓ Direct API calls
Core Components (FindDevices, StreamMuse, StreamE4, StreamRecorder)
    ↓ LSL Protocol
Hardware Devices
```

**Key design decisions:**

1. **Separation of Concerns**
   - UI only handles display and user input
   - Controller manages all device/recording logic
   - Core components handle hardware communication

2. **Thread Safety**
   - Qt signals for cross-thread communication
   - Background threads for blocking operations
   - No UI blocking during device discovery/connection

3. **Process-Based Streaming**
   - Each device runs in separate Process (BaseStreamer)
   - Crash isolation - one device failure doesn't affect others
   - True parallelism for multi-device streaming

4. **Error Handling**
   - Try-except blocks at every hardware interaction
   - User-friendly error dialogs
   - Graceful degradation (partial device failures OK)

5. **LSL for Synchronization**
   - Industry-standard protocol for physiological data
   - Microsecond-precision timestamps
   - Network-transparent streaming

---

## 🔧 Troubleshooting

### "No devices found"

**Muse:**
- Ensure BLED112 dongle is plugged in
- Check Device Manager (Windows) for "Bluegiga Bluetooth Low Energy"
- Make sure Muse is charged and in pairing mode

**E4:**
- Launch "Empatica E4 Streaming Server" first
- Connect E4 to BLE dongle in the server
- Check that EmpaticaBLEServer.exe is running

**BITalino:**
- Ensure Bluetooth is enabled
- BITalino should be discoverable
- Try pairing in system Bluetooth settings first

---

### "Connection timeout"

- Device may be in use by another application
- Try resetting the device
- Check if device is within range (Bluetooth: ~10m)
- For Muse: try different BLED112 dongle port

---

### "Recording failed to start"

- Ensure at least one device is connected and streaming
- Check that Documents folder has write permissions
- Verify LSL streams are active (right panel shows streams)

---

### UI not responding

- Device discovery can take 10-30 seconds (scanning all interfaces)
- Device connection can take 5-15 seconds (handshake protocols)
- UI will resume once operations complete
- Check status message for progress

---

## 📊 Output Files Explained

After recording, check:
```
Documents/StreamSense/[timestamp]/
├── RawData/              # HDF5 files with raw sensor data
│   ├── Muse-A01B_EEG.h5
│   ├── Muse-A01B_PPG.h5
│   ├── E4-12345_BVP.h5
│   └── ...
└── Dataset/              # Processed datasets
    ├── eeg_dataset.pkl   # MNE RawArray objects
    ├── ppg_dataset.pkl   # Pandas DataFrames
    └── ...
```

**Load in Python:**
```python
import pickle
import pandas as pd
import mne

# Load EEG data
with open('Dataset/eeg_dataset.pkl', 'rb') as f:
    eeg_data = pickle.load(f)

# Access specific device
muse_eeg = eeg_data['Muse-A01B_EEG']['data']  # MNE RawArray
sampling_rate = eeg_data['Muse-A01B_EEG']['sfreq']

# Load PPG data
with open('Dataset/ppg_dataset.pkl', 'rb') as f:
    ppg_data = pickle.load(f)

# Access as DataFrame
muse_ppg = ppg_data['Muse-A01B_PPG']['data']  # pandas DataFrame
```

---

## 🎯 Questions to Expect and How to Answer

### "How did you handle multi-threading in the UI?"

> "The UI uses Qt signals and slots for thread-safe communication. Blocking operations like device discovery run in background threads, emitting signals to update the UI on the main thread. This prevents UI freezing while maintaining safety."

---

### "How do you ensure timestamp synchronization across devices?"

> "All devices use Lab Streaming Layer (LSL), which provides a shared network time protocol. Each sample gets an LSL timestamp at capture time, synchronized across the network to microsecond precision. This enables accurate cross-device correlation analysis."

---

### "What happens if a device disconnects mid-recording?"

> "The BaseStreamer architecture isolates each device in its own Process. If one fails, others continue. The StreamRecorder detects disconnections using timeout thresholds and implements exponential backoff reconnection. For brief disconnections (<2 minutes), it interpolates missing samples using spline or linear interpolation based on signal type."

---

### "How is this different from existing tools?"

> "Most tools support single devices or require manual synchronization. StreamSense provides:
> 1. Multi-vendor support (Muse, E4, BITalino) in one platform
> 2. Automatic LSL-based synchronization
> 3. Extensible architecture via BaseStreamer pattern
> 4. Professional UI for easy operation
> 5. Analysis-ready output (MNE format for EEG)
>
> It's designed specifically for social neuroscience - measuring synchrony between people's physiological signals."

---

## 📚 Next Steps

After your demo, mention:

- **Roadmap**: "I'm expanding support to Polar H10, Emotiv, OpenBCI, and even mobile phone sensors"
- **Open Source**: "The architecture is designed for community contributions - adding a new device only takes 1-2 days"
- **Research Applications**: "I built this for my physiological synchrony research, but it's applicable to neuroscience, HCI, psychology..."
- **Documentation**: "I've written comprehensive guides including a Multi-Device Synchronization Guide with analysis examples"

---

## ✨ Pro Tips

1. **Practice the demo** - run through it 5 times before the interview
2. **Have backup data** - pre-record a session in case hardware fails
3. **Know your numbers**:
   - Muse: 256 Hz EEG, 5 channels
   - E4: 64 Hz BVP, 4 Hz GSR
   - BITalino: Up to 1000 Hz, 6 channels
4. **Explain edge cases** - show you thought about robustness
5. **Connect to their domain** - adapt the physiological synchrony example to their research/product area

---

## 🎓 Related Documentation

- `MULTI_DEVICE_SYNCHRONIZATION_GUIDE.md` - Analysis techniques
- `DEVICE_SUPPORT_ROADMAP.md` - Expansion plans
- `streamer/README.md` - BaseStreamer API
- `CONCURRENCY_GUIDELINES.md` - Architecture decisions

---

**Good luck with your job interview! You've got this! 🚀**

*The UI backend is fully integrated and ready for professional demonstrations.*

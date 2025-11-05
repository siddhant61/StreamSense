# Device Support Roadmap

**Goal**: Make StreamSense the universal platform for physiological synchrony research

---

## Current Devices ✅

| Device | Sensors | Status | Quality |
|--------|---------|--------|---------|
| **Muse Headband** | EEG (5ch), PPG, ACC, GYRO | ✅ Implemented | Research-grade |
| **Empatica E4** | BVP, GSR, TEMP, ACC | ✅ Implemented | Research-grade |

---

## Priority 1: Next 2 Weeks ⭐⭐⭐⭐⭐

### BioTalino (r)evolution
- **Sensors**: ECG, EDA, EMG, EEG, ACC, Light, 6 analog inputs
- **Cost**: €149-€299
- **Complexity**: Low (Python SDK available)
- **Use cases**: Multi-modal biosignal research, affordable EMG/ECG
- **Implementation time**: 2-3 days
- **Why first**: Open-source, multi-modal, easy integration, fills sensor gaps

### Polar H10
- **Sensors**: ECG, HR, HRV, ACC
- **Cost**: ~$90
- **Complexity**: Low (Bluetooth LE)
- **Use cases**: Gold standard heart rate measurement
- **Implementation time**: 1-2 days
- **Why second**: Best HR accuracy, simple integration

---

## Priority 2: Next Month ⭐⭐⭐⭐

### Emotiv EPOC X / Insight
- **Sensors**: EEG (5-14 channels), Gyro
- **Cost**: $299-$849
- **Complexity**: Medium (Cortex API, licensing)
- **Use cases**: More EEG channels than Muse, facial expression detection
- **Implementation time**: 3-4 days
- **Why**: Popular Muse alternative, more research channels

### OpenBCI Cyton/Ganglion
- **Sensors**: EEG/ECG/EMG (8-16 channels)
- **Cost**: $199-$999
- **Complexity**: Medium (Python SDK available)
- **Use cases**: Research-grade multi-channel EEG, ultra-flexible
- **Implementation time**: 3-4 days
- **Why**: Open-source gold standard, maximum flexibility

---

## Priority 3: Future Expansion ⭐⭐⭐

### Mobile Phone Sensors
- **Sensors**: ACC, Gyro, Camera (PPG), Microphone
- **Cost**: Free (users own devices)
- **Complexity**: High (needs companion app, WebSocket streaming)
- **Use cases**: Movement synchrony, remote PPG, interaction studies
- **Implementation time**: 5-7 days (app + integration)
- **Why**: Ubiquitous, enables remote studies, creative applications

### Tobii Eye Tracker
- **Sensors**: Gaze position, pupil dilation
- **Cost**: $229-$25,000
- **Complexity**: Medium (SDK available)
- **Use cases**: Attention, arousal, joint attention studies
- **Implementation time**: 3-5 days
- **Why**: Unique data modality, growing in research

### NeuroSky MindWave
- **Sensors**: Single-channel EEG
- **Cost**: $99
- **Complexity**: Low
- **Use cases**: Entry-level EEG, education
- **Implementation time**: 1-2 days
- **Why**: Very affordable, good for education/outreach

---

## Priority 4: Advanced Devices ⭐⭐

### Smart Watches (Apple Watch, Fitbit, Garmin)
- **Sensors**: HR, HRV, SpO2, ACC, Gyro
- **Cost**: $150-$800 (users own)
- **Complexity**: High (limited APIs, real-time restrictions)
- **Use cases**: Long-term monitoring, daily life studies
- **Implementation time**: 7-10 days per platform
- **Why later**: API limitations, not real-time, significant development effort

### fNIRS (Functional Near-Infrared Spectroscopy)
- **Sensors**: Brain hemodynamics
- **Cost**: $10,000-$100,000
- **Complexity**: High
- **Use cases**: Brain imaging during naturalistic tasks
- **Implementation time**: 10+ days
- **Why later**: Very specialized, expensive, complex

---

## Device Plugin Architecture

### BaseStreamer Pattern (Already Implemented! ✅)

All new devices follow the same pattern:

```python
class StreamNewDevice(BaseStreamer):
    def __init__(self, device_name, synchronized_start_time, root_output_folder):
        super().__init__(device_name, synchronized_start_time, root_output_folder)
        # Device-specific initialization

    def _stream_wrapper(self):
        # 1. Connect to device
        # 2. Setup LSL outlets
        # 3. Signal connected
        # 4. Stream loop
        pass

    def _setup_lsl_outlets(self):
        # Create LSL outlets for each sensor
        pass
```

### Benefits of This Architecture:
- ✅ Consistent interface across all devices
- ✅ Automatic multi-device synchronization
- ✅ Context manager support
- ✅ Robust error handling
- ✅ Process-based isolation
- ✅ Easy to test
- ✅ Clear documentation pattern

---

## Implementation Checklist

For each new device:

- [ ] **Research phase** (1-2 hours)
  - [ ] Find Python SDK/library
  - [ ] Review API documentation
  - [ ] Test basic connection
  - [ ] Identify data streams

- [ ] **Implementation** (4-8 hours)
  - [ ] Create `StreamXXX` class inheriting BaseStreamer
  - [ ] Implement `_stream_wrapper()`
  - [ ] Implement `_setup_lsl_outlets()`
  - [ ] Handle device-specific errors
  - [ ] Add reconnection logic if needed

- [ ] **Testing** (2-4 hours)
  - [ ] Create mock device for testing
  - [ ] Write unit tests (following test patterns)
  - [ ] Test with real hardware
  - [ ] Verify LSL timestamp alignment

- [ ] **Integration** (1-2 hours)
  - [ ] Add to FindDevices discovery
  - [ ] Add to main.py menu
  - [ ] Update documentation
  - [ ] Add to examples

- [ ] **Documentation** (1-2 hours)
  - [ ] Device-specific README
  - [ ] Add to MULTI_DEVICE_SYNCHRONIZATION_GUIDE
  - [ ] Example use cases
  - [ ] Troubleshooting section

**Total per device**: ~1-3 days depending on complexity

---

## Quality Standards

### All device implementations must:
- ✅ Inherit from BaseStreamer
- ✅ Have 80%+ test coverage
- ✅ Use Event-based stop signaling (not bool)
- ✅ Include reconnection logic
- ✅ Have comprehensive docstrings
- ✅ Follow LSL naming conventions
- ✅ Handle errors gracefully
- ✅ Include example code
- ✅ Document sensor specifications

---

## Community Contribution Guidelines

### How Others Can Add Devices:

1. **Fork repository**
2. **Create device streamer** following BaseStreamer pattern
3. **Add tests** (minimum 80% coverage)
4. **Add documentation** (README + examples)
5. **Submit PR** with:
   - Device information
   - SDK/library dependencies
   - Hardware requirements
   - Test results
   - Example recording

### Example PR Template:
```markdown
## New Device: BioTalino

**Device info**:
- Name: BioTalino (r)evolution
- Sensors: ECG, EDA, EMG, EEG, ACC
- SDK: `bitalino` Python package
- Cost: €149-€299

**Implementation**:
- [x] StreamBioTalino class
- [x] Unit tests (85% coverage)
- [x] Integration tests
- [x] Documentation
- [x] Example recording

**Hardware tested**: Yes, with 6-channel BioTalino kit
**LSL streams verified**: ECG, EDA, EMG, ACC working

**Breaking changes**: None
```

---

## Long-term Vision

### Year 1: Foundation
- ✅ Muse, E4 (Done!)
- 🔄 BioTalino, Polar H10
- 🔄 Emotiv, OpenBCI

### Year 2: Expansion
- Mobile phone integration
- Eye trackers
- Smart watches (limited API support)
- Community-contributed devices

### Year 3: Ecosystem
- StreamSense becomes **the** multi-device research platform
- 20+ supported devices
- Active community
- Pre-configured study protocols
- Cloud synchronization for remote studies

---

## Success Metrics

**Device ecosystem health**:
- Number of supported devices
- Community contributions per month
- Device documentation completeness
- Average time to add new device
- User-reported compatibility issues

**Research impact**:
- Publications using StreamSense
- Number of simultaneous devices in typical study
- Cross-device synchrony accuracy
- Geographic distribution of users

---

## Current Status: January 2025

✅ **Architecture ready** (BaseStreamer + LSL)
✅ **Documentation complete** (developer guides)
✅ **Testing framework** (pytest + mocks)
✅ **Multi-device sync proven** (Muse + E4 working)

**Next step**: Add BioTalino support (highest priority, easiest integration)

---

*This roadmap is living document. Update as devices are added and priorities shift.*

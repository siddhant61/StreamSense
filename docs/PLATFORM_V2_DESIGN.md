# StreamSense Platform v2 — Design

> Status: **in progress** (foundation). Branch: `claude/streamsense-platform`.
> Builds on the stabilized base (PR #21).

## Goal

A single, modern control surface to run a personal multi-modal research rig:

| Modality | Device | Mode | Transport |
|----------|--------|------|-----------|
| EEG/PPG/ACC/GYRO | **Muse S** | **live** | BLE (`muselsl`/`pygatt`) → LSL |
| ECG/EDA/EMG/EEG/ACC | **BITalino** | **live** | BLE/serial (`bitalino`) → LSL |
| Body/skeleton + IMU; RGB/depth video | **Azure Kinect** | **live** | `pyk4a` + Body Tracking SDK → LSL (joints/IMU) + `.mkv` sidecar + sync markers |
| BVP/EDA/HR/IBI/TEMP/ACC/tags | **Empatica E4** | **import only** | Empatica withdrew the streaming server → no live path; import E4 Connect session archives post-hoc |

LSL remains the live bus; the recorder writes the synchronized session.

## Architecture

```
                +-------------------+        WebSocket (live status)
   Browser  <-->|  Web frontend     |<--------------------------+
   (Vite/React) |  (frontend/)      |                           |
                +---------+---------+                           |
                          | REST (/api/*)                       |
                +---------v-----------------------------+       |
                |  FastAPI app (api/app.py)             |-------+
                +---------+-----------------------------+
                          | calls (framework-agnostic)
                +---------v-----------------------------+
                |  DeviceManager (core/device_manager)  |  <-- pure Python, unit-tested headless
                |  - discover/connect/record/status     |
                |  - EventBus (listeners)               |
                +----+-------------------+--------------+
                     | drivers (lazy hw imports)
       +-------------+------+-------------+-----------------+
       | MuseDriver | BitalinoDriver | KinectDriver(*) | E4ImportDriver |
       +------------+----------------+-----------------+----------------+
              \           \                |                 (import)
               \           \               v
                ` -> streamer.* (BaseStreamer subclasses) -> LSL -> StreamRecorder
   (*) Kinect driver: PR-2
```

**Key principle:** `core/` never imports FastAPI or any hardware library at module load. Drivers
import hardware deps *lazily* inside methods and report availability via `available()`, so the
core + API import and test cleanly in a headless CI with no devices installed.

## API surface (PR-1)

- `GET  /api/health`
- `GET  /api/devices` — known devices + state
- `POST /api/discover` — `{types?: [...]}` → discovered devices
- `POST /api/devices/{id}/connect` / `/disconnect`
- `POST /api/recording/start` / `/stop`
- `GET  /api/status` — system + recording + per-device
- `GET  /api/streams` — active LSL streams
- `WS   /ws` — live `device_update` / `recording` / `status` / `log` events

## Honest constraints

- **E4 is post-hoc only** (server withdrawn) — it cannot be part of the live synchronized capture; it is aligned by timestamps/tags after the fact.
- **Signal quality** is reported as `null` until a real per-stream metric is implemented (no fabricated values).
- Raw Kinect video is **not** pushed over LSL (bandwidth) — recorded to file + LSL sync markers.

## PR series

1. **Foundation (done, PR #22):** `core/` device manager + drivers (Muse/BITalino real, Kinect/E4 declared) + FastAPI API + headless tests + web frontend scaffold.
2. **Kinect streamer (in progress):** `StreamKinect` (BaseStreamer) — body-tracking joints (32×8=256 ch) + IMU → LSL, RGB/depth → `.mkv` sidecar, per-frame `SYNC` markers on the LSL clock for post-hoc alignment. All hardware calls isolated behind an injectable `KinectBackend`; loop + shaping + specs unit-tested headless, `PyK4ABackend` (pyk4a + pykinect_azure body tracking) awaits on-device verification.
3. **E4 importer:** parse E4 Connect archives → align into session dataset.
4. **Acquisition optimization (in progress):** unified `SessionClock`, `ExponentialBackoff` + stop-aware `retry_with_backoff` (per-device reconnect), and a real `signal_quality` SQI (0..1, `None` when unknown) wired into `DeviceManager` — fabricated 92/87/85 removed from the legacy UI. Sleep audit: of 41 `time.sleep`, **24 are in the retiring E4 live path** (replaced wholesale by PR-3 import, not refactored); `BaseStreamer` already uses a **queue/event handshake** for connection (no coordination sleep); the remainder are genuine BLE/serial settle delays, kept and documented. New event/queue handshake replaces fixed-delay reconnect loops.
5. **Frontend build-out:** device cards, signal quality, stream monitor, Kinect preview, session manager.

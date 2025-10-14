# StreamSense Deep Audit & Delivery Roadmap

## Current State Overview
- The operational entry point (`main.py`) is a command-line orchestrator that depends on global state, Windows-only libraries (`wmi`), and manual thread coordination, highlighting the absence of a cross-platform runtime or dedicated GUI shell.【F:main.py†L1-L159】
- Live visualization exists only as a thin wrapper around Vispy canvas creation, and the viewer currently mutates a sequence of `StreamInfo` objects as if it were a dictionary, so detected streams cannot be validated or filtered reliably before plotting.【F:viewer/view_streams.py†L18-L118】
- Device adapters stream directly to LSL but override timestamps with ad-hoc offsets instead of synchronizing against the shared LSL clock, which risks drift and breaks cross-device alignment (e.g., Empatica E4 uses elapsed device time, Muse pushes raw device timestamps without alignment).【F:streamer/stream_e4.py†L153-L215】【F:streamer/stream_muse.py†L200-L289】

## Key Technical Gaps & Risks
1. **Runtime & UX limitations**  
   - No graphical control surface or headless API; all workflows require manual CLI menu navigation in `main.py`, blocking novice operators and automation scenarios.【F:main.py†L167-L195】
   - Hard Windows dependencies (`wmi`, Empatica BLE server check) prevent Linux/macOS deployment, which also blocks containerized or cloud streaming setups.【F:main.py†L102-L156】

2. **Visualization pipeline fragility**  
   - `ViewStreams.start_viewing` tries to treat the stream list as a mapping, so it silently drops handles to verified streams; error handling is broad `except: pass`, masking discovery failures.【F:viewer/view_streams.py†L92-L100】
   - Rendering launches a separate process per canvas but never tears down Vispy's event loop cleanly, which complicates multi-stream dashboards and hinders embedding into a GUI shell.【F:viewer/view_streams.py†L58-L117】

3. **Timing & synchronization debt**  
   - Empatica timestamps are rewritten to `elapsed_time` from the first sample instead of the LSL clock, so reconnections and long recordings cannot be merged accurately with other sensors.【F:streamer/stream_e4.py†L160-L215】
   - Muse outlets push the raw device timestamps without compensating for transport lag or start offsets, leading to inconsistent sample spacing and duplicate filtering hacks (`recent_data_cache`).【F:streamer/stream_muse.py†L200-L289】

4. **Extensibility bottlenecks**  
   - Streamer implementations are monolithic; new devices would require copy/paste adapters rather than reusable interfaces for BLE, serial, or network transports.【F:streamer/stream_muse.py†L1-L412】【F:streamer/stream_e4.py†L1-L249】
   - Recording, event logging, and experiments couple tightly to the existing device set, so additional sensors (smart watches, BITalino, etc.) would need cross-cutting updates in recorder, viewer, and CLI flows.【F:main.py†L43-L158】

## Proposed Roadmap
1. **Foundation & Infrastructure (Milestone A)**  
   - Replace global state in `main.py` with a controller service that exposes explicit lifecycle hooks for discovery, streaming, recording, and visualization.  
   - Introduce a platform abstraction layer for device discovery so Windows-specific code (e.g., `wmi`) can be swapped with cross-platform providers; mock hardware via dependency injection for tests.  
   - Add baseline automated tests and a dependency-locked environment to guard regressions before feature expansion.

2. **Visualization & Operator Experience (Milestone B)**  
   - Refactor `view_streams.py` to return concrete stream metadata, eliminate the list/dict mutation bug, and surface discovery errors.  
   - Wrap the Vispy canvases inside a Qt (or web) dashboard that can display multiple streams, status indicators, and controls in a single process for session operators.  
   - Provide a REST/gRPC layer for remote control and future GUI clients.

3. **Clock Synchronization & Data Integrity (Milestone C)**  
   - Align Empatica and Muse timestamp handling with LSL best practices by referencing `local_clock()` offsets at connection time and persisting those offsets across reconnects.  
   - Build a synchronization monitor that visualizes per-stream latency, jitter, and drift, feeding back into the recorder to persist correction metadata.  
   - Extend the recorder to optionally resample streams to common time bases, preparing for multimodal analytics.

4. **Sensor Expansion Framework (Milestone D)**  
   - Design a plugin interface for streamers with reusable BLE/serial/WebSocket backends and shared retry, buffering, and timestamp utilities.  
   - Implement adapters for high-priority devices: consumer smartwatches (Wear OS/Apple Watch via companion bridge), BITalino (BLE/USB), and other lab sensors, each exposing discovery metadata for automation.  
   - Update the recorder and visualization layers to rely on generic stream descriptors so new devices flow through without bespoke code.

5. **Advanced Operations & Research Tooling (Milestone E)**  
   - Integrate protocol scripting (e.g., oddball paradigms) with synchronized triggers and the event logger using the unified controller service.  
   - Add session templates, tagging, and annotation APIs to streamline study setup.  
   - Package the system with deployment recipes (desktop installer, Docker, cloud) targeting labs and remote data collection teams.

## Suggested Next Steps
1. Approve Milestone A scope and create detailed tasks for controller refactor, platform abstraction, and CI setup.  
2. Schedule design workshops for visualization architecture and device plugin interfaces.  
3. Prioritize external device targets (smartwatch platforms, BITalino models, other sensors) to inform adapter sequencing in Milestone D.

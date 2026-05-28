# Deliverable 4: Technical Debt Report (Refreshed — 2026-05-28)

> Supersedes the Aug-2025 and Nov-2025 versions. Every item carries `file:line` evidence.
> Severity: 🔴 critical · 🟠 high · 🟡 medium · ⚪ low.

---

## 1. Correctness Bugs (not just smells)

| # | Sev | Issue | Evidence |
|---|-----|-------|----------|
| C1 | 🔴 | **UI cannot connect E4** — `StreamE4(device_id=…, output_path=…)` but ctor is `StreamE4(e4, root_output_folder, synchronized_start_time)` → `TypeError`, swallowed as "Failed to connect". | `ui/streamsense_controller.py:258-262` vs `streamer/stream_e4.py:40` |
| C2 | 🔴 | **`data_processor.py` runs work at import** against `D:/Study Data/...` → `FileNotFoundError`; breaks `import data_processor` and `tests/test_data_processor.py` collection. | `data_processor.py:606-607` |
| C3 | 🔴 | **CLI import-crashes off Windows** — `import wmi` at module top. | `main.py:14` |
| C4 | 🟠 | **CLI can hang forever** — after `start_streaming()` returns `False` (timeout), `connected_event.wait()` blocks with no timeout. | `main.py:113-114`, contract `base_streamer.py:128-132` |
| C5 | 🟠 | **Wrong thread target** — `threading.Thread(target=self._monitor_connection(muse, …))` passes the *call result* (None/generator), not a callable. | `streamer/stream_muse.py:310` |
| C6 | 🟡 | **Unbounded thread creation** — a new `threading.Timer` per sample for cache eviction. | `streamer/stream_muse.py:250-251` |

## 2. Cross-Platform Debt (🔴 blocks the documented "Win/macOS/Linux" claim)

- `main.py:14` `import wmi`; `main.py:138-155` polls `Win32_Process()` for `EmpaticaBLEServer.exe`.
- `main.py:291` event-logger launch via `["start","cmd","/k"]` (Windows shell).
- `helper/e4_helper.py:26` hardcoded `D:/E4StreamingServer1.0.4.5400/EmpaticaBLEServer.exe`.
- `helper/muse_helper.py` `subprocess.call('start bluemuse:', shell=True)` (multiple sites).
- `archive/e4_basic_flow.py:7` same Windows EXE path.
- Good counter-example: `helper/serial_helper.py:31-39` conditionally imports `termios`.

## 3. Concurrency Debt

- **Hybrid model**: `multiprocessing.Process` per device + ad-hoc `threading`, daemon threads (E4 `handle_incoming_data`, server monitors) created without joins.
- **41 `time.sleep()`** synchronization points across `streamer/`, `helper/`, `recorder/`, `main.py`.
- **Tests deadlock under coverage**: any run including process-spawning tests (`test_base_streamer`, etc.) hangs >150s with `--cov`; per-file without coverage they pass. This blocks coverage-gated CI until fixed (e.g. `--cov-context`/`concurrency=multiprocessing` config or dependency-injected processes).

## 4. Design / Maintainability

- **Monolithic UI**: `ui/streamsense_ui.py` 771 LOC; **28 inline `setStyleSheet`** blocks with duplicated style strings; hardcoded geometry (1400×900, 450px panels).
- **Fake telemetry**: signal-quality values 92/87/85 are hardcoded (`controller.py:250,268,295`) — misleading in a research/demo tool.
- **Capability asymmetry**: BITalino is UI-only (CLI has 0 references); E4 works only via CLI. The two entry points expose different, partly-broken feature sets.
- **Duplication**:
  - `event_logger.py` ≈ `stream_info.py` (near-identical `log_event`).
  - `find_streams()` duplicated in `viewer/view_streams.py` and `viewer/plot_streams.py`.
  - Audit artifacts: `audit/feature_map.json` is byte-identical to `deliverables/3_*.json` (md5 `553c97…`); `audit/project_manifest.csv` == `deliverables/1_*.csv` (md5 `fb0704…`).
- **Stub / dead code**: `helper/plot_helper.py` (4 lines, 54 bytes); `archive/data_helper.py`, `archive/e4_basic_flow.py` (unreferenced).

## 5. Test & Process Debt

- Coverage uneven: `serial_helper.py` (846 LOC) 0%, `muse_helper.py` (707 LOC) 0%, `plot_streams.py` 0%, `stream_bitalino.py` 0% (no test file), `recorder` effectively ~0% (1 trivial test). Strong: `view_streams` 93%, `base_streamer` 79%.
- `tests/test_data_processor.py` cannot be collected (see C2).
- No CI workflow, no linter/formatter/type-check config, no lock file.

## 6. Repository Hygiene

- 8 **empty** `Logs/*.log` files committed; 8 `Logs/Slide20-27.JPG` slides misplaced under `Logs/`.
- `docs/screenshots/09_*.png` and `10_*.png` are **byte-identical** (md5 `e4bf6724…`); all 10 PNGs share the identical 3,787,081-byte size (suspect duplicate exports).
- 4 overlapping screenshot scripts; `scripts/capture_screenshots_headless.py` redundant.

## 7. Security

- 🟡 **Secret in git history**: the E4 API key (`7abb651d…`) was removed from code (commits `705b663`, `dea3ae7`; current `e4_helper.py:28` uses `os.getenv('E4_API_KEY')`) but **remains in git history** → recommend **rotating the key**, not just deleting the file.
- 🟡 **No dependency lock / audit**: pinned ranges only; no `pip-audit`/lock file → non-reproducible builds, unscanned CVEs.
- ⚪ Direct BLE/serial access requires elevated OS permissions (document, don't sandbox-break).

## 8. Hotspot Ranking (where to spend effort first)

| Rank | File | LOC | Why |
|------|------|-----|-----|
| 1 | `helper/e4_helper.py` | 678 | Windows EXE path, orphan processes, 20% cov, drives broken E4 path |
| 2 | `helper/muse_helper.py` | 707 | 0% cov, Windows shell calls, hardcoded packet sizes |
| 3 | `ui/streamsense_ui.py` | 771 | monolith, style duplication, fake metrics |
| 4 | `helper/serial_helper.py` | 846 | largest file, 0% cov, BGAPI complexity |
| 5 | `data_processor.py` | 608 | broken-on-import + orphaned |

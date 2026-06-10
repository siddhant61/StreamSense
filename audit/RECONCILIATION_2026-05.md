# Audit Reconciliation — 2026-05-28 vs. Prior Audits

This audit was run in **verify & supersede** mode. Below is how the current findings relate to
`audit/nov_2025_comprehensive_audit.md` (2025-11-05) and the prior `audit/deliverables/*`.

## 1. The prior audit is materially STALE

The Nov-2025 report states **"Total Files: 61"** and a **10-feature** matrix built around
`main.py` + helpers + recorder. Since then the repository has grown to **115 files** and gained
entire subsystems the prior audit does not mention:

| Added since Nov-2025 (not in prior audit) | Evidence |
|---|---|
| PyQt5 UI dashboard | `ui/streamsense_ui.py` (771 LOC), `ui/streamsense_controller.py` (449 LOC) |
| BITalino support | `streamer/stream_bitalino.py` (312 LOC) |
| `BaseStreamer` abstraction + migration | `streamer/base_streamer.py`; StreamMuse/E4/BITalino inherit it |
| Full mock-based test suite | `tests/` 14 `.py` + `tests/mocks/` (was "2 test files") |
| Screenshot/docs tooling | `scripts/*` (4 scripts), `docs/screenshots/*` |

→ The prior matrix (10 features, "2 tests") **no longer describes the project** and is superseded
by `deliverables/3_Feature_Status_Matrix.json` (15 features, 107 passing tests).

## 2. Prior findings now RESOLVED (verified)

| Prior claim | Current reality |
|---|---|
| "No requirements.txt" (Aug-2025 deliverable 2) | `requirements.txt` exists with **pinned ranges** |
| "0 tests / ~10% coverage" | **8 test files, 107 passing**, real mock layer |
| "Hardcoded API key in `archive/e4_basic_flow.py:8`" | **Removed** (commits `705b663`, `dea3ae7`); code now uses `os.getenv('E4_API_KEY')`. *Residual:* key still in git history → **rotate it** (new finding). |
| "CLI uses global state" | Refactored to `AppState` dataclass (`main.py:37-67`) |

## 3. Prior findings CONFIRMED (still true)

- Windows-only platform lock (`wmi`, `start cmd /k`, `D:/…EmpaticaBLEServer.exe`).
- Hybrid threading/multiprocessing complexity; heavy `time.sleep()` synchronization (**41** sites in core).
- Orphaned utilities (`stream_info.py`/`event_logger.py` not integrated; `data_processor.py` not wired in).
- `helper/plot_helper.py` is a 54-byte stub.
- Misplaced `Logs/Slide*.JPG`; empty `Logs/*.log`.

## 4. NEW findings the prior audit missed

1. **UI→E4 connect is Broken** — `StreamE4(device_id=…, output_path=…)` vs ctor `(e4, root_output_folder, …)` → `TypeError`. (`controller.py:258-262`)
2. **`data_processor.py` executes at import** (`:606-607`) → crashes import and test collection.
3. **CLI hang** — `connected_event.wait()` with no timeout after a failed start (`main.py:113-114`).
4. **`stream_muse.py:310`** passes a call result (not a callable) as a thread target.
5. **Fake UI signal quality** (hardcoded 92/87/85).
6. **Feature asymmetry** — BITalino is UI-only; E4 is CLI-only-working.
7. **Coverage deadlock** — process-spawning tests hang the suite under `--cov`.
8. **Audit-artifact duplication** — `feature_map.json`==deliverable 3; `project_manifest.csv`==deliverable 1 (byte-identical).
9. **Screenshot duplication** — `09_*.png`==`10_*.png` (md5 identical).

## 5. Net status delta

Prior overall: *"Stabilizing — no production-ready features."* Current assessment: the project has
**advanced in surface area** (UI, BITalino, tests) but **regressed in integration integrity** — there
are now **3 hard-Broken paths** (UI→E4, data_processor import, CLI off-Windows) that did not exist
or were not identified before. The good news: all three are **small, well-localized fixes** (see
roadmap P0).

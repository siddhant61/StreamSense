# StreamSense — Granular Audit Task Execution Log

**Protocol:** Granular Audit Protocol (read-only, evidence-based)
**Branch:** `claude/determined-gates-Zkfw7`
**Started:** 2026-05-28
**Mode:** Verify & supersede prior `audit/` artifacts. Reconcile findings.
**Tracker:** in-repo markdown (this file)

Status legend: ⬜ todo · 🔄 in-progress · ✅ done · ⏭️ skipped · ❌ blocked

---

## Phase 0 — Setup & Governance
- ✅ 0.1 Initialize task tracker (`audit/TASK_LOG.md`)
- ✅ 0.2 Record scope decision (verify & supersede prior audit)
- ✅ 0.3 Snapshot toolchain — Python 3.11.15; no deps preinstalled; pip has network; deps installed for test run

## Phase 1 — Reconnaissance, Indexing & Environment
- ✅ 1.1 Recursive file manifest — 115 files; see deliverable 1 CSV
- ✅ 1.2 Categorize every path
- ✅ 1.3 Flag duplicates/backups — feature_map.json==deliv3 (md5 553c97…), project_manifest.csv==deliv1 (fb0704…), tech_debt diverged; screenshots 09==10; archive/; stream_info≈event_logger
- ✅ 1.4 Git analysis — 67 commits, 2023(5)/2024(5)/2025(57); only master+work branch; heavy AI-assisted
- ✅ 1.5 Dependency map — requirements.txt has 22 pinned ranges; classified core/optional/platform
- ✅ 1.6 Version-currency — numpy pinned <2.0; no lock file; git-history secret residue
- ✅ 1.7 Intended environment — Windows-first; README claims cross-platform (contradicted by code)

## Phase 2 — Architectural Reconstruction & Intent
- ✅ 2.1 Tech-stack inventory
- ✅ 2.2 Synthesize intended features & architecture (README + docs/)
- ✅ 2.3 Entry points — main.py (CLI), ui/streamsense_ui.py (PyQt5)
- ✅ 2.4 Reconstruct actual architecture — modular monolith, process-per-device, LSL bus
- ✅ 2.5 Data models / formats — HDF5 raw, pickle datasets, MNE for EEG, CSV markers
- ✅ 2.6 Integration seams — CLI→streamers; UI→controller→streamers→LSL→recorder

## Phase 3 — Deep Dive & Feature Mapping
- ✅ 3.1 Backend dive (Explore agent, evidence-dense)
- ✅ 3.2 Per-device trace — all 3 inherit BaseStreamer; E4 delegates to legacy stream()
- ✅ 3.3 Frontend dive — all 6 signals wired; monolithic 771-LOC; 28 inline stylesheets
- ✅ 3.4 Actual features identified
- ✅ 3.5 Intended vs actual ledger
- ✅ 3.6 Feature→code mapping (deliverable 3)
- ✅ 3.7 Dead/orphaned — data_processor (orphaned+broken), archive/*, plot_helper stub

## Phase 4 — Status Assessment & Technical Debt
- ✅ 4.1 Classify features & components (Status Ontology + evidence)
- ✅ 4.2 Integration status — UI→E4 BROKEN (param mismatch); BITalino CLI-missing
- ✅ 4.3 Test execution — 107 passed, 7 skipped; test_data_processor errors at collect
- ✅ 4.4 Map coverage back to features — per-module % recorded in deliverable 3/4
- ✅ 4.5 Static analysis — 41 time.sleep sync points; import-time side effects; fake metrics; thread-target bug
- ✅ 4.6 Quantify & rank technical debt — hotspot ranking in deliverable 4 §8

## Phase 5 — Deliverables
- ✅ 5.1 Project Structure Manifest (refreshed, 115 files) — deliverables/1_*.csv
- ✅ 5.2 Architecture Overview + 2 Mermaid diagrams — deliverables/2_*.md
- ✅ 5.3 Feature Status Matrix (15 features) — deliverables/3_*.json
- ✅ 5.4 Technical Debt Report — deliverables/4_*.md
- ✅ 5.5 Actionable Stabilization Roadmap (P0–P4) — deliverables/5_*.md
- ✅ 5.6 Task Execution Log export (this file)
- ✅ 5.7 Reconcile vs prior audit — audit/RECONCILIATION_2026-05.md

---

## Execution Notes / Evidence

### Toolchain
Python 3.11.15; deps installed via pip (network OK) for the test run only — no source modified.

### Test execution (Phase 4.3) — per-file, `--timeout=15`
| File | Result |
|------|--------|
| test_base_streamer.py | 21 passed |
| test_device_discovery.py | 20 passed |
| test_e4_streaming.py | 19 passed (10.3s) |
| test_muse_streaming.py | 8 passed, 7 skipped |
| test_hardware_mocks.py | 24 passed |
| test_visualization.py | 14 passed |
| test_stream_recorder.py | 1 passed |
| test_data_processor.py | **ERROR at collection** (data_processor.py:606-607 import side-effect) |
**Totals: 107 passed, 7 skipped, 1 errored file.** Full suite hangs under `--cov` (process-spawning tests). Pure-mock subset coverage = **19%**.

### Confirmed Broken paths (evidence)
- UI→E4: `controller.py:258-262` vs `stream_e4.py:40` → TypeError.
- `data_processor.py:606-607` executes at import.
- `main.py:14` `import wmi` (off-Windows crash).

### Reconciliation
Prior Nov-2025 audit stale (61 files/10 features → now 115/15); secret remediated in code but lives in git history (rotate). See RECONCILIATION_2026-05.md.

### Read-only guarantee (audit phase)
During Phases 0–5 no application source was modified. Audit artifacts live under `audit/` only. Test-run caches are git-ignored.

---

## Post-Audit: P0 Stabilization (applied 2026-05-28, authorized by user)
Switched from read-only audit to remediation at the user's request ("proceed as you see fit").

- ✅ P0-1 UI→E4 ctor signature fixed (`ui/streamsense_controller.py`)
- ✅ P0-2 `data_processor.py` import side-effect removed (behind `main()`)
- ✅ P0-3 `main.py` `import wmi` guarded + cross-platform event-logger launch + E4 flow skips off-Windows
- ✅ P0-4 `main.py` CLI hang fixed (use `start_streaming()` bool, drop unconditional `connected_event.wait()`)
- ✅ P0-5 `stream_muse.py` thread-target bug fixed + eviction Timer daemonized/guarded
- ✅ bonus: two latent `detect_gaps` bugs fixed (pandas≥2.0 `replace(method=)`; seq-0 mis-flag)
- ✅ Added `tests/test_import_safety.py` (3 regression tests)
- ✅ Verified: **112 passed, 7 skipped, 0 fail/err** (was 107 + 1 errored file)

See `audit/deliverables/5_Actionable_Stabilization_Roadmap.md` → "Stabilization Progress" for detail.

## Post-Audit: P1 Test-Trustworthiness (applied 2026-05-28)
- ✅ P1-1 Coverage deadlock fixed — unit/integration split (`pytest.ini` markers), `.coveragerc`; `pytest -m "not integration" --cov` completes ~5s at **21%** (was infinite hang).
- ✅ P1-1b Whole-suite shutdown hang fixed at root — autouse reaper fixture in `tests/conftest.py` terminates stray multiprocessing children. Full `pytest` now **112 passed, 7 skipped, EXIT 0** (was RC=124 hang).
- ✅ P1-3 CI workflow `.github/workflows/tests.yml` (ubuntu, py3.10/3.11) — unit+coverage, integration, pip-audit.
- ✅ P1-4 `requirements-dev.txt` (CI test stack); validated by fresh-venv install + full CI simulation (both steps RC=0, coverage.xml produced).
- Finding: working stack is numpy 2.x / pandas 3.x, newer than requirements.txt pins (<2.0 / <3.0) — reconcile (tracked in roadmap).
- ✅ P1-2 BITalino + recorder unit tests added — `test_bitalino_streaming.py` (8), `test_recorder_unit.py` (5). stream_bitalino 0%→59%, recorder 11%→26%, TOTAL→25%. Full suite **125 passed, 7 skipped, EXIT 0**.

### Verification (P1)
Fresh venv (requirements-dev.txt only): `pytest -m "not integration" --cov` → 66 passed/7 skipped/21%/RC=0; `pytest -m integration` → 46 passed/RC=0; full `pytest` → 112 passed/7 skipped/EXIT 0.

### CI red→green fixes (PR #21, first runs failed)
- **py3.11 exit 4 (usage error)**: `pytest.ini addopts` had `--timeout/--timeout-method` (pytest-timeout flags). The plugin wasn't active on the runner → `unrecognized arguments` → exit 4. Reproduced locally with `-p no:timeout`. Fix: removed plugin-dependent flags from `addopts` (reaper fixture already prevents hangs; full suite still EXIT 0 without them). Kept `--strict-markers`.
- **py3.10 install failure**: pinned numpy 2.4.6 / scipy 1.17.1 / pandas 3.0.3 have **no cp310 wheels** (confirmed via pip dry-run; numpy 3.10-max is 2.2.6). Fix: matrix `["3.10","3.11"]` → `["3.11","3.12"]` (cp312 wheels confirmed available).

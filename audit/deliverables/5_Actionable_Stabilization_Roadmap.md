# Deliverable 5: Actionable Stabilization Roadmap (Refreshed — 2026-05-28)

> Supersedes prior roadmaps. Ordered by **return-on-effort**: fix what is silently broken
> first, then make the suite trustworthy, then pay down structural debt. Each item is a
> ready-to-file issue with acceptance criteria. Effort: S (<½ day), M (1–3 days), L (1–2 weeks).

---

## P0 — Stop-the-bleeding (correctness; small, high-impact)

| ID | Effort | Task | Acceptance criteria |
|----|--------|------|---------------------|
| P0-1 | S | Fix UI→E4 call (`controller.py:258-262`): use `StreamE4(e4=address, root_output_folder=…, synchronized_start_time=…)` or align signatures. | Connecting an E4 from the UI starts a streamer without `TypeError`; covered by a controller test. |
| P0-2 | S | Remove import-time execution in `data_processor.py:606-607`; move under `if __name__ == "__main__":` with argparse path. | `import data_processor` succeeds anywhere; `tests/test_data_processor.py` collects and runs. |
| P0-3 | S | Make Windows-only imports/launches conditional: guard `import wmi` (`main.py:14`) behind `platform.system()=='Windows'`; cross-platform event-logger launch (replace `start cmd /k`). | `python -c "import main"` succeeds on Linux/macOS; CLI starts on all 3 OSes. |
| P0-4 | S | Fix CLI hang: pass a timeout / break when `start_streaming()` returns False instead of unconditional `connected_event.wait()` (`main.py:113-114,183`). | A failed device connect returns control to the menu within the timeout. |
| P0-5 | S | Fix `stream_muse.py:310` thread target (wrap in lambda/partial); cap or remove per-sample `Timer` (`:250-251`). | Monitor thread runs; no unbounded thread growth (assert thread count stable in a timed test). |

## P1 — Make the test suite trustworthy (unblocks everything else)

| ID | Effort | Task | Acceptance criteria |
|----|--------|------|---------------------|
| P1-1 | M | Resolve coverage deadlock: configure `coverage` for multiprocessing (`concurrency=multiprocessing`, `sigterm`) or inject a fake Process in streamer tests; add `--timeout` to default pytest config. | `pytest --cov` over the whole suite completes < 90s and emits a TOTAL. |
| P1-2 | M | Add a `test_bitalino_streaming.py` and meaningful `recorder` tests; raise `recorder`, `stream_bitalino` off 0%. | Each streamer + recorder has ≥1 behavioral test; suite still green. |
| P1-3 | S | Add CI (GitHub Actions): install `requirements.txt`, run `pytest` with timeout on Linux (and Windows runner for device paths). | CI runs on push/PR; required to merge. |
| P1-4 | S | Add `requirements-lock.txt` (`pip freeze`) and run `pip-audit` in CI. | Reproducible install; CVE report visible in CI. |

## P2 — Integration & feature parity

| ID | Effort | Task | Acceptance criteria |
|----|--------|------|---------------------|
| P2-1 | M | Wire BITalino into the CLI (`main.py`) for parity with the UI, or explicitly document CLI scope. | CLI can discover/stream BITalino, or README states the limitation. |
| P2-2 | M | Replace fake UI signal-quality (92/87/85) with a real metric or an explicit "demo" badge. | UI shows measured quality, or clearly labels placeholder. |
| P2-3 | S | Integrate event logging into the app lifecycle instead of a detached Windows console; de-duplicate `event_logger.py`/`stream_info.py` into one module. | One marker-logging module, launchable cross-platform, sharing session state. |
| P2-4 | M | Add an LSL availability/health check before record/view; surface a clear error instead of silent failure. | Recording without an LSL stream produces a user-visible, tested error. |

## P3 — Structural debt & cross-platform

| ID | Effort | Task | Acceptance criteria |
|----|--------|------|---------------------|
| P3-1 | M | Extract a hardware-config layer for E4 (env-driven server path, no `D:/…` literal); document `E4_API_KEY` + server setup. | No hardcoded device paths; E4 server location configurable. |
| P3-2 | L | Decompose `ui/streamsense_ui.py`: move `Colors`→`ui/theme.py`, extract `DeviceCard`/`StreamWidget`/`LSLMonitorThread`, centralize stylesheets. | UI file < 300 LOC; styles defined once; UI smoke test passes. |
| P3-3 | M | Reduce `time.sleep()`-based synchronization in favor of events/queues where feasible. | Core sleep count materially reduced; streamers still pass tests. |
| P3-4 | S | 🔐 **Rotate the E4 API key** (it remains in git history) and document rotation. | New key issued; old key invalidated. |

## P4 — Repository hygiene (read-only audit *recommends*; do not auto-delete)

| ID | Effort | Task |
|----|--------|------|
| P4-1 | S | Archive/remove dead code: `archive/*`, `helper/plot_helper.py` stub, `scripts/capture_screenshots_headless.py`. |
| P4-2 | S | De-duplicate audit artifacts: delete `audit/feature_map.json` & `audit/project_manifest.csv` (byte-identical to `deliverables/`), or replace with pointers. |
| P4-3 | S | Move `Logs/Slide20-27.JPG` to `experiments/assets/`; stop committing empty `Logs/*.log`; regenerate or dedupe `docs/screenshots/*` (09==10). |

---

## Suggested sequence

```mermaid
flowchart LR
    P0[P0 correctness] --> P1[P1 trustworthy tests + CI]
    P1 --> P2[P2 integration/parity]
    P1 --> P4[P4 hygiene]
    P2 --> P3[P3 structural + x-platform]
```

**Headline:** 5 small P0 fixes remove 3 "Broken" statuses and a hang. P1 then makes regressions
visible. Everything after is incremental and safe once CI + coverage are trustworthy.

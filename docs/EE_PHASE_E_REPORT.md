# Eagle Eye Phase E Report

## Final Status
- Phase E gate status: GREEN.
- Full matrix summary #1: `47 passed, 1 skipped in 459.14s (0:07:39)`.
- Full matrix summary #2: `47 passed, 1 skipped in 457.99s (0:07:37)`.
- Full matrix summary #3: `47 passed, 1 skipped in 480.45s (0:08:00)`.
- Post-optimization validation run: `47 passed, 1 skipped in 80.56s (0:01:20)`.

## Frozen Threshold Set
- Config hash: `35239fb360066624c7c8c8ee38cadcbf496d30e0d85808bf5394b212b6670a16`.
- Frozen keys in active default profile:
  - `base_min_sessions=60`
  - `base_max_width_pct=0.18`
  - `volume_breakout_mult=2.5`
  - `cmf_floor=0.05`
  - `atr_squeeze_pctile=0.20`
  - `avoid_reclaim_clear_closes=2`
  - `trend_join_window=40`

## P1 -> P7 Defect Ledger
- P1 detector stall / transition visibility:
  - Root cause: phase transitions lacked complete traceability under mixed gate outcomes.
  - Fix: added deterministic transition tracing and reconciliation docs.
  - Guard: `tests/integration/test_eagle_eye_regression.py` gates and `docs/EE_DETECTOR_TRACE_P1.md`.
- P2 gate determinism drift:
  - Root cause: synthetic/trace mismatch under repeated reruns.
  - Fix: deterministic synthetic generation + reconciliation tracking.
  - Guard: integration regression matrix and documented parity checks.
- P3 warmup gate ordering:
  - Root cause: warmup eligibility and pattern logic timing could overlap.
  - Fix: explicit warmup alignment gate before pattern paths.
  - Guard: `tests/fixtures/generate_synthetic.py::_assert_warmup_alignment` and `tests/unit/test_eagle_eye_phase_e.py::test_u1d_join_preempts_base_when_both_true_inside_window`.
- P4 EXIT re-arm lifecycle:
  - Root cause: post-exit lifecycle could allow early re-qualification windows.
  - Fix: tightened lifecycle expectations and regression coverage.
  - Guard: `tests/integration/test_eagle_eye_regression.py::test_r5_mabanee_regression_gate`.
- P5 base lifecycle invalidation:
  - Root cause: retained base context could survive invalidation paths too long.
  - Fix: explicit invalidation handling for retained base context.
  - Guard: `tests/unit/test_eagle_eye_phase_e.py::test_u1l_invalidated_base_during_avoid_lands_neutral`.
- P6 AVOID reclaim release:
  - Root cause: reclaim clear path did not always preserve intended resume semantics.
  - Fix: retained pre-avoid phase/base context with explicit resume-vs-neutral landing logic.
  - Guard: `tests/unit/test_eagle_eye_phase_e.py::test_u1j_avoid_clear_resumes_phase_when_base_survives` and `docs/EE_AVOID_TRACE_P6.md`.
- P7 resume-to-watch close-out:
  - Root cause: resumed base context could miss immediate watch path evaluation.
  - Fix: resumed-path watch evaluation + lifecycle evidence hardening.
  - Guard: `tests/unit/test_eagle_eye_phase_e.py::test_u1k_reversal_entry_after_avoid_clear` and `tests/integration/test_eagle_eye_regression.py::test_r2_bpcc_regression_gate`.

## X1 Magnitude Anchors (Permanent)
- Added permanent real-chart magnitude anchors in fixture self-checks:
  - BPCC base-top->peak >= +12%
  - TIJARA >= +80%
  - ZAIN >= +12%
  - SANAM >= +45%
  - MABANEE >= +25%
- Implemented in `tests/fixtures/generate_synthetic.py` via `_assert_markup_anchor(...)` and per-symbol checks.

## X2 V4 Adversarial Results (R6)
- Added fixtures: `synthetic_chop.csv`, `synthetic_fakeout.csv`, `synthetic_pump.csv`.
- Added self-checks:
  - CHOP: no post-warmup 60-session window with normalized net-flow > 0.10.
  - FAKEOUT: crossing-bar rv < 1.5, no rv >= 2.5 in fakeout window, fade back inside base within 3 bars.
  - PUMP: one-day gap > 8% and no qualifying 60-session base in pre-spike adversarial segment.
- R6 assertion outcome: zero `ACCUMULATION_ALERT`, zero `BREAKOUT_CONFIRMED`, zero opened positions across CHOP/FAKEOUT/PUMP.
- Near-miss evidence_json entries: none were emitted for adversarial symbols in the final green run.

## Determinism Evidence
- Pass/fail set across three required full matrix runs: identical (`47 passed, 1 skipped`).
- Synthetic-suite per-symbol signal counts (runs 2 and 3): identical.
  - BPCC: `AVOID_SET=6`, `BREAKOUT_CONFIRMED=2`, `DISTRIBUTION_WARNING=2`, `EXIT=2`, `PHASE_ONLY=29`
  - JOINER: `AVOID_SET=1`, `DISTRIBUTION_WARNING=1`, `EXIT=1`, `PHASE_ONLY=11`
  - MABANEE: `ACCUMULATION_ALERT=1`, `AVOID_SET=6`, `BREAKOUT_CONFIRMED=1`, `DISTRIBUTION_WARNING=1`, `EXIT=1`, `PHASE_ONLY=23`
  - SANAM: `ACCUMULATION_ALERT=1`, `AVOID_SET=1`, `BREAKOUT_CONFIRMED=1`, `DISTRIBUTION_WARNING=1`, `EXIT=1`, `PHASE_ONLY=7`
  - TIJARA: `ACCUMULATION_ALERT=1`, `AVOID_SET=1`, `BREAKOUT_CONFIRMED=1`, `DISTRIBUTION_WARNING=1`, `EXIT=1`, `PHASE_ONLY=11`
  - ZAIN: `ACCUMULATION_ALERT=1`, `BREAKOUT_CONFIRMED=1`, `DISTRIBUTION_WARNING=1`, `PHASE_ONLY=8`
  - CHOP/FAKEOUT/PUMP: no entry signals and no positions opened.

## Runtime Before / After (V5)
- Baseline before runtime optimization: ~7:37 to ~8:00 full matrix (with R6).
- Profiling snapshot (cProfile, top self-time):
  - `sqlite3.Connection.commit`: 130.806s
  - `sqlite3.Cursor.execute`: 129.537s
  - `sqlite3.Connection.close`: 90.808s
  - `_thread.lock.acquire`: 34.459s
  - `_sqlite3.connect`: 25.590s
- Mechanical optimizations applied:
  - module-scoped synthetic integration fixture load/backtest (eliminate per-test regeneration/backtest)
  - SQLite dev pragmas (`synchronous=NORMAL`, `temp_store=MEMORY`, cache sizing)
  - process-local shared SQLite connection reuse in `app/core/database.py`
- Post-optimization runtime: `1:20` full matrix.
- Detector logic changes for speed: none.
- U9 equivalence guard status: green in matrix.

## Config Disclosure State
- Runtime gates and traces were executed with production-default thresholds (frozen set above).
- No threshold tuning was used to pass R6 or runtime optimization.

## Deliverables Checklist
- X1 complete: split commits created, scratch root cleaned, u1k disclosure addressed, anchors enforced.
- X2 complete: adversarial fixtures + self-checks + R6 gate.
- X3 complete: third deterministic run and runtime optimization below hard cap.
- X4 complete: report, change requests moved to proposed, scan-preview endpoint + test.

## X4 Audit Change Requests (Proposed)
- CR #2: P3 Warmup Gate Amendment -> `proposed` (trace: `docs/EE_WARMUP_GATE_TRACE_P3.md`)
- CR #3: P4 EXIT Re-Arm Amendment -> `proposed` (trace: `docs/EE_EXIT_REARM_TRACE_P4.md`)
- CR #4: P5 Base Lifecycle Invalidation Amendment -> `proposed` (trace: `docs/EE_BASE_INVALIDATION_TRACE_P5.md`)
- CR #5: P6 AVOID Reclaim Release Amendment -> `proposed` (trace: `docs/EE_AVOID_TRACE_P6.md`)
- CR #6: P7 Resume-to-Watch Amendment -> `proposed` (trace: `docs/EE_RESUME_TO_WATCH_TRACE_P7.md`)

## X4 Admin Preview Endpoint
- Added `GET /api/v1/eagle-eye/signals/scan-preview` (admin).
- Behavior:
  - Loads CSVs from `data/kse/`.
  - Resets preview runtime tables.
  - Runs canonical Eagle Eye pipeline.
  - Returns ranked watchlist items with full indicator evidence plus latest signal evidence.
  - Always returns `"advice": false`.
  - Returns explicit empty-directory/empty-file message when no CSVs are available.
- Endpoint test coverage:
  - `tests/unit/test_eagle_eye_phase_e.py::test_u4b_scan_preview_admin_endpoint`

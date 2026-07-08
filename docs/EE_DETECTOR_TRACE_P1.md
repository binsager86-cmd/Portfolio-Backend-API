# Eagle Eye Detector Trace Pass 1

## Scope and Constraints
- Pass order followed: T1 -> T2 -> T3 -> matrix -> T4 -> matrix checkpoint status.
- Thresholds frozen (no threshold edits).
- No fixture edits in this pass.

## Validation Checkpoints
- compileall: pass (`python -m compileall app scripts tests`)
- unit indicator CMF tests: pass (`7 passed`)
- unit callsite lint guard: pass (`1 passed`)
- matrix after T1-T3: `6 failed, 12 passed, 1 skipped`
  - failing: R1, R2, R3, R4, R5, R7

## T2: CMF Sanity Result
- Added hand-computed CMF definition tests in [tests/unit/test_eagle_eye_indicator_service.py](tests/unit/test_eagle_eye_indicator_service.py).
- Result: CMF math is behaving per definition (unit tests green).
- Evidence in watch traces shows CMF varies with signed money flow and can cross floor; failures are not explained by a broken CMF implementation.

## T3: Per-Symbol Numeric Trace Summary
Source: `python -m scripts.debug_gates --trace-watch ALL --trace-watch-mf > trace_watch_all_p1.txt`

### TIJARA (R1)
- silent trace marker:
  - `TRACE_WATCH_SILENT symbol=TIJARA reason=no_rows_emitted phase_distribution={"EXIT": 362}`
- phase rows emitted: `0`
- implication: symbol remains in EXIT distribution in this pass; no accumulation rows emitted.

### BPCC (R2)
- phase_rows: `200`
- acc_gate_true_rows: `8`
- watch_pass_true_rows: `0`
- join_open_rows: `1/200`
- join_open_all_true_rows: `0`
- first row shows BASE_FORMING with accumulation gate false.
- last row (1633564800) still fails WATCH (`near_base=False`) and join conditions.

### ZAIN (R3)
- phase_rows: `332`
- acc_gate_true_rows: `35`
- watch_pass_true_rows: `0`
- join_open_rows: `40/332`
- join_open_all_true_rows: `21`
- despite many open/all-true join rows, no breakout emitted in regression test.

### SANAM (R4)
- silent trace marker:
  - `TRACE_WATCH_SILENT symbol=SANAM reason=no_rows_emitted phase_distribution={"EXIT": 342}`
- phase rows emitted: `0`
- implication: symbol remains in EXIT distribution in this pass.

### MABANEE (R5)
- phase_rows: `199`
- acc_gate_true_rows: `23`
- watch_pass_true_rows: `5`
- join_open_rows: `19/199`
- join_open_all_true_rows: `0`
- lifecycle does not progress to breakout in this pass; watch true occurs but no breakout-confirmed event.

### JOINER / synthetic_joiner (R7)
- phase_rows: `369`
- acc_gate_true_rows: `0`
- watch_pass_true_rows: `0`
- join_open_rows: `40/369`
- join_open_all_true_rows: `18`
- first row starts NEUTRAL; later rows enter BASE_FORMING.
- final row remains BASE_FORMING with join open false and trend false.
- key contradiction to test expectation: join conditions are observed as open+all-true in trace rows, yet `joined_externally` is still missing in regression assertions.

## T4 Focus (JOINER)
- Quantitative finding: JOINER has `open_all_true_rows=18` inside trace output, but regression R7 still reports no `joined_externally` lifecycle event.
- This points to an event emission / lifecycle write path mismatch rather than raw gate truth absence.

## Stop Rule Status
- R-tests still red after T1-T4 checkpoint.
- Per directive, this pass stops here for review with numeric trace evidence captured.

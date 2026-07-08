# Eagle Eye Trace — P7 Resume To Watch

## Scope
- Phase: P7 detector close-out.
- Area: AVOID clear resume semantics and resume-to-watch reachability.

## Root Cause
- On AVOID clear, resumed base context did not reliably reach watch/confirm in the intended same-bar flow, creating regression failures.

## Fix Summary
- Preserved pre-avoid base context, resumed `BASE_FORMING` on valid clear, and allowed resumed paths to evaluate watch conditions.
- Added lifecycle evidence for `avoid_cleared_resume` and resumed-phase transitions.

## Guard Tests
- `tests/unit/test_eagle_eye_phase_e.py::test_u1k_reversal_entry_after_avoid_clear`
- `tests/integration/test_eagle_eye_regression.py::test_r2_bpcc_regression_gate`

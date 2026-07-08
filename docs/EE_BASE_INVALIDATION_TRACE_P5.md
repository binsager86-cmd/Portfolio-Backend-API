# Eagle Eye Trace — P5 Base Lifecycle Invalidation

## Scope
- Phase: P5 base lifecycle invalidation.
- Area: invalidation while base-like context is retained in non-neutral phases.

## Root Cause
- Base retention state could persist beyond invalidation conditions in edge paths, delaying neutralization.

## Fix Summary
- Added explicit invalidation checks on retained base context paths and lifecycle logging for invalidated landings.

## Guard Tests
- `tests/unit/test_eagle_eye_phase_e.py::test_u1l_invalidated_base_during_avoid_lands_neutral`
- `tests/integration/test_eagle_eye_regression.py::test_r2_bpcc_regression_gate`

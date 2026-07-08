# Eagle Eye Trace — P3 Warmup Gate

## Scope
- Phase: P3 warmup gating.
- Area: scanner warmup readiness and trend-join eligibility timing.

## Root Cause
- Warmup readiness depended on mixed indicator availability and could allow pattern logic to run before full window prerequisites were stable.

## Fix Summary
- Enforced explicit warmup readiness checks before pattern path execution.
- Added deterministic warmup alignment checks in synthetic fixture generation.

## Guard Tests
- `tests/unit/test_eagle_eye_phase_e.py::test_u1d_join_preempts_base_when_both_true_inside_window`
- `tests/fixtures/generate_synthetic.py::_assert_warmup_alignment`

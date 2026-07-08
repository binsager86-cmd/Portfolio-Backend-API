# Eagle Eye Phase E Parity Audit (2026-07-05)

## V1 parity baseline (production defaults)

Run profile used for parity:
- Engine config reset to production defaults in every regression test.
- Synthetic override whitelist enforced: only min_daily_value_kwd remains.
- Command: pytest tests/unit/test_eagle_eye_indicator_service.py tests/unit/test_eagle_eye_phase_e.py tests/unit/test_eagle_eye_audit_service.py tests/integration/test_eagle_eye_regression.py -q

Result:
- Unit suites: green.
- Regression gates: R1-R5 all red.
- Summary: 5 failed, 22 passed, 1 skipped in 212.16s.

Failure mapping:
- R1 TIJARA: ACCUMULATION_ALERT missing.
- R2 BPCC: BREAKOUT_CONFIRMED missing.
- R3 ZAIN: BREAKOUT_CONFIRMED missing.
- R4 SANAM: ACCUMULATION_ALERT missing.
- R5 MABANEE: joined_externally MARKUP signal missing.

Gate trace snapshot under this parity setup:
- TIJARA: no all-pass accumulation window.
- BPCC: no all-pass accumulation window.
- ZAIN: no all-pass accumulation window.
- SANAM: no all-pass accumulation window.
- MABANEE: no all-pass accumulation window.

Directive outcome:
- This is a real production-default finding.
- Per directive V1 step 3, work is stopped at this point and non-whitelisted overrides are not reintroduced.

## V3 breakout-reference semantics (current implementation)

Current code path references:
- Base reference assignment: app/services/eagle_eye/scanner_service.py:300
- Base freeze write on entry: app/services/eagle_eye/scanner_service.py:517
- BREAKOUT_WATCH block using base_high_ref: app/services/eagle_eye/scanner_service.py:367

Current pseudocode:

```text
base_high_ref = state.base_high if present else range_high_60 else range_high_120

if phase == NEUTRAL and base criteria pass:
	phase = BASE_FORMING
	state.base_high = range_high_60
	state.base_low = range_low_60

if phase == BREAKOUT_WATCH:
	mandatory gate M1 uses close > base_high_ref
	revert uses close < 0.99 * base_high_ref
```

Conformance vs required semantics:
- Freeze at base entry: partially present. state.base_high/state.base_low are written at BASE_FORMING entry.
- Permitted ratchet-in-base rule: missing. There is no explicit constrained ratchet rule while in BASE_FORMING/ACCUMULATION.
- Immutability in WATCH/CONFIRMING: conditionally true only when state.base_high is already set; otherwise fallback can use rolling range_high_60/120.

## A4 trace findings requested

SANAM base-loss trace finding:
- Under strict parity rerun, SANAM does not show a base-loss transition. Observed trace is only NEUTRAL -> BASE_FORMING at 1636675200, then it stalls in BASE_FORMING with no ACCUMULATION_ALERT. No specific "lost base" invalidation bar is produced in this parity run.

BPCC anchor ratchet finding:
- Under strict parity rerun, BPCC shows NEUTRAL -> BASE_FORMING at 1633564800 and stalls there. Final state has base_high=611.222, base_low=559.273. Because it never reaches BREAKOUT_WATCH/CONFIRMING in parity, no breakout-anchor ratcheting behavior is exercised in this run.

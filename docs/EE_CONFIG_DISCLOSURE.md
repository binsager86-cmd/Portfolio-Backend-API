# EE Config Disclosure (V1)

## Scope

This disclosure compares production defaults from ee_engine_config with synthetic regression overrides from tests/integration/test_eagle_eye_regression.py.

Policy enforced:
- Only scale-dependent overrides are allowed.
- Allowed whitelist: min_daily_value_kwd and (if needed) backtest cost keys.
- Non-whitelisted detector overrides were removed.

## Default vs Synthetic Override Matrix

| Key | Default (ee_engine_config) | Synthetic Override | Differs | Justification |
|---|---:|---:|---|---|
| adx_trigger | 22 | none | no | none |
| allow_self_review | false | none | no | none |
| atr_squeeze_pctile | 0.2 | none | no | none |
| base_max_width_pct | 0.18 | none | no | none |
| base_min_sessions | 60 | none | no | none |
| bt_commission_bps | 25 | none | no | none |
| bt_slippage_bps | 30 | none | no | none |
| climax_partial | true | none | no | none |
| cmf_floor | 0.05 | none | no | none |
| max_portfolio_heat | 0.06 | none | no | none |
| max_positions | 8 | none | no | none |
| max_sector_concentration | 0.4 | none | no | none |
| min_daily_value_kwd | 100000.0 | 100000.0 | no | Whitelist scale-dependent key. Override slot retained for fixture scale portability; current value matches production default, so no behavioral delta. |
| ml_gate_enabled | false | none | no | none |
| ml_min_labeled_signals | 150 | none | no | none |
| ml_prob_min | 0.45 | none | no | none |
| obv_divergence_lookback | 40 | none | no | none |
| pilot_enabled | true | none | no | none |
| risk_per_trade | 0.01 | none | no | none |
| rsi_regime | 55 | none | no | none |
| trend_join_window | 40 | none | no | none |
| volume_breakout_mult | 2.5 | none | no | none |

Trend-join window semantics:
- `trend_join_window` is now counted from `warmup_ready_date` (the first bar where both `sma200` and `range_low_120` are valid), not from raw symbol history start.

## Removed Non-Whitelisted Overrides

Removed from SYNTHETIC_OVERRIDES to satisfy V1 policy:
- base_price_slope_floor
- base_price_slope_ceiling
- base_min_width_pct
- base_to_range_low120_cap
- trend_join_window_grace

## Step 3 Knob Adjudication

Evidence source:
- docs/EE_V1_STALL_TRACE.md generated with trace-watch under production defaults.
- Scanner source currently does not read any of the five removed keys.

| Key | Classification | Decision | Evidence |
|---|---|---|---|
| base_price_slope_floor | (b) fixture-fitting crutch | keep deleted | trace-watch prints this key as absent and not read by current scanner; no active gate depends on it. |
| base_price_slope_ceiling | (b) fixture-fitting crutch | keep deleted | trace-watch prints this key as absent and not read by current scanner; no active gate depends on it. |
| base_min_width_pct | (b) fixture-fitting crutch | keep deleted | trace-watch prints this key as absent and not read by current scanner; no active gate depends on it. |
| base_to_range_low120_cap | (b) fixture-fitting crutch | keep deleted | trace-watch prints this key as absent and not read by current scanner; no active gate depends on it. |
| trend_join_window_grace | (b) fixture-fitting crutch | keep deleted | trend-join already has trend_join_window in production defaults; trace-watch confirms grace key is absent and not read. |

Change-request handling:
- None filed for these five because they are currently removed and classified as fixture-fitting under this pass.

## Pass-2 Adjudication Addendum

- Historical signal from deleted knob `base_price_slope_ceiling` is acknowledged as having pointed at a real detector gap.
- Resolution path is structural, not threshold reinstatement: trend-join precedence over base-detection inside the join window (ordering fix), with tests/docs updated accordingly.
- `base_price_slope_ceiling` remains deleted.

## Parity Run Result (after whitelist enforcement)

Command:
- pytest tests/unit/test_eagle_eye_indicator_service.py tests/unit/test_eagle_eye_phase_e.py tests/unit/test_eagle_eye_audit_service.py tests/integration/test_eagle_eye_regression.py -q

Outcome:
- 5 failed, 22 passed, 1 skipped in 212.16s.
- Failing gates: R1, R2, R3, R4, R5.

Interpretation:
- This is a production-default detection gap, not a fixture-only issue.
- Per directive, no non-whitelisted override was reintroduced.

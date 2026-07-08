# EE Gate Reconciliation (Pass 2)

Scope:
- Source of truth: app/services/eagle_eye/scanner_service.py
- Runtime evidence source: trace_watch_all_p1.txt and docs/EE_DETECTOR_TRACE_P1.md
- Status keys:
	- IN-SPEC: explicit in current detector directive/spec
	- UNDOCUMENTED: implemented, but not currently codified in detector spec text
	- LEFTOVER: behavior traced to removed legacy knob-era logic

## A1 Evaluation Order (Implemented)
1. Distribution-warning exit checks
2. ACCUMULATION -> BREAKOUT_WATCH trigger
3. BREAKOUT_WATCH confirm window (M1-M5 then C1-C6)
4. BASE_FORMING -> ACCUMULATION gate
5. NEUTRAL trend-join gate (windowed)
6. NEUTRAL base-detection gate

Pass-2 amendment applied in code semantics/tests:
- While phase is NEUTRAL and join window is open, trend-join is evaluated before base detection.

## Full Condition Table

| Area | Implemented condition | File:line | Spec reference | Status |
|---|---|---|---|---|
| AVOID set | close < sma200 and sma200_slope < 0 and ema10 < ema30 | app/services/eagle_eye/scanner_service.py:283 | AVOID safety regime | UNDOCUMENTED |
| AVOID clear | avoid_clear_streak >= 20 -> NEUTRAL | app/services/eagle_eye/scanner_service.py:288-291 | cooldown clear rule | UNDOCUMENTED |
| WATCH trigger | rv_hits >= 2 of last 5 where rel_volume >= 1.5 | app/services/eagle_eye/scanner_service.py:353-355 | watch trigger (proximity + volume) | IN-SPEC |
| WATCH trigger | close >= 0.97 * base_high_ref | app/services/eagle_eye/scanner_service.py:354 | watch proximity | IN-SPEC |
| M1 | close > base_high_ref | app/services/eagle_eye/scanner_service.py:384 | mandatory core M1 | IN-SPEC |
| M2 | rel_volume >= volume_breakout_mult | app/services/eagle_eye/scanner_service.py:385 | mandatory core M2 | IN-SPEC |
| M3 | ema10 > ema30 | app/services/eagle_eye/scanner_service.py:386 | mandatory core M3 | IN-SPEC |
| M4 | gap_pct_base <= 0.08 | app/services/eagle_eye/scanner_service.py:387 | mandatory core M4 | IN-SPEC |
| M5 | liquidity_ok | app/services/eagle_eye/scanner_service.py:388 | mandatory core M5 | IN-SPEC |
| C1 | rsi >= rsi_regime | app/services/eagle_eye/scanner_service.py:392 | confirmatory C1 | IN-SPEC |
| C2 | rsi_rising | app/services/eagle_eye/scanner_service.py:393 | confirmatory C2 | IN-SPEC |
| C3 | adx >= adx_trigger and plus_di > minus_di | app/services/eagle_eye/scanner_service.py:394 | confirmatory C3 | IN-SPEC |
| C4 | adx > adx_5_back | app/services/eagle_eye/scanner_service.py:395 | confirmatory C4 | IN-SPEC |
| C5 | macd_hist > 0 or recent MACD cross | app/services/eagle_eye/scanner_service.py:396 | confirmatory C5 | IN-SPEC |
| C6 | close_top40 | app/services/eagle_eye/scanner_service.py:397 | confirmatory C6 | IN-SPEC |
| BASE->ACC gate | accumulation_divergence OR flat-price-plus-flow fallback | app/services/eagle_eye/scanner_service.py:447-452 | accumulation structure gate | IN-SPEC |
| BASE->ACC gate | cmf_hits >= 5 (cmf_10 > cmf_floor over last 10) | app/services/eagle_eye/scanner_service.py:445-446 | accumulation flow gate | IN-SPEC |
| BASE->ACC gate | squeeze_ok (bb_width <= 0.12 OR atr_pct_percentile <= atr_squeeze_pctile) | app/services/eagle_eye/scanner_service.py:453-456 | compression gate | IN-SPEC |
| BASE->ACC gate | close >= ema30 OR close >= 0.97*sma200 | app/services/eagle_eye/scanner_service.py:462 | trend support in base | IN-SPEC |
| BASE->ACC gate | liquidity_ok and score >= 60 | app/services/eagle_eye/scanner_service.py:463-464 | liquidity + quality floor | IN-SPEC |
| Trend-join window | sessions_since_start <= trend_join_window | app/services/eagle_eye/scanner_service.py:493 | join-window eligibility | IN-SPEC |
| Trend-join C1 | close > sma200 | app/services/eagle_eye/scanner_service.py:494 | join core | IN-SPEC |
| Trend-join C2 | sma200_slope > 0 | app/services/eagle_eye/scanner_service.py:495 | join core | IN-SPEC |
| Trend-join C3 | ema10 > ema30 | app/services/eagle_eye/scanner_service.py:496 | join core | IN-SPEC |
| Trend-join C4 | range_low_120 > 0 | app/services/eagle_eye/scanner_service.py:497 | warmup readiness | IN-SPEC |
| Trend-join C5 | close >= range_low_120 * 1.15 | app/services/eagle_eye/scanner_service.py:498 | join distance floor | IN-SPEC |
| Base entry | sma200 > 0 | app/services/eagle_eye/scanner_service.py:511 | base readiness | IN-SPEC |
| Base entry | ema30 > 0 | app/services/eagle_eye/scanner_service.py:512 | base readiness | IN-SPEC |
| Base entry | width <= base_max_width_pct | app/services/eagle_eye/scanner_service.py:514 | base width cap | IN-SPEC |
| Base entry | sessions_in_range >= base_min_sessions | app/services/eagle_eye/scanner_service.py:515 | base residency | IN-SPEC |
| Base entry | base_low_60 <= close <= base_high_60 | app/services/eagle_eye/scanner_service.py:516 | in-range close | IN-SPEC |
| Base reference | base_high_ref = state.base_high OR range_high_60 | app/services/eagle_eye/scanner_service.py:300 | frozen/active base reference | IN-SPEC |

## Dispositions for Prior UNDOCUMENTED/LEFTOVER Items

| Item | Prior status | Pass-2 disposition | Evidence / rationale |
|---|---|---|---|
| WATCH hard-break branch with EMA dependency | LEFTOVER | REMOVED (already absent in current scanner) | WATCH trigger now only proximity + volume at app/services/eagle_eye/scanner_service.py:353-355 |
| Adaptive base reference fallback to range_high_120 | LEFTOVER | REMOVED in Pass 2 | base_high_ref now only frozen base_high or range_high_60 fallback at app/services/eagle_eye/scanner_service.py:300 |
| Bearish-stack AVOID condition | UNDOCUMENTED | KEEP via CR-SCAN-AVOID-001 | Trace-backed risk guard for downtrend suppression; no contradictory red-test evidence yet |
| AVOID clear streak (20) | UNDOCUMENTED | KEEP via CR-SCAN-AVOID-002 | Required to allow recovery from AVOID state; preserves state-machine liveness |

## Pass-2 T1 Completion Notes

- Required WATCH cleanup is satisfied: no EMA condition remains in WATCH trigger.
- All listed detector conditions are now reconciled with explicit status and disposition.

## Pass 5 Addendum (Frozen Order + Disclosure)

### G1.1 Test Assertion Disclosure (verbatim diff)

Observed diff from `tests/unit/test_eagle_eye_phase_e.py` in prior pass window:

```diff
@@
 def test_u1c_confirming_window_confirms_or_reverts():
@@
	assert r_revert["signal_type"] is None


+def test_u1d_breakout_uses_frozen_base_high_not_rolling_120_high():
+    ...
+    r_watch = evaluate_symbol(symbol, base_td + 4 * 86400, 80.0, cfg)
+    assert r_watch["phase"] == "BREAKOUT_WATCH"
+    assert r_watch["transition"] == ("BASE_FORMING", "BREAKOUT_WATCH")
+    ...
+    r_confirm = evaluate_symbol(symbol, base_td + 7 * 86400, 80.0, cfg)
+    assert r_confirm["signal_type"] == "BREAKOUT_CONFIRMED"

@@ def test_u5_risk_suppression_emits_signal_and_skips_position():
-    exec_sql(
-        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, updated_at, state_json) VALUES (?, 'BREAKOUT_WATCH', ?, ?, '{}')",
-        (symbol, latest - 86400, latest),
-    )
+    exec_sql(
+        "INSERT INTO ee_symbol_state (symbol, phase, phase_since, base_high, base_low, updated_at, state_json) VALUES (?, 'BREAKOUT_WATCH', ?, 110.0, 90.0, ?, '{}')",
+        (symbol, latest - 86400, latest),
+    )
```

Justification per change:

1. `test_u1d...` was added to enforce frozen-base breakout reference semantics (state `base_high`, not rolling `range_high_120`).
2. `test_u5...` setup was adjusted to seed `base_high/base_low` so BREAKOUT_WATCH mandatory checks remain meaningful after frozen-base migration.
3. No U5 assertion was weakened; assertion lines in U5 are unchanged.

Ruling:

1. No assertion weakening accepted in this disclosure set.
2. If any future assertion relaxation appears without directive-approved contract change, it must be reverted in-pass.

### G1.2 Mid-pass order flips and frozen evaluation order

Recorded flips during prior pass:

1. Flip A: WATCH evaluation moved ahead of BASE_FORMING->ACCUMULATION.
2. Flip B: order restored to ACCUMULATION before WATCH.

FROZEN order (change-controlled):

1. Distribution-warning exit checks.
2. BREAKOUT_WATCH confirm window.
3. BASE_FORMING -> ACCUMULATION gate.
4. BASE_FORMING/ACCUMULATION -> BREAKOUT_WATCH trigger.
5. NEUTRAL trend-join gate.
6. NEUTRAL base-detection gate.

Policy:

1. This order is now frozen.
2. Any future order change requires a formal change request before code edits.

### G1.3 Config-channel ruling (env var path)

Ruling:

1. Detector runtime config enters only via `ee_engine_config` plus explicit `config_overrides`.
2. `os.environ`/`os.getenv` reads under `app/services/eagle_eye/**` are prohibited for detector knobs.

Verification snapshot:

1. Grep over `app/services/eagle_eye/**` found no `os.environ`/`os.getenv` reads.
2. Grep over whole repo found no `EE_MIN_DAILY_VALUE_KWD` token in source files.
3. Conclusion: prior command-line env export was superstition (not consumed by Eagle Eye engine path).

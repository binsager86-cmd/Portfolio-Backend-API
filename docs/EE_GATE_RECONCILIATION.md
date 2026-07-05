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

# Eagle Eye Phase E Parity Audit

Scope: Rule-level conformance against Sections 3.1, 4, 5, 6, 7 from Eagle Eye amendments.

Status keys:
- EXACT: implemented to spec behavior.
- SIMPLIFIED: implemented but materially different.
- MISSING: not implemented.

| Spec ref | Rule | Implementation location (file:function) | Status | If simplified: exact difference |
|---|---|---|---|---|
| 3.1 | accumulation_divergence: norm LR slopes over 40, abs(price_slope)<0.02 and (obv_slope>0.10 or anv_slope>0.10) | app/services/eagle_eye/indicator_service.py:compute_symbol_indicators | EXACT | |
| 3.1 | distribution_divergence: price HH over last 20 vs prior 20 and OBV LH or RSI LH | app/services/eagle_eye/indicator_service.py:compute_symbol_indicators | EXACT | Uses pivot-high fallback via forward-fill windows, allowed by spec note |
| 4 | State machine phases NEUTRAL->...->EXIT plus AVOID | app/services/eagle_eye/scanner_service.py:evaluate_symbol | SIMPLIFIED | Some transitions are present but missing explicit failed-breakout revert flow, 20-session AVOID clear, and full breakout gate conditions |
| 4.1 | BASE_FORMING entry: range_width_pct(60)<=0.18, sessions_in_range>=60, close in range, not AVOID | app/services/eagle_eye/scanner_service.py:evaluate_symbol | SIMPLIFIED | Width check only; does not enforce sessions_in_range>=60 and explicit close-in-range constraint |
| 4.1 | BASE_FORMING->ACCUMULATION all 5 conditions including CMF persistence 5/10 and squeeze/bb quintile | app/services/eagle_eye/scanner_service.py:evaluate_symbol | SIMPLIFIED | Uses cmf current-bar threshold only, not 5 of last 10 persistence; bb quintile not implemented as percentile; other checks partial |
| 4.1 | ACCUMULATION_ALERT emitted risk medium as watch/pilot | app/services/eagle_eye/scanner_service.py:_emit_signal | EXACT | |
| 4.1 | ACCUMULATION->BREAKOUT_WATCH: close within 3% of range_high_120 and rel_volume>=1.5 on >=2/5 sessions | app/services/eagle_eye/scanner_service.py:evaluate_symbol | SIMPLIFIED | Checks current bar only, not 2 of last 5 sessions |
| 4.1 | BREAKOUT_WATCH->BREAKOUT_CONFIRMED all 7 same-bar rules | app/services/eagle_eye/scanner_service.py:evaluate_symbol | SIMPLIFIED | Missing ADX rising over 5 sessions, MACD cross-within-5 fallback, close-in-top-40%-of-range condition |
| 4.1 + 6.4 | Chase guard: no entry if breakout bar gap >8% above base high | app/services/eagle_eye/scanner_service.py:evaluate_symbol | MISSING | No gap-chase rule currently enforced |
| 4.1 | BREAKOUT_FAILED guard within 5 sessions then revert + close pilot | app/services/eagle_eye/scanner_service.py:evaluate_symbol | MISSING | No failure window tracking or BREAKOUT_FAILED signal emission |
| 4.1 | BREAKOUT_CONFIRMED->MARKUP: 10 sessions post-breakout with >=8 closes above EMA30 | app/services/eagle_eye/scanner_service.py:evaluate_symbol | SIMPLIFIED | Transitions immediately when close>=ema30 on a single bar |
| 4.1 | MARKUP->DISTRIBUTION_WARNING triggers set (distribution divergence OR climax OR RSI collapse OR CMF negative persistence) | app/services/eagle_eye/scanner_service.py:evaluate_symbol | SIMPLIFIED | Implements distribution_divergence or immediate cmf<-0.05 only; missing climax and RSI-collapse and 5-day CMF persistence while flat/up |
| 4.1 | DISTRIBUTION_WARNING->EXIT triggers set including two closes below EMA30 with volume condition | app/services/eagle_eye/scanner_service.py:evaluate_symbol | SIMPLIFIED | Uses single close below EMA30 or trail breach; missing 2-close EMA30+volume condition |
| 4.1 | AVOID filter blocks all long signals | app/services/eagle_eye/scanner_service.py:evaluate_symbol | SIMPLIFIED | Entry checks respect avoid flag but no explicit suppression signal and clear-window mechanics incomplete |
| 4.1 | AVOID clear after 20 sessions of condition false | app/services/eagle_eye/scanner_service.py:evaluate_symbol | MISSING | No rolling clear counter |
| 5 | Composite score blocks and weights 30/25/20/15/10 | app/services/eagle_eye/rating_service.py:compute_rating_from_indicator | SIMPLIFIED | Uses same block weights, but component formulas are heuristic approximations not full spec component set |
| 5 | Score bands A/B/C/D | app/services/eagle_eye/rating_service.py:compute_rating_from_indicator | EXACT | |
| 5 | Publish gating: ACCUMULATION_ALERT score>=60 and BREAKOUT full-size score>=70 | app/services/eagle_eye/scanner_service.py:evaluate_symbol | EXACT | |
| 5 | Explainability with full components_json and stubs flagged when missing feeds | app/services/eagle_eye/rating_service.py:store_rating | SIMPLIFIED | Components persisted, but stubbed feed markers (e.g., PE/news) are not explicitly marked with stubbed=true |
| 6.1 | Entry ladder T1/T2/T3 with 25/50/25 and no averaging down | app/services/eagle_eye/entry_exit_service.py:maybe_open_or_add_position | SIMPLIFIED | T1/T2/T3 sizing exists; T3 trigger logic absent in scanner; no explicit loser-add prevention check beyond tranche dedupe |
| 6.2 | Pilot/core initial stop exact nearest-of-two with minimum 1.5 ATR distance for pilot | app/services/eagle_eye/scanner_service.py:evaluate_symbol | SIMPLIFIED | Stop formulas approximated; nearest-of-two + 1.5 ATR floor rule not exact |
| 6.3 | MARKUP trail 3.0 ATR chandelier and do not exit on RSI>70 alone | app/services/eagle_eye/scanner_service.py:evaluate_symbol | EXACT | RSI>70 standalone exit is not used |
| 6.3 | DISTRIBUTION_WARNING trail tightened to 2.0 ATR and EMA30 rule armed then | app/services/eagle_eye/scanner_service.py:evaluate_symbol | SIMPLIFIED | Trail tightening present; EMA30 logic currently active without strict armed-two-close implementation |
| 6.3 | EXIT closes 100%; optional climax partial 33% via config | app/services/eagle_eye/entry_exit_service.py:close_open_position | SIMPLIFIED | Full close on EXIT implemented; climax-partial not implemented |
| 6.3 | Time stop: 40 sessions post-breakout and < entry+1 ATR exits | app/services/eagle_eye/scanner_service.py:evaluate_symbol | MISSING | No 40-session dead-breakout timer |
| 6.4 | Never rules: no entries while AVOID; no adding to losers; no >8% chase gap entries | app/services/eagle_eye/scanner_service.py:evaluate_symbol, app/services/eagle_eye/entry_exit_service.py:maybe_open_or_add_position | SIMPLIFIED | AVOID block present; loser-add prevention and 8% chase guard not fully enforced |
| 7.1 | Liquidity filter: median 20d value>=min, price>=50 fils, zero-volume<=3 in last 60 | app/services/eagle_eye/risk_service.py:liquidity_filter | EXACT | |
| 7.1 | Liquidity filter applied before publishable signals | app/services/eagle_eye/scanner_service.py:evaluate_symbol | SIMPLIFIED | Applied for accumulation; breakout path can still emit transitions without explicit publish suppress for all long signal types |
| 7.2 | Position sizing formula (equity*risk)/(entry-stop) with volume cap and limits | app/services/eagle_eye/risk_service.py:compute_position_size | SIMPLIFIED | Core formula and 5% liquidity cap exist, but sector concentration/portfolio heat checks not fully enforced |
| 7.2 | Max concurrent positions 8 and score-ranked suppression emits SIGNAL_SUPPRESSED_RISK | app/services/eagle_eye/risk_service.py:can_open_new_position, app/services/eagle_eye/scanner_service.py:evaluate_symbol | SIMPLIFIED | Max=8 and suppression signal exist; explicit ranking-based overflow selection not implemented |

## Zero-tolerance check summary

Zero-tolerance rules currently NOT exact:
- Failed-breakout guard and BREAKOUT_FAILED workflow: MISSING.
- AVOID 20-session clear rule: MISSING.
- BREAKOUT_CONFIRMED all-7 gate including close-in-top-40%-of-range and ADX-rise/MACD-cross conditions: SIMPLIFIED.
- 8% chase guard from 6.4: MISSING.
- Exit-time-stop 40 sessions and strict arming semantics in distribution EMA30 rule: MISSING/SIMPLIFIED.

Approved simplifications currently present but need explicit components marker:
- Rating components for PE/news-density are effectively neutral stubs but do not yet mark stubbed=true in components_json.

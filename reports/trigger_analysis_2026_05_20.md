# Trigger Analysis - 2026-05-20

Scope: analysis only. No product code, scanner logic, or Behavioral DNA code was modified for this task.

## Executive Summary

Universe screened: 121 eligible Boursa Kuwait names using the existing Eagle Eye data universe and liquidity gate.

- Eligible stocks: 121
- Tier split: 107 PREMIER, 14 MAIN
- Eligibility rule reused from current ML pipeline: at least 500 trading days, at least 50 detected moves, and not ILLIQUID
- Outcome studied: whether a trigger day is followed by a forward max gain of at least 15% or 25%
- Primary ranking metric: 20-day forward hit rate for `+25%` versus each stock's own weighted base rate

The most concrete market-wide triggers were:

1. `breakout_20d_relvol_15`: close breaks above the prior 20-day high and relative volume is above 1.5x. This was the best practical trigger because it combined strong lift with real scale: `12.0%` hit rate for `+25% in 20d` versus a `7.7%` weighted base rate, a `+4.3` percentage-point edge across `1,702` fires. That is about `1.56x` the baseline hit rate.
2. `supertrend_flip_bull`: supertrend flips from bearish to bullish. Raw pooled lift was strongest at `11.7%` versus `6.2%`, a `+5.5` point edge, but it only fired `111` times and should be treated as sparse rather than broad.
3. `ema_ribbon_aligns_bullish`: the ribbon changes into bullish alignment. It produced `9.5%` versus `7.2%`, a `+2.3` point edge across `959` fires.
4. `obv_slope_60_cross_up`: the 60-day OBV slope crosses from negative to positive. It produced `8.9%` versus `7.1%`, a `+1.8` point edge across `617` fires, and held up better than most signals at 60 days.
5. `macd_bull_cross`: MACD line crosses above signal. It was broadly available and slightly positive, `8.1%` versus `7.2%`, but the edge was small enough to treat as secondary confirmation rather than a standalone high-conviction trigger.

If the question is "which 3-5 triggers should we build around first," the answer is:

1. `breakout_20d_relvol_15`
2. `supertrend_flip_bull` with a low-sample warning
3. `ema_ribbon_aligns_bullish`
4. `obv_slope_60_cross_up`
5. `macd_bull_cross` only as a supporting filter, not as a lead trigger

## Method

- Indicators came from the existing Eagle Eye indicator stack.
- Trigger events were evaluated at day `T` using only values available at `T`.
- Forward outcomes were computed from the maximum close gain inside the next `20` and `60` trading days.
- Market-wide baselines were pooled by weighting each stock's own base rate by the number of times the trigger fired in that stock.
- Combination tests used trailing 3-day windows ending at `T`, not future windows.
- `accumulation_score` triggers were explicitly excluded because that indicator currently uses a full-series percentile rank, which introduces look-ahead leakage for event studies.

## Trigger Library

| Trigger | Definition |
| --- | --- |
| `macd_bull_cross` | `macd_line[t-1] <= macd_signal[t-1] and macd_line[t] > macd_signal[t]` |
| `rsi_cross_50_up` | `rsi[t-1] < 50 and rsi[t] >= 50` |
| `rsi_cross_30_up` | `rsi[t-1] < 30 and rsi[t] >= 30` |
| `dmi_bull_cross` | `plus_di[t-1] <= minus_di[t-1] and plus_di[t] > minus_di[t]` |
| `supertrend_flip_bull` | `supertrend[t-1] == -1 and supertrend[t] == 1` |
| `price_reclaims_ema50` | `close[t-1] <= ema_50[t-1] and close[t] > ema_50[t]` |
| `close_above_ichimoku_cloud` | `ichimoku_cloud_pos[t-1] <= 0 and ichimoku_cloud_pos[t] == 1` |
| `obv_slope_60_cross_up` | `obv_slope_60[t-1] <= 0 and obv_slope_60[t] > 0` |
| `cmf_cross_above_010` | `cmf[t-1] <= 0.10 and cmf[t] > 0.10` |
| `ema_ribbon_aligns_bullish` | `ema_ribbon_aligned[t-1] != 1 and ema_ribbon_aligned[t] == 1` |
| `ichimoku_tk_bull_cross` | `ichimoku_tk_cross[t-1] <= 0 and ichimoku_tk_cross[t] > 0` |
| `breakout_20d_relvol_15` | `close[t] > max(high[t-20:t-1]) and close[t-1] <= max(high[t-21:t-2]) and rel_volume[t] > 1.5` |

Excluded trigger:

- `accumulation_score_cross_75`: excluded because `accumulation_score` currently relies on full-series percentile normalization, which leaks future information into the historical trigger label.

## Ranked Market-Wide Table

Primary sort: edge for `+25% in 20d`, measured as trigger hit rate minus weighted base rate.

| Trigger | Fires | Hit +25 in 20d | Base +25 in 20d | Edge | Hit +15 in 20d | Edge | Hit +25 in 60d | Edge | Breadth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `supertrend_flip_bull` | 111 | 11.7% | 6.2% | +5.5 pts | 19.8% | +6.6 pts | 19.0% | +0.5 pts | sparse |
| `breakout_20d_relvol_15` | 1,702 | 12.0% | 7.7% | +4.3 pts | 23.4% | +7.0 pts | 24.6% | +1.5 pts | 62/121 positive-stock lifts |
| `ema_ribbon_aligns_bullish` | 959 | 9.5% | 7.2% | +2.3 pts | 18.6% | +2.7 pts | 22.5% | +0.0 pts | 39/100 positive-stock lifts |
| `obv_slope_60_cross_up` | 617 | 8.9% | 7.1% | +1.8 pts | 19.0% | +3.8 pts | 24.3% | +3.6 pts | 24/77 positive-stock lifts |
| `macd_bull_cross` | 3,255 | 8.1% | 7.2% | +0.9 pts | 16.4% | +0.8 pts | 23.3% | +1.5 pts | 53/121 positive-stock lifts |
| `dmi_bull_cross` | 2,936 | 7.5% | 6.7% | +0.8 pts | 15.9% | +1.1 pts | 20.7% | +0.1 pts | 52/121 positive-stock lifts |
| `close_above_ichimoku_cloud` | 2,235 | 7.9% | 7.1% | +0.8 pts | 17.5% | +1.9 pts | 23.4% | +2.0 pts | 44/121 positive-stock lifts |
| `ichimoku_tk_bull_cross` | 2,077 | 8.1% | 7.3% | +0.8 pts | 18.8% | +3.2 pts | 23.7% | +2.1 pts | 47/121 positive-stock lifts |
| `rsi_cross_50_up` | 5,080 | 7.4% | 7.2% | +0.3 pts | 15.3% | -0.3 pts | 21.4% | -0.2 pts | 44/121 positive-stock lifts |
| `cmf_cross_above_010` | 2,714 | 7.1% | 6.7% | +0.3 pts | 16.1% | +1.3 pts | 21.4% | +0.6 pts | 44/121 positive-stock lifts |
| `price_reclaims_ema50` | 3,751 | 7.1% | 7.3% | -0.2 pts | 15.6% | -0.1 pts | 21.6% | -0.1 pts | 36/121 positive-stock lifts |
| `rsi_cross_30_up` | 765 | 6.3% | 6.6% | -0.4 pts | 14.0% | -0.1 pts | 19.3% | +0.1 pts | 12/73 positive-stock lifts |

Interpretation:

- Positive and actionable: `breakout_20d_relvol_15`, `ema_ribbon_aligns_bullish`, `obv_slope_60_cross_up`
- Positive but sample-limited: `supertrend_flip_bull`
- Mostly noise / mild confirmation only: `macd_bull_cross`, `dmi_bull_cross`, `close_above_ichimoku_cloud`, `ichimoku_tk_bull_cross`, `rsi_cross_50_up`, `cmf_cross_above_010`
- Negative or no better than random: `price_reclaims_ema50`, `rsi_cross_30_up`

## Best Combinations

These are 3-day trailing co-occurrence windows, not same-day exact intersections.

### Best 2-Trigger Combinations

| Combo | Fires | Hit +25 in 20d | Base +25 in 20d | Edge | Hit +25 in 60d | Edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `supertrend_flip_bull + breakout_20d_relvol_15` | 87 | 13.8% | 6.2% | +7.6 pts | 20.7% | +2.5 pts |
| `breakout_20d_relvol_15 + ema_ribbon_aligns_bullish` | 259 | 15.4% | 8.1% | +7.4 pts | 30.0% | +5.7 pts |
| `breakout_20d_relvol_15 + obv_slope_60_cross_up` | 168 | 15.5% | 8.6% | +6.9 pts | 28.5% | +5.4 pts |
| `breakout_20d_relvol_15 + macd_bull_cross` | 503 | 13.7% | 7.7% | +6.1 pts | 27.2% | +4.3 pts |
| `supertrend_flip_bull + macd_bull_cross` | 47 | 10.6% | 5.2% | +5.5 pts | 17.4% | +0.3 pts |

### Best 3-Trigger Combinations

| Combo | Fires | Hit +25 in 20d | Base +25 in 20d | Edge | Hit +25 in 60d | Edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `breakout_20d_relvol_15 + obv_slope_60_cross_up + macd_bull_cross` | 32 | 21.9% | 9.5% | +12.4 pts | 33.3% | +9.2 pts |
| `breakout_20d_relvol_15 + obv_slope_60_cross_up + dmi_bull_cross` | 63 | 20.6% | 9.8% | +10.9 pts | 35.7% | +10.2 pts |
| `supertrend_flip_bull + breakout_20d_relvol_15 + dmi_bull_cross` | 38 | 15.8% | 6.9% | +8.9 pts | 25.0% | +7.5 pts |
| `breakout_20d_relvol_15 + ema_ribbon_aligns_bullish + macd_bull_cross` | 98 | 16.3% | 7.7% | +8.6 pts | 32.0% | +8.1 pts |
| `breakout_20d_relvol_15 + macd_bull_cross + dmi_bull_cross` | 226 | 16.4% | 8.0% | +8.4 pts | 31.2% | +8.5 pts |

Main takeaway from combinations: the breakout signal is the common anchor. The best pairs and triples almost all include `breakout_20d_relvol_15`, then improve further when trend-alignment signals join it.

## Tier Split

### PREMIER

| Trigger | Fires | Hit +25 in 20d | Base +25 in 20d | Edge |
| --- | ---: | ---: | ---: | ---: |
| `supertrend_flip_bull` | 102 | 11.8% | 6.4% | +5.4 pts |
| `breakout_20d_relvol_15` | 1,537 | 12.4% | 8.0% | +4.4 pts |
| `obv_slope_60_cross_up` | 545 | 9.4% | 7.2% | +2.2 pts |
| `ema_ribbon_aligns_bullish` | 844 | 9.6% | 7.5% | +2.1 pts |
| `macd_bull_cross` | 2,880 | 8.4% | 7.4% | +1.0 pts |

### MAIN

| Trigger | Fires | Hit +25 in 20d | Base +25 in 20d | Edge |
| --- | ---: | ---: | ---: | ---: |
| `ema_ribbon_aligns_bullish` | 115 | 8.7% | 4.8% | +3.9 pts |
| `rsi_cross_30_up` | 40 | 12.5% | 9.0% | +3.5 pts |
| `ichimoku_tk_bull_cross` | 242 | 8.7% | 5.1% | +3.5 pts |
| `breakout_20d_relvol_15` | 165 | 7.9% | 5.1% | +2.8 pts |
| `cmf_cross_above_010` | 286 | 7.0% | 5.0% | +2.0 pts |

Tier note: MAIN results are directionally interesting but much less reliable because they come from only `14` eligible names. PREMIER should carry more product-design weight.

## Practical Conclusions

- If only one event should be prototyped first, use `breakout_20d_relvol_15`.
- If a second high-value confirmation is added, favor `ema_ribbon_aligns_bullish` or `obv_slope_60_cross_up`.
- `supertrend_flip_bull` is worth monitoring, but not as the main production trigger until it proves itself out-of-sample because the sample is thin.
- Pure reclaim / oversold bounce signals were not compelling here. `price_reclaims_ema50` and `rsi_cross_30_up` did not beat baseline.

## Caveats

- In-sample only: this study was run on the same historical universe used to discover the patterns. It is not an out-of-sample or walk-forward validation.
- Thin-volume unreliability: the worst illiquid names were filtered out, but MAIN-tier names are still thin enough that trigger reliability is less stable than PREMIER.
- Multiple-testing risk: even with only 12 single triggers, the pair and triple search space is much larger. Some high-lift combinations will appear by chance.
- Look-ahead bias protection: trigger definitions were restricted to values observable at day `T`, and the combination windows were trailing only.
- Explicit leakage exclusion: `accumulation_score`-based triggers were excluded because the current implementation uses full-series percentile normalization and therefore is not safe for event testing.
- Breadth caveat: `supertrend_flip_bull` had the highest pooled lift, but with only `111` fires it is better treated as an opportunistic signal than a market-wide workhorse.

## Bottom Line

The clearest answer from this run is that Boursa Kuwait's most actionable trigger is not a generic oscillator cross. It is a breakout event: price taking out the prior 20-day high on materially elevated relative volume. Trend-alignment triggers help, especially ribbon alignment and OBV slope recovery, but the breakout is the recurring backbone in both single-signal and combo results.

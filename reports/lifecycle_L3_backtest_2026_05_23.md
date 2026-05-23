# Lifecycle L3 Backtest - CONFIRMED Entry

Generated: 2026-05-23

## 1. Exact Rules Used

- Entry signal: transition into `state == CONFIRMED` from the default lifecycle labeler config, with no retuning.
- Signal timing: signal on day D, entry at day D+1 open from OHLCV. No signal-day fills.
- Stop policy: fixed 8.0% stop from entry. This keeps the exit explicit and avoids introducing another adaptive parameter while testing the entry.
- Time exit: next available open after 20 full trading days if the stop was not hit first.
- Stop execution rule: if a post-entry bar traded through the stop (`low <= stop`), the exit filled at the stop price on that bar.
- Cost model (moderate): Premier 0.15% commission + 0.10% slippage each leg; Main 0.15% + 0.30% each leg. Reused from the existing backtester.
- Cost model (harsh): same commission, doubled slippage.
- Random baseline: for each actual entry date, choose a random non-signal trade path from the same date when possible, with same-month fallback only if a date is undersupplied.
- Liquidity cap: 10% of rolling 20-day traded value (`close * volume`). Shrink above cap; skip below 100 KWD.
- Portfolio model: starting equity 100,000 KWD, target 10% of current equity per trade, no leverage.
- Universe: all cached tickers with at least 250 bars. This remains survivor-biased because it depends on the current OHLCV cache.

## 2. Tradeable Signal Count

- Eligible universe size: 137 tickers.
- Raw CONFIRMED transitions: 93.
- Tradeable after liquidity: 93 (100.0%).
- Liquidity shrinks: 0.
- Liquidity skips: 0.

## 3. Full Real Metrics

- Scenario: CONFIRMED moderate
- Executed trades: 93
- Win rate: 37.63%
- Avg win: 9.98%
- Avg loss: -5.57%
- Win/loss ratio: 1.791
- Expectancy: 0.280% per trade (16.82 KWD)
- Mean net return: 0.280%
- Median net return: -1.524%
- Total net P&L: 1563.85 KWD
- Equity start/end: 100000.00 -> 101563.85 KWD
- Max drawdown: -15.44%
- CAGR: 0.53%

### Outcome Distribution

- <=-10%: 0
- -10% to -5%: 32
- -5% to 0%: 26
- 0% to 5%: 17
- >5%: 18
- Big losers (<= -8% net): 31

### Sample Trades

- INTEGRATED: signal 2023-06-11, entry 2023-06-12 @ 404.000, exit 2023-07-16 @ 443.000, TIME_EXIT, net 9.13%, costs 52.41 KWD
- AZNOULA: signal 2023-06-19, entry 2023-06-20 @ 196.000, exit 2023-07-26 @ 201.000, TIME_EXIT, net 2.04%, costs 51.09 KWD
- MUNTAZAHAT: signal 2023-07-05, entry 2023-07-06 @ 73.200, exit 2023-08-07 @ 74.000, TIME_EXIT, net 0.59%, costs 50.87 KWD
- MARKAZ: signal 2023-07-09, entry 2023-07-10 @ 102.000, exit 2023-08-09 @ 103.000, TIME_EXIT, net 0.48%, costs 50.77 KWD
- BOURSA: signal 2023-07-23, entry 2023-07-24 @ 1922.000, exit 2023-08-21 @ 1990.000, TIME_EXIT, net 3.03%, costs 51.40 KWD
- WARBACAP: signal 2023-09-19, entry 2023-09-20 @ 314.000, exit 2023-09-20 @ 288.880, STOP_LOSS, net -8.48%, costs 48.74 KWD
- NIH: signal 2023-09-24, entry 2023-09-25 @ 124.000, exit 2023-10-08 @ 114.080, STOP_LOSS, net -8.48%, costs 48.35 KWD
- MAZAYA: signal 2023-09-19, entry 2023-09-20 @ 55.400, exit 2023-10-19 @ 55.000, TIME_EXIT, net -1.22%, costs 50.58 KWD
- SOKOUK: signal 2023-10-03, entry 2023-10-04 @ 40.400, exit 2023-11-01 @ 40.500, TIME_EXIT, net -0.25%, costs 50.30 KWD
- NOOR: signal 2023-11-02, entry 2023-11-05 @ 185.000, exit 2023-12-03 @ 200.000, TIME_EXIT, net 7.59%, costs 51.85 KWD

## 4. Baseline Comparison

- Entry-date matched random entry baseline (25 runs of 74 trades): mean expectancy 0.741% per trade, median expectancy 0.495%, positive total-P&L runs 60.0%, average same-month fallbacks 0, average shortfall 19.
- KSE proxy buy-and-hold over 2023-06-12 to 2026-05-17: 28.61% return, CAGR 8.97%.

## 5. Sensitivity Checks

### Cost Stress

- Scenario: CONFIRMED harsh
- Executed trades: 93
- Win rate: 36.56%
- Avg win: 10.06%
- Avg loss: -5.68%
- Win/loss ratio: 1.771
- Expectancy: 0.075% per trade (-3.81 KWD)
- Mean net return: 0.075%
- Median net return: -1.723%
- Total net P&L: -354.02 KWD
- Equity start/end: 100000.00 -> 99645.98 KWD
- Max drawdown: -16.12%
- CAGR: -0.12%

### Exclude Best Stock

- Best stock by net P&L in the moderate run: IFAHR
- Scenario: CONFIRMED moderate ex-IFAHR
- Executed trades: 91
- Win rate: 36.26%
- Avg win: 8.02%
- Avg loss: -5.57%
- Win/loss ratio: 1.440
- Expectancy: -0.641% per trade (-69.58 KWD)
- Mean net return: -0.641%
- Median net return: -1.557%
- Total net P&L: -6331.85 KWD
- Equity start/end: 100000.00 -> 93668.15 KWD
- Max drawdown: -15.97%
- CAGR: -2.21%

## 6. Verdict

CONFIRMED shows a positive moderate-cost expectancy (0.280% per trade), but the edge looks fragile. It does not beat the random-entry baseline, it does not stay positive under harsh costs, and it does not survive removing the best stock. That is not strong enough to call the edge robust.

## Limitations

- The universe comes from the current OHLCV cache, so this remains survivor-biased.
- Market tier uses the existing repo heuristic based on median daily volume because there is no full historical exchange-tier table in the local DB.
- The index comparison uses a monthly proxy CSV because the KSE index series is absent from `ee_ohlcv_cache`.
- If trade count is small, treat any apparent edge as weak evidence rather than proof.

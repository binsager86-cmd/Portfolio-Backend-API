# Behavioral DNA Fix Validation — 2026-05-20

## Part 1 Summary

### Current event population before the fix

- `detect_moves()` records an event only after a forward gain is confirmed.
  - Evidence: `app/services/eagle_eye/move_detector.py` filters with `if gain_pct < threshold: continue`.
- Fakeouts are recorded separately by `detect_fakeouts()`.
- In the old DNA math, fakeouts were not part of the success-rate denominator.
- So the old unit of analysis was a hindsight-confirmed move, not a setup at time `T`.

### Correct denominator now used

The new DNA base rates use setup occurrences, not confirmed moves:

- A setup is the current live signal fingerprint for the stock.
- The fingerprint is selected from the stock's currently active precursor signals, ranked by historical usefulness.
- Up to 3 active signals are used, and the match is relaxed only if needed to reach enough historical setup occurrences.
- A setup occurrence is the first day of a contiguous run where all selected setup signals are active.
- Forward outcome = maximum close-to-close gain over the next `H` trading days.
- Success rate for each threshold is `hits / total_setups`.

### Horizon confirmation

`H = 180` trading days.

Reason:

- The existing DNA move logic already uses `CONFIG.MAX_MOVE_LOOKAHEAD_DAYS = 180`.
- The current target ladder includes `+50%` and `+100%`, which is not compatible with a very short horizon like 20 days for most names.

### Sample-size guard

- If fewer than 20 matching setup occurrences are found, the stock is marked `INSUFFICIENT_HISTORY`.
- In that state, the API returns the setup fingerprint and sample count, but no percentage table.

## Part 1 Validation

### Unit-level validation

- `pytest tests/test_eagle_eye_dna_extractor.py -q` passed.
- The regression test now proves:
  - denominator = all setup occurrences
  - success rates are cumulative and monotonic decreasing
  - percentages are hidden when setup count is below the minimum

### Real-stock validation

#### CLEANING

- `history_status`: `INSUFFICIENT_HISTORY`
- `matching_setups`: `13`
- `setup_signals`: `wyckoff_in_markup`
- `setup_horizon_days`: `180`
- Threshold table: not shown because setup count is below the 20-occurrence minimum.

Interpretation:

- This is the expected corrected behavior.
- The old screen showed extreme percentages for CLEANING because it was profiling hindsight-selected moves.
- Under the new denominator, CLEANING does not have enough matching setup history to justify historical hit-rate percentages.

#### ENERGYH

- `history_status`: `ok`
- `matching_setups`: `24`
- `setup_signals`: `obv_60d_slope_strongly_positive`, `volume_breakout_15x`
- `setup_horizon_days`: `180`

| Target | Hits | Total Setups | Success Rate | Avg Gain All | Avg Gain on Hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| +10% | 22 | 24 | 91.7% | 82.4% | 89.2% |
| +15% | 20 | 24 | 83.3% | 82.4% | 96.8% |
| +25% | 16 | 24 | 66.7% | 82.4% | 116.1% |
| +50% | 10 | 24 | 41.7% | 82.4% | 163.0% |
| +100% | 7 | 24 | 29.2% | 82.4% | 198.7% |

Checks:

- Success rates are monotonic decreasing: **YES**
- Rates are no longer circular 100% winner-profiling numbers: **YES**

#### ALSAFAT

- `history_status`: `INSUFFICIENT_HISTORY`
- `matching_setups`: `13`
- `setup_signals`: `wyckoff_in_markup`, `volume_breakout_2x`, `volume_breakout_15x`
- `setup_horizon_days`: `180`
- Threshold table: not shown because setup count is below the 20-occurrence minimum.

## Part 2 Summary

The mobile Behavioral DNA screen now reflects the corrected math:

- Headline explains these are historical base rates for this setup within the configured horizon.
- Matching setup count and forward horizon are shown prominently.
- Average gain across all setups is shown as the main expectation metric.
- Target outcomes are rendered as success-rate bars with explicit `hits / total setups` counts.
- "What fires before this stock moves" is replaced with setup-frequency bars showing how often each precursor was present across the matched setups.
- Stocks with fewer than 20 matching setups now show an insufficient-history state instead of misleading percentages.

## Final Sanity Check

The corrected CLEANING result is:

- **INSUFFICIENT_HISTORY**
- **13 matching setups**
- **setup fingerprint: `wyckoff_in_markup`**

That is the number to sanity-check against the old winner-only DNA table. Under the corrected denominator, CLEANING does not currently have enough setup history to justify displayed target percentages.
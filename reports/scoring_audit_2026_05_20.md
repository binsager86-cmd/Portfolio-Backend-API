# Eagle Eye Scoring Audit — 2026-05-20

## Scope

This audit compares the production scanner confidence path against the Behavioral DNA historical statistics path to explain the observed contradiction:

1. Scanner confidence has clustered around 56-67 and has never exceeded 67 in production.
2. Behavioral DNA has shown 100% hit rates for large targets on some stocks.

No new fixes were made as part of this audit. Note that the current worktree already contains uncommitted local DNA-stat changes from an earlier investigation, so this report distinguishes:

- the production scanner scoring path now in use
- the historical DNA generation logic and what it actually measures

## Executive Summary

All three hypotheses are real, but not equally:

- **Hypothesis A — Confidence calculation is broken / artificially suppressed:** **TRUE**
- **Hypothesis B — Historical statistics are inflated / biased:** **TRUE**
- **Hypothesis C — Both are wrong in different ways:** **TRUE**

The contradiction is not because one side is slightly off. It exists because the two numbers are not measuring the same thing, and both pipelines have issues:

- The scanner confidence path is compressed by design and by implementation errors.
- The Behavioral DNA statistics are still selection-biased even after cumulative-target cleanup.

The higher-priority trust problem is on the **historical statistics side**. The scanner score is range-compressed, but the DNA hit rates are not a valid forward-looking probability of a current setup.

## Step 1 — Confidence Score Audit

### Owning code

- Production scoring entrypoint: `app/services/eagle_eye/ingest.py`
  - `compute_confidence(latest, stage, dna=None)` at line 412
  - liquidity adjustment at line 421
  - thin-volume dampener logging at line 443
- Rule engine: `app/services/eagle_eye/rating_engine.py`
  - `compute_confidence()` at line 539
  - confluence score at line 577
  - fixed `historical_base_rate = 0.5` at line 580
  - fixed `dna_match = 0.5` at line 621
  - stage caps at lines 649-660
  - structural readiness cap at line 679

### Inputs and weights

The rule engine uses these weighted components:

- 0.25 `confluence_score`
- 0.20 `historical_base_rate`
- 0.15 `accumulation_score`
- 0.15 `risk_reward_score`
- 0.10 `regime_alignment`
- 0.10 `stage_score`
- 0.05 `dna_pattern_match`

### What actually feeds production today

In the current production path, `compute_all_ratings()` calls:

`compute_confidence(latest, stage, dna=None)`

That has three important consequences:

1. `historical_base_rate` is not computed from DNA. It is hardcoded to `0.5`, contributing a fixed **10.0 points**.
2. `dna_match` is not computed from DNA. It is hardcoded to `0.5`, contributing a fixed **2.5 points**.
3. `_risk_reward_ratio` is usually missing in the indicator dict, so `risk_reward_score` falls back to a default `0.6`, contributing a fixed **9.0 points**.

So three of the seven score components are effectively fixed in production.

### Confidence compression bug in confluence

`confluence_score` is supposed to reflect bullish evidence across indicator categories, but the implementation only increments bullish counts for these categories:

- `trend`
- `momentum`
- `volume_flow`

If other categories are present and have checked indicators, they still contribute to the denominator but can never contribute positive bullish counts. In live rows, this means categories like `volatility` and `institutional` drag the confluence score down to zero contribution even when populated.

This is not just a design choice. It is an implementation asymmetry that compresses scores.

### Hard caps / ceilings / dampeners

The scoring path has multiple suppressors:

1. **Stage caps** in `rating_engine.py`
   - `EARLY_BREAKOUT`: 100
   - `MARKUP_TRENDING`: 90
   - `STEALTH_ACCUMULATION`: 75
   - `DORMANT`: 40
   - `ACCELERATION_CLIMAX`: 55
   - `DISTRIBUTION_TOPPING`: 30
   - `MARKDOWN_DECLINE`: 20
   - `CAPITULATION_EXHAUSTION`: 50

2. **Structural readiness cap**
   - If ATR percentile, relative volume, and price/RSI readiness are not met, confidence is capped at **55**.

3. **Liquidity / volume adjustment in `ingest.py`**
   - `ILLIQUID`: multiply by 0.50
   - `WATCH_ONLY`: multiply by 0.70
   - unconfirmed volume: multiply by 0.85
   - very high relative-volume percentile: multiply by 1.10, capped at 100

4. **Thin-volume-on-rise dampener**
   - If `rel_liq < 0.5` and `today_return > 0.02`, confidence is capped at **60**.

### Theoretical maximum confidence

There are two different answers depending on what “theoretical” means.

#### Absolute function maximum

If `compute_confidence()` is called with:

- DNA present and fully bullish
- `_risk_reward_ratio >= 2.0`
- `EARLY_BREAKOUT`
- `RISK_ON`
- no structural cap
- no liquidity suppression

then the function can mathematically reach **100**.

#### Actual production-path maximum

That is not what production is doing.

Given the current scanner callsite:

- `dna=None`
- no injected risk/reward ratio
- neutral regime
- live rows typically populate 5 categories, but only 3 categories can score bullish in confluence

the practical raw ceiling is much lower.

For the current production shape:

- max confluence contribution: **15.0**
- historical base rate: **10.0 fixed**
- accumulation: **15.0 max**
- risk/reward: **9.0 fixed**
- regime: **6.0 fixed**
- stage: **10.0 max for EARLY_BREAKOUT**, **8.0 for MARKUP_TRENDING**
- DNA match: **2.5 fixed**

That gives:

- **67.5 raw max** for `EARLY_BREAKOUT`
- **65.5 raw max** for `MARKUP_TRENDING`

If the post-score `HIGH_RV * 1.10` boost applies, those become:

- **74.25 practical max** for `EARLY_BREAKOUT`
- **72.05 practical max** for `MARKUP_TRENDING`

This explains why production has lived in the high-50s to high-60s. The score is not truly 0-100 in practice.

### Three recent stocks — actual confidence breakdowns

Current top scanner rows:

1. `CLEANING` — 66.76 — `MARKUP_TRENDING`
2. `ENERGYH` — 66.20 — `MARKUP_TRENDING`
3. `ALSAFAT` — 65.49 — `EARLY_BREAKOUT`

#### CLEANING

- Cached confidence: **66.76**
- Raw score before post-adjustments: **60.686**
- Stage cap: **90** (not binding)
- Structural cap: **not applied**
- Liquidity adjustment: `HIGH_RV * 1.10`
- Thin-volume dampener: **not applied**

Contribution breakdown:

- Confluence: **12.57**
- Historical base rate: **10.00**
- Accumulation: **12.616**
- Risk/reward: **9.00**
- Regime: **6.00**
- Stage: **8.00**
- DNA match: **2.50**

Firing conditions / category evidence:

- Trend: 13/13 bullish
  - includes `ema_8`, `ema_21`, `ema_50`, `ema_100`, `ema_200`, `ema_ribbon_aligned`, `macd_line`, `macd_histogram`, `adx`, `plus_di`
- Momentum: 8/10 bullish
  - includes `rsi`, `stoch_k`, `stoch_d`, `stoch_rsi`, `cci`, `roc`, `tsi`, `connors_rsi`
- Volume flow: 5/7 bullish
  - includes `obv`, `cmf`, `mfi`, `vwap`, `force_index`
- Volatility: checked, but **0 bullish by implementation**
- Institutional: checked, but **0 bullish by implementation**

#### ENERGYH

- Cached confidence: **66.20**
- Raw score before post-adjustments: **60.176**
- Liquidity adjustment: `HIGH_RV * 1.10`
- Thin-volume dampener: **not applied**

Contribution breakdown:

- Confluence: **11.80**
- Historical base rate: **10.00**
- Accumulation: **12.876**
- Risk/reward: **9.00**
- Regime: **6.00**
- Stage: **8.00**
- DNA match: **2.50**

#### ALSAFAT

- Cached confidence: **65.49**
- Raw score before post-adjustments: **59.542**
- Stage: `EARLY_BREAKOUT`
- Liquidity adjustment: `HIGH_RV * 1.10`
- Thin-volume dampener: **not applied**

Contribution breakdown:

- Confluence: **12.185**
- Historical base rate: **10.00**
- Accumulation: **9.857**
- Risk/reward: **9.00**
- Regime: **6.00**
- Stage: **10.00**
- DNA match: **2.50**

## Step 2 — Dampener Audit

### Thin-volume-on-rise dampener

Audit result for the current top 50 scanner stocks:

- **0 / 50** had the thin-volume-on-rise dampener applied.

So the explicit `rel_liq < 0.5 AND today_return > 0.02` cap is **not** what is holding the top of the scanner at 67 today.

### Other suppressors / adjustments active today

These are materially affecting the live distribution:

- **Structural readiness cap hit on 30 / 50** names
- **Liquidity/volume adjustment applied on 48 / 50** names

Examples:

- `AAYAN` — `UNCONFIRMED * 0.85`
- `AAYANRE` — `UNCONFIRMED * 0.85`
- `ALDEERA` — `UNCONFIRMED * 0.85`
- `ALIMTIAZ` — `UNCONFIRMED * 0.85`
- `CLEANING`, `ENERGYH`, `ALSAFAT`, `RASIYAT`, `KPPC` — `HIGH_RV * 1.10`

Conclusion: the named thin-volume dampener is not the main live culprit today. The much larger live effect comes from the structural cap plus the fact that the raw score itself is already compressed.

## Step 3 — Historical Statistics Audit

### Owning code

- Move detector: `app/services/eagle_eye/move_detector.py`
  - future threshold check at line 79
  - event creation with `threshold_pct=threshold` at line 126
  - threshold bucketing suppression via `used_starts.add((i, t))` at line 143
- DNA aggregation: `app/services/eagle_eye/dna_extractor.py`
  - cumulative threshold evaluation at line 194
  - success rate denominator at line 212
  - `avg_gain_all_pct` at line 216

### Re-check of earlier bugs

#### Bug A — Cumulative target evaluation

- **Current local code:** fixed. Targets are now evaluated cumulatively with:
  - `tier_real = [s for s in real_moves if float(s.event.gain_pct) >= threshold]`
- **Production cached DNA / prior behavior:** this bug was real before the local patch. Older stored outputs reflected exact-threshold buckets.

Verdict: **REAL historically, fixed locally in the current worktree.**

#### Bug B — Avg gain only over winners

- **Current local code:** partially fixed.
  - `avg_gain_all_pct` is now present.
  - `avg_gain_on_hits_pct` is also present.
- **Production cached DNA / prior behavior:** the displayed `avg_gain_pct` was winner-only.

Verdict: **REAL historically, partially addressed locally.**

#### Bug C — Failed events in denominator

- **Current local code:** still not fully correct if the intended denominator is all candidate events including failures.
  - `success_rate = len(tier_real) / max(1, len(real_moves)) * 100`
  - That excludes fakeouts from the success-rate denominator.
- `avg_gain_all_pct` does include fakeouts/losses in the return average, but success rate does not.

Verdict: **REAL.**

### Deeper issue: the dataset itself is selection-biased

Even after cumulative-target cleanup, the bigger problem is upstream in `detect_moves()`.

The move detector only emits an event after looking forward and confirming that a future peak exceeded the threshold:

`if gain_pct < threshold: continue`

That means the Behavioral DNA event library is not a neutral set of all possible setups. It is a library of **already-realized successful moves**, plus separately detected fakeouts.

That creates a much larger bias than the display bugs:

- The event set is selected with look-ahead knowledge.
- Large winners dominate the library.
- The resulting success rates are descriptive of the detected move library, not predictive probabilities for a fresh live signal.

This is why even the corrected CLEANING DNA still shows extreme hit rates.

## Step 4 — CLEANING Sanity Check

### Current live score

`CLEANING` is the top scanner stock today at **66.76** with stage **MARKUP_TRENDING**.

Current production-path breakdown:

- Raw confidence before live adjustments: **60.686**
- Stage cap: **90**
- Structural cap: **not applied**
- Liquidity adjustment: `HIGH_RV * 1.10`
- Thin-volume-on-rise dampener: **not applied**
- Final confidence: **66.75** (matches cached 66.76)

### CLEANING theoretical max

Under the **actual production path** for a stock like CLEANING:

- DNA contribution is fixed, not dynamic
- risk/reward contribution is fixed, not dynamic
- confluence is dragged down by categories that can never score bullish

For a `MARKUP_TRENDING` live row with the current implementation, the effective raw ceiling is about **65.5**, and about **72.05** after the `HIGH_RV * 1.10` boost.

So CLEANING is not dramatically under-scored relative to the current production engine. It is already close to that compressed ceiling.

### CLEANING corrected historical statistics

Local recompute for CLEANING with the current cumulative-target DNA extractor:

- Total events: **318**

| Target | Success Rate | Occurrences | Avg Gain All | Avg Gain on Hits |
| --- | ---: | ---: | ---: | ---: |
| +10% | 100.0% | 318 | 102.2% | 153.9% |
| +15% | 100.0% | 318 | 102.2% | 153.9% |
| +25% | 100.0% | 318 | 102.2% | 153.9% |
| +50% | 97.2% | 309 | 102.2% | 157.1% |
| +100% | 81.8% | 260 | 102.2% | 174.1% |

Top signal fire rates:

- `wyckoff_in_markup` — 60.7% (193/318)
- `accumulation_above_75` — 17.3% (55/318)
- `obv_60d_slope_strongly_positive` — 91.5% (291/318)
- `above_ichimoku_cloud` — 91.5% (291/318)
- `plus_di_dominates` — 91.5% (291/318)
- `wyckoff_in_accumulation` — 31.8% (101/318)
- `rsi_bullish_divergence` — 100.0% (318/318)
- `supertrend_bullish` — 100.0% (318/318)

Interpretation:

This does **not** show that the confidence engine is underrating CLEANING by a little. It shows that the DNA statistics are still drawn from a highly filtered library of already-realized moves. Those numbers are not directly comparable to the live 66.76 scanner score.

## Step 5 — Hypothesis Verdicts

### Hypothesis A — Confidence calculation is broken

**TRUE**

Reasons:

- production scoring ignores DNA entirely (`dna=None` at the callsite)
- production scoring uses a default risk/reward contribution instead of live computed risk/reward
- confluence only credits 3 categories while other populated categories still drag down the denominator
- the resulting production score is effectively compressed into a narrow band

### Hypothesis B — Historical statistics are inflated

**TRUE**

Reasons:

- prior exclusive-threshold bug was real
- prior winner-only average bug was real
- success-rate denominator still excludes fakeouts
- most importantly, the move library is selected using future threshold hits, so the event set is look-ahead biased

### Hypothesis C — Both are wrong in different ways

**TRUE**

The scanner score is compressed and only partially connected to the intended inputs. The DNA statistics are still not a trustworthy forward-looking probability surface.

## Recommendation

### Recommendation 1 — Change both, not one

Do **not** try to make the UI trust one number and discard the other. Both pipelines need revision, for different reasons.

### Recommendation 2 — Confidence calculation

The scanner confidence should be revised so that it actually reflects the intended model inputs:

- pass DNA into production scoring if DNA is intended to matter
- inject real risk/reward instead of the fixed default
- either score all populated categories explicitly or remove non-scored categories from the confluence denominator
- document the true live range if the score is intentionally not 0-100

### Recommendation 3 — Historical statistics display

The DNA display should not present current target hit rates as if they are forward probabilities of a new live setup unless the dataset is rebuilt from neutral candidate setups.

If the current move library is retained, the UI copy should describe it as:

- a profile of **past realized move behavior**
- not a calibrated probability of a fresh live signal

To make it trustworthy for decision support, the historical statistics need a different denominator:

- all qualifying candidate setups at time `t`
- then measure forward outcomes from those setups
- not only setups that are known in hindsight to have become moves

## Final Call

The cleaner explanation for the contradiction is:

- the scanner confidence is range-compressed and partially hardcoded
- the historical DNA table is still selection-biased and therefore overstated as a probability measure

If forced to choose which side is currently less trustworthy as a decision aid, the answer is:

**The historical statistics are less trustworthy than the scanner score.**

They still look like probabilities, but they are generated from a hindsight-selected event library.
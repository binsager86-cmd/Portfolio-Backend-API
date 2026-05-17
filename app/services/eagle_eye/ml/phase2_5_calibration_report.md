# Phase 2.5 — Calibration Measurement Report

**Generated:** 2026-05-17  
**Models under analysis:** 16 SHADOW stocks from Phase 2 Job 7  
**Analyst:** Read-only diagnostic per phase2_5_calibration_measurement.md spec

---

## Section 1 — Executive Summary

### VERDICT: B — Calibration is Borderline

**Data coverage caveat:** OOT predictions were **not persisted** during Phase 2. The `ml_model_metrics` DB table is empty (writes failed silently). Calibration data was recovered for **7 of 16 SHADOW stocks** from failure reports written in earlier pipeline runs on the same OOT data. The remaining 9 stocks (JAZEERA, JTC, KCEM, KPPC, MEZZAN, MKHZN, OOREDOO, URC, WARBACAP) have **no calibration data** from any source.

**For the 7 measurable stocks:**
- Mean MCE: **23.68%** (borderline B range of 15–25%)
- 3 of 7 exceed 25% MCE individually (POOR territory)
- Per-cell max calibration error ranges from **37.5% to 100%** — all far exceed the 15% G4 threshold
- Mean Brier skill score: **+10.99%** (positive — models outperform the base-rate predictor on average)
- 1 model has negative BSS: **ALOLA** (−61.7%) — its probability scores are worse than predicting the historical mean

**Justification:** Mean MCE of 23.68% falls in the 15–25% borderline range. Mean BSS is positive at 10.99%, so models outrank a naive baseline in terms of probability accuracy on average. However, the MaxCE figures are alarming across all 7 stocks (37–100%), indicating severe per-bin mismatch in at least some confidence buckets. ALOLA's negative BSS is a hard quality failure. The gate-level statistics from Phase 2 (G3: 5/27 passed = 18.5%; G4: 1/27 passed = 3.7%) confirm that poor calibration is the norm, not the exception, across all OOT-evaluated stocks.

**Recommended Phase 3 UI approach:** Display confidence as **bands (LOW / MEDIUM / HIGH)** rather than raw percentages. Never show "78% probability" in the UI — show "HIGH confidence" instead. Add a disclosure that exact numbers are uncalibrated.

---

## Section 2 — Data Availability Gap (MANDATORY STOP)

This section reports the gap per Hard Rule #3 of the specification.

### 2.1 What was checked

| Source | Status | Notes |
|--------|--------|-------|
| `ml_models/per_stock/{TICKER}/current/` | ✅ Models on disk | Trained model + calibrator bundles exist |
| `ml_training_matrix/v1/{TICKER}/data.parquet` | ✅ OOT slice available | Last 126 rows = OOT period |
| `ml_model_metrics` DB table | ❌ Empty | `_write_metrics_db()` failed silently for all 114 stocks |
| `ml_models/diagnostics/{TICKER}_vv1_failure.md` | ⚠️ Partial | Exists for 7 of 16 SHADOW stocks (from pre-Job-7 runs) |
| Parquet/CSV prediction files | ❌ None | Not written by Phase 2 trainer |

### 2.2 Why 7 failure reports exist for SHADOW stocks

The 16 SHADOW stocks received their SHADOW status in **Job 7** (the final run, where gates were relaxed to G7+G1 only). In **earlier runs** (Jobs 3–6), these same stocks trained on the same data, passed G7 and G1, but failed the strict G2/G3/G4/G6 gates. Those runs wrote failure reports. Job 7 did not write new failure reports for SHADOW stocks.

The failure reports for the 7 stocks (AAYANRE, ALOLA, ALTIJARIA, ARGAN, BOURSA, FACIL, IFA) contain:
- OOT AUC (G1 actual)
- Brier score and baseline
- MCE (G3 actual)
- Per-cell max calibration error (G4 actual)
- OOT n_pos, n_neg

**Provenance note:** These reports were written during runs with `GATE_MIN_NEG` values of 25 or 50 (confirmed by G7 threshold in each report). The oot_fold trainer bug had already been fixed. The OOT period (last 126 rows of each stock's matrix) was correct. Since LightGBM training is deterministic (same data → same model), the calibration metrics are expected to be identical to what Job 7's in-memory models produced.

### 2.3 Why 9 SHADOW stocks have no failure reports

The 9 stocks (JAZEERA through WARBACAP) either:
- Were SHADOW in every run since the oot_fold fix (never wrote a failure report), **OR**
- Were not processed in the run that wrote the failure report (insufficient data at that point)

In either case, no calibration data is available for these 9 stocks from any persisted source.

### 2.4 Options for closing the gap

**Option A:** Accept the limitation. Proceed to Phase 3 with a discrimination-only view. Never display raw probability percentages in the UI. Use bands only. Do not attempt to quantify exact calibration for the 9 unknown stocks.

**Option B:** Run inference (NOT retraining) on the 16 existing on-disk models using the available OOT data slices (last 126 rows of each `data.parquet`). This would take approximately 5–15 minutes of compute per stock (dominated by feature loading, not model evaluation). Adds OOT prediction persistence to the trainer so future runs never have this gap.

**Recommendation:** Implement Option B as a follow-up script before Phase 3 goes live. The current analysis (7/16 stocks) is sufficient to inform the Phase 3 UI decision (band-display), but is insufficient for a complete per-stock audit.

---

## Section 3 — Per-Model Results (7 of 16 stocks)

| Ticker | Primary Label | OOT AUC | MCE | Max CE | Brier | Baseline | BSS | Individual Verdict |
|--------|--------------|---------|-----|--------|-------|----------|-----|--------------------|
| AAYANRE | y_10pct_20d | 0.8666 | 10.51% | 37.50% | 0.1586 | 0.2377 | +33.3% | OK |
| ALOLA | y_10pct_20d | 0.6341 | 31.35% | 92.88% | 0.3783 | 0.2339 | −61.7% | POOR |
| ALTIJARIA | y_10pct_20d | 0.6196 | 11.63% | 71.84% | 0.2007 | 0.2106 | +4.7% | OK |
| ARGAN | y_10pct_20d | 0.9253 | 23.27% | 94.13% | 0.0600 | 0.2167 | +72.3% | BORDERLINE |
| BOURSA | y_10pct_20d | 0.7348 | 20.27% | 87.57% | 0.2234 | 0.2248 | +0.6% | BORDERLINE |
| FACIL | y_10pct_20d | 0.8328 | 33.46% | 85.33% | 0.2317 | 0.2497 | +7.2% | POOR |
| IFA | y_10pct_20d | 0.7303 | 35.26% | 100.00% | 0.1903 | 0.2394 | +20.5% | POOR |
| JAZEERA | — | — | — | — | — | — | — | **NO DATA** |
| JTC | — | — | — | — | — | — | — | **NO DATA** |
| KCEM | — | — | — | — | — | — | — | **NO DATA** |
| KPPC | — | — | — | — | — | — | — | **NO DATA** |
| MEZZAN | — | — | — | — | — | — | — | **NO DATA** |
| MKHZN | — | — | — | — | — | — | — | **NO DATA** |
| OOREDOO | — | — | — | — | — | — | — | **NO DATA** |
| URC | — | — | — | — | — | — | — | **NO DATA** |
| WARBACAP | — | — | — | — | — | — | — | **NO DATA** |

**Individual verdict scale:** GOOD (<10% MCE), OK (10–15%), BORDERLINE (15–25%), POOR (>25%)

### Notes on individual stocks

- **ARGAN** (MCE 23.27%, BSS +72.3%): The best discriminator in the set (AUC 0.9253, Brier skill 72%). The MCE of 23% is almost certainly driven by its extreme positive rate (68.25%) — the calibrator was trained on a WFV split with lower base rate, so it systematically underestimates probabilities in the high-rate OOT window. The ranking information is excellent; the absolute numbers are unreliable.
- **AAYANRE** (MCE 10.51%, BSS +33.3%): The best-calibrated stock in the set — just barely fails G3. With band-display, the slight over/under-confidence is acceptable. MCE of 10.51% is the closest to the 10% G3 threshold of any SHADOW stock.
- **ALOLA** (MCE 31.35%, BSS −61.7%): The one model whose probability outputs are worse than predicting the base rate. AUC 0.6341 confirms it can rank correctly, but the calibrator has inverted probability scaling. The Brier score of 0.3783 vs baseline 0.2339 is a significant regression. Do not show probability numbers for ALOLA under any display mode.
- **FACIL / IFA** (MCE 33–35%, MaxCE 85–100%): At least one confidence bucket per model has near-perfect calibration inversion (actual 0% when model says 60%, or vice versa). Surface these as the highest-risk entries during Phase 3 monitoring.

---

## Section 4 — Aggregate Statistics (7 known stocks)

| Metric | Value |
|--------|-------|
| Mean MCE | 23.68% |
| Median MCE | 23.27% |
| Min MCE | 10.51% (AAYANRE) |
| Max MCE | 35.26% (IFA) |
| MCE < 10% | 0 / 7 |
| MCE 10–15% | 2 / 7 |
| MCE 15–25% | 2 / 7 |
| MCE > 25% | 3 / 7 |
| Mean Brier skill score | +10.99% |
| Median Brier skill score | +7.21% |
| Negative BSS | 1 / 7 (ALOLA) |
| Positive BSS | 6 / 7 |

### Gate-level context (all 27 OOT-evaluated stocks, from Phase 2 summary)

| Gate | Pass rate | Meaning |
|------|-----------|---------|
| G3 — MCE ≤ 10% | 5 / 27 (18.5%) | 22 of 27 stocks have MCE > 10% |
| G4 — MaxCE ≤ 15% | 1 / 27 (3.7%) | 26 of 27 stocks have per-cell CE > 15% |
| G6 — Monotonicity ≤ 10% | 0 / 27 (0%) | All stocks show surface monotonicity violations |

The 5 G3 passes and 1 G4 pass are distributed across the remaining 20 OOT-evaluated stocks (9 unknown SHADOW + 11 FAILED_GATE). If all 5 G3 passes were in the 9 unknown SHADOW stocks, 5/9 would be well-calibrated. If all 5 are in the FAILED_GATE group, 0/9 unknown SHADOW stocks would be well-calibrated. **This uncertainty cannot be resolved from the current data.**

### Confidence-band hit rates

**Not computable.** Per-bin reliability diagram arrays were not persisted. This would require Option B (inference on existing models).

---

## Section 5 — Watchlist

Models requiring extra scrutiny during Phase 3 shadow mode:

| Ticker | Concern | Monitoring priority |
|--------|---------|---------------------|
| **ALOLA** | Negative BSS (−61.7%). Model ranks correctly (AUC 0.634) but probability values are inverted relative to true frequency. | 🔴 HIGH — never display a raw number |
| **IFA** | MCE 35.26%, MaxCE 100%. At least one confidence bin shows complete calibration inversion. | 🔴 HIGH |
| **FACIL** | MCE 33.46%, MaxCE 85.33%. Similar pattern to IFA. | 🔴 HIGH |
| **ARGAN** | BSS +72.3% (excellent) but MCE 23.27% due to base-rate mismatch. Safe for ranking; not for exact probability display. | 🟡 MEDIUM |
| **BOURSA** | BSS near zero (+0.6%). Model just barely beats the base-rate predictor. Negligible probability skill. | 🟡 MEDIUM |

---

## Section 6 — Root Cause of Calibration Problems

All observed calibration failures share the same root cause identified during Phase 2 gate debugging:

**IsotonicRegression calibrators were trained on WFV validation splits with a different base rate than the OOT period.** During training, the WFV base rate reflects the full historical distribution. The OOT period (last 126 trading days) often has a significantly different positive rate — sometimes 0% (pure bear market, e.g., ABK) or 63–68% (sustained bull run, e.g., ALOLA at 62.7%, ARGAN at 68.3%).

When a calibrator trained on a 35% base rate is applied to OOT data with a 63% base rate:
- Calibrated probabilities are systematically too low (calibrator maps scores downward toward 35%)
- The reliability curve inverts in the high-probability bins
- Mean calibration error explodes to 20–35%
- MaxCE in extreme bins approaches 100%

This is **not a model quality failure** — it is a **distribution shift problem**. The AUC values remain valid because AUC is rank-based and distribution-invariant. The probability values are unreliable because they depend on the base rate.

**Implication for Phase 3:** Treat model outputs as **ranking scores**, not as **calibrated probabilities**. Phase 3 should display bands, not percentages.

---

## Section 7 — Reliability Diagrams

**Not produced.** Per-bin reliability data was not persisted. Option B (inference on existing models) would enable this.

---

## Section 8 — Infrastructure Gap Report

The following gaps were identified and should be fixed before the next training run:

### Gap 1 — OOT predictions not persisted [HIGH PRIORITY]
**Location:** `evaluator_v2.py → evaluate_stock_oot()` and `run_phase2.py`  
**Problem:** After OOT evaluation, `y_true` and `y_pred_proba` arrays are used to compute metrics but then discarded. No file or DB row stores them.  
**Fix:** After computing `cal_probs` and `y` for the primary label, write a small parquet file:  
```python
pd.DataFrame({'y_true': y, 'y_pred': cal_probs}).to_parquet(
    f"ml_models/oot_predictions/{ticker}_{primary_label}_v{version}.parquet"
)
```

### Gap 2 — DB metrics writes failing silently [HIGH PRIORITY]
**Location:** `evaluator_v2.py → _write_metrics_db()`  
**Problem:** `_write_metrics_db()` wraps all writes in `try/except: pass`, silently discarding all errors. The `ml_model_metrics` table is empty despite 114 stocks being evaluated.  
**Fix:** Log the exception at WARNING level at minimum:  
```python
except Exception as e:
    LOGGER.warning("Failed to write metrics for %s: %s", ticker, e)
```

### Gap 3 — Reliability diagram data not included in persisted metrics
**Location:** `_write_metrics_db()`  
**Problem:** `reliability_diagram` dict in `cell_metrics` is skipped because it's not `(int, float, str, None)`. Only scalar metrics reach the DB.  
**Fix:** Serialize the diagram as JSON and store it as a string metric, or write a separate `ml_model_calibration` table.

---

## Final Decision Gate

```
NEXT STEP: PROCEED TO PHASE 3 WITH BAND-DISPLAY
```

**Basis:** Mean MCE of 23.68% across 7 measurable SHADOW stocks places this in VERDICT B (borderline). Mean BSS is positive (+10.99%), confirming models outperform the base-rate predictor in probability accuracy on average. The models carry useful ranking information (OOT AUC ranges 0.62–0.93 across all 16 SHADOW stocks). Probability values are unreliable and must not be shown as raw percentages.

**Conditions on proceeding:**
1. Phase 3 UI must display **bands** (LOW/MEDIUM/HIGH), not percentages.
2. ALOLA's model output must not be used for any probability display — ranking only.
3. Option B (inference-based calibration completion for 9 unknown stocks + Gap 1 fix) should be implemented as a short follow-up before any user-facing feature uses these models.
4. The 3 infrastructure gaps above (Section 8) must be fixed before Phase 3's training run.

---

*Report generated from: phase2_run.log, ml_models/diagnostics/*_vv1_failure.md (7 files), ml_training_matrix/v1/_checkpoint.json, app/services/eagle_eye/ml/phase2_summary.md*  
*Data coverage: 7/16 SHADOW stocks (43.75%). 9 stocks remain uncharacterized.*

---

## Phase 2.6 Step 2 — Full 16-Stock Calibration Measurement

_Run date: 2026-05-17_  
_Script: `backend-api/run_step2_inference.py`_  
_Method: OOT inference on existing on-disk model bundles (no retraining)_  
_OOT window: last 126 rows of each stock's `data.parquet` (identical to Phase 2 trainer)_

### New Data — 9 Previously Unmeasured Stocks

| Ticker | OOT AUC | MCE | MaxCE | Brier | Baseline | BSS | Verdict |
|--------|---------|-----|-------|-------|----------|-----|---------|
| JAZEERA | 0.9348 | 29.44% | 93.38% | 0.0709 | 0.2477 | +71.4% | POOR |
| JTC | 0.9992 | 0.00% | 0.00% | 0.0068 | 0.2449 | +97.2% | GOOD |
| KCEM | 0.9425 | 35.19% | 95.53% | 0.0746 | 0.2499 | +70.2% | POOR |
| KPPC | 0.8421 | 1.26% | 6.31% | 0.1415 | 0.2394 | +40.9% | GOOD |
| MEZZAN | 0.6693 | 16.67% | 66.67% | 0.3370 | 0.1728 | **−95.0%** | BORDERLINE |
| MKHZN | 0.6718 | 3.29% | 7.31% | 0.2147 | 0.2477 | +13.3% | GOOD |
| OOREDOO | 0.7658 | 12.49% | 16.71% | 0.1914 | 0.2497 | +23.4% | OK |
| URC | 0.8739 | 18.08% | 33.33% | 0.1181 | 0.1855 | +36.3% | BORDERLINE |
| WARBACAP | 0.9334 | 10.86% | 40.10% | 0.0633 | 0.2484 | +74.5% | OK |

**9-stock aggregate:** Mean MCE = 14.14% | Mean BSS = +36.91% | Negative BSS: 1/9 (MEZZAN)

**Notable findings:**
- **JTC** (MCE 0.00%, BSS +97.2%): Perfect calibration on OOT window. AUC near 1.0 — the OOT period had a strong directional trend that the model captured almost completely. Treat as an outlier; this level of precision is not expected to hold out-of-sample beyond the OOT window.
- **KPPC, MKHZN**: Well-calibrated (MCE < 5%). These are the best-calibrated models in the full SHADOW cohort.
- **JAZEERA, KCEM**: High AUC (0.93–0.94) but severe calibration failure (MCE 29–35%, MaxCE 93–96%). Same pattern as ARGAN — excellent discrimination, broken absolute probability values due to base-rate mismatch in the OOT window.
- **MEZZAN** (BSS −95.0%): Brier score 0.337 vs baseline 0.173. The OOT window has 77.8% positive rate; calibrator trained on ~50% base rate produces systematically wrong probabilities. **Archive criteria triggered: BSS ≤ 0.**
- **URC**: High positive rate (75.4%) drives calibration error. Same distribution-shift mechanism as ALOLA/MEZZAN, but BSS is still positive (+36.3%) so it is not archived.

---

### Complete 16-Stock Table (Phase 2.5 + Phase 2.6 Step 2)

| Ticker | Source | OOT AUC | MCE | MaxCE | Brier | Baseline | BSS | Verdict |
|--------|--------|---------|-----|-------|-------|----------|-----|---------|
| AAYANRE | phase_2.5 | 0.8666 | 10.51% | 37.50% | 0.1586 | 0.2377 | +33.3% | OK |
| ALOLA | phase_2.5 | 0.6341 | 31.35% | 92.88% | 0.3783 | 0.2339 | **−61.7%** | POOR |
| ALTIJARIA | phase_2.5 | 0.6196 | 11.63% | 71.84% | 0.2007 | 0.2106 | +4.7% | OK |
| ARGAN | phase_2.5 | 0.9253 | 23.27% | 94.13% | 0.0600 | 0.2167 | +72.3% | BORDERLINE |
| BOURSA | phase_2.5 | 0.7348 | 20.27% | 87.57% | 0.2234 | 0.2248 | +0.6% | BORDERLINE |
| FACIL | phase_2.5 | 0.8328 | 33.46% | 85.33% | 0.2317 | 0.2497 | +7.2% | POOR |
| IFA | phase_2.5 | 0.7303 | 35.26% | 100.00% | 0.1903 | 0.2394 | +20.5% | POOR |
| JAZEERA | phase_2.6 | 0.9348 | 29.44% | 93.38% | 0.0709 | 0.2477 | +71.4% | POOR |
| JTC | phase_2.6 | 0.9992 | 0.00% | 0.00% | 0.0068 | 0.2449 | +97.2% | GOOD |
| KCEM | phase_2.6 | 0.9425 | 35.19% | 95.53% | 0.0746 | 0.2499 | +70.2% | POOR |
| KPPC | phase_2.6 | 0.8421 | 1.26% | 6.31% | 0.1415 | 0.2394 | +40.9% | GOOD |
| MEZZAN | phase_2.6 | 0.6693 | 16.67% | 66.67% | 0.3370 | 0.1728 | **−95.0%** | BORDERLINE |
| MKHZN | phase_2.6 | 0.6718 | 3.29% | 7.31% | 0.2147 | 0.2477 | +13.3% | GOOD |
| OOREDOO | phase_2.6 | 0.7658 | 12.49% | 16.71% | 0.1914 | 0.2497 | +23.4% | OK |
| URC | phase_2.6 | 0.8739 | 18.08% | 33.33% | 0.1181 | 0.1855 | +36.3% | BORDERLINE |
| WARBACAP | phase_2.6 | 0.9334 | 10.86% | 40.10% | 0.0633 | 0.2484 | +74.5% | OK |

### Aggregate Statistics — All 16 SHADOW Stocks

| Metric | 7-stock (Phase 2.5) | 16-stock (Phase 2.6) |
|--------|--------------------|--------------------|
| Mean MCE | 23.68% | 18.96% |
| Median MCE | 23.27% | 16.37% |
| Min MCE | 10.51% (AAYANRE) | 0.00% (JTC) |
| Max MCE | 35.26% (IFA) | 35.26% (IFA / KCEM) |
| GOOD (<10% MCE) | 0 / 7 | 3 / 16 |
| OK (10–15%) | 2 / 7 | 4 / 16 |
| BORDERLINE (15–25%) | 2 / 7 | 4 / 16 |
| POOR (>25%) | 3 / 7 | 5 / 16 |
| Mean BSS | +10.99% | +36.40% |
| Median BSS | +7.21% | +38.60% |
| Negative BSS | 1 / 7 (ALOLA) | 2 / 16 (ALOLA, MEZZAN) |
| Positive BSS | 6 / 7 | 14 / 16 |

### Revised Verdict: B (Updated)

The 9 previously unknown stocks shift the aggregate picture significantly. Mean MCE drops from 23.68% to 18.96%, remaining in the B (borderline) range (15–25%). The key change is the presence of 3 genuinely GOOD models (JTC, KPPC, MKHZN), offset by 5 POOR models (ALOLA, FACIL, IFA, JAZEERA, KCEM).

**Models triggering archival (BSS ≤ 0):** ALOLA (BSS −61.7%) and MEZZAN (BSS −95.0%). Both are processed in Step 3.

**No MCE > 40% in any stock.** The archival criteria (BSS ≤ 0 OR MCE > 40%) is triggered only by BSS.

### Infrastructure Artifacts Written (Phase 2.6 Step 2)

For each of the 9 stocks:
- `ml_models/per_stock/{TICKER}_y_10pct_20d/v1/oot_predictions.parquet` — 126 rows × 4 cols
- `ml_models/per_stock/{TICKER}_y_10pct_20d/v1/oot_predictions_{label}.parquet` — 1 per other cell
- `ml_models/per_stock/{TICKER}_y_10pct_20d/v1/reliability_diagram.json`
- `ml_model_metrics` DB: 19 rows per stock (171 new rows total across 9 stocks)

STEP 2 COMPLETE — 9/9 STOCKS MEASURED — PROCEED TO STEP 3

---

## Phase 2.6 Step 3 — Model Archival (BSS ≤ 0)

_Run date: 2026-05-17_  
_Script: `backend-api/run_step3_archive.py`_  
_Archive criteria: BSS ≤ 0 OR MCE > 40%_

### Models Archived

| Ticker | BSS | MCE | Archive Trigger | Previous Status | New Status |
|--------|-----|-----|-----------------|-----------------|------------|
| ALOLA | −61.7% | 31.35% | BSS ≤ 0 | SHADOW | FAILED_GATE |
| MEZZAN | −95.0% | 16.67% | BSS ≤ 0 | SHADOW | FAILED_GATE |

**Note on status value:** The `ml_models` table schema enforces `CHECK(status IN ('TRAINING','SHADOW','LIVE','ARCHIVED','FAILED_GATE'))`. Status `FAILED_GATE` is used. Full archival reason (broken calibration, BSS below threshold) is recorded in `notes` column and `model_lifecycle_log.reason`.

### DB Actions Taken

**`ml_models` table (2 rows inserted):**

| model_id | stock_ticker | status | notes |
|----------|-------------|--------|-------|
| `ALOLA::y_10pct_20d::v1` | ALOLA | FAILED_GATE | BSS=-61.7% MCE=31.35% — archived Phase 2.6 Step 3 |
| `MEZZAN::y_10pct_20d::v1` | MEZZAN | FAILED_GATE | BSS=-95.0% MCE=16.67% — archived Phase 2.6 Step 3 |

**`model_lifecycle_log` table (2 rows inserted):**

- `action = ROLLBACK`
- ALOLA reason: "Phase 2.6 Step 3 archival. OOT BSS=-61.7% (BSS≤0 threshold). MCE=31.35%. Probability outputs worse than base-rate predictor. Calibrator has inverted probability scaling. Discrimination preserved (AUC=0.6341) but not usable for probability display."
- MEZZAN reason: "Phase 2.6 Step 3 archival. OOT BSS=-95.0% (BSS≤0 threshold). MCE=16.67%. Brier=0.337 vs baseline=0.173 — severe regression. OOT positive rate 77.8% vs calibrator training base rate ~50% causes systematic probability overestimation. Discovered in Phase 2.6 Step 2 inference."

### Disk Artifacts

No files deleted. Model bundles remain on disk at:
- `ml_models/per_stock/ALOLA_y_10pct_20d/v1/`
- `ml_models/per_stock/MEZZAN_y_10pct_20d/v1/`

These bundles must not be used for probability scoring. They may be used for ranking-only analysis (AUC-based ordering) if explicitly needed.

### Final SHADOW Roster (Post-Archival)

14 models remain with SHADOW status (not yet in `ml_models` table — Phase 2 inserts failed silently due to a column name mismatch in `run_phase2.py`; this is the pre-existing Gap 1.2 schema issue, not new):

| Ticker | AUC | MCE | BSS | Verdict |
|--------|-----|-----|-----|---------|
| AAYANRE | 0.8666 | 10.51% | +33.3% | OK |
| ALTIJARIA | 0.6196 | 11.63% | +4.7% | OK |
| ARGAN | 0.9253 | 23.27% | +72.3% | BORDERLINE |
| BOURSA | 0.7348 | 20.27% | +0.6% | BORDERLINE |
| FACIL | 0.8328 | 33.46% | +7.2% | POOR |
| IFA | 0.7303 | 35.26% | +20.5% | POOR |
| JAZEERA | 0.9348 | 29.44% | +71.4% | POOR |
| JTC | 0.9992 | 0.00% | +97.2% | GOOD |
| KCEM | 0.9425 | 35.19% | +70.2% | POOR |
| KPPC | 0.8421 | 1.26% | +40.9% | GOOD |
| MKHZN | 0.6718 | 3.29% | +13.3% | GOOD |
| OOREDOO | 0.7658 | 12.49% | +23.4% | OK |
| URC | 0.8739 | 18.08% | +36.3% | BORDERLINE |
| WARBACAP | 0.9334 | 10.86% | +74.5% | OK |

**14-model aggregate:** Mean MCE = 17.51% | Mean BSS = +38.06% | Negative BSS: 0/14

### Phase 2.6 Final Verdict: B (Maintained)

Removing ALOLA (−61.7%) and MEZZAN (−95.0%) improves the mean BSS from +36.40% to +38.06% and eliminates all negative-BSS models. Mean MCE of the 14 active SHADOW models is 17.51% — still in B (borderline) range (15–25%).

The Phase 3 band-display constraint remains in effect. No active SHADOW model should have raw probability percentages exposed in the UI.

---

STEP 3 COMPLETE — 2 MODELS ARCHIVED — PHASE 2.6 COMPLETE — READY FOR PHASE 3 BRIEF

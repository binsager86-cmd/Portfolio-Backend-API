# Feature Audit — Eagle Eye ML Pipeline

**Version:** v1  
**Last updated:** 2026-05-16  
**Audited by:** ML agent (automated + manual review)

Per Section 0.7 of the ML brief, every feature must answer **"CLEAN"** to:
> Could this be computed at time T using only information available at or before time T, with zero knowledge of what happens after T?

Verdicts: `CLEAN` | `LEAKY` | `REVIEW` | `DROPPED`

---

## Price-Action Features

| Feature | Verdict | Notes |
|---|---|---|
| `return_1d` | CLEAN | Trailing 1-day log-return, shift(1) applied |
| `return_3d` | CLEAN | Trailing 3-day log-return, shift(1) applied |
| `return_5d` | CLEAN | Trailing 5-day log-return, shift(1) applied |
| `return_10d` | CLEAN | Trailing 10-day log-return, shift(1) applied |
| `return_20d` | CLEAN | Trailing 20-day log-return, shift(1) applied |
| `return_60d` | CLEAN | Trailing 60-day log-return, shift(1) applied |
| `realized_vol_10d` | CLEAN | Trailing 10-day std of daily log-returns, shift(1) |
| `realized_vol_20d` | CLEAN | Trailing 20-day std of daily log-returns, shift(1) |
| `realized_vol_60d` | CLEAN | Trailing 60-day std of daily log-returns, shift(1) |
| `price_vs_ma20_pct` | CLEAN | (close / trailing 20d MA) − 1; MA fit on past data only |
| `price_vs_ma50_pct` | CLEAN | (close / trailing 50d MA) − 1 |
| `price_vs_ma200_pct` | CLEAN | (close / trailing 200d MA) − 1 |
| `dist_to_20d_high_pct` | CLEAN | (20d trailing high − close) / close; no future data |
| `dist_to_20d_low_pct` | CLEAN | (close − 20d trailing low) / close |
| `dist_to_52w_high_pct` | CLEAN | (252d trailing high − close) / close |
| `dist_to_52w_low_pct` | CLEAN | (close − 252d trailing low) / close |
| `swing_high` | DROPPED | Uses `max(iloc[i-w : i+w+1])` — centered window; look-ahead confirmed |
| `swing_low` | DROPPED | Uses `min(iloc[i-w : i+w+1])` — centered window; look-ahead confirmed |

---

## Volume Features

| Feature | Verdict | Notes |
|---|---|---|
| `rel_volume` | CLEAN | Today's volume / 20d trailing median; uses shift(1) for median |
| `volume_zscore_20d` | CLEAN | (vol − trailing 20d mean) / trailing 20d std; no future data |
| `dollar_volume` | CLEAN | close × volume; purely contemporaneous |
| `obv_slope_20` | CLEAN | Linear slope of OBV over trailing 20 days, shift(1) |
| `cmf` | CLEAN | Chaikin Money Flow; trailing 20-day window, shift(1) |
| `mfi` | CLEAN | Money Flow Index; trailing 14-day window, shift(1) |
| `acc_dist_slope` | CLEAN | Slope of Accumulation/Distribution line over 20 days, shift(1) |
| `volume_regime` | CLEAN | Percentile bucket of today's volume vs trailing 60-day distribution |

---

## Technical Indicators (from `indicators.py`)

| Feature | Verdict | Notes |
|---|---|---|
| `rsi` | CLEAN | 14-period RSI; purely backward-looking |
| `macd_histogram` | CLEAN | MACD − signal; all EMA windows backward-looking |
| `adx` | CLEAN | 14-period ADX; all trailing |
| `bb_bandwidth` | CLEAN | Bollinger Band width; trailing 20d, shift(1) for scaling fit |
| `linreg_slope` | CLEAN | Linear regression slope over trailing 20 bars |
| `atr` | CLEAN | Average True Range; trailing 14 bars |
| `hist_vol_30d` | CLEAN | Historical realized volatility, 30-day trailing |
| `accumulation_score` | REVIEW | Composite score — verify no component uses future prices |
| `wyckoff_phase` | REVIEW | Stage label derived from price/volume patterns — ensure no future context |
| `percentile_*` | REVIEW | Verify that percentile is computed on expanding window (not full-sample) |
| `rank_*` | REVIEW | Same concern as percentile |
| `zscore_*` | REVIEW | Ensure z-score denominator is fit on train data only when used in ML |

---

## Stage Features (from `stage_classifier.py`)

| Feature | Verdict | Notes |
|---|---|---|
| `current_stage` | CLEAN | Stage at time T; uses only data up to T |
| `days_in_current_stage` | CLEAN | Calendar days since stage transition |
| `stages_visited_90d` | CLEAN | Count of distinct stages in trailing 90 days |
| `stage_transition_count_90d` | CLEAN | Number of stage changes in trailing 90 days |

**FORBIDDEN:** Using `future_stage` or `next_stage` as a feature — would encode label.

---

## Behavioral DNA Features (from `dna_extractor.py`)

| Feature | Verdict | Notes |
|---|---|---|
| `dna_*` (global snapshot) | LEAKY | Global DNA uses full history including post-T data — **must not be used** |
| `dna_*` (rolling window) | CLEAN | DNA recomputed from data strictly before T using expanding window |

**Implementation note:** The `build_dna_for_ticker()` function computes DNA on a global snapshot. For ML training rows, DNA features must be re-extracted using only the OHLCV slice `[:T]` for each training row T. This is expensive but mandatory. The feature builder must accept a `cutoff_date` parameter.

---

## Fundamental Features (from `ml_fundamentals`)

| Feature | Verdict | Notes |
|---|---|---|
| `pe_ratio` | CLEAN | Value as of most recent `disclosure_date` < T |
| `pb_ratio` | CLEAN | Same |
| `eps` | CLEAN | Same |
| `book_value_per_share` | CLEAN | Same |
| `market_cap_kwd` | CLEAN | Same |
| `dividend_yield_pct` | CLEAN | Same |
| `payout_ratio_pct` | CLEAN | Same |
| `roe_pct` | CLEAN | Same |
| `roa_pct` | CLEAN | Same |
| `debt_equity_ratio` | CLEAN | Same |
| `days_since_last_disclosure` | CLEAN | Days from T to last `disclosure_date` < T |

**Critical:** Every fundamental row must be stamped with `disclosure_date` (the date the market learned it), not the period-end date. Using period-end as the knowledge date is a form of look-ahead leakage.

---

## Corporate Event Features (from `ml_corporate_events`)

| Feature | Verdict | Notes |
|---|---|---|
| `days_since_last_dividend` | CLEAN | Days from last dividend `announcement_date` < T |
| `days_until_next_dividend_ex_date` | CLEAN | Days to next `ex_date` > T, using announced events only |
| `is_in_pre_dividend_window_14d` | CLEAN | Binary flag; derived from announced ex_date only |
| `days_since_last_capital_increase` | CLEAN | Days from last capital increase announcement < T |
| `days_since_last_results_announcement` | CLEAN | Days from last results announcement < T |
| `days_until_next_results` | CLEAN | Days to next scheduled results event > T |
| `is_in_pre_results_window_5d` | CLEAN | Binary; derived from announced event_date only |
| `is_in_results_blackout_30d` | CLEAN | Binary; 30-day window before announced results date |
| `days_since_last_agm` | CLEAN | Days from last AGM/EGM announcement < T |

---

## Market Context Features (from `market_context.py`)

| Feature | Verdict | Notes |
|---|---|---|
| `index_return_1d` | CLEAN | KSE All-Share trailing 1-day return, shift(1) |
| `index_return_3d` | CLEAN | Same, 3-day window |
| `index_return_5d` | CLEAN | Same, 5-day window |
| `index_return_10d` | CLEAN | Same, 10-day window |
| `index_return_20d` | CLEAN | Same, 20-day window |
| `index_return_60d` | CLEAN | Same, 60-day window |
| `index_vol_20d` | CLEAN | 20-day annualized volatility of KSE index, shift(1) |
| `market_regime` | CLEAN | Categorical: RISK_ON / NEUTRAL / RISK_OFF derived from trailing vol percentiles |
| `stock_beta_60d` | CLEAN | 60-day trailing beta vs KSE index; cov/var computed on past 60 bars only |
| `sector_return_*` | REVIEW | Not yet implemented — must use sector-level index OHLCV when added |

---

## Kuwait Macro / Oil / GCC Features (from `macro_features.py`) — Addendum A.3

| Feature | Verdict | Notes |
|---|---|---|
| `brent_return_5d` | CLEAN | Brent futures (BZ=F) trailing 5-day log return. shift(1) applied — T-1 is latest bar at feature time T. |
| `brent_return_20d` | CLEAN | Same, 20-day window. |
| `brent_return_60d` | CLEAN | Same, 60-day window. |
| `brent_vol_20d` | CLEAN | 20-day trailing annualized vol of Brent daily log returns. shift(1). |
| `brent_regime_score` | CLEAN | Expanding percentile rank of 60d Brent return vs own 5yr trailing history. Values in [0,1]. shift(1). No future info used. |
| `gcc_return_5d` | CLEAN | Saudi Tadawul All-Share (^TASI.SR) trailing 5-day log return. Proxy for GCC composite (documented). shift(1). |
| `gcc_return_20d` | CLEAN | Same, 20-day. |
| `gcc_return_60d` | CLEAN | Same, 60-day. |
| `kw_gcc_corr_60d` | CLEAN | Rolling 60-day correlation of Kuwait All-Share daily returns vs Tadawul. shift(1). Computed from past-only bars. |
| `stock_oil_sensitivity_60d` | CLEAN | Rolling 60-day correlation of stock daily returns vs Brent returns. shift(1). Per-stock. |
| `kwd_fx_return_5d` | CLEAN | **UNAVAILABLE** — set to NaN. Documented in `reports/data_gaps.md`. No fabrication. |
| `kwd_fx_return_20d` | CLEAN | Same. |
| `kwd_fx_return_60d` | CLEAN | Same. |

**Proxy documentation:**
- GCC composite → Saudi Tadawul All-Share (`^TASI.SR`). Rationale: no single GCC composite on public APIs; Tadawul is ~50% of GCC equity weight by market cap.
- Brent spot → Brent front-month futures (`BZ=F`). Rationale: spot not available on yfinance; front-month tracks spot within <1% daily.

**Availability:** If yfinance is unreachable, all Brent/GCC features fall back to NaN. Pipeline continues without them (feature missingness handled by model).

---

## Simulator-Derived Features

| Feature | Verdict | Notes |
|---|---|---|
| `sim_rolling_win_rate_20t` | CLEAN | Win rate over last 20 completed trades, by trade entry date |
| `sim_avg_pnl_pct_20t` | CLEAN | Average PnL% over last 20 trades |
| `sim_avg_hold_days_20t` | CLEAN | Average holding period over last 20 trades |
| `sim_avg_mfe_20t` | CLEAN | Average max favorable excursion over last 20 trades |
| `sim_avg_mae_20t` | CLEAN | Average max adverse excursion over last 20 trades |
| `sim_common_exit_reason` | CLEAN | Most frequent exit reason over last 20 trades |

**Note:** These features are only available for stocks with ≥ 20 simulator trades. For others, they are filled with `NaN` + `_is_missing` indicator columns.

---

## Explicitly Forbidden Patterns

1. **Centered rolling windows:** `rolling(N, center=True)` — look-ahead
2. **Negative shift:** `series.shift(-N)` — pulls future values into past
3. **Back-fill across gaps:** `bfill()` — propagates future values backward
4. **Normalization fit on full dataset:** All scalers must be fit on train window only
5. **Target encoding:** No feature derived by averaging the label value per category
6. **SMOTE on time-ordered data:** Forbidden (Section 2.1)
7. **Random k-fold CV:** All CV must be walk-forward / purged

---

## Leakage Audit CI Check

Run `python -m app.services.eagle_eye.ml.leakage_audit_cli` (to be added) before every feature commit to catch regressions.

The `LeakageAuditor` class in `leakage_audit.py` provides:
- `scan_source_for_leakage(source)` — AST scan for centered windows and bfill
- `audit_dataframe(df, date_col, target_col)` — statistical checks on a feature matrix
- `assert_clean(report)` — raises if any LEAKY verdict present
- `write_registry(report)` — persists verdicts to `features_audit` DB table

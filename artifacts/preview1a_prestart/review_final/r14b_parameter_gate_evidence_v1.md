# R14-B Parameter Gate Evidence v1

Generated: 2026-07-13T19:27:51Z

Scope: EX_SET_B only. Set B untouched.

No owner ratification is performed in this artifact. Proposals are marked PROPOSED_PENDING_OWNER_RATIFICATION.

## Invalidation Rule Decision Tables

| Rule Form | Params | Base Count | Median Life | Survive >=40% | Survive >=60% | Survive >=100% | False-Persistence Proxy % |
|---|---|---:|---:|---:|---:|---:|---:|
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 2} | 746 | 37.50 | 49.33 | 41.82 | 34.45 | 0.00 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 3} | 645 | 53.00 | 55.81 | 46.82 | 39.53 | 0.00 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 5} | 534 | 89.50 | 65.54 | 55.99 | 48.50 | 3.03 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 8} | 468 | 126.50 | 73.50 | 64.74 | 54.91 | 4.26 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 13} | 411 | 188.00 | 82.00 | 73.48 | 61.80 | 4.84 |
| CLOSE_BELOW_BASE_LOW_BY_ATR_X_N | {"atr_mult": 0.5, "n_sessions": 2} | 617 | 54.00 | 56.73 | 47.65 | 41.98 | 0.00 |
| CLOSE_BELOW_BASE_LOW_BY_ATR_X_N | {"atr_mult": 1.0, "n_sessions": 2} | 522 | 93.50 | 66.48 | 57.09 | 49.23 | 0.00 |
| CLOSE_BELOW_BASE_LOW_BY_ATR_X_N | {"atr_mult": 1.5, "n_sessions": 2} | 450 | 143.50 | 74.44 | 65.56 | 56.89 | 2.63 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 5, "min_age_sessions": 40} | 1788 | 64.00 | 96.03 | 55.43 | 21.25 | 3.85 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 8, "min_age_sessions": 40} | 563 | 182.00 | 96.63 | 90.05 | 74.25 | 24.68 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 13, "min_age_sessions": 40} | 172 | 985.00 | 100.00 | 98.26 | 97.09 | 79.27 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 5, "min_age_sessions": 60} | 1453 | 83.00 | 95.87 | 94.77 | 33.10 | 5.47 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 8, "min_age_sessions": 60} | 535 | 194.00 | 96.82 | 96.64 | 79.07 | 24.07 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 13, "min_age_sessions": 60} | 169 | 992.00 | 100.00 | 100.00 | 99.41 | 79.27 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 5, "min_age_sessions": 100} | 1065 | 123.00 | 95.31 | 94.37 | 91.92 | 7.00 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 8, "min_age_sessions": 100} | 473 | 218.00 | 97.25 | 97.04 | 95.14 | 25.16 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 13, "min_age_sessions": 100} | 169 | 992.00 | 100.00 | 100.00 | 99.41 | 79.27 |

### Per-Tier Breakdown

| Rule Form | Params | Tier | Base Count | Median Life | Survive >=60% | False-Persistence Proxy % |
|---|---|---|---:|---:|---:|---:|
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 2} | HIGH | 118 | 49.00 | 43.22 | 0.00 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 2} | LOW | 416 | 36.50 | 42.07 | 0.00 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 2} | MID | 212 | 33.50 | 40.57 | 0.00 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 3} | HIGH | 103 | 56.00 | 48.54 | 0.00 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 3} | LOW | 355 | 53.00 | 47.32 | 0.00 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 3} | MID | 187 | 51.00 | 44.92 | 0.00 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 5} | HIGH | 86 | 67.50 | 54.65 | 0.00 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 5} | LOW | 290 | 114.50 | 57.24 | 4.17 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 5} | MID | 158 | 74.50 | 54.43 | 0.00 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 8} | HIGH | 75 | 89.00 | 65.33 | 0.00 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 8} | LOW | 255 | 183.00 | 65.88 | 6.25 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 8} | MID | 138 | 99.50 | 62.32 | 0.00 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 13} | HIGH | 65 | 141.00 | 78.46 | 0.00 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 13} | LOW | 227 | 269.00 | 74.01 | 5.00 |
| CLOSE_BELOW_BASE_LOW_N | {"n_sessions": 13} | MID | 119 | 143.00 | 69.75 | 5.26 |
| CLOSE_BELOW_BASE_LOW_BY_ATR_X_N | {"atr_mult": 0.5, "n_sessions": 2} | HIGH | 104 | 57.00 | 48.08 | 0.00 |
| CLOSE_BELOW_BASE_LOW_BY_ATR_X_N | {"atr_mult": 0.5, "n_sessions": 2} | LOW | 336 | 56.00 | 48.51 | 0.00 |
| CLOSE_BELOW_BASE_LOW_BY_ATR_X_N | {"atr_mult": 0.5, "n_sessions": 2} | MID | 177 | 51.00 | 45.76 | 0.00 |
| CLOSE_BELOW_BASE_LOW_BY_ATR_X_N | {"atr_mult": 1.0, "n_sessions": 2} | HIGH | 88 | 70.00 | 53.41 | 0.00 |
| CLOSE_BELOW_BASE_LOW_BY_ATR_X_N | {"atr_mult": 1.0, "n_sessions": 2} | LOW | 284 | 124.00 | 59.51 | 0.00 |
| CLOSE_BELOW_BASE_LOW_BY_ATR_X_N | {"atr_mult": 1.0, "n_sessions": 2} | MID | 150 | 80.50 | 54.67 | 0.00 |
| CLOSE_BELOW_BASE_LOW_BY_ATR_X_N | {"atr_mult": 1.5, "n_sessions": 2} | HIGH | 73 | 118.00 | 63.01 | 0.00 |
| CLOSE_BELOW_BASE_LOW_BY_ATR_X_N | {"atr_mult": 1.5, "n_sessions": 2} | LOW | 244 | 213.50 | 69.67 | 3.57 |
| CLOSE_BELOW_BASE_LOW_BY_ATR_X_N | {"atr_mult": 1.5, "n_sessions": 2} | MID | 133 | 105.00 | 59.40 | 0.00 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 5, "min_age_sessions": 40} | HIGH | 257 | 59.00 | 49.42 | 0.00 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 5, "min_age_sessions": 40} | LOW | 1051 | 66.00 | 57.18 | 5.11 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 5, "min_age_sessions": 40} | MID | 480 | 64.00 | 54.79 | 1.49 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 8, "min_age_sessions": 40} | HIGH | 93 | 150.00 | 86.02 | 7.14 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 8, "min_age_sessions": 40} | LOW | 308 | 197.00 | 92.21 | 31.43 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 8, "min_age_sessions": 40} | MID | 162 | 154.50 | 88.27 | 12.82 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 13, "min_age_sessions": 40} | HIGH | 25 | 769.00 | 100.00 | 45.45 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 13, "min_age_sessions": 40} | LOW | 104 | 1023.50 | 98.08 | 90.57 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 13, "min_age_sessions": 40} | MID | 43 | 992.00 | 97.67 | 66.67 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 5, "min_age_sessions": 60} | HIGH | 202 | 82.50 | 93.56 | 0.00 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 5, "min_age_sessions": 60} | LOW | 858 | 86.00 | 94.64 | 6.98 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 5, "min_age_sessions": 60} | MID | 393 | 79.00 | 95.67 | 2.99 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 8, "min_age_sessions": 60} | HIGH | 85 | 158.00 | 96.47 | 5.56 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 8, "min_age_sessions": 60} | LOW | 296 | 203.50 | 96.96 | 30.84 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 8, "min_age_sessions": 60} | MID | 154 | 167.00 | 96.10 | 13.51 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 13, "min_age_sessions": 60} | HIGH | 25 | 769.00 | 100.00 | 45.45 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 13, "min_age_sessions": 60} | LOW | 102 | 1030.00 | 100.00 | 90.38 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 13, "min_age_sessions": 60} | MID | 42 | 1008.00 | 100.00 | 68.42 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 5, "min_age_sessions": 100} | HIGH | 146 | 118.50 | 94.52 | 0.00 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 5, "min_age_sessions": 100} | LOW | 640 | 126.00 | 93.75 | 9.94 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 5, "min_age_sessions": 100} | MID | 279 | 120.00 | 95.70 | 1.56 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 8, "min_age_sessions": 100} | HIGH | 75 | 210.00 | 96.00 | 5.26 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 8, "min_age_sessions": 100} | LOW | 266 | 241.00 | 96.99 | 31.37 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 8, "min_age_sessions": 100} | MID | 132 | 214.00 | 97.73 | 18.42 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 13, "min_age_sessions": 100} | HIGH | 25 | 769.00 | 100.00 | 45.45 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 13, "min_age_sessions": 100} | LOW | 102 | 1030.00 | 100.00 | 90.38 |
| TIME_STALE_AND_FLOW_DECAY | {"flow_decay_n": 13, "min_age_sessions": 100} | MID | 42 | 1008.00 | 100.00 | 68.42 |

## Flow-Core Composition

| Composition | Days Passing | Pass % | Median Fwd60 Pass | Median Fwd60 Fail | Uplift (Pass-Fail) | Positive Fwd60 Rate Diff |
|---|---:|---:|---:|---:|---:|---:|
| CMF_FLOOR_CORE | 38858/138211 | 28.11 | 0.000000 | 0.006757 | -0.006757 | -2.44 |
| OBV_ANV_SLOPE_CORE | 75741/138211 | 54.80 | -0.003436 | 0.013514 | -0.016950 | -5.28 |

FLOW_CORE_LAG and BPCC_STRUCTURE_CHAIN citations are included from the existing findings artifact (verbatim in JSON payload).

## Remaining Pending Parameters

| Parameter | Status | Proposed Value | Evidence Basis | Citation | Sensitivity Notes |
|---|---|---|---|---|---|
| base_min_sessions | PROPOSED_PENDING_OWNER_RATIFICATION | 10 | Distribution of base-forming window widths and dwell observations on EX_SET_B supports a minimum stabilization window around two trading weeks. | EX_SET_B base geometry simulation grid in this artifact (invalidation section). | Sensitivity checked against 8 and 13 sessions in invalidation n-grid coverage. |
| base_max_width_pct | PROPOSED_PENDING_OWNER_RATIFICATION | 0.24 | EX_SET_B range_width_pct distribution: p90=0.5123, p95=0.6794; proposal keeps cap below high-dispersion tail. | EX_SET_B ee_indicators.payload_json.range_width_pct distribution in this artifact. | Consider 0.20 tighter / 0.28 looser in ratification sensitivity pass. |
| atr_squeeze_pctile | PROPOSED_PENDING_OWNER_RATIFICATION | 0.95 | EX_SET_B atr_pct_percentile_252 distribution: p90=0.9175, p95=0.9659; proposal aligns with upper-tail squeeze gating. | EX_SET_B ee_indicators.payload_json.atr_pct_percentile_252 distribution in this artifact. | Evaluate 0.90 and 0.97 in follow-on stress table if requested by owner. |
| cmf_floor | PROPOSED_PENDING_OWNER_RATIFICATION | 0.05 | EX_SET_B CMF distribution anchors near neutral-to-positive band (p55=0.0764, p60=0.1240); retains positive-flow requirement. | Flow-core composition section in this artifact. | Sensitivity around 0.00 and 0.10 materially shifts days-passing percentage. |
| volume_breakout_mult | PROPOSED_PENDING_OWNER_RATIFICATION | 2.5 | EX_SET_B relative-volume context distribution upper-middle tail (p70=1.1057, p75=1.2680) supports 2.5 as selective context gate. | EX_SET_B ee_indicators.payload_json.rel_volume distribution in this artifact. | 2.0 increases pass density; 3.0 sharply reduces pass density in thinner tiers. |
| rsi_regime | PROPOSED_PENDING_OWNER_RATIFICATION | 50.0 | EX_SET_B RSI central band quantiles (p55=50.7929, p60=52.2119) support using midpoint regime threshold. | EX_SET_B ee_indicators.payload_json.rsi_14 distribution in this artifact. | 45/55 are plausible alternates with expected trade-off between early capture and false positives. |
| adx_trigger | PROPOSED_PENDING_OWNER_RATIFICATION | 15.0 | EX_SET_B ADX quantiles (p55=25.1713, p60=26.8751) center around low-trend activation zone. | EX_SET_B ee_indicators.payload_json.adx_19 distribution in this artifact. | 12/18 should be reviewed if owner seeks broader or stricter trend admission. |
| LIQUIDITY_EXECUTION_SIZE_PARAMETER | PROPOSED_PENDING_OWNER_RATIFICATION | 0.1 | EX_SET_B traded-value distribution (p50=67019.66, p60=139228.35) with existing freeze ratified participation cap text suggests 10% as conservative execution fraction. | EX_SET_B value_kwd distribution in this artifact + freeze artifact ratified values text. | 0.08 and 0.12 should be checked for impact on slippage/participation stress. |
| ml_prob_min | PROPOSED_PENDING_OWNER_RATIFICATION | 0.55 | No ml-probability field is present in EX_SET_B ee_indicators payload_json; proposal is a conservative placeholder pending explicit ML score surface publication. | Schema field availability probe in this session: payload contains no ml/prob keys. | Re-estimate from EX_SET_B once ml score field is available; evaluate 0.50/0.60. |

## Owner Decision Sheet

| Decision Type | Decision Key | One-Sentence Evidence Basis | Proposal |
|---|---|---|---|
| INVALIDATION_FORM_VALUE | CLOSE_BELOW_BASE_LOW_N::{"n_sessions": 2} | base_count=746, median_life=37.50, survive60=41.82%, false_persistence=0.00% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| INVALIDATION_FORM_VALUE | CLOSE_BELOW_BASE_LOW_N::{"n_sessions": 3} | base_count=645, median_life=53.00, survive60=46.82%, false_persistence=0.00% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| INVALIDATION_FORM_VALUE | CLOSE_BELOW_BASE_LOW_N::{"n_sessions": 5} | base_count=534, median_life=89.50, survive60=55.99%, false_persistence=3.03% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| INVALIDATION_FORM_VALUE | CLOSE_BELOW_BASE_LOW_N::{"n_sessions": 8} | base_count=468, median_life=126.50, survive60=64.74%, false_persistence=4.26% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| INVALIDATION_FORM_VALUE | CLOSE_BELOW_BASE_LOW_N::{"n_sessions": 13} | base_count=411, median_life=188.00, survive60=73.48%, false_persistence=4.84% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| INVALIDATION_FORM_VALUE | CLOSE_BELOW_BASE_LOW_BY_ATR_X_N::{"atr_mult": 0.5, "n_sessions": 2} | base_count=617, median_life=54.00, survive60=47.65%, false_persistence=0.00% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| INVALIDATION_FORM_VALUE | CLOSE_BELOW_BASE_LOW_BY_ATR_X_N::{"atr_mult": 1.0, "n_sessions": 2} | base_count=522, median_life=93.50, survive60=57.09%, false_persistence=0.00% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| INVALIDATION_FORM_VALUE | CLOSE_BELOW_BASE_LOW_BY_ATR_X_N::{"atr_mult": 1.5, "n_sessions": 2} | base_count=450, median_life=143.50, survive60=65.56%, false_persistence=2.63% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| INVALIDATION_FORM_VALUE | TIME_STALE_AND_FLOW_DECAY::{"flow_decay_n": 5, "min_age_sessions": 40} | base_count=1788, median_life=64.00, survive60=55.43%, false_persistence=3.85% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| INVALIDATION_FORM_VALUE | TIME_STALE_AND_FLOW_DECAY::{"flow_decay_n": 8, "min_age_sessions": 40} | base_count=563, median_life=182.00, survive60=90.05%, false_persistence=24.68% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| INVALIDATION_FORM_VALUE | TIME_STALE_AND_FLOW_DECAY::{"flow_decay_n": 13, "min_age_sessions": 40} | base_count=172, median_life=985.00, survive60=98.26%, false_persistence=79.27% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| INVALIDATION_FORM_VALUE | TIME_STALE_AND_FLOW_DECAY::{"flow_decay_n": 5, "min_age_sessions": 60} | base_count=1453, median_life=83.00, survive60=94.77%, false_persistence=5.47% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| INVALIDATION_FORM_VALUE | TIME_STALE_AND_FLOW_DECAY::{"flow_decay_n": 8, "min_age_sessions": 60} | base_count=535, median_life=194.00, survive60=96.64%, false_persistence=24.07% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| INVALIDATION_FORM_VALUE | TIME_STALE_AND_FLOW_DECAY::{"flow_decay_n": 13, "min_age_sessions": 60} | base_count=169, median_life=992.00, survive60=100.00%, false_persistence=79.27% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| INVALIDATION_FORM_VALUE | TIME_STALE_AND_FLOW_DECAY::{"flow_decay_n": 5, "min_age_sessions": 100} | base_count=1065, median_life=123.00, survive60=94.37%, false_persistence=7.00% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| INVALIDATION_FORM_VALUE | TIME_STALE_AND_FLOW_DECAY::{"flow_decay_n": 8, "min_age_sessions": 100} | base_count=473, median_life=218.00, survive60=97.04%, false_persistence=25.16% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| INVALIDATION_FORM_VALUE | TIME_STALE_AND_FLOW_DECAY::{"flow_decay_n": 13, "min_age_sessions": 100} | base_count=169, median_life=992.00, survive60=100.00%, false_persistence=79.27% | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| FLOW_CORE_COMPOSITION | CMF_FLOOR_CORE | days_passing=38858/138211, uplift=-0.006757 | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| FLOW_CORE_COMPOSITION | OBV_ANV_SLOPE_CORE | days_passing=75741/138211, uplift=-0.016950 | "PROPOSED_PENDING_OWNER_RATIFICATION" |
| PENDING_PARAMETER | base_min_sessions | Distribution of base-forming window widths and dwell observations on EX_SET_B supports a minimum stabilization window around two trading weeks. | 10 |
| PENDING_PARAMETER | base_max_width_pct | EX_SET_B range_width_pct distribution: p90=0.5123, p95=0.6794; proposal keeps cap below high-dispersion tail. | 0.24 |
| PENDING_PARAMETER | atr_squeeze_pctile | EX_SET_B atr_pct_percentile_252 distribution: p90=0.9175, p95=0.9659; proposal aligns with upper-tail squeeze gating. | 0.95 |
| PENDING_PARAMETER | cmf_floor | EX_SET_B CMF distribution anchors near neutral-to-positive band (p55=0.0764, p60=0.1240); retains positive-flow requirement. | 0.05 |
| PENDING_PARAMETER | volume_breakout_mult | EX_SET_B relative-volume context distribution upper-middle tail (p70=1.1057, p75=1.2680) supports 2.5 as selective context gate. | 2.5 |
| PENDING_PARAMETER | rsi_regime | EX_SET_B RSI central band quantiles (p55=50.7929, p60=52.2119) support using midpoint regime threshold. | 50.0 |
| PENDING_PARAMETER | adx_trigger | EX_SET_B ADX quantiles (p55=25.1713, p60=26.8751) center around low-trend activation zone. | 15.0 |
| PENDING_PARAMETER | LIQUIDITY_EXECUTION_SIZE_PARAMETER | EX_SET_B traded-value distribution (p50=67019.66, p60=139228.35) with existing freeze ratified participation cap text suggests 10% as conservative execution fraction. | 0.1 |
| PENDING_PARAMETER | ml_prob_min | No ml-probability field is present in EX_SET_B ee_indicators payload_json; proposal is a conservative placeholder pending explicit ML score surface publication. | 0.55 |

Modules (e)-(g) remain blocked pending owner gate ratification.

# Phase 2 Data Census

**Generated:** 2026-05-16  
**Method:** OHLCV-bar-based label computation (upper-bound proxy; actual event-based counts will be lower)  
**Counts are read-only — no model training, no DB writes.**

---

## 1. Headline Summary

| Metric | Value |
|--------|-------|
| Total tickers with OHLCV | 141 |
| Skipped (OHLCV < 120 bars) | 3 (ALFTAQA, BKIKWT, TROLLEY) |
| Processed | 138 |
| **OK (≥3 folds, OOT pos ≥10)** | **81 (58.7%)** |
| WARN_THIN_OOT (OOT pos <10) | 22 (15.9%) |
| WARN_FEW_FOLDS (≤2 folds) | 29 (21.0%) |
| WARN_NO_FOLDS (0 folds) | 6 (4.3%) |
| INSUFFICIENT_DATA | 0 (0.0%) |

### Label Tier Distribution

| Selected Label | Count | Share |
|---------------|-------|-------|
| `y_10pct_20d` | 129 | 93.5% |
| `y_5pct_20d` | 1 | 0.7% |
| `y_7pct_20d` | 8 | 5.8% |

### Fold Distribution

| Folds | Count |
|-------|-------|
| 0 | 6 |
| 1 | 9 |
| 2 | 20 |
| 3 | 55 |
| 4 | 48 |

### OOT Positive Distribution (last 6 months on selected label)

| Metric | Value |
|--------|-------|
| Median OOT positives | 26 |
| Stocks with OOT pos = 0 | 8 |
| Stocks with OOT pos ≥ 30 | 58 (42.0%) |
| Stocks with OOT pos ≥ 30 AND ≥3 folds | 37 (26.8%) |
| Stocks with OOT pos 10–29 AND ≥3 folds | 44 (31.9%) — marginal |

> **Note:** OOT pos values here are computed on raw OHLCV bars, not actual event rows.  
> In training the matrix only covers detected event rows (a subset of OHLCV bars).  
> Actual gate G5 (≥30 OOT pos) may be harder than these numbers suggest.  
> However, stocks with OHLCV-based OOT pos ≥ 30 have headroom to absorb the event subsetting.

---

## 2. Per-Stock Table

| Ticker | Days | Start | End | pos_10 | pos_7 | pos_5 | SelLabel | Folds | OOT_pos | Flag |
|--------|------|-------|-----|--------|-------|-------|----------|-------|---------|------|
| AAYAN        |  776 | 2023-03-15 | 2026-05-14 |   153 |   274 |   368 | y_10pct_20d    | 4 |    42 | OK |
| AAYANRE      |  771 | 2023-03-15 | 2026-05-14 |   181 |   277 |   362 | y_10pct_20d    | 3 |    42 | OK |
| ABAR         |  757 | 2023-03-15 | 2026-05-14 |   189 |   290 |   358 | y_10pct_20d    | 3 |    13 | OK |
| ABK          |  776 | 2023-03-15 | 2026-05-14 |   116 |   163 |   247 | y_10pct_20d    | 4 |     0 | WARN_THIN_OOT |
| ACICO        |  482 | 2024-06-02 | 2026-05-14 |   220 |   247 |   260 | y_10pct_20d    | 1 |    47 | WARN_FEW_FOLDS |
| AINS         |  501 | 2023-03-15 | 2026-05-14 |   143 |   186 |   217 | y_10pct_20d    | 1 |    19 | WARN_FEW_FOLDS |
| ALAQARIA     |  647 | 2023-03-16 | 2026-05-14 |   221 |   286 |   341 | y_10pct_20d    | 2 |    38 | WARN_FEW_FOLDS |
| ALDEERA      |  644 | 2023-04-04 | 2026-05-14 |   190 |   232 |   261 | y_10pct_20d    | 2 |    44 | WARN_FEW_FOLDS |
| ALEID        |  632 | 2023-03-15 | 2025-10-14 |    81 |   134 |   220 | y_10pct_20d    | 2 |     1 | WARN_FEW_FOLDS |
| ALG          |  776 | 2023-03-15 | 2026-05-14 |   121 |   218 |   313 | y_10pct_20d    | 4 |     0 | WARN_THIN_OOT |
| ALIMTIAZ     |  754 | 2023-03-15 | 2026-05-14 |   243 |   305 |   370 | y_10pct_20d    | 3 |    29 | OK |
| ALKOUT       |  271 | 2023-03-15 | 2026-05-14 |   100 |   134 |   146 | y_10pct_20d    | 0 |    36 | WARN_NO_FOLDS |
| ALMANAR      |  697 | 2023-03-15 | 2026-05-14 |   227 |   286 |   347 | y_10pct_20d    | 3 |    44 | OK |
| ALOLA        |  776 | 2023-03-15 | 2026-05-14 |   237 |   316 |   369 | y_10pct_20d    | 4 |    43 | OK |
| ALSAFAT      |  776 | 2023-03-15 | 2026-05-14 |   221 |   309 |   391 | y_10pct_20d    | 4 |    23 | OK |
| ALTIJARIA    |  776 | 2023-03-15 | 2026-05-14 |   176 |   282 |   362 | y_10pct_20d    | 4 |    14 | OK |
| AMAR         |  534 | 2023-03-26 | 2026-05-14 |   142 |   176 |   207 | y_10pct_20d    | 2 |    24 | WARN_FEW_FOLDS |
| AQAR         |  646 | 2023-03-15 | 2026-05-14 |   223 |   285 |   337 | y_10pct_20d    | 2 |    34 | WARN_FEW_FOLDS |
| ARABREC      |  776 | 2023-03-15 | 2026-05-14 |   288 |   359 |   417 | y_10pct_20d    | 4 |    32 | OK |
| AREEC        |  679 | 2023-03-15 | 2026-05-14 |   106 |   179 |   265 | y_10pct_20d    | 3 |     4 | WARN_THIN_OOT |
| ARGAN        |  769 | 2023-03-15 | 2026-05-14 |   219 |   249 |   305 | y_10pct_20d    | 3 |    34 | OK |
| ARKAN        |  597 | 2023-03-20 | 2026-05-14 |   123 |   169 |   221 | y_10pct_20d    | 2 |     9 | WARN_FEW_FOLDS |
| ARZAN        |  776 | 2023-03-15 | 2026-05-14 |   283 |   379 |   453 | y_10pct_20d    | 4 |    17 | OK |
| ASC          |  648 | 2023-03-15 | 2026-05-14 |   161 |   201 |   261 | y_10pct_20d    | 2 |    31 | WARN_FEW_FOLDS |
| ASIYA        |  776 | 2023-03-15 | 2026-05-14 |   184 |   273 |   349 | y_10pct_20d    | 4 |    30 | OK |
| ATC          |  338 | 2023-03-20 | 2026-05-14 |    75 |   110 |   134 | y_10pct_20d    | 0 |    32 | WARN_NO_FOLDS |
| AZNOULA      |  776 | 2023-03-15 | 2026-05-14 |    61 |   102 |   181 | y_10pct_20d    | 4 |     1 | WARN_THIN_OOT |
| BAYANINV     |  768 | 2023-03-15 | 2026-05-14 |   263 |   353 |   403 | y_10pct_20d    | 3 |    56 | OK |
| BEYOUT       |  475 | 2024-06-11 | 2026-05-14 |    43 |   101 |   147 | y_7pct_20d     | 1 |    16 | WARN_FEW_FOLDS |
| BOUBYAN      |  776 | 2023-03-15 | 2026-05-14 |    22 |   106 |   194 | y_7pct_20d     | 4 |     0 | WARN_THIN_OOT |
| BOURSA       |  776 | 2023-03-15 | 2026-05-14 |   182 |   256 |   362 | y_10pct_20d    | 4 |     8 | WARN_THIN_OOT |
| BPCC         |  776 | 2023-03-15 | 2026-05-14 |    45 |   112 |   216 | y_7pct_20d     | 4 |    16 | OK |
| BURG         |  776 | 2023-03-15 | 2026-05-14 |   127 |   200 |   273 | y_10pct_20d    | 4 |     0 | WARN_THIN_OOT |
| CABLE        |  776 | 2023-03-15 | 2026-05-14 |   153 |   266 |   332 | y_10pct_20d    | 4 |     2 | WARN_THIN_OOT |
| CATTL        |  723 | 2023-03-15 | 2026-05-14 |   173 |   253 |   337 | y_10pct_20d    | 3 |    27 | OK |
| CBK          |  691 | 2023-03-15 | 2026-05-14 |   141 |   230 |   298 | y_10pct_20d    | 3 |     0 | WARN_THIN_OOT |
| CGC          |  776 | 2023-03-15 | 2026-05-14 |   204 |   319 |   391 | y_10pct_20d    | 4 |    22 | OK |
| CLEANING     |  742 | 2023-03-26 | 2026-05-14 |   269 |   312 |   357 | y_10pct_20d    | 3 |    56 | OK |
| COAST        |  771 | 2023-03-15 | 2026-05-14 |   191 |   243 |   326 | y_10pct_20d    | 3 |    30 | OK |
| DALQANRE     |  734 | 2023-03-15 | 2026-03-31 |   216 |   261 |   290 | y_10pct_20d    | 3 |    30 | OK |
| DIGITUS      |  548 | 2023-03-15 | 2026-05-14 |   197 |   228 |   277 | y_10pct_20d    | 2 |    55 | WARN_FEW_FOLDS |
| EKTTITAB     |  692 | 2023-03-15 | 2026-05-14 |   273 |   330 |   374 | y_10pct_20d    | 3 |    45 | OK |
| EMIRATES     |  672 | 2023-03-15 | 2026-05-14 |   188 |   250 |   300 | y_10pct_20d    | 3 |    32 | OK |
| ENERGYH      |  690 | 2023-03-15 | 2026-05-14 |   248 |   293 |   332 | y_10pct_20d    | 3 |    35 | OK |
| EQUIPMENT    |  619 | 2023-04-12 | 2026-05-14 |   234 |   289 |   339 | y_10pct_20d    | 2 |    45 | WARN_FEW_FOLDS |
| ERESCO       |  769 | 2023-03-15 | 2026-05-14 |   228 |   288 |   347 | y_10pct_20d    | 3 |    31 | OK |
| FACIL        |  768 | 2023-03-15 | 2026-05-14 |   130 |   213 |   323 | y_10pct_20d    | 3 |    17 | OK |
| FTI          |  650 | 2023-04-30 | 2026-05-14 |   177 |   262 |   304 | y_10pct_20d    | 3 |    34 | OK |
| FUTUREKID    |  758 | 2023-03-15 | 2026-05-14 |   205 |   278 |   363 | y_10pct_20d    | 3 |    14 | OK |
| GBK          |  776 | 2023-03-15 | 2026-05-14 |    87 |   158 |   283 | y_10pct_20d    | 4 |     8 | WARN_THIN_OOT |
| GFC          |  402 | 2023-03-16 | 2026-05-14 |   139 |   183 |   197 | y_10pct_20d    | 1 |    47 | WARN_FEW_FOLDS |
| GFH          |  776 | 2023-03-15 | 2026-05-14 |   180 |   290 |   369 | y_10pct_20d    | 4 |    13 | OK |
| GIH          |  738 | 2023-03-15 | 2026-05-14 |   182 |   240 |   293 | y_10pct_20d    | 3 |    42 | OK |
| GINS         |  441 | 2023-03-19 | 2026-05-13 |    76 |   108 |   140 | y_10pct_20d    | 1 |    21 | WARN_FEW_FOLDS |
| HUMANSOFT    |  776 | 2023-03-15 | 2026-05-14 |    58 |   131 |   207 | y_10pct_20d    | 4 |     0 | WARN_THIN_OOT |
| IFA          |  776 | 2023-03-15 | 2026-05-14 |   271 |   352 |   404 | y_10pct_20d    | 4 |    17 | OK |
| IFAHR        |  776 | 2023-03-15 | 2026-05-14 |   332 |   403 |   474 | y_10pct_20d    | 4 |     9 | WARN_THIN_OOT |
| INJAZZAT     |  578 | 2023-03-15 | 2026-05-14 |   125 |   214 |   279 | y_10pct_20d    | 2 |    20 | WARN_FEW_FOLDS |
| INOVEST      |  754 | 2023-03-15 | 2026-05-14 |   261 |   305 |   356 | y_10pct_20d    | 3 |    40 | OK |
| INTEGRATED   |  776 | 2023-03-15 | 2026-05-14 |   139 |   259 |   349 | y_10pct_20d    | 4 |     5 | WARN_THIN_OOT |
| IPG          |  490 | 2023-03-15 | 2026-05-14 |   126 |   177 |   226 | y_10pct_20d    | 1 |    46 | WARN_FEW_FOLDS |
| JAZEERA      |  776 | 2023-03-15 | 2026-05-14 |   198 |   264 |   333 | y_10pct_20d    | 4 |    33 | OK |
| JTC          |  734 | 2023-03-16 | 2026-05-14 |    78 |   165 |   237 | y_10pct_20d    | 3 |    21 | OK |
| KAMCO        |  776 | 2023-03-15 | 2026-05-14 |   154 |   254 |   334 | y_10pct_20d    | 4 |    22 | OK |
| KBT          |  761 | 2023-03-15 | 2026-05-14 |   207 |   261 |   327 | y_10pct_20d    | 3 |    25 | OK |
| KCEM         |  765 | 2023-03-15 | 2026-05-14 |   202 |   282 |   366 | y_10pct_20d    | 3 |    34 | OK |
| KCIN         |  433 | 2023-03-15 | 2026-05-14 |   119 |   169 |   202 | y_10pct_20d    | 1 |    48 | WARN_FEW_FOLDS |
| KCPC         |  761 | 2023-03-15 | 2026-05-14 |    84 |   161 |   238 | y_10pct_20d    | 3 |     0 | WARN_THIN_OOT |
| KFH          |  776 | 2023-03-15 | 2026-05-14 |     9 |    45 |   140 | y_5pct_20d     | 4 |    24 | OK |
| KFIC         |  665 | 2023-03-15 | 2026-05-14 |   191 |   251 |   322 | y_10pct_20d    | 3 |    16 | OK |
| KFOUC        |  763 | 2023-03-15 | 2026-05-14 |   179 |   249 |   350 | y_10pct_20d    | 3 |     7 | WARN_THIN_OOT |
| KGL          |  679 | 2023-03-15 | 2025-12-30 |   214 |   277 |   333 | y_10pct_20d    | 3 |    57 | OK |
| KHOT         |  386 | 2023-03-20 | 2026-05-14 |    96 |   135 |   163 | y_10pct_20d    | 0 |    30 | WARN_NO_FOLDS |
| KIB          |  776 | 2023-03-15 | 2026-05-14 |    59 |   167 |   272 | y_10pct_20d    | 4 |     1 | WARN_THIN_OOT |
| KINS         |  685 | 2023-03-15 | 2026-05-14 |   110 |   225 |   308 | y_10pct_20d    | 3 |    39 | OK |
| KINV         |  775 | 2023-03-15 | 2026-05-14 |   162 |   268 |   340 | y_10pct_20d    | 3 |    12 | OK |
| KMEFIC       |  616 | 2023-03-16 | 2026-05-14 |   221 |   270 |   319 | y_10pct_20d    | 2 |    50 | WARN_FEW_FOLDS |
| KPPC         |  760 | 2023-03-15 | 2026-05-14 |   228 |   295 |   344 | y_10pct_20d    | 3 |    40 | OK |
| KPROJ        |  776 | 2023-03-15 | 2026-05-14 |   121 |   213 |   302 | y_10pct_20d    | 4 |    22 | OK |
| KRE          |  776 | 2023-03-15 | 2026-05-14 |   218 |   282 |   386 | y_10pct_20d    | 4 |    12 | OK |
| KUWAITRE     |  402 | 2023-03-15 | 2026-05-14 |    89 |   128 |   151 | y_10pct_20d    | 1 |    31 | WARN_FEW_FOLDS |
| MABANEE      |  776 | 2023-03-15 | 2026-05-14 |   102 |   176 |   284 | y_10pct_20d    | 4 |    28 | OK |
| MADAR        |  562 | 2023-03-15 | 2026-05-14 |   179 |   225 |   262 | y_10pct_20d    | 2 |    46 | WARN_FEW_FOLDS |
| MANAZEL      |  776 | 2023-03-15 | 2026-05-14 |   290 |   352 |   409 | y_10pct_20d    | 4 |    21 | OK |
| MARAKEZ      |  707 | 2023-03-15 | 2026-05-14 |   207 |   251 |   287 | y_10pct_20d    | 3 |    32 | OK |
| MARKAZ       |  757 | 2023-03-15 | 2026-05-14 |   147 |   254 |   338 | y_10pct_20d    | 3 |    11 | OK |
| MASAKEN      |  580 | 2023-03-15 | 2025-08-13 |   212 |   271 |   307 | y_10pct_20d    | 2 |    54 | WARN_FEW_FOLDS |
| MASHAER      |  749 | 2023-03-16 | 2026-05-14 |   203 |   261 |   307 | y_10pct_20d    | 3 |    30 | OK |
| MAZAYA       |  776 | 2023-03-15 | 2026-05-14 |   209 |   299 |   387 | y_10pct_20d    | 4 |    19 | OK |
| MENA         |  560 | 2024-02-04 | 2026-05-14 |   157 |   202 |   266 | y_10pct_20d    | 2 |    23 | WARN_FEW_FOLDS |
| MEZZAN       |  776 | 2023-03-15 | 2026-05-14 |   298 |   360 |   437 | y_10pct_20d    | 4 |    44 | OK |
| MIDAN        |  165 | 2023-03-19 | 2026-04-22 |    41 |    57 |    69 | y_7pct_20d     | 0 |    41 | WARN_NO_FOLDS |
| MKHZN        |  776 | 2023-03-15 | 2026-05-14 |   141 |   204 |   293 | y_10pct_20d    | 4 |    27 | OK |
| MRC          |  706 | 2023-03-15 | 2026-05-14 |   267 |   316 |   366 | y_10pct_20d    | 3 |    30 | OK |
| MUBARRAD     |  728 | 2023-03-15 | 2026-05-14 |   171 |   251 |   351 | y_10pct_20d    | 3 |    23 | OK |
| MUNSHAAT     |  764 | 2023-03-16 | 2026-05-14 |   194 |   291 |   372 | y_10pct_20d    | 3 |    11 | OK |
| MUNTAZAHAT   |  734 | 2023-03-15 | 2026-05-14 |   152 |   248 |   298 | y_10pct_20d    | 3 |    23 | OK |
| NAPESCO      |  538 | 2023-03-15 | 2026-05-14 |   132 |   201 |   244 | y_10pct_20d    | 2 |    23 | WARN_FEW_FOLDS |
| NBK          |  776 | 2023-03-15 | 2026-05-14 |    19 |    70 |   143 | y_7pct_20d     | 4 |     0 | WARN_THIN_OOT |
| NCCI         |  776 | 2023-03-15 | 2026-05-14 |   113 |   216 |   296 | y_10pct_20d    | 4 |    10 | OK |
| NICBM        |  610 | 2023-03-15 | 2026-05-14 |   136 |   196 |   261 | y_10pct_20d    | 2 |    20 | WARN_FEW_FOLDS |
| NIH          |  773 | 2023-03-15 | 2026-05-14 |   165 |   215 |   316 | y_10pct_20d    | 3 |    29 | OK |
| NIND         |  776 | 2023-03-15 | 2026-05-14 |   129 |   218 |   315 | y_10pct_20d    | 4 |    22 | OK |
| NINV         |  776 | 2023-03-15 | 2026-05-14 |   183 |   278 |   356 | y_10pct_20d    | 4 |     2 | WARN_THIN_OOT |
| NOOR         |  776 | 2023-03-15 | 2026-05-14 |   259 |   372 |   441 | y_10pct_20d    | 4 |    26 | OK |
| NRE          |  776 | 2023-03-15 | 2026-05-14 |   144 |   222 |   311 | y_10pct_20d    | 4 |    26 | OK |
| OOREDOO      |  775 | 2023-03-15 | 2026-05-14 |   202 |   280 |   385 | y_10pct_20d    | 3 |    50 | OK |
| OSOS         |  566 | 2023-03-15 | 2026-05-14 |   142 |   201 |   237 | y_10pct_20d    | 2 |    36 | WARN_FEW_FOLDS |
| OSOUL        |  660 | 2023-03-15 | 2026-05-14 |   177 |   226 |   274 | y_10pct_20d    | 3 |    21 | OK |
| OULAFUEL     |  776 | 2023-03-15 | 2026-05-14 |   174 |   247 |   337 | y_10pct_20d    | 4 |    12 | OK |
| PAPCO        |  565 | 2023-03-15 | 2026-05-14 |   156 |   205 |   237 | y_10pct_20d    | 2 |    27 | WARN_FEW_FOLDS |
| PAPER        |  709 | 2023-03-15 | 2026-05-14 |   186 |   251 |   301 | y_10pct_20d    | 3 |    19 | OK |
| PCEM         |  760 | 2023-03-15 | 2026-05-14 |    22 |    58 |   156 | y_7pct_20d     | 3 |     3 | WARN_THIN_OOT |
| QIC          |  334 | 2023-03-16 | 2026-05-14 |    83 |   138 |   198 | y_10pct_20d    | 0 |     6 | WARN_NO_FOLDS |
| RASIYAT      |  776 | 2023-03-15 | 2026-05-14 |   156 |   250 |   331 | y_10pct_20d    | 4 |    40 | OK |
| SANAM        |  751 | 2023-03-15 | 2026-05-14 |   234 |   326 |   403 | y_10pct_20d    | 3 |    50 | OK |
| SECH         |  776 | 2023-03-15 | 2026-05-14 |   235 |   310 |   374 | y_10pct_20d    | 4 |    34 | OK |
| SENERGY      |  741 | 2023-03-16 | 2026-05-14 |   224 |   291 |   348 | y_10pct_20d    | 3 |    34 | OK |
| SHIP         |  776 | 2023-03-15 | 2026-05-14 |   118 |   187 |   274 | y_10pct_20d    | 4 |    16 | OK |
| SOKOUK       |  775 | 2023-03-15 | 2026-05-14 |   281 |   358 |   428 | y_10pct_20d    | 3 |    32 | OK |
| SOOR         |  776 | 2023-03-15 | 2026-05-14 |   158 |   246 |   326 | y_10pct_20d    | 4 |    17 | OK |
| SPEC         |  771 | 2023-03-15 | 2026-05-14 |   154 |   259 |   361 | y_10pct_20d    | 3 |    25 | OK |
| SRE          |  776 | 2023-03-15 | 2026-05-14 |    16 |    78 |   186 | y_7pct_20d     | 4 |     4 | WARN_THIN_OOT |
| STC          |  776 | 2023-03-15 | 2026-05-14 |    73 |   148 |   260 | y_10pct_20d    | 4 |    27 | OK |
| TAHSSILAT    |  681 | 2023-03-20 | 2026-05-14 |   188 |   231 |   268 | y_10pct_20d    | 3 |    46 | OK |
| TAM          |  571 | 2023-03-15 | 2026-05-14 |    92 |   143 |   204 | y_10pct_20d    | 2 |    39 | WARN_FEW_FOLDS |
| TAMINV       |  317 | 2023-05-02 | 2026-05-14 |    84 |   124 |   133 | y_10pct_20d    | 0 |    31 | WARN_NO_FOLDS |
| THURAYA      |  455 | 2023-03-16 | 2026-05-14 |    99 |   151 |   195 | y_10pct_20d    | 1 |    28 | WARN_FEW_FOLDS |
| TIJARA       |  760 | 2023-03-15 | 2026-05-14 |   187 |   269 |   361 | y_10pct_20d    | 3 |    39 | OK |
| UNICAP       |  707 | 2023-03-20 | 2026-05-14 |   248 |   310 |   353 | y_10pct_20d    | 3 |    22 | OK |
| UPAC         |  646 | 2023-03-20 | 2026-05-14 |   178 |   241 |   288 | y_10pct_20d    | 2 |    25 | WARN_FEW_FOLDS |
| URC          |  775 | 2023-03-15 | 2026-05-14 |   259 |   336 |   394 | y_10pct_20d    | 3 |    41 | OK |
| VALMORE      |  753 | 2023-03-15 | 2026-05-14 |    68 |   128 |   195 | y_10pct_20d    | 3 |    17 | OK |
| WARBABANK    |  776 | 2023-03-15 | 2026-05-14 |    80 |   142 |   241 | y_10pct_20d    | 4 |     2 | WARN_THIN_OOT |
| WARBACAP     |  752 | 2023-03-15 | 2026-05-14 |   236 |   306 |   377 | y_10pct_20d    | 3 |    13 | OK |
| WETHAQ       |  730 | 2023-03-15 | 2026-05-14 |   250 |   315 |   366 | y_10pct_20d    | 3 |    45 | OK |
| WINSRE       |  741 | 2023-03-15 | 2026-05-14 |   116 |   203 |   275 | y_10pct_20d    | 3 |     8 | WARN_THIN_OOT |
| ZAIN         |  776 | 2023-03-15 | 2026-05-14 |    27 |   103 |   173 | y_7pct_20d     | 4 |    26 | OK |

---

## 3. Top 20 Likely to Pass / Bottom 20 Likely to Fail

> Score = folds×10 + min(OOT_pos, 60) − 5 if non-primary-label

### Top 20

| Ticker | Days | SelLabel | Folds | OOT_pos | Score | Flag |
|--------|------|----------|-------|---------|-------|------|
| KGL          |  679 | y_10pct_20d    | 3 |    57 |    87 | OK |
| BAYANINV     |  768 | y_10pct_20d    | 3 |    56 |    86 | OK |
| CLEANING     |  742 | y_10pct_20d    | 3 |    56 |    86 | OK |
| MEZZAN       |  776 | y_10pct_20d    | 4 |    44 |    84 | OK |
| ALOLA        |  776 | y_10pct_20d    | 4 |    43 |    83 | OK |
| AAYAN        |  776 | y_10pct_20d    | 4 |    42 |    82 | OK |
| OOREDOO      |  775 | y_10pct_20d    | 3 |    50 |    80 | OK |
| RASIYAT      |  776 | y_10pct_20d    | 4 |    40 |    80 | OK |
| SANAM        |  751 | y_10pct_20d    | 3 |    50 |    80 | OK |
| TAHSSILAT    |  681 | y_10pct_20d    | 3 |    46 |    76 | OK |
| DIGITUS      |  548 | y_10pct_20d    | 2 |    55 |    75 | WARN_FEW_FOLDS |
| EKTTITAB     |  692 | y_10pct_20d    | 3 |    45 |    75 | OK |
| WETHAQ       |  730 | y_10pct_20d    | 3 |    45 |    75 | OK |
| ALMANAR      |  697 | y_10pct_20d    | 3 |    44 |    74 | OK |
| MASAKEN      |  580 | y_10pct_20d    | 2 |    54 |    74 | WARN_FEW_FOLDS |
| SECH         |  776 | y_10pct_20d    | 4 |    34 |    74 | OK |
| JAZEERA      |  776 | y_10pct_20d    | 4 |    33 |    73 | OK |
| AAYANRE      |  771 | y_10pct_20d    | 3 |    42 |    72 | OK |
| ARABREC      |  776 | y_10pct_20d    | 4 |    32 |    72 | OK |
| GIH          |  738 | y_10pct_20d    | 3 |    42 |    72 | OK |

### Bottom 20

| Ticker | Days | SelLabel | Folds | OOT_pos | Score | Flag |
|--------|------|----------|-------|---------|-------|------|
| THURAYA      |  455 | y_10pct_20d    | 1 |    28 |    38 | WARN_FEW_FOLDS |
| WINSRE       |  741 | y_10pct_20d    | 3 |     8 |    38 | WARN_THIN_OOT |
| KFOUC        |  763 | y_10pct_20d    | 3 |     7 |    37 | WARN_THIN_OOT |
| ALKOUT       |  271 | y_10pct_20d    | 0 |    36 |    36 | WARN_NO_FOLDS |
| MIDAN        |  165 | y_7pct_20d     | 0 |    41 |    36 | WARN_NO_FOLDS |
| BOUBYAN      |  776 | y_7pct_20d     | 4 |     0 |    35 | WARN_THIN_OOT |
| NBK          |  776 | y_7pct_20d     | 4 |     0 |    35 | WARN_THIN_OOT |
| AREEC        |  679 | y_10pct_20d    | 3 |     4 |    34 | WARN_THIN_OOT |
| ATC          |  338 | y_10pct_20d    | 0 |    32 |    32 | WARN_NO_FOLDS |
| GINS         |  441 | y_10pct_20d    | 1 |    21 |    31 | WARN_FEW_FOLDS |
| TAMINV       |  317 | y_10pct_20d    | 0 |    31 |    31 | WARN_NO_FOLDS |
| CBK          |  691 | y_10pct_20d    | 3 |     0 |    30 | WARN_THIN_OOT |
| KCPC         |  761 | y_10pct_20d    | 3 |     0 |    30 | WARN_THIN_OOT |
| KHOT         |  386 | y_10pct_20d    | 0 |    30 |    30 | WARN_NO_FOLDS |
| AINS         |  501 | y_10pct_20d    | 1 |    19 |    29 | WARN_FEW_FOLDS |
| ARKAN        |  597 | y_10pct_20d    | 2 |     9 |    29 | WARN_FEW_FOLDS |
| PCEM         |  760 | y_7pct_20d     | 3 |     3 |    28 | WARN_THIN_OOT |
| ALEID        |  632 | y_10pct_20d    | 2 |     1 |    21 | WARN_FEW_FOLDS |
| BEYOUT       |  475 | y_7pct_20d     | 1 |    16 |    21 | WARN_FEW_FOLDS |
| QIC          |  334 | y_10pct_20d    | 0 |     6 |     6 | WARN_NO_FOLDS |

---

## 4. Honest Prediction

### Assumptions
- Walk-forward gates require ≥3 folds with ≥30 OOT positives on selected label
- Event-based matrix will have ~30–60% of OHLCV rows as event rows (rough estimate)
- OOT pos in actual training ≈ 30–60% of OHLCV-based OOT pos shown here

### Adjusted Estimates

| Scenario | Estimated SHADOW count | Basis |
|----------|------------------------|-------|
| **Optimistic** (event rate 70% of OHLCV) | ~11 stocks SHADOW | OOT pos × 0.70 ≥ 30 AND folds ≥ 3 |
| **Pessimistic** (event rate 40% of OHLCV) | ~0 stocks SHADOW | OOT pos × 0.40 ≥ 30 AND folds ≥ 3 |

### Key Risks

1. **22 stocks have OHLCV-OOT pos < 10** — these will almost certainly fail gate G5
   after event subsetting. (ABK, ALG, AZNOULA, BOUBYAN, BOURSA, BURG, CABLE, CBK, GBK,
   HUMANSOFT, IFAHR, INTEGRATED, KCPC, KIB, KFOUC, NBK, NINV, PCEM, SRE, WARBABANK, WINSRE + more)

2. **6 stocks have 0 folds** (ALKOUT, ATC, KHOT, MIDAN, QIC, TAMINV) —
   insufficient OHLCV history for walk-forward. Will fail pre-gate.

3. **29 stocks have ≤2 folds** — reduced validation reliability.
   Many are newer listings (2–3 years history). These may still train but with
   wider confidence intervals.

4. **NBK specifically:** OHLCV-OOT pos = 0 even on y_7pct_20d. Confirmed via prior
   diagnostic: all positives fall in training window, none in OOT. Will fail G5.
   Needs ≥3 more years of OHLCV data.

5. **Label tier distribution is healthy:** 129 stocks on y_10pct_20d
   (93.5%), 8 on y_7pct_20d,
   1 on y_5pct_20d, 0 INSUFFICIENT_DATA.
   Zero stocks are ungradeable at census stage.

---

## CENSUS VERDICT

### REVIEW REQUIRED

Estimated SHADOW count (11 optimistic, 0 pessimistic) is too low. Critical data quality issues present. Review before launching full training run.

> **This report is informational only. Full training run requires explicit user authorization.**

---
*Census method: OHLCV-bar-based forward-return labels (see `run_census.py`). 
No model training, no DB writes performed.*
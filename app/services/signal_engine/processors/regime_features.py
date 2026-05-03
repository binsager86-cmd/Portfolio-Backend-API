"""Regime feature extraction for the Kuwait HMM detector.

Produces the three-dimensional feature matrix used by hmm_regime_detector:
  [log_return, atr_percentile, flow_ratio]

No TA-Lib dependency — all features use numpy for speed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.services.signal_engine.config.model_params import ATR_PERCENTILE_WINDOW, ATR_PERIOD


def _true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Vectorised true-range calculation."""
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return tr


def _wilder_atr(tr: np.ndarray, period: int) -> np.ndarray:
    """Wilder-smoothed ATR (same as TA-Lib ATR)."""
    n = len(tr)
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    # seed with SMA of first `period` TRs
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def extract_regime_features(rows: list[dict[str, Any]]) -> np.ndarray:
    """Build the (N, 3) feature matrix for HMM training / prediction.

    Features (each element aligns 1:1 with the input rows):
      col 0 — log_return:    daily log(close[t] / close[t-1]), 0 for first bar
      col 1 — atr_percentile: ATR_14 percentile in last ATR_PERCENTILE_WINDOW bars (0-1)
      col 2 — flow_ratio:    (close - open) / true_range where TR > 0, else 0
                              Proxy for net institutional flow direction.

    Args:
        rows: OHLCV rows sorted ascending by date, length ≥ ATR_PERIOD.

    Returns:
        np.ndarray of shape (N, 3).  NaN values in warmup rows for atr_percentile.
    """
    n = len(rows)
    o = np.array([float(r.get("open") or 0.0) for r in rows])
    h = np.array([float(r.get("high") or 0.0) for r in rows])
    l = np.array([float(r.get("low") or 0.0) for r in rows])
    c = np.array([float(r.get("close") or 0.0) for r in rows])

    # ── log_return ─────────────────────────────────────────────────────────────
    log_ret = np.zeros(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        prev_c = np.roll(c, 1)
        prev_c[0] = c[0]
        ratio = np.where(prev_c > 0, c / prev_c, 1.0)
        log_ret = np.log(np.where(ratio > 0, ratio, 1.0))

    # ── ATR percentile ─────────────────────────────────────────────────────────
    tr = _true_range(h, l, c)
    atr = _wilder_atr(tr, ATR_PERIOD)

    atr_pct = np.full(n, np.nan)
    for i in range(ATR_PERIOD - 1, n):
        window_start = max(0, i - ATR_PERCENTILE_WINDOW + 1)
        window = atr[window_start : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 1:
            atr_pct[i] = float(np.sum(valid <= atr[i])) / len(valid)

    # ── flow_ratio ────────────────────────────────────────────────────────────
    flow = np.zeros(n)
    hl_range = h - l
    safe_range = np.where(hl_range > 0, hl_range, 1.0)
    flow = np.where(hl_range > 0, (c - o) / safe_range, 0.0)
    # 3-day rolling mean to reduce noise
    flow_smoothed = np.full(n, np.nan)
    for i in range(2, n):
        flow_smoothed[i] = np.mean(flow[i - 2 : i + 1])
    flow_smoothed[0] = flow[0]
    flow_smoothed[1] = np.mean(flow[:2])

    features = np.column_stack([log_ret, atr_pct, flow_smoothed])
    return features

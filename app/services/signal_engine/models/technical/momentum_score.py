"""Momentum scorer for the Kuwait Signal Engine.

Components: RSI(14), MACD(12,26,9), ROC(10).

Raw score [0, 100]:
  > 60 → positive momentum
  40-60 → neutral
  < 40 → negative / exhausted momentum

Pre-computed indicators expected in rows:
  rsi_14, macd, macd_signal, macd_hist
ROC is computed from close prices directly (not pre-computed in indicators_service).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.services.signal_engine.config.model_params import (
    ROC_PERIOD,
    RSI_BULL_MOMENTUM_HIGH,
    RSI_BULL_MOMENTUM_LOW,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
)


def _rsi_score(last: dict[str, Any]) -> tuple[int, str]:
    """Score RSI momentum (max 40 pts)."""
    rsi = last.get("rsi_14")
    if rsi is None:
        return 20, "rsi_missing"
    v = float(rsi)
    if RSI_BULL_MOMENTUM_LOW <= v <= RSI_BULL_MOMENTUM_HIGH:
        return 40, f"healthy_bull_momentum_rsi_{v:.1f}"
    if RSI_BULL_MOMENTUM_HIGH < v < RSI_OVERBOUGHT:
        return 32, f"strong_but_extended_rsi_{v:.1f}"
    if v >= RSI_OVERBOUGHT:
        return 10, f"overbought_rsi_{v:.1f}"
    if 40.0 <= v < RSI_BULL_MOMENTUM_LOW:
        return 22, f"recovering_rsi_{v:.1f}"
    if RSI_OVERSOLD <= v < 40.0:
        return 12, f"weak_rsi_{v:.1f}"
    # v < RSI_OVERSOLD — deeply oversold (possible reversal, but bearish for cont.)
    return 5, f"deeply_oversold_rsi_{v:.1f}"


def _macd_score(last: dict[str, Any]) -> tuple[int, str]:
    """Score MACD momentum (max 35 pts)."""
    macd = last.get("macd")
    signal = last.get("macd_signal")
    hist = last.get("macd_hist")

    if macd is None or signal is None:
        return 15, "macd_missing"

    m, s = float(macd), float(signal)
    h = float(hist) if hist is not None else m - s

    if m > s and h > 0:
        # Check if histogram is growing (accelerating momentum)
        return 35, "macd_bullish_accelerating"
    if m > s and h <= 0:
        return 22, "macd_above_signal_decelerating"
    if m < s and h > 0:
        return 18, "macd_crossover_imminent"
    # m < s and h <= 0 — bearish
    return 5, "macd_bearish"


def _roc_score(rows: list[dict[str, Any]]) -> tuple[int, str]:
    """Score Rate of Change — ROC(10) (max 25 pts)."""
    if len(rows) < ROC_PERIOD + 1:
        return 12, "roc_insufficient_data"

    closes = [float(r.get("close") or 0.0) for r in rows]
    c_now = closes[-1]
    c_prev = closes[-(ROC_PERIOD + 1)]

    if c_prev == 0:
        return 12, "roc_zero_division"

    roc_pct = (c_now - c_prev) / c_prev * 100.0

    if roc_pct > 5.0:
        return 25, f"strong_positive_roc_{roc_pct:.1f}pct"
    if roc_pct > 2.0:
        return 20, f"moderate_positive_roc_{roc_pct:.1f}pct"
    if roc_pct > 0.5:
        return 15, f"mild_positive_roc_{roc_pct:.1f}pct"
    if roc_pct > -1.0:
        return 8, f"flat_roc_{roc_pct:.1f}pct"
    if roc_pct > -3.0:
        return 4, f"mild_negative_roc_{roc_pct:.1f}pct"
    return 0, f"strong_negative_roc_{roc_pct:.1f}pct"


def compute_momentum_score(rows: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    """Compute the raw momentum score and component breakdown.

    Args:
        rows: OHLCV + indicator rows sorted ascending by date.

    Returns:
        Tuple of (raw_score: int [0, 100], details: dict).
    """
    if not rows:
        return 50, {"error": "no_rows"}

    last = rows[-1]

    rsi_pts, rsi_desc = _rsi_score(last)
    macd_pts, macd_desc = _macd_score(last)
    roc_pts, roc_desc = _roc_score(rows)

    raw = min(100, rsi_pts + macd_pts + roc_pts)

    details = {
        "rsi_pts": rsi_pts,
        "rsi_desc": rsi_desc,
        "macd_pts": macd_pts,
        "macd_desc": macd_desc,
        "roc_pts": roc_pts,
        "roc_desc": roc_desc,
        "raw_score": raw,
    }
    return raw, details

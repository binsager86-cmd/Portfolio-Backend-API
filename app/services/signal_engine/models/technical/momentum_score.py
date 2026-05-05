"""Momentum scorer for the Kuwait Signal Engine.

Components: RSI(14), MACD(12,26,9), ROC(10), Stochastic.

Raw score [0, 100]:
  > 60 → positive momentum
  40-60 → neutral
  < 40 → negative / exhausted momentum

Option A weighting (Kuwait Trend-Priority):
  MACD 40% | RSI 25% | ROC 25% | Stoch 10%

Weights are baked into each component's max-points so the sum is naturally 0–100:
  MACD  max 40 pts  — captures slow block-accumulation; trend-following edge in Kuwait
  RSI   max 25 pts  — overbought/oversold guard, not primary driver
  ROC   max 25 pts  — breakout momentum with slight noise-filtering
  Stoch max 10 pts  — short-term timing, minimal to avoid oscillator double-counting

Pre-computed indicators expected in rows:
  rsi_14, macd, macd_signal, macd_hist, stoch_k, stoch_d
ROC is computed from close prices directly.
"""
from __future__ import annotations

from typing import Any

from app.services.signal_engine.config.model_params import ROC_PERIOD


def _rsi_score(last: dict[str, Any]) -> tuple[int, str]:
    """Max 25 pts (Option A: RSI=25%). Overbought threshold raised to 70."""
    rsi = last.get("rsi_14")
    if rsi is None:
        return 12, "rsi_missing"
    v = float(rsi)
    if 50.0 <= v <= 65.0:
        return 25, f"healthy_bull_momentum_rsi_{v:.1f}"
    if 65.0 < v < 70.0:
        return 20, f"strong_but_extended_rsi_{v:.1f}"
    if v >= 70.0:
        return 6, f"overbought_rsi_{v:.1f}"
    if 40.0 <= v < 50.0:
        return 13, f"recovering_rsi_{v:.1f}"
    if 35.0 <= v < 40.0:
        return 7, f"weak_rsi_{v:.1f}"
    return 3, f"deeply_oversold_rsi_{v:.1f}"


def _macd_score(last: dict[str, Any]) -> tuple[int, str]:
    """Max 40 pts (Option A: MACD=40%). Captures trend persistence / block accumulation."""
    macd = last.get("macd")
    signal = last.get("macd_signal")
    hist = last.get("macd_hist")

    if macd is None or signal is None:
        return 17, "macd_missing"

    m, s = float(macd), float(signal)
    h = float(hist) if hist is not None else m - s

    if m > s and h > 0:
        return 40, "macd_bullish_accelerating"
    if m > s and h <= 0:
        return 25, "macd_above_signal_decelerating"
    if m < s and h > 0:
        return 20, "macd_crossover_imminent"
    return 5, "macd_bearish"


def _roc_score(rows: list[dict[str, Any]]) -> tuple[int, str]:
    """Max 25 pts (Option A: ROC=25%). Breakout momentum; ROC(10) bars."""
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
        return 7, f"flat_roc_{roc_pct:.1f}pct"
    if roc_pct > -3.0:
        return 4, f"mild_negative_roc_{roc_pct:.1f}pct"
    return 0, f"strong_negative_roc_{roc_pct:.1f}pct"


def _stoch_score(last: dict[str, Any]) -> tuple[int, str]:
    """Max 10 pts (Option A: Stoch=10%). Minimal weight to reduce RSI overlap."""
    sk = last.get("stoch_k")
    sd = last.get("stoch_d")
    if sk is None or sd is None:
        return 5, "stoch_missing"
    k, d = float(sk), float(sd)
    if k > d:
        if 40.0 <= k <= 70.0:
            return 10, f"stoch_bullish_zone_k{k:.0f}"
        if k < 40.0:
            return 8, f"stoch_recovering_oversold_k{k:.0f}"
        if k < 80.0:
            return 6, f"stoch_extended_not_overbought_k{k:.0f}"
        return 3, f"stoch_overbought_k{k:.0f}"
    else:
        if k > 60.0:
            return 2, f"stoch_bearish_elevated_k{k:.0f}"
        return 0, f"stoch_bearish_k{k:.0f}"


def compute_momentum_score(rows: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    """Option A: MACD 40% | RSI 25% | ROC 25% | Stoch 10%.

    Component max-pts already encode the weights; the sum is naturally 0–100.

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
    stoch_pts, stoch_desc = _stoch_score(last)

    raw = min(100, max(0, rsi_pts + macd_pts + roc_pts + stoch_pts))

    details = {
        "rsi_pts": rsi_pts,
        "rsi_desc": rsi_desc,
        "macd_pts": macd_pts,
        "macd_desc": macd_desc,
        "roc_pts": roc_pts,
        "roc_desc": roc_desc,
        "stoch_pts": stoch_pts,
        "stoch_desc": stoch_desc,
        "raw_score": raw,
    }
    return raw, details


"""Trend strength scorer for the Kuwait Signal Engine.

Evaluates EMA alignment, ADX momentum, and higher-high / higher-low
swing structure to produce a raw trend score in [0, 100].

Score interpretation:
  > 60 → bullish trend
  40-60 → neutral / transitioning
  < 40 → bearish trend

Pre-computed indicators expected in rows (from attach_indicators):
  ema_20, ema_50, sma_200, adx_14
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.services.signal_engine.config.model_params import (
    ADX_STRONG_MIN,
    ADX_TRENDING_MIN,
    ADX_VERY_STRONG_MIN,
    PIVOT_LOOKBACK,
)


def _ema_alignment_score(last: dict[str, Any]) -> tuple[int, str]:
    """Score EMA alignment (max 40 pts).

    Returns (score, description).
    """
    close = float(last.get("close") or 0.0)
    ema20 = last.get("ema_20")
    ema50 = last.get("ema_50")
    sma200 = last.get("sma_200")

    if ema20 is None or ema50 is None:
        return 20, "ema_missing"

    e20, e50 = float(ema20), float(ema50)
    s200 = float(sma200) if sma200 is not None else None

    if s200 is not None:
        if close > e20 > e50 > s200:
            return 40, "full_bullish_alignment"
        if close > e20 > e50:
            return 30, "short_term_bullish"
        if e20 > e50 > s200 and close < e20:
            return 22, "bullish_structure_pullback"
        if close > e20 and e20 < e50:
            return 15, "price_above_ema20_only"
        if close < e20 < e50 < s200:
            return 0, "full_bearish_alignment"
        if close < e20 < e50:
            return 5, "short_term_bearish"
        return 12, "mixed"
    else:
        # No SMA200 (insufficient history)
        if close > e20 > e50:
            return 28, "short_term_bullish_no_200"
        if close < e20 < e50:
            return 7, "short_term_bearish_no_200"
        return 15, "mixed_no_200"


def _adx_score(last: dict[str, Any]) -> tuple[int, str]:
    """Score ADX trend strength (max 30 pts)."""
    adx = last.get("adx_14")
    if adx is None:
        return 10, "adx_missing"
    v = float(adx)
    if v >= ADX_VERY_STRONG_MIN:
        return 30, f"very_strong_trend_adx_{v:.1f}"
    if v >= ADX_STRONG_MIN:
        return 24, f"strong_trend_adx_{v:.1f}"
    if v >= ADX_TRENDING_MIN:
        return 15, f"trending_adx_{v:.1f}"
    return 5, f"weak_adx_{v:.1f}"


def _swing_structure_score(rows: list[dict[str, Any]]) -> tuple[int, str]:
    """Score higher-highs / higher-lows swing structure (max 30 pts).

    Uses the last PIVOT_LOOKBACK * 4 bars to identify the most recent
    two swing highs and two swing lows, then checks their direction.
    """
    lookback = PIVOT_LOOKBACK
    window = rows[-(lookback * 6):]  # need several pivots worth of bars
    if len(window) < lookback * 2 + 1:
        return 15, "insufficient_history"

    closes = [float(r.get("close") or 0.0) for r in window]
    highs = [float(r.get("high") or 0.0) for r in window]
    lows = [float(r.get("low") or 0.0) for r in window]
    n = len(highs)

    swing_highs: list[float] = []
    swing_lows: list[float] = []

    for i in range(lookback, n - lookback):
        if highs[i] == max(highs[i - lookback: i + lookback + 1]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i - lookback: i + lookback + 1]):
            swing_lows.append(lows[i])

    bullish_hh = len(swing_highs) >= 2 and swing_highs[-1] > swing_highs[-2]
    bullish_hl = len(swing_lows) >= 2 and swing_lows[-1] > swing_lows[-2]
    bearish_lh = len(swing_highs) >= 2 and swing_highs[-1] < swing_highs[-2]
    bearish_ll = len(swing_lows) >= 2 and swing_lows[-1] < swing_lows[-2]

    if bullish_hh and bullish_hl:
        return 30, "higher_highs_and_higher_lows"
    if bullish_hh or bullish_hl:
        return 20, "partial_bullish_structure"
    if bearish_lh and bearish_ll:
        return 0, "lower_highs_and_lower_lows"
    if bearish_lh or bearish_ll:
        return 5, "partial_bearish_structure"
    return 12, "no_clear_swing_structure"


def compute_trend_score(rows: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    """Compute the raw trend score and component breakdown.

    Args:
        rows: OHLCV + indicator rows sorted ascending by date.

    Returns:
        Tuple of (raw_score: int [0, 100], details: dict).
    """
    if not rows:
        return 50, {"error": "no_rows"}

    last = rows[-1]

    ema_pts, ema_desc = _ema_alignment_score(last)
    adx_pts, adx_desc = _adx_score(last)
    swing_pts, swing_desc = _swing_structure_score(rows)

    raw = min(100, ema_pts + adx_pts + swing_pts)

    details = {
        "ema_alignment_pts": ema_pts,
        "ema_alignment_desc": ema_desc,
        "adx_pts": adx_pts,
        "adx_desc": adx_desc,
        "swing_structure_pts": swing_pts,
        "swing_structure_desc": swing_desc,
        "raw_score": raw,
    }
    return raw, details

"""Trend strength scorer for the Kuwait Signal Engine.

Evaluates EMA alignment, ADX momentum, and higher-high / higher-low
swing structure to produce a raw trend score in [0, 100].

Three multiplicative modifiers are applied after the base score:
  1. Kaufman Efficiency Ratio  — rewards clean directional moves, penalises noise.
  2. Trend age / maturity      — boosts fresh EMA crossovers, penalises exhaustion.
  3. EMA stretch guard         — penalises price that is over-extended from EMA20.

Score interpretation:
  > 60 → bullish trend
  40-60 → neutral / transitioning
  < 40 → bearish trend

Pre-computed indicators expected in rows (from attach_indicators):
  ema_20, ema_50, sma_200, adx_14, atr_14
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.services.signal_engine.config.model_params import (
    ADX_STRONG_MIN,
    ADX_TRENDING_MIN,
    ADX_VERY_STRONG_MIN,
    ER_HIGH,
    ER_LOW,
    ER_MID,
    ER_PERIOD,
    PIVOT_LOOKBACK,
    STRETCH_MODERATE_ATR,
    STRETCH_SEVERE_ATR,
    TREND_AGE_FLOOR_MULT,
    TREND_AGE_PEAK_MULT,
    TREND_AGE_SCALE,
)


# ── Modifier helpers ──────────────────────────────────────────────────────────

def _kaufman_er(rows: list[dict[str, Any]], period: int = ER_PERIOD) -> float:
    """Kaufman Efficiency Ratio: directional move / sum of bar-to-bar noise.

    Returns a value in [0.0, 1.0]:
      0.0 = pure random walk (maximum noise, zero net direction)
      1.0 = perfectly trending (every bar moves in the same direction)

    Uses the last ``period + 1`` closes.  Returns 0.5 (neutral) when
    insufficient data is available.
    """
    n = len(rows)
    if n < period + 1:
        return 0.5
    recent = rows[-(period + 1):]
    closes = [float(r.get("close") or 0.0) for r in recent]
    directional = abs(closes[-1] - closes[0])
    noise = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))
    return directional / noise if noise > 0 else 0.5


def _bars_since_ema20_50_cross(rows: list[dict[str, Any]]) -> int:
    """Count bars elapsed since the most recent EMA20-crosses-above-EMA50 event.

    Walks backwards through the row window looking for the transition:
      row[i-1]: ema_20 ≤ ema_50  (bearish or flat)
      row[i]  : ema_20 > ema_50  (bullish crossover)

    Returns:
      0                → crossover happened on the very last bar (freshest)
      n-1 (≈ lookback) → crossover was the oldest detectable bar
      len(rows)        → no bullish crossover found in the window
                          (trend is either fully aged or currently bearish)
    """
    n = len(rows)
    if n < 2:
        return n
    for i in range(n - 1, 0, -1):
        e20c = rows[i].get("ema_20")
        e50c = rows[i].get("ema_50")
        e20p = rows[i - 1].get("ema_20")
        e50p = rows[i - 1].get("ema_50")
        if None in (e20c, e50c, e20p, e50p):
            continue
        if float(e20p) <= float(e50p) and float(e20c) > float(e50c):
            return n - 1 - i   # bars elapsed since that crossover
    return n   # no bullish cross found → treat as fully aged / bearish


# ── Base sub-component scorers (unchanged weights: 40 / 30 / 30) ─────────────

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

    from numpy.lib.stride_tricks import sliding_window_view

    highs_arr = np.array([float(r.get("high") or 0.0) for r in window])
    lows_arr  = np.array([float(r.get("low")  or 0.0) for r in window])
    n = len(highs_arr)
    w = 2 * lookback + 1
    windows_h = sliding_window_view(highs_arr, w)  # shape (n - w + 1, w)
    windows_l = sliding_window_view(lows_arr,  w)
    center_h = highs_arr[lookback: n - lookback]
    center_l = lows_arr[lookback:  n - lookback]
    swing_highs: list[float] = center_h[center_h == windows_h.max(axis=1)].tolist()
    swing_lows:  list[float] = center_l[center_l == windows_l.min(axis=1)].tolist()

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


# ── Main scorer ───────────────────────────────────────────────────────────────

def compute_trend_score(
    rows: list[dict[str, Any]],
    stock_sector: str = "",
    banking_trend_raw: float = 0.0,
) -> tuple[int, dict[str, Any]]:
    """Compute the raw trend score with multiplicative context modifiers.

    Args:
        rows: OHLCV + indicator rows sorted ascending by date.
        stock_sector: Sector string (e.g. ``"Banking"``).  Passed by
            ``signal_generator`` after the banking-data fetch so the
            sector lead-lag modifier can be applied in one call.
        banking_trend_raw: Raw banking-index trend score [0–100].  A value
            of 0.0 (default) disables the sector modifier (neutral × 1.0).

    Returns:
        Tuple of (final_score: int [0, 100], details: dict).

    Modifier pipeline (applied after base score):
        1. Kaufman ER      — penalises choppy / noisy price action.
        2. Trend age       — boosts fresh EMA20/50 crossovers, penalises aged trends.
        3. EMA stretch     — penalises over-extension from EMA20 (mean-reversion risk).
        4. Sector lead-lag — discounts non-banking stocks when banking is weak (optional).
    """
    if not rows:
        return 50, {"error": "no_rows"}

    last = rows[-1]

    # ── Base sub-components (40 / 30 / 30 — weights unchanged) ───────────────
    ema_pts, ema_desc = _ema_alignment_score(last)
    adx_pts, adx_desc = _adx_score(last)
    swing_pts, swing_desc = _swing_structure_score(rows)
    base_raw = min(100, ema_pts + adx_pts + swing_pts)

    # ── Modifier 1: Kaufman Efficiency Ratio ──────────────────────────────────
    er = _kaufman_er(rows)
    if er > ER_HIGH:
        er_mult = 1.15
        er_label = "clean_trend"
    elif er > ER_MID:
        er_mult = 1.05
        er_label = "moderate_trend"
    elif er < ER_LOW:
        er_mult = 0.80
        er_label = "noisy_random_walk"
    else:
        er_mult = 1.00
        er_label = "neutral"

    # ── Modifier 2: Trend age / maturity ─────────────────────────────────────
    bars_since_cross = _bars_since_ema20_50_cross(rows)
    age_mult = max(
        TREND_AGE_FLOOR_MULT,
        TREND_AGE_PEAK_MULT - (bars_since_cross / TREND_AGE_SCALE),
    )

    # ── Modifier 3: EMA stretch (mean-reversion guard) ────────────────────────
    close = float(last.get("close") or 0.0)
    ema20_val = last.get("ema_20")
    atr14_val = last.get("atr_14")
    if ema20_val is not None and atr14_val and float(atr14_val) > 0:
        stretch = abs(close - float(ema20_val)) / float(atr14_val)
        if stretch > STRETCH_SEVERE_ATR:
            stretch_mult = 0.45
            stretch_label = f"severely_extended_{stretch:.2f}x_atr"
        elif stretch > STRETCH_MODERATE_ATR:
            stretch_mult = 0.75
            stretch_label = f"moderately_extended_{stretch:.2f}x_atr"
        else:
            stretch_mult = 1.00
            stretch_label = f"within_bounds_{stretch:.2f}x_atr"
    else:
        stretch = 0.0
        stretch_mult = 1.00
        stretch_label = "atr_unavailable"

    # ── Modifier 4: Kuwait sector lead-lag (optional) ─────────────────────────
    # Applied when banking context is provided by signal_generator (step 8a).
    # Non-banking stocks are discounted when banking-sector trend is weak.
    sector_is_banking = str(stock_sector).lower() == "banking"
    if banking_trend_raw > 0 and not sector_is_banking and banking_trend_raw < 60:
        sector_mult = 0.85
        sector_label = f"non_banking_weak_bank_index_{banking_trend_raw:.0f}"
    else:
        sector_mult = 1.00
        sector_label = "not_applied"

    # ── Final score ───────────────────────────────────────────────────────────
    combined_mult = er_mult * age_mult * stretch_mult * sector_mult
    final_score = int(min(100, max(0, base_raw * combined_mult)))

    details: dict[str, Any] = {
        # Base sub-components (unchanged — audit trail)
        "ema_alignment_pts":   ema_pts,
        "ema_alignment_desc":  ema_desc,
        "adx_pts":             adx_pts,
        "adx_desc":            adx_desc,
        "swing_structure_pts": swing_pts,
        "swing_structure_desc": swing_desc,
        "base_raw":            base_raw,
        # Multipliers (full audit trail)
        "er_value":            round(er, 4),
        "er_mult":             er_mult,
        "er_label":            er_label,
        "bars_since_ema_cross": bars_since_cross,
        "age_mult":            round(age_mult, 4),
        "stretch_atr":         round(stretch, 4),
        "stretch_mult":        stretch_mult,
        "stretch_label":       stretch_label,
        "sector_mult":         sector_mult,
        "sector_label":        sector_label,
        "combined_mult":       round(combined_mult, 4),
        "raw_score":           final_score,
    }
    return final_score, details


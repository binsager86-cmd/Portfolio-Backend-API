"""Support/resistance analysis and entry-level calculator.

Provides:
  compute_sr_score()     — raw S/R quality score [0, 100]
  compute_entry_stop_tp() — entry zone, stop-loss, TP1, TP2 in fils

S/R levels are identified via:
  1. Swing-pivot clustering (local price extremes ± PIVOT_LOOKBACK bars)
  2. Volume-Profile POC approximation (price bucket with highest total volume)
  3. Anchored VWAP from last significant swing low / high

All output prices are tick-aligned per Kuwait rules.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.services.signal_engine.config.kuwait_constants import align_to_tick
from app.services.signal_engine.config.model_params import (
    ENTRY_BUFFER_PCT,
    PIVOT_CLUSTER_PCT,
    PIVOT_LOOKBACK,
    SR_PROXIMITY_PCT,
    STOP_ATR_MULTIPLIER,
    TP1_RR_MULTIPLIER,
    TP2_RR_MULTIPLIER,
    VWAP_ANCHOR_LOOKBACK,
)


# ── Swing Pivot Detection ─────────────────────────────────────────────────────

def _find_swing_pivots(
    highs: list[float],
    lows: list[float],
    lookback: int,
) -> tuple[list[float], list[float]]:
    """Identify swing highs and swing lows in the data.

    Returns:
        (swing_highs, swing_lows) as lists of price values.
    """
    n = len(highs)
    sh: list[float] = []
    sl: list[float] = []
    for i in range(lookback, n - lookback):
        if highs[i] == max(highs[i - lookback: i + lookback + 1]):
            sh.append(highs[i])
        if lows[i] == min(lows[i - lookback: i + lookback + 1]):
            sl.append(lows[i])
    return sh, sl


def _cluster_levels(prices: list[float], cluster_pct: float) -> list[float]:
    """Merge nearby price levels into clusters (return cluster medians)."""
    if not prices:
        return []
    sorted_prices = sorted(prices)
    clusters: list[list[float]] = [[sorted_prices[0]]]
    for p in sorted_prices[1:]:
        ref = clusters[-1][-1]
        if ref > 0 and abs(p - ref) / ref <= cluster_pct:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [float(np.median(c)) for c in clusters]


# ── Volume Profile POC Approximation ────────────────────────────────────────

def _volume_profile_poc(
    rows: list[dict[str, Any]],
    n_buckets: int = 20,
) -> float | None:
    """Approximate the Point-of-Control (price with max accumulated volume)."""
    prices = [float(r.get("close") or 0.0) for r in rows]
    vols = [float(r.get("volume") or 0.0) for r in rows]
    if not prices or max(prices) == min(prices):
        return None
    lo, hi = min(prices), max(prices)
    bucket_size = (hi - lo) / n_buckets
    buckets = np.zeros(n_buckets)
    for p, v in zip(prices, vols):
        idx = int((p - lo) / (hi - lo) * (n_buckets - 1))
        buckets[idx] += v
    poc_idx = int(np.argmax(buckets))
    return lo + poc_idx * bucket_size + bucket_size / 2


# ── Anchored VWAP ─────────────────────────────────────────────────────────────

def _anchored_vwap(rows: list[dict[str, Any]], anchor_lookback: int) -> float | None:
    """Compute VWAP anchored to the lowest close in the lookback window."""
    window = rows[-anchor_lookback:]
    if not window:
        return None
    closes = [float(r.get("close") or 0.0) for r in window]
    anchor_idx = int(np.argmin(closes))
    segment = window[anchor_idx:]
    if not segment:
        return None
    typical = [(float(r.get("high") or 0) + float(r.get("low") or 0) + float(r.get("close") or 0)) / 3
               for r in segment]
    vols = [float(r.get("volume") or 0.0) for r in segment]
    cum_pv = sum(t * v for t, v in zip(typical, vols))
    cum_v = sum(vols)
    return cum_pv / cum_v if cum_v > 0 else None


# ── Main S/R Scorer ──────────────────────────────────────────────────────────

def compute_sr_score(
    rows: list[dict[str, Any]],
) -> tuple[int, dict[str, Any], list[float], list[float]]:
    """Compute the raw support/resistance score and identify key levels.

    Args:
        rows: OHLCV rows sorted ascending by date.

    Returns:
        (raw_score [0, 100], details, support_levels, resistance_levels)
    """
    if len(rows) < PIVOT_LOOKBACK * 2 + 2:
        return 50, {"error": "insufficient_data"}, [], []

    highs = [float(r.get("high") or 0.0) for r in rows]
    lows = [float(r.get("low") or 0.0) for r in rows]
    close = float(rows[-1].get("close") or 0.0)

    sh, sl = _find_swing_pivots(highs, lows, PIVOT_LOOKBACK)
    resistance_raw = [p for p in sh if p > close]
    support_raw = [p for p in sl if p < close]

    resistance_levels = sorted(_cluster_levels(resistance_raw, PIVOT_CLUSTER_PCT))
    support_levels = sorted(_cluster_levels(support_raw, PIVOT_CLUSTER_PCT), reverse=True)

    # Add volume POC and anchored VWAP as extra reference levels
    poc = _volume_profile_poc(rows[-60:])
    avwap = _anchored_vwap(rows, VWAP_ANCHOR_LOOKBACK)

    details: dict[str, Any] = {
        "support_levels": [round(s, 1) for s in support_levels[:5]],
        "resistance_levels": [round(r, 1) for r in resistance_levels[:5]],
        "volume_poc": round(poc, 1) if poc else None,
        "anchored_vwap": round(avwap, 1) if avwap else None,
    }

    # ── 1. Proximity to nearest support (max 40 pts) ──────────────────────────
    support_pts = 0
    nearest_support = support_levels[0] if support_levels else None
    if nearest_support:
        dist_pct = (close - nearest_support) / close if close > 0 else 1.0
        if dist_pct <= SR_PROXIMITY_PCT:
            support_pts = 40      # price is at support — ideal entry
        elif dist_pct <= 0.05:
            support_pts = 25      # slightly above support
        elif dist_pct <= 0.10:
            support_pts = 12      # 5-10 % above support
        else:
            support_pts = 5       # extended above support
    details["support_proximity_pts"] = support_pts
    details["nearest_support"] = round(nearest_support, 1) if nearest_support else None

    # ── 2. Resistance clearance ahead (max 35 pts) ────────────────────────────
    resistance_pts = 35
    nearest_resistance = resistance_levels[0] if resistance_levels else None
    if nearest_resistance and close > 0:
        gap_pct = (nearest_resistance - close) / close
        if gap_pct < 0.02:
            resistance_pts = 0    # immediate resistance — block signal
        elif gap_pct < 0.05:
            resistance_pts = 10   # tight resistance ahead
        elif gap_pct < 0.10:
            resistance_pts = 22   # moderate clearance
        # else: clear path → full 35 pts
    details["resistance_clearance_pts"] = resistance_pts
    details["nearest_resistance"] = round(nearest_resistance, 1) if nearest_resistance else None

    # ── 3. Volume profile confirmation (max 25 pts) ──────────────────────────
    vp_pts = 10  # baseline
    if poc and abs(poc - close) / close <= SR_PROXIMITY_PCT:
        vp_pts = 25   # price at POC = strong volume-based support
    elif avwap and close > avwap:
        vp_pts = 18   # price above anchored VWAP = bullish
    details["volume_profile_pts"] = vp_pts

    raw = min(100, support_pts + resistance_pts + vp_pts)
    details["raw_score"] = raw

    return raw, details, support_levels, resistance_levels


# ── Entry / Stop / TP Calculator ─────────────────────────────────────────────

def compute_entry_stop_tp(
    rows: list[dict[str, Any]],
    direction: str,
    nearest_resistance: float | None = None,
    nearest_support: float | None = None,
) -> dict[str, Any]:
    """Calculate tick-aligned entry zone, stop-loss, TP1, and TP2.

    ATR-based stop placement with tick-grid rounding throughout.

    Args:
        rows: OHLCV rows sorted ascending.
        direction: "BUY" or "SELL".
        nearest_resistance: Nearest resistance level above price.
        nearest_support:    Nearest support level below price.

    Returns:
        Dict with entry_low, entry_mid, entry_high, stop_loss,
        tp1, tp2, risk_per_share, risk_reward_ratio.
    """
    last = rows[-1]
    close = float(last.get("close") or 0.0)
    atr_raw = last.get("atr_14")
    atr = float(atr_raw) if atr_raw is not None else close * 0.015

    buffer = close * ENTRY_BUFFER_PCT
    entry_low = align_to_tick(close - buffer)
    entry_high = align_to_tick(close + buffer)
    entry_mid = align_to_tick((entry_low + entry_high) / 2.0)

    risk = atr * STOP_ATR_MULTIPLIER

    if direction == "BUY":
        stop_loss = align_to_tick(entry_mid - risk)
        tp1 = align_to_tick(entry_mid + risk * TP1_RR_MULTIPLIER)
        tp2 = align_to_tick(entry_mid + risk * TP2_RR_MULTIPLIER)
        # Cap TP1 just below nearest resistance if relevant
        if nearest_resistance and tp1 > nearest_resistance:
            tp1 = align_to_tick(nearest_resistance * 0.99)
    else:  # SELL
        stop_loss = align_to_tick(entry_mid + risk)
        tp1 = align_to_tick(entry_mid - risk * TP1_RR_MULTIPLIER)
        tp2 = align_to_tick(entry_mid - risk * TP2_RR_MULTIPLIER)
        # Floor TP1 just above nearest support if relevant
        if nearest_support and tp1 < nearest_support:
            tp1 = align_to_tick(nearest_support * 1.01)

    actual_risk = abs(entry_mid - stop_loss)
    actual_reward = abs(tp1 - entry_mid)
    rr = round(actual_reward / actual_risk, 2) if actual_risk > 0 else 0.0

    return {
        "entry_low": entry_low,
        "entry_mid": entry_mid,
        "entry_high": entry_high,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "risk_per_share": round(actual_risk, 1),
        "risk_reward_ratio": rr,
    }

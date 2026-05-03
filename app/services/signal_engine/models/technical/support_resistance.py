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
    TP3_RR_MULTIPLIER,
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


# ── Multi-method TP calculator ────────────────────────────────────────────────

def _fib_target(
    rows: list[dict[str, Any]],
    direction: str,
    ratio: float,
    lookback: int = 60,
) -> float | None:
    """Fibonacci extension target from recent swing."""
    window = rows[-lookback:]
    highs = [float(r.get("high") or 0.0) for r in window]
    lows = [float(l.get("low") or 0.0) for l in window]
    if not highs or not lows:
        return None
    swing_high = max(highs)
    swing_low = min(l for l in lows if l > 0)
    diff = swing_high - swing_low
    if diff <= 0:
        return None
    if direction == "BUY":
        return align_to_tick(swing_low + diff * ratio)
    else:
        return align_to_tick(swing_high - diff * ratio)


def _atr_target(
    entry_mid: float,
    atr: float,
    direction: str,
    multiplier: float,
) -> float:
    if direction == "BUY":
        return align_to_tick(entry_mid + atr * multiplier)
    return align_to_tick(entry_mid - atr * multiplier)


def _psych_target(
    entry_mid: float,
    direction: str,
    n: int = 1,
) -> float | None:
    """Nth round-number level above (BUY) or below (SELL) entry."""
    if entry_mid <= 0:
        return None
    if entry_mid < 20:
        step = 1.0
    elif entry_mid < 100:
        step = 5.0
    elif entry_mid < 500:
        step = 10.0
    elif entry_mid < 1000:
        step = 25.0
    else:
        step = 50.0
    base = (int(entry_mid / step) + (1 if direction == "BUY" else 0)) * step
    if direction == "BUY":
        return align_to_tick(base + (n - 1) * step)
    else:
        return align_to_tick(base - n * step)


def _52w_extreme(rows: list[dict[str, Any]], direction: str) -> float | None:
    window = rows[-252:]
    if direction == "BUY":
        vals = [float(r.get("high") or 0.0) for r in window]
        return align_to_tick(max(vals)) if vals else None
    else:
        vals = [float(r.get("low") or 0.0) for r in window if (r.get("low") or 0) > 0]
        return align_to_tick(min(vals)) if vals else None


def _median_of_valid(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None and v > 0]
    if not valid:
        return None
    return float(np.median(valid))


def _confluence_count(values: list[float | None], median: float, tolerance: float) -> int:
    valid = [v for v in values if v is not None and v > 0]
    if median <= 0:
        return 0
    return sum(1 for v in valid if abs(v - median) / median <= tolerance)


def compute_tp_methods(
    rows: list[dict[str, Any]],
    direction: str,
    entry_mid: float,
    stop_loss: float,
    volume_profile: dict[str, Any] | None = None,
    nearest_sr: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate TP1, TP2, TP3 using multi-method confluence.

    Each TP uses 4-5 independent methods; final value = median of valid methods.
    Confluence count = number of methods within tolerance of the median.

    Args:
        rows:           OHLCV rows sorted ascending.
        direction:      "BUY" or "SELL".
        entry_mid:      Entry mid-price in fils.
        stop_loss:      Stop-loss price in fils.
        volume_profile: Output of calculate_volume_profile() (optional).
        nearest_sr:     Dict with nearest_resistance / nearest_support keys.

    Returns:
        {
            "tp1": float, "tp1_methods": {...}, "tp1_confluence": int,
            "tp2": float, "tp2_methods": {...}, "tp2_confluence": int,
            "tp3": float, "tp3_methods": {...}, "tp3_confluence": int,
            "risk_per_share": float, "risk_reward_ratio": float,
        }
    """
    if not rows or entry_mid <= 0:
        return {}

    last = rows[-1]
    atr_raw = last.get("atr_14")
    atr = float(atr_raw) if atr_raw is not None else entry_mid * 0.015
    risk = abs(entry_mid - stop_loss) if stop_loss > 0 else atr * STOP_ATR_MULTIPLIER
    vp = volume_profile or {}

    # ── TP1 (conservative — 4 methods) ───────────────────────────────────────
    tp1_rr = align_to_tick(entry_mid + risk * TP1_RR_MULTIPLIER) if direction == "BUY" else align_to_tick(entry_mid - risk * TP1_RR_MULTIPLIER)
    tp1_fib = _fib_target(rows, direction, 1.272)
    tp1_atr = _atr_target(entry_mid, atr, direction, 1.5)
    tp1_hvn: float | None = None
    hvns = [p for p in vp.get("hvn_levels", []) if (p > entry_mid if direction == "BUY" else p < entry_mid)]
    if hvns:
        tp1_hvn = align_to_tick(min(hvns) if direction == "BUY" else max(hvns))

    tp1_vals = [tp1_rr, tp1_fib, tp1_atr, tp1_hvn]
    tp1_median = _median_of_valid(tp1_vals) or tp1_rr
    tp1_conf = _confluence_count(tp1_vals, tp1_median, 0.02)

    # Cap TP1 below nearest resistance for BUY
    if direction == "BUY" and nearest_sr:
        nr = nearest_sr.get("nearest_resistance")
        if nr and tp1_median >= nr:
            tp1_median = align_to_tick(nr * 0.99)
    elif direction == "SELL" and nearest_sr:
        ns = nearest_sr.get("nearest_support")
        if ns and tp1_median <= ns:
            tp1_median = align_to_tick(ns * 1.01)

    # ── TP2 (moderate — 5 methods) ────────────────────────────────────────────
    tp2_rr = align_to_tick(entry_mid + risk * TP2_RR_MULTIPLIER) if direction == "BUY" else align_to_tick(entry_mid - risk * TP2_RR_MULTIPLIER)
    tp2_fib = _fib_target(rows, direction, 1.618)
    tp2_atr = _atr_target(entry_mid, atr, direction, 2.5)
    tp2_poc: float | None = None
    poc = vp.get("poc")
    if poc and (poc > entry_mid if direction == "BUY" else poc < entry_mid):
        tp2_poc = align_to_tick(poc)
    tp2_swing = _fib_target(rows, direction, 1.0)  # full swing re-test

    tp2_vals = [tp2_rr, tp2_fib, tp2_atr, tp2_poc, tp2_swing]
    tp2_median = _median_of_valid(tp2_vals) or tp2_rr
    tp2_conf = _confluence_count(tp2_vals, tp2_median, 0.03)

    # ── TP3 (aggressive — 5 methods) ─────────────────────────────────────────
    tp3_rr = align_to_tick(entry_mid + risk * TP3_RR_MULTIPLIER) if direction == "BUY" else align_to_tick(entry_mid - risk * TP3_RR_MULTIPLIER)
    tp3_fib = _fib_target(rows, direction, 2.618)
    tp3_atr = _atr_target(entry_mid, atr, direction, 4.0)
    tp3_psych = _psych_target(entry_mid, direction, n=2)
    tp3_52w = _52w_extreme(rows, direction)

    tp3_vals = [tp3_rr, tp3_fib, tp3_atr, tp3_psych, tp3_52w]
    tp3_median = _median_of_valid(tp3_vals) or tp3_rr
    tp3_conf = _confluence_count(tp3_vals, tp3_median, 0.05)

    # Ensure TP3 > TP2 > TP1 (BUY) or TP3 < TP2 < TP1 (SELL)
    if direction == "BUY":
        tp2_median = max(tp2_median, tp1_median * 1.01)
        tp3_median = max(tp3_median, tp2_median * 1.01)
    else:
        tp2_median = min(tp2_median, tp1_median * 0.99)
        tp3_median = min(tp3_median, tp2_median * 0.99)

    actual_risk = risk
    rr = round(abs(tp1_median - entry_mid) / actual_risk, 2) if actual_risk > 0 else 0.0

    return {
        "tp1": round(tp1_median, 1),
        "tp1_methods": {
            "rr_1_5x": round(tp1_rr, 1),
            "fib_127": round(tp1_fib, 1) if tp1_fib else None,
            "atr_1_5x": round(tp1_atr, 1),
            "hvn_nearest": round(tp1_hvn, 1) if tp1_hvn else None,
        },
        "tp1_confluence": tp1_conf,
        "tp2": round(tp2_median, 1),
        "tp2_methods": {
            "rr_3_0x": round(tp2_rr, 1),
            "fib_161": round(tp2_fib, 1) if tp2_fib else None,
            "atr_2_5x": round(tp2_atr, 1),
            "volume_poc": round(tp2_poc, 1) if tp2_poc else None,
            "swing_retest": round(tp2_swing, 1) if tp2_swing else None,
        },
        "tp2_confluence": tp2_conf,
        "tp3": round(tp3_median, 1),
        "tp3_methods": {
            "rr_4_0x": round(tp3_rr, 1),
            "fib_261": round(tp3_fib, 1) if tp3_fib else None,
            "atr_4_0x": round(tp3_atr, 1),
            "psychological": round(tp3_psych, 1) if tp3_psych else None,
            "fifty_two_week": round(tp3_52w, 1) if tp3_52w else None,
        },
        "tp3_confluence": tp3_conf,
        "risk_per_share": round(actual_risk, 1),
        "risk_reward_ratio": rr,
    }

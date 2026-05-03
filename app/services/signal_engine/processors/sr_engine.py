"""Enhanced Support/Resistance Engine — Kuwait Signal Engine.

Produces rich S/R level metadata beyond the basic swing-pivot scorer:
  - Swing highs/lows (with volume-strength classification)
  - Fibonacci retracement/extension levels
  - Classic pivot points (daily: PP, R1-R3, S1-S3)
  - Moving average proximity levels (EMA20, EMA50, SMA100)
  - Psychological round-number levels (Kuwait fils grid)
  - Volume HVN/LVN levels from volume profile

Each level carries:
  {price, type, strength, strength_score, volume_cluster, distance_from_entry_pct}

Public API:
  calculate_full_sr_levels(rows, volume_profile, entry_price) -> dict
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.services.signal_engine.config.kuwait_constants import align_to_tick
from app.services.signal_engine.config.model_params import PIVOT_CLUSTER_PCT, PIVOT_LOOKBACK

# ── Strength score map ────────────────────────────────────────────────────────
_STRENGTH_SCORE: dict[str, int] = {
    "very_strong": 90,
    "strong": 70,
    "moderate": 50,
    "weak": 30,
}


def _level(
    price: float,
    level_type: str,
    strength: str,
    volume_cluster: bool = False,
    distance_pct: float = 0.0,
) -> dict[str, Any]:
    return {
        "price": round(price, 1),
        "type": level_type,
        "strength": strength,
        "strength_score": _STRENGTH_SCORE.get(strength, 50),
        "volume_cluster": volume_cluster,
        "distance_from_entry_pct": round(distance_pct * 100, 2),
    }


def _cluster_and_merge(
    levels: list[dict[str, Any]],
    tolerance: float = 0.005,
) -> list[dict[str, Any]]:
    """Merge levels within tolerance% of each other — keep highest strength."""
    if not levels:
        return []
    levels_sorted = sorted(levels, key=lambda x: x["price"])
    merged: list[dict[str, Any]] = [levels_sorted[0]]
    for lv in levels_sorted[1:]:
        ref = merged[-1]["price"]
        if ref > 0 and abs(lv["price"] - ref) / ref <= tolerance:
            # Keep the one with higher strength_score
            if lv["strength_score"] > merged[-1]["strength_score"]:
                merged[-1] = lv
        else:
            merged.append(lv)
    return merged


# ── Swing pivot levels ────────────────────────────────────────────────────────

def _swing_levels(
    rows: list[dict[str, Any]],
    entry_price: float,
    avg_volume: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    highs = [float(r.get("high") or 0.0) for r in rows]
    lows = [float(r.get("low") or 0.0) for r in rows]
    vols = [float(r.get("volume") or 0.0) for r in rows]
    n = len(highs)
    lb = PIVOT_LOOKBACK

    resistance: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []

    for i in range(lb, n - lb):
        h = highs[i]
        l = lows[i]
        v = vols[i]
        vol_strong = v > avg_volume * 1.5

        if h == max(highs[max(0, i - lb): i + lb + 1]) and h > entry_price:
            strength = "strong" if vol_strong else "moderate"
            d = (h - entry_price) / entry_price if entry_price > 0 else 0.0
            resistance.append(_level(h, "Swing High", strength, vol_strong, d))

        if l == min(lows[max(0, i - lb): i + lb + 1]) and l < entry_price:
            strength = "strong" if vol_strong else "moderate"
            d = (entry_price - l) / entry_price if entry_price > 0 else 0.0
            support.append(_level(l, "Swing Low", strength, vol_strong, d))

    return resistance, support


# ── Fibonacci levels ──────────────────────────────────────────────────────────
_FIB_RATIOS = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618, 2.618]
_FIB_STRENGTH: dict[float, str] = {
    0.618: "strong", 0.382: "strong",
    0.500: "moderate", 0.786: "moderate",
    0.236: "weak", 1.000: "moderate",
    1.272: "moderate", 1.618: "strong", 2.618: "moderate",
}


def _fibonacci_levels(
    rows: list[dict[str, Any]],
    entry_price: float,
    lookback: int = 60,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    window = rows[-lookback:]
    highs = [float(r.get("high") or 0.0) for r in window]
    lows = [float(r.get("low") or 0.0) for r in window]
    if not highs or not lows:
        return [], []

    swing_high = max(highs)
    swing_low = min(l for l in lows if l > 0)
    diff = swing_high - swing_low
    if diff <= 0:
        return [], []

    resistance: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []

    for ratio in _FIB_RATIOS:
        # Retracement (from swing high down)
        retrace = swing_high - diff * ratio
        strength = _FIB_STRENGTH.get(ratio, "weak")
        if retrace > entry_price:
            d = (retrace - entry_price) / entry_price if entry_price > 0 else 0.0
            resistance.append(_level(retrace, f"Fib {ratio:.3f}", strength, False, d))
        elif retrace < entry_price:
            d = (entry_price - retrace) / entry_price if entry_price > 0 else 0.0
            support.append(_level(retrace, f"Fib {ratio:.3f}", strength, False, d))

        # Extension above swing high
        if ratio > 1.0:
            ext = swing_low + diff * ratio
            if ext > entry_price:
                d = (ext - entry_price) / entry_price if entry_price > 0 else 0.0
                resistance.append(_level(ext, f"Fib Ext {ratio:.3f}", strength, False, d))

    return resistance, support


# ── Classic pivot points ──────────────────────────────────────────────────────

def _pivot_levels(
    rows: list[dict[str, Any]],
    entry_price: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Use previous session (second-to-last bar)
    if len(rows) < 2:
        return [], []
    prev = rows[-2]
    high = float(prev.get("high") or 0.0)
    low = float(prev.get("low") or 0.0)
    close = float(prev.get("close") or 0.0)
    if high <= 0 or low <= 0 or close <= 0:
        return [], []

    pp = (high + low + close) / 3.0
    r1 = 2 * pp - low
    r2 = pp + (high - low)
    r3 = high + 2 * (pp - low)
    s1 = 2 * pp - high
    s2 = pp - (high - low)
    s3 = low - 2 * (high - pp)

    resistance: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []

    pivot_res = [(r1, "Pivot R1", "moderate"), (r2, "Pivot R2", "strong"), (r3, "Pivot R3", "very_strong")]
    pivot_sup = [(s1, "Pivot S1", "moderate"), (s2, "Pivot S2", "strong"), (s3, "Pivot S3", "very_strong")]

    for price, ptype, strength in pivot_res:
        if price > entry_price:
            d = (price - entry_price) / entry_price if entry_price > 0 else 0.0
            resistance.append(_level(price, ptype, strength, False, d))
    for price, ptype, strength in pivot_sup:
        if price < entry_price and price > 0:
            d = (entry_price - price) / entry_price if entry_price > 0 else 0.0
            support.append(_level(price, ptype, strength, False, d))

    return resistance, support


# ── Moving average levels ─────────────────────────────────────────────────────

def _ma_levels(
    rows: list[dict[str, Any]],
    entry_price: float,
    proximity_pct: float = 0.05,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    last = rows[-1]
    ma_map = {
        "EMA 20":  last.get("ema_20"),
        "EMA 50":  last.get("ema_50"),
        "SMA 100": last.get("sma_100"),
    }
    resistance: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []

    for name, val in ma_map.items():
        if val is None:
            continue
        price = float(val)
        if price <= 0:
            continue
        dist = abs(price - entry_price) / entry_price if entry_price > 0 else 1.0
        if dist > proximity_pct:
            continue  # too far away to be relevant
        strength = "moderate" if dist < 0.03 else "weak"
        d = (price - entry_price) / entry_price if entry_price > 0 else 0.0
        if price > entry_price:
            resistance.append(_level(price, name, strength, False, abs(d)))
        else:
            support.append(_level(price, name, strength, False, abs(d)))

    return resistance, support


# ── Psychological round-number levels ─────────────────────────────────────────

def _psychological_levels(
    entry_price: float,
    n_levels: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate round-number levels (Kuwait fils — grid depends on price tier)."""
    if entry_price <= 0:
        return [], []

    # Round to the nearest sensible grid
    if entry_price < 20:
        step = 1.0
    elif entry_price < 100:
        step = 5.0
    elif entry_price < 500:
        step = 10.0
    elif entry_price < 1000:
        step = 25.0
    else:
        step = 50.0

    base = round(entry_price / step) * step
    resistance: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []

    for i in range(1, n_levels + 1):
        above = base + i * step
        below = base - i * step
        d_above = (above - entry_price) / entry_price if entry_price > 0 else 0.0
        d_below = (entry_price - below) / entry_price if entry_price > 0 else 0.0
        if above > entry_price:
            resistance.append(_level(above, "Psychological", "moderate", False, d_above))
        if below > 0 and below < entry_price:
            support.append(_level(below, "Psychological", "moderate", False, d_below))

    return resistance, support


# ── Volume HVN/LVN levels ─────────────────────────────────────────────────────

def _volume_levels(
    volume_profile: dict[str, Any],
    entry_price: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resistance: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []

    poc = volume_profile.get("poc")
    if poc:
        d = abs(poc - entry_price) / entry_price if entry_price > 0 else 0.0
        if poc > entry_price:
            resistance.append(_level(poc, "Volume POC", "strong", True, d))
        elif poc < entry_price:
            support.append(_level(poc, "Volume POC", "strong", True, d))

    for price in volume_profile.get("hvn_levels", []):
        d = abs(price - entry_price) / entry_price if entry_price > 0 else 0.0
        if price > entry_price:
            resistance.append(_level(price, "HVN", "strong", True, d))
        else:
            support.append(_level(price, "HVN", "strong", True, d))

    for price in volume_profile.get("lvn_levels", []):
        d = abs(price - entry_price) / entry_price if entry_price > 0 else 0.0
        # LVNs act as price gaps — resistance if above, support if below
        if price > entry_price:
            resistance.append(_level(price, "LVN", "weak", False, d))
        else:
            support.append(_level(price, "LVN", "weak", False, d))

    return resistance, support


# ── Main public function ──────────────────────────────────────────────────────

def calculate_full_sr_levels(
    rows: list[dict[str, Any]],
    volume_profile: dict[str, Any],
    entry_price: float,
    max_levels: int = 6,
) -> dict[str, Any]:
    """Build rich support/resistance level lists for the UI S/R map.

    Args:
        rows:           OHLCV rows sorted ascending.
        volume_profile: Output of calculate_volume_profile().
        entry_price:    Entry mid-price in fils.
        max_levels:     Max levels to return per side after merging.

    Returns:
        {
          "resistance": [level, ...],   # sorted ascending (nearest first)
          "support":    [level, ...],   # sorted descending (nearest first)
          "nearest_resistance": float | None,
          "nearest_support":    float | None,
        }
    """
    if not rows or entry_price <= 0:
        return {"resistance": [], "support": [], "nearest_resistance": None, "nearest_support": None}

    vols = [float(r.get("volume") or 0.0) for r in rows]
    avg_volume = float(np.mean(vols)) if vols else 1.0

    all_resistance: list[dict[str, Any]] = []
    all_support: list[dict[str, Any]] = []

    for fn in [
        lambda: _swing_levels(rows, entry_price, avg_volume),
        lambda: _fibonacci_levels(rows, entry_price),
        lambda: _pivot_levels(rows, entry_price),
        lambda: _ma_levels(rows, entry_price),
        lambda: _psychological_levels(entry_price),
        lambda: _volume_levels(volume_profile, entry_price),
    ]:
        r, s = fn()
        all_resistance.extend(r)
        all_support.extend(s)

    # Merge close levels and sort
    merged_res = _cluster_and_merge(all_resistance, tolerance=PIVOT_CLUSTER_PCT)
    merged_sup = _cluster_and_merge(all_support, tolerance=PIVOT_CLUSTER_PCT)

    # Sort: resistance ascending (nearest first), support descending (nearest first)
    merged_res.sort(key=lambda x: x["price"])
    merged_sup.sort(key=lambda x: x["price"], reverse=True)

    # Cap
    merged_res = merged_res[:max_levels]
    merged_sup = merged_sup[:max_levels]

    return {
        "resistance": merged_res,
        "support": merged_sup,
        "nearest_resistance": merged_res[0]["price"] if merged_res else None,
        "nearest_support": merged_sup[0]["price"] if merged_sup else None,
    }

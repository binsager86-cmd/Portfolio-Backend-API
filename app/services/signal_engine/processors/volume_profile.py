"""Volume Profile Calculator — Kuwait Signal Engine.

Calculates:
  - Point of Control (POC): price bucket with highest accumulated volume
  - High Volume Nodes (HVN): volume peaks above threshold × average
  - Low Volume Nodes (LVN): volume valleys below threshold × average
  - Value Area (70% of volume centred on POC)
  - Full volume-by-price dict for downstream use
"""
from __future__ import annotations

from typing import Any

import numpy as np


def calculate_volume_profile(
    rows: list[dict[str, Any]],
    n_buckets: int = 40,
    hvn_threshold: float = 1.5,
    lvn_threshold: float = 0.5,
    value_area_pct: float = 70.0,
) -> dict[str, Any]:
    """Compute volume profile from OHLCV rows.

    Args:
        rows:            OHLCV rows sorted ascending. Uses last 60 rows.
        n_buckets:       Number of price bins.
        hvn_threshold:   Multiplier above average volume to qualify as HVN.
        lvn_threshold:   Multiplier below average volume to qualify as LVN.
        value_area_pct:  Percentage of total volume that defines the value area.

    Returns:
        {
            "poc": float,
            "hvn_levels": list[float],
            "lvn_levels": list[float],
            "value_area_high": float,
            "value_area_low": float,
            "volume_by_price": dict[float, float],
        }
        or empty dict on insufficient data.
    """
    window = rows[-60:] if len(rows) >= 60 else rows
    if len(window) < 5:
        return {}

    # Build price→volume map distributing each bar's volume across its range
    prices_flat: list[float] = []
    for r in window:
        lo = float(r.get("low") or 0.0)
        hi = float(r.get("high") or 0.0)
        vol = float(r.get("volume") or 0.0)
        close = float(r.get("close") or 0.0)
        if lo <= 0 or hi <= lo:
            prices_flat.append(close)
        else:
            # Use 5 interpolated points across the bar
            for p in np.linspace(lo, hi, 5):
                prices_flat.append(float(p))

    if not prices_flat:
        return {}

    all_lows = [float(r.get("low") or r.get("close") or 0.0) for r in window]
    all_highs = [float(r.get("high") or r.get("close") or 0.0) for r in window]
    global_lo = min(p for p in all_lows if p > 0)
    global_hi = max(p for p in all_highs if p > 0)

    if global_hi <= global_lo:
        return {}

    bucket_size = (global_hi - global_lo) / n_buckets
    buckets: dict[int, float] = {i: 0.0 for i in range(n_buckets)}

    for r in window:
        lo = float(r.get("low") or 0.0)
        hi = float(r.get("high") or 0.0)
        vol = float(r.get("volume") or 0.0)
        close = float(r.get("close") or 0.0)
        if lo <= 0:
            lo = hi = close
        if hi <= lo:
            hi = lo

        # Spread bar volume across buckets it touches
        lo_bin = max(0, int((lo - global_lo) / bucket_size))
        hi_bin = min(n_buckets - 1, int((hi - global_lo) / bucket_size))
        n_bins = hi_bin - lo_bin + 1
        vol_per_bin = vol / n_bins if n_bins > 0 else vol
        for b in range(lo_bin, hi_bin + 1):
            buckets[b] += vol_per_bin

    # Convert bucket indices to mid-prices
    volume_by_price: dict[float, float] = {}
    for b, v in buckets.items():
        mid_price = round(global_lo + (b + 0.5) * bucket_size, 2)
        volume_by_price[mid_price] = v

    if not volume_by_price:
        return {}

    # POC — bucket with highest volume
    poc = max(volume_by_price, key=lambda p: volume_by_price[p])

    avg_vol = float(np.mean(list(volume_by_price.values())))
    sorted_prices = sorted(volume_by_price.keys())

    # HVN — local volume peaks above hvn_threshold × average
    hvn_levels: list[float] = []
    for i, price in enumerate(sorted_prices[1:-1], 1):
        v = volume_by_price[price]
        prev_v = volume_by_price[sorted_prices[i - 1]]
        next_v = volume_by_price[sorted_prices[i + 1]]
        if v >= prev_v and v >= next_v and v >= avg_vol * hvn_threshold:
            hvn_levels.append(price)

    # LVN — local volume valleys below lvn_threshold × average
    lvn_levels: list[float] = []
    for i, price in enumerate(sorted_prices[1:-1], 1):
        v = volume_by_price[price]
        prev_v = volume_by_price[sorted_prices[i - 1]]
        next_v = volume_by_price[sorted_prices[i + 1]]
        if v <= prev_v and v <= next_v and v <= avg_vol * lvn_threshold:
            lvn_levels.append(price)

    # Value Area — 70% of total volume centred on POC
    total_vol = sum(volume_by_price.values())
    target_vol = total_vol * (value_area_pct / 100.0)
    va_prices = sorted(volume_by_price.keys(), key=lambda p: volume_by_price[p], reverse=True)
    accumulated = 0.0
    value_area_prices: list[float] = []
    for price in va_prices:
        accumulated += volume_by_price[price]
        value_area_prices.append(price)
        if accumulated >= target_vol:
            break

    return {
        "poc": round(poc, 2),
        "hvn_levels": [round(p, 2) for p in sorted(hvn_levels)],
        "lvn_levels": [round(p, 2) for p in sorted(lvn_levels)],
        "value_area_high": round(max(value_area_prices), 2) if value_area_prices else round(global_hi, 2),
        "value_area_low": round(min(value_area_prices), 2) if value_area_prices else round(global_lo, 2),
        "volume_by_price": {round(k, 2): round(v, 1) for k, v in volume_by_price.items()},
    }

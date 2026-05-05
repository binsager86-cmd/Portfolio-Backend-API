"""Volume and money-flow scorer for the Kuwait Signal Engine.

Components: CMF(20), OBV slope, RVOL (Relative Volume), Auction Intensity.

Option A weighting (Kuwait Block-Trade optimised):
  CMF  35% — bar-by-bar flow; captures institutional absorption vs distribution
  OBV  25% — trend alignment filter; reduced from 35% to prevent cumulative lag
  RVOL 25% — current / 20-day median volume; filters low-volume traps
  Auction 15% — closing-auction block execution; confirmation only

Replaces A/D Line (was 20%) with RVOL to eliminate OBV/A-D redundancy (~0.90 corr).

Raw score [0, 100]:
  > 60 → bullish accumulation
  40-60 → neutral
  < 40 → distribution / selling pressure

Pre-computed indicators expected in rows:
  obv, cmf_20
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.services.signal_engine.config.model_params import OBV_SLOPE_BARS


def _obv_score(rows: list[dict[str, Any]]) -> tuple[int, str]:
    """Score OBV trend via linear-regression slope (max 25 pts)."""
    if len(rows) < OBV_SLOPE_BARS + 1:
        return 12, "obv_insufficient_data"

    recent = rows[-(OBV_SLOPE_BARS + 1):]
    obvs = [r.get("obv") for r in recent]
    if any(v is None for v in obvs):
        return 12, "obv_missing"

    vals = np.array([float(v) for v in obvs])
    x = np.arange(len(vals), dtype=float)
    y_mean = vals.mean()
    if y_mean == 0:
        return 12, "obv_zero"
    slope, _ = np.polyfit(x, vals, 1)
    slope_pct = slope / abs(y_mean) * 100.0

    if slope_pct > 1.5:
        return 25, f"obv_strongly_rising_{slope_pct:.1f}pct_per_bar"
    if slope_pct > 0.3:
        return 18, f"obv_rising_{slope_pct:.1f}pct_per_bar"
    if slope_pct > -0.3:
        return 12, "obv_flat"
    if slope_pct > -1.5:
        return 6, f"obv_declining_{slope_pct:.1f}pct_per_bar"
    return 0, f"obv_strongly_declining_{slope_pct:.1f}pct_per_bar"


def _cmf_score(last: dict[str, Any]) -> tuple[int, str]:
    """Score Chaikin Money Flow (max 35 pts)."""
    cmf = last.get("cmf_20")
    if cmf is None:
        return 14, "cmf_missing"
    v = float(cmf)
    if v > 0.20:
        return 35, f"strong_accumulation_cmf_{v:.3f}"
    if v > 0.10:
        return 28, f"accumulation_cmf_{v:.3f}"
    if v > 0.03:
        return 20, f"mild_accumulation_cmf_{v:.3f}"
    if v > -0.03:
        return 14, f"neutral_cmf_{v:.3f}"
    if v > -0.10:
        return 7, f"mild_distribution_cmf_{v:.3f}"
    if v > -0.20:
        return 3, f"distribution_cmf_{v:.3f}"
    return 0, f"strong_distribution_cmf_{v:.3f}"


def _rvol_score(rows: list[dict[str, Any]]) -> tuple[int, str]:
    """Relative Volume confirmation (max 25 pts).

    Filters low-volume traps and confirms institutional participation.
    RVOL = current_volume / 20-day median volume.
    """
    if len(rows) < 21:
        return 12, "rvol_insufficient_data"

    volumes = [float(r.get("volume") or 0.0) for r in rows]
    current_vol = volumes[-1]
    median_vol = float(np.median(volumes[:-1]))  # exclude current day

    if median_vol <= 0:
        return 12, "rvol_zero_median"

    rvol = current_vol / median_vol

    if rvol >= 2.0:
        return 25, f"exceptional_volume_rvol_{rvol:.1f}x"
    if rvol >= 1.5:
        return 20, f"strong_volume_rvol_{rvol:.1f}x"
    if rvol >= 1.2:
        return 15, f"above_average_rvol_{rvol:.1f}x"
    if rvol >= 0.8:
        return 10, f"normal_volume_rvol_{rvol:.1f}x"
    if rvol >= 0.5:
        return 5, f"low_volume_rvol_{rvol:.1f}x"
    return 0, f"thin_volume_rvol_{rvol:.1f}x"


def _auction_score(intensity: float) -> tuple[int, str]:
    """Score auction intensity proxy (max 15 pts)."""
    if intensity > 1.8:
        return 15, f"high_institutional_auction_{intensity:.2f}"
    if intensity >= 1.0:
        return 10, f"normal_auction_{intensity:.2f}"
    return 3, f"low_institutional_auction_{intensity:.2f}"


def _orderbook_adjustment(ob_data: dict[str, Any] | None) -> tuple[int, str]:
    """Order book imbalance adjustment (±10 pts, +5 liquidity wall bonus).

    Args:
        ob_data: Dict with keys ``imbalance_ratio`` ∈ [-1,+1] and optional
                 ``liquidity_wall`` sub-dict.  Pass None when OB is unavailable.

    Returns:
        (adjustment, description) — adjustment capped at [-10, +15].
    """
    if not ob_data:
        return 0, "orderbook_unavailable"

    ratio = float(ob_data.get("imbalance_ratio") or 0.0)
    wall = ob_data.get("liquidity_wall")

    if ratio > 0.3:
        adj, desc = +10, f"strong_bid_pressure_ob_{ratio:.2f}"
    elif ratio < -0.3:
        adj, desc = -10, f"strong_ask_pressure_ob_{ratio:.2f}"
    else:
        adj, desc = 0, f"balanced_ob_{ratio:.2f}"

    # Liquidity wall adds +5 (confirms direction; total capped at +15)
    if wall:
        adj = min(+15, adj + 5)
        desc += f"+wall_{wall.get('side', '')}@{wall.get('price', '')}"

    return adj, desc


def compute_volume_flow_score(
    rows: list[dict[str, Any]],
    auction_intensity: float,
    orderbook_imbalance: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Compute the raw volume/flow score and component breakdown.

    Args:
        rows: OHLCV + indicator rows sorted ascending by date.
        auction_intensity: Pre-computed auction intensity from auction_proxy or real OB.
        orderbook_imbalance: Optional dict with ``imbalance_ratio`` and ``liquidity_wall``
                             from order book analysis.  Pass None when OB is unavailable.

    Returns:
        Tuple of (raw_score: int [0, 100], details: dict).

    Scoring (max 100 base):
        CMF(20)    : 35 pts  — primary flow signal
        OBV slope  : 25 pts  — trend alignment
        RVOL       : 25 pts  — breakout confirmation (replaces A/D Line)
        Auction    : 15 pts  — closing-auction block execution
        OB adjust  : ±10 pts (+5 wall bonus, total ±15)
    """
    if not rows:
        return 50, {"error": "no_rows"}

    last = rows[-1]

    cmf_pts, cmf_desc = _cmf_score(last)
    obv_pts, obv_desc = _obv_score(rows)
    rvol_pts, rvol_desc = _rvol_score(rows)
    auc_pts, auc_desc = _auction_score(auction_intensity)
    ob_adj, ob_desc = _orderbook_adjustment(orderbook_imbalance)

    raw = min(100, max(0, cmf_pts + obv_pts + rvol_pts + auc_pts + ob_adj))

    details = {
        "cmf_pts": cmf_pts,
        "cmf_desc": cmf_desc,
        "obv_pts": obv_pts,
        "obv_desc": obv_desc,
        "rvol_pts": rvol_pts,
        "rvol_desc": rvol_desc,
        "auction_pts": auc_pts,
        "auction_desc": auc_desc,
        "orderbook_adjustment": ob_adj,
        "orderbook_desc": ob_desc,
        "auction_intensity": auction_intensity,
        "orderbook_imbalance": orderbook_imbalance,
        "raw_score": raw,
    }
    return raw, details


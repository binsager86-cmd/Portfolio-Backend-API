"""Volume and money-flow scorer for the Kuwait Signal Engine.

Components: OBV trend slope, CMF(20), and auction intensity proxy.

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
    """Score OBV trend via linear-regression slope (max 35 pts)."""
    if len(rows) < OBV_SLOPE_BARS + 1:
        return 15, "obv_insufficient_data"

    recent = rows[-(OBV_SLOPE_BARS + 1):]
    obvs = [r.get("obv") for r in recent]
    if any(v is None for v in obvs):
        return 15, "obv_missing"

    vals = np.array([float(v) for v in obvs])
    x = np.arange(len(vals), dtype=float)
    # Linear regression slope (normalised by mean OBV magnitude)
    x_mean = x.mean()
    y_mean = vals.mean()
    if y_mean == 0:
        return 15, "obv_zero"
    slope_norm = np.sum((x - x_mean) * (vals - y_mean)) / np.sum((x - x_mean) ** 2)
    slope_pct = slope_norm / abs(y_mean) * 100.0  # slope as % of mean OBV per bar

    if slope_pct > 1.5:
        return 35, f"obv_strongly_rising_{slope_pct:.1f}pct_per_bar"
    if slope_pct > 0.3:
        return 25, f"obv_rising_{slope_pct:.1f}pct_per_bar"
    if slope_pct > -0.3:
        return 15, "obv_flat"
    if slope_pct > -1.5:
        return 7, f"obv_declining_{slope_pct:.1f}pct_per_bar"
    return 0, f"obv_strongly_declining_{slope_pct:.1f}pct_per_bar"


def _cmf_score(last: dict[str, Any]) -> tuple[int, str]:
    """Score Chaikin Money Flow (max 35 pts)."""
    cmf = last.get("cmf_20")
    if cmf is None:
        return 15, "cmf_missing"
    v = float(cmf)
    if v > 0.20:
        return 35, f"strong_accumulation_cmf_{v:.3f}"
    if v > 0.10:
        return 28, f"accumulation_cmf_{v:.3f}"
    if v > 0.03:
        return 20, f"mild_accumulation_cmf_{v:.3f}"
    if v > -0.03:
        return 15, f"neutral_cmf_{v:.3f}"
    if v > -0.10:
        return 8, f"mild_distribution_cmf_{v:.3f}"
    if v > -0.20:
        return 3, f"distribution_cmf_{v:.3f}"
    return 0, f"strong_distribution_cmf_{v:.3f}"


def _auction_score(intensity: float) -> tuple[int, str]:
    """Score auction intensity proxy (max 30 pts)."""
    if intensity > 1.8:
        return 30, f"high_institutional_auction_{intensity:.2f}"
    if intensity >= 1.0:
        return 20, f"normal_auction_{intensity:.2f}"
    return 5, f"low_institutional_auction_{intensity:.2f}"


def compute_volume_flow_score(
    rows: list[dict[str, Any]],
    auction_intensity: float,
) -> tuple[int, dict[str, Any]]:
    """Compute the raw volume/flow score and component breakdown.

    Args:
        rows: OHLCV + indicator rows sorted ascending by date.
        auction_intensity: Pre-computed auction intensity from auction_proxy.

    Returns:
        Tuple of (raw_score: int [0, 100], details: dict).
    """
    if not rows:
        return 50, {"error": "no_rows"}

    last = rows[-1]

    obv_pts, obv_desc = _obv_score(rows)
    cmf_pts, cmf_desc = _cmf_score(last)
    auc_pts, auc_desc = _auction_score(auction_intensity)

    raw = min(100, obv_pts + cmf_pts + auc_pts)

    details = {
        "obv_pts": obv_pts,
        "obv_desc": obv_desc,
        "cmf_pts": cmf_pts,
        "cmf_desc": cmf_desc,
        "auction_pts": auc_pts,
        "auction_desc": auc_desc,
        "auction_intensity": auction_intensity,
        "raw_score": raw,
    }
    return raw, details

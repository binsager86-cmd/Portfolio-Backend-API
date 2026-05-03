"""Historical-simulation CVaR calculator for the Kuwait Signal Engine.

Uses the non-parametric (historical) method — sort rolling daily returns,
take the mean of the worst alpha % as Conditional Value-at-Risk.

Also exputes standard VaR for the CVaR/VaR ratio check.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from app.services.signal_engine.config.risk_config import (
    CVAR_ALPHA,
    CVAR_ILLIQUID_WIDEN_FACTOR,
    CVAR_LOOKBACK_DAYS,
    CVAR_VAR_RATIO_REDUCE_THRESHOLD,
    CVAR_VAR_REDUCE_FACTOR,
    LIQUIDITY_THRESHOLD_KD,
)


def calculate_cvar(
    rows: list[dict[str, Any]],
    alpha: float = CVAR_ALPHA,
    adtv_kd: float | None = None,
) -> dict[str, Any]:
    """Compute historical-simulation CVaR and VaR.

    Args:
        rows:    OHLCV rows sorted ascending, length ≥ 2.
        alpha:   Tail probability (default 0.05 → 95 % CVaR).
        adtv_kd: 20-day average daily traded value in KD.
                 When below LIQUIDITY_THRESHOLD_KD, CVaR is widened by
                 CVAR_ILLIQUID_WIDEN_FACTOR.

    Returns:
        Dict with keys:
          var_95:          Value-at-Risk at (1-alpha) confidence (positive = loss)
          cvar_95:         CVaR (expected shortfall) (positive = loss)
          cvar_ils:        CVaR in fils (multiply by entry price)
          is_illiquid_adj: whether the illiquidity adjustment was applied
          position_size_reduction: extra reduction factor if CVaR ≫ VaR
    """
    # Build daily return series from available history (last CVAR_LOOKBACK_DAYS)
    window = rows[-CVAR_LOOKBACK_DAYS:]
    closes = [float(r.get("close") or 0.0) for r in window]

    returns: list[float] = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            ret = (closes[i] - closes[i - 1]) / closes[i - 1]
            returns.append(ret)

    if len(returns) < 10:
        return {
            "var_95": None,
            "cvar_95": None,
            "cvar_fils": None,
            "is_illiquid_adj": False,
            "position_size_reduction": 1.0,
        }

    returns_arr = np.array(returns)
    sorted_ret = np.sort(returns_arr)    # ascending (worst first for negative)

    n = len(sorted_ret)
    cutoff_idx = max(1, int(math.floor(alpha * n)))
    tail = sorted_ret[:cutoff_idx]

    var_95 = float(-np.percentile(returns_arr, alpha * 100))     # positive loss
    cvar_95 = float(-np.mean(tail)) if len(tail) > 0 else var_95 # positive loss

    illiquid_adj = False
    if adtv_kd is not None and adtv_kd < LIQUIDITY_THRESHOLD_KD:
        cvar_95 *= CVAR_ILLIQUID_WIDEN_FACTOR
        illiquid_adj = True

    # Translate to expected fils loss on a position entry at current close
    current_close = float(rows[-1].get("close") or 0.0)
    cvar_fils = round(cvar_95 * current_close, 1) if current_close > 0 else None

    # Extra reduction factor if CVaR is disproportionately larger than VaR
    pos_reduction = 1.0
    if var_95 > 0 and cvar_95 / var_95 > CVAR_VAR_RATIO_REDUCE_THRESHOLD:
        pos_reduction = CVAR_VAR_REDUCE_FACTOR

    return {
        "var_95": round(var_95, 5),
        "cvar_95": round(cvar_95, 5),
        "cvar_fils": cvar_fils,
        "is_illiquid_adj": illiquid_adj,
        "position_size_reduction": pos_reduction,
    }

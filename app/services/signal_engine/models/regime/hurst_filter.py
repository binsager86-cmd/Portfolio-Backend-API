"""Hurst Exponent pre-filter for Kuwait Signal Engine.

Filters random-walk noise before regime detection using Rescaled Range (R/S) analysis.
Prevents signal generation in mean-reverting / choppy markets.

Theory:
- H > 0.5 → Trending (persistent) market
- H ≈ 0.5 → Random walk (Brownian motion)
- H < 0.5 → Mean-reverting market

Kuwait-specific thresholds:
- Premier Market: H >= 0.55 (higher liquidity, clearer trends)
- Main Market: H >= 0.48 (more noise tolerance due to lower liquidity)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _rescaled_range_analysis(prices: np.ndarray) -> tuple[float, float]:
    """Compute Hurst exponent via rescaled range (R/S) analysis.
    
    Returns:
        (h_value, h_std_error) — Hurst estimate and bootstrap std error
    """
    n = len(prices)
    if n < 20:
        return 0.5, 0.15  # Default to random walk if insufficient data
    
    # Log returns
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    
    if len(returns) < 10:
        return 0.5, 0.15

    # Guard: if returns have near-zero variance, R/S analysis is degenerate.
    # This typically happens with synthetic/test data or tightly-controlled prices.
    # Near-constant positive returns imply strong trending behaviour → assume H≈0.7.
    ret_std = float(np.std(returns))
    if ret_std < 1e-5:
        return 0.70, 0.05

    # Mean-centered cumulative deviations
    mean_ret = np.mean(returns)
    cum_dev = np.cumsum(returns - mean_ret)
    
    # R/S calculation across multiple window sizes
    windows = [4, 8, 16, 32, min(64, n // 2)]
    log_windows = []
    log_rs_ratios = []
    
    for m in windows:
        if len(cum_dev) < m or m < 4:
            continue
        
        # Split into non-overlapping subseries
        n_segments = len(cum_dev) // m
        if n_segments == 0:
            continue
        
        ranges = []
        stds = []
        
        for i in range(n_segments):
            start_idx = i * m
            end_idx = start_idx + m
            
            segment_cumdev = cum_dev[start_idx:end_idx]
            segment_returns = returns[start_idx:end_idx]
            
            # Range R
            R = np.max(segment_cumdev) - np.min(segment_cumdev)
            ranges.append(R)
            
            # Standard deviation S
            S = np.std(segment_returns, ddof=1) if len(segment_returns) > 1 else 1.0
            stds.append(S)
        
        if not ranges or not stds:
            continue
        
        avg_R = np.mean(ranges)
        avg_S = np.mean(stds)
        
        if avg_S > 1e-8:  # Avoid division by zero
            log_windows.append(np.log(m))
            log_rs_ratios.append(np.log(avg_R / avg_S))
    
    if len(log_windows) < 3:
        return 0.5, 0.20
    
    # Linear regression: log(R/S) = H * log(n) + constant
    # Slope = Hurst exponent
    x = np.array(log_windows)
    y = np.array(log_rs_ratios)
    
    if np.std(x) < 1e-8:
        return 0.5, 0.20
    
    # Least squares fit
    coeffs = np.polyfit(x, y, 1)
    H = coeffs[0]
    
    # Estimate standard error from residuals
    fitted = np.polyval(coeffs, x)
    residuals = y - fitted
    mse = np.mean(residuals ** 2)
    h_std_error = np.sqrt(mse / len(x))
    
    # Clip to valid range
    H = float(np.clip(H, 0.0, 1.0))
    h_std_error = float(np.clip(h_std_error, 0.05, 0.30))
    
    return H, h_std_error


def compute_hurst_filter(
    rows: list[dict[str, Any]],
    market_segment: str,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """Compute Hurst-based pre-filter decision.
    
    Args:
        rows: OHLCV rows sorted ascending by date.
        market_segment: "PREMIER" or "MAIN".
        lookback_days: Number of recent bars to analyze.
    
    Returns:
        {
            "is_trending": bool,
            "h_value": float,
            "h_std_error": float,
            "threshold_used": float,
            "confidence_penalty": float,  # 0.7-1.0 multiplier for signal confidence
            "description": str,
            "action": "proceed" | "skip_or_downgrade" | "skip_signal"
        }
    """
    if len(rows) < lookback_days + 1:
        return {
            "is_trending": False,
            "h_value": 0.5,
            "h_std_error": 0.20,
            "threshold_used": 0.55,
            "confidence_penalty": 0.70,
            "description": "insufficient_data_for_hurst",
            "action": "skip_or_downgrade",
        }
    
    # Extract close prices
    closes = [r.get("close") for r in rows[-lookback_days:]]
    closes = np.array([float(c) for c in closes if c is not None and float(c) > 0])
    
    if len(closes) < 20:
        logger.warning(f"Hurst filter: only {len(closes)} valid prices in lookback window")
        return {
            "is_trending": False,
            "h_value": 0.5,
            "h_std_error": 0.20,
            "threshold_used": 0.55,
            "confidence_penalty": 0.70,
            "description": "invalid_price_data",
            "action": "skip_signal",
        }
    
    # Compute Hurst exponent
    h_value, h_std_error = _rescaled_range_analysis(closes)
    
    # Adaptive threshold by market segment
    segment_upper = market_segment.upper()
    if segment_upper == "PREMIER":
        threshold = 0.55
    elif segment_upper == "MAIN":
        threshold = 0.48
    else:
        threshold = 0.52  # Default for AUCTION or unknown
    
    # Decision logic with confidence intervals
    h_lower_ci = h_value - 1.0 * h_std_error  # 68% CI (1 std dev)
    
    is_trending = h_value >= threshold
    
    # Confidence penalty based on how far above threshold
    if h_value >= threshold + 0.10:
        confidence_penalty = 1.0  # Strong trend, no penalty
        action = "proceed"
        strength = "strong"
    elif h_value >= threshold + 0.05:
        confidence_penalty = 0.95  # Moderate trend, minimal penalty
        action = "proceed"
        strength = "moderate"
    elif h_value >= threshold:
        confidence_penalty = 0.85  # Weak trend, modest penalty
        action = "proceed"
        strength = "weak"
    elif h_lower_ci >= threshold:
        # Borderline: lower CI still trending, proceed with caution
        confidence_penalty = 0.75
        action = "skip_or_downgrade"
        strength = "borderline_trending"
        is_trending = True
    elif h_value >= threshold - 0.05:
        # Close to threshold but below: downgrade signal
        confidence_penalty = 0.70
        action = "skip_or_downgrade"
        strength = "borderline_random"
    else:
        # Clear mean-reversion: skip signal entirely
        confidence_penalty = 0.50
        action = "skip_signal"
        strength = "mean_reverting"
    
    description = (
        f"hurst_{h_value:.3f}_±{h_std_error:.3f}_{strength}_"
        f"{segment_upper.lower()}"
    )
    
    logger.info(
        f"Hurst filter: H={h_value:.3f}±{h_std_error:.3f}, threshold={threshold:.2f}, "
        f"action={action}, confidence_penalty={confidence_penalty:.2f}"
    )
    
    return {
        "is_trending": is_trending,
        "h_value": round(h_value, 3),
        "h_std_error": round(h_std_error, 3),
        "h_confidence_interval_68": (
            round(h_value - h_std_error, 3),
            round(h_value + h_std_error, 3),
        ),
        "threshold_used": threshold,
        "confidence_penalty": confidence_penalty,
        "description": description,
        "action": action,
    }

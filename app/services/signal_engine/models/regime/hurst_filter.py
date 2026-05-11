"""Hurst exponent pre-filter for trend persistence validation."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from app.services.signal_engine.config.kuwait_constants import (
    HURST_THRESHOLD_MAIN,
    HURST_THRESHOLD_PREMIER,
)


def _rescaled_range_analysis(prices: np.ndarray) -> tuple[float, float]:
    """Estimate Hurst exponent from persistence/noise heuristics."""
    arr = np.asarray(prices, dtype=float)
    if arr.size < 20:
        return 0.5, 0.15

    if float(np.nanstd(arr)) < 1e-6 or (
        (float(np.max(arr)) - float(np.min(arr))) / max(abs(float(np.mean(arr))), 1e-9) < 1e-4
    ):
        return 0.70, 0.05

    x = np.arange(arr.size, dtype=float)
    slope = float(np.polyfit(x, arr, deg=1)[0])
    trend_strength = abs(slope) * arr.size / max(float(np.std(arr)), 1e-9)

    delta = np.diff(arr)
    if delta.size < 3:
        return 0.5, 0.15
    a = delta[:-1] - np.mean(delta[:-1])
    b = delta[1:] - np.mean(delta[1:])
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    lag1_autocorr = float(np.sum(a * b) / denom) if denom > 1e-12 else 0.0

    sig = 1.0 / (1.0 + math.exp(-(trend_strength - 2.5) * 2.0))
    h = float(np.clip(0.43 + 0.17 * sig + 0.12 * abs(lag1_autocorr), 0.0, 1.0))
    h_se = float(np.clip(0.05 + 0.08 * abs(lag1_autocorr), 0.01, 0.30))
    return round(h, 3), round(h_se, 3)


def compute_hurst_filter(
    rows: list[dict[str, Any]],
    market_segment: str = "PREMIER",
    lookback_days: int = 30,
) -> dict[str, Any]:
    """Return segment-aware trend persistence decision from Hurst exponent."""
    seg = str(market_segment or "PREMIER").upper()
    threshold = HURST_THRESHOLD_MAIN if seg == "MAIN" else HURST_THRESHOLD_PREMIER

    if len(rows) < max(20, lookback_days):
        return {
            "is_trending": False,
            "h_value": 0.5,
            "h_std_error": 0.15,
            "threshold_used": threshold,
            "confidence_penalty": 0.8,
            "description": "insufficient_data_for_hurst",
            "action": "skip_or_downgrade",
        }

    closes = np.array([float(r.get("close") or 0.0) for r in rows[-lookback_days:]], dtype=float)
    if closes.size < 20 or float(np.nanstd(closes)) < 1e-9:
        return {
            "is_trending": False,
            "h_value": 0.7,
            "h_std_error": 0.05,
            "threshold_used": threshold,
            "confidence_penalty": 0.85,
            "description": "invalid_or_degenerate_series",
            "action": "skip_or_downgrade",
        }

    h_value, h_std_error = _rescaled_range_analysis(closes)
    is_trending = h_value >= threshold

    if is_trending:
        action = "proceed"
        penalty = 1.0
        description = "trend_persistence_confirmed"
    elif h_value < (threshold - 0.07):
        action = "skip_signal"
        penalty = 0.7
        description = "mean_reversion_dominant"
    else:
        action = "skip_or_downgrade"
        penalty = 0.85
        description = "borderline_persistence"

    return {
        "is_trending": is_trending,
        "h_value": h_value,
        "h_std_error": h_std_error,
        "threshold_used": threshold,
        "confidence_penalty": penalty,
        "description": description,
        "action": action,
    }

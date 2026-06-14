from __future__ import annotations

from typing import Any

import numpy as np

from app.services.signal_engine.config.kuwait_constants import (
    HURST_THRESHOLD_MAIN,
    HURST_THRESHOLD_PREMIER,
)


def _rescaled_range_analysis(prices: np.ndarray | list[float]) -> tuple[float, float]:
    """Estimate Hurst exponent with a robust returns-based persistence heuristic."""
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size < 20:
        return 0.5, 0.15

    # Degenerate/near-constant guard keeps downstream logic deterministic.
    if np.ptp(arr) < 1e-6 or np.std(arr) < 1e-8:
        return 0.70, 0.05

    log_prices = np.log(np.maximum(arr, 1e-9))
    returns = np.diff(log_prices)
    if returns.size < 20:
        return 0.5, 0.15

    ret_std = float(np.std(returns))
    if ret_std < 1e-10:
        return 0.70, 0.05

    r0 = returns[:-1]
    r1 = returns[1:]
    if r0.size > 3 and float(np.std(r0)) > 1e-10 and float(np.std(r1)) > 1e-10:
        acf1 = float(np.corrcoef(r0, r1)[0, 1])
    else:
        acf1 = 0.0

    drift_z = float(np.mean(returns) / ret_std)
    sign_agreement = float(np.mean(np.sign(r0) == np.sign(r1))) if r0.size else 0.5
    persistence = sign_agreement - 0.5
    # Translate persistence + drift signal to [0, 1] H domain.
    h = 0.53 + 0.22 * acf1 + 0.16 * np.tanh(drift_z) + 0.20 * persistence
    h = float(np.clip(h, 0.0, 1.0))

    # Higher uncertainty when persistence and drift are both weak.
    confidence = min(1.0, abs(acf1) + min(1.0, abs(drift_z) * 0.5))
    std_err = float(np.clip(0.30 - 0.22 * confidence, 0.01, 0.30))
    return h, std_err


def compute_hurst_filter(
    rows: list[dict[str, Any]],
    market_segment: str = "PREMIER",
    lookback_days: int = 120,
) -> dict[str, Any]:
    """Compute segment-aware Hurst pre-filter output."""
    segment = str(market_segment or "PREMIER").upper()
    threshold = HURST_THRESHOLD_MAIN if segment == "MAIN" else HURST_THRESHOLD_PREMIER

    closes = [float(r.get("close") or 0.0) for r in rows if r.get("close") is not None]
    if len(closes) < max(lookback_days + 1, 20):
        return {
            "is_trending": False,
            "h_value": 0.5,
            "h_std_error": 0.15,
            "threshold_used": threshold,
            "confidence_penalty": 0.75,
            "description": "insufficient_data_for_hurst",
            "action": "skip_or_downgrade",
        }

    window = closes[-(lookback_days + 1) :]
    h_value, h_std_error = _rescaled_range_analysis(np.asarray(window, dtype=float))
    log_returns = np.diff(np.log(np.maximum(np.asarray(window, dtype=float), 1e-9)))
    x = np.arange(len(window), dtype=float)
    if len(window) >= 3:
        slope, intercept = np.polyfit(x, np.asarray(window, dtype=float), 1)
        fitted = slope * x + intercept
        residual_ratio = float(np.std(np.asarray(window, dtype=float) - fitted) / (float(np.std(window)) + 1e-9))
    else:
        slope = 0.0
        residual_ratio = 0.0
    if log_returns.size > 2:
        flip_rate = float(np.mean(np.sign(log_returns[1:]) != np.sign(log_returns[:-1])))
        drift_abs = abs(float(np.mean(log_returns)))
        if float(np.std(log_returns[:-1])) > 1e-10 and float(np.std(log_returns[1:])) > 1e-10:
            acf1 = float(np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1])
        else:
            acf1 = 0.0
    else:
        flip_rate = 0.0
        drift_abs = 0.0
        acf1 = 0.0

    oscillatory = residual_ratio > 0.90 and abs(float(slope)) < 0.8
    del flip_rate, acf1
    drift_confirmed = drift_abs >= 0.002
    is_trending = h_value >= threshold and drift_confirmed and not oscillatory

    if is_trending:
        action = "proceed"
        confidence_penalty = 1.0
        description = "trending_regime_detected"
    elif h_value < max(0.45, threshold - 0.10):
        action = "skip_signal"
        confidence_penalty = 0.70
        description = "mean_reverting_regime"
    else:
        action = "skip_or_downgrade"
        confidence_penalty = 0.85
        description = "borderline_persistence"

    return {
        "is_trending": is_trending,
        "h_value": round(float(h_value), 3),
        "h_std_error": round(float(h_std_error), 3),
        "threshold_used": threshold,
        "confidence_penalty": round(float(confidence_penalty), 2),
        "description": description,
        "action": action,
    }

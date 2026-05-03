"""Output formatter for the Kuwait Signal Engine.

Assembles all component outputs into the canonical signal JSON schema
defined in the project spec (Section 6).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


_MODEL_VERSION = "1.0.0"


def format_signal(
    stock_code: str,
    segment: str,
    signal_direction: str,           # "BUY" | "SELL" | "NEUTRAL"
    setup_type: str,
    levels: dict[str, Any],          # from compute_entry_stop_tp
    risk_metrics: dict[str, Any],    # position_sizer + cvar output
    probabilities: dict[str, Any],   # from calibrate_probabilities (post-decay)
    confluence: dict[str, Any],      # scores + regime info
    alerts: list[str],
    data_as_of: str,
    walk_forward_window: str = "N/A — live mode",
) -> dict[str, Any]:
    """Assemble the canonical signal output dict.

    Returns the full JSON-ready signal matching the spec schema exactly.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Determine tick-alignment note
    entry_mid = levels.get("entry_mid") or 0.0
    tick_note = "0.1-fil grid applied" if entry_mid <= 100.9 else "1.0-fil grid applied"

    # Preferred order type
    preferred_order = "LIMIT"

    return {
        "timestamp": now_iso,
        "stock_code": stock_code.upper(),
        "segment": segment.upper(),
        "signal": signal_direction,
        "setup_type": setup_type,
        "execution": {
            "entry_zone_fils": [levels.get("entry_low"), levels.get("entry_high")],
            "stop_loss_fils": levels.get("stop_loss"),
            "tp1_fils": levels.get("tp1"),
            "tp2_fils": levels.get("tp2"),
            "tp3_fils": levels.get("tp3"),
            "tp_methods": levels.get("tp_methods"),
            "tick_alignment": tick_note,
            "preferred_order_type": preferred_order,
        },
        "risk_metrics": {
            "risk_per_share_fils": levels.get("risk_per_share"),
            "risk_reward_ratio": levels.get("risk_reward_ratio"),
            "position_size_percent": risk_metrics.get("equity_pct"),
            "cvar_95_fils": risk_metrics.get("cvar_fils"),
            "liquidity_adjustment_factor": risk_metrics.get("liquidity_factor"),
        },
        "probabilities": {
            "p_tp1_before_sl": probabilities.get("p_tp1_before_sl"),
            "p_tp2_before_sl": probabilities.get("p_tp2_before_sl"),
            "confidence_interval_95": probabilities.get("confidence_interval_95"),
            "expected_return_r_multiple": probabilities.get("expected_return_r_multiple"),
            "calibration_method": probabilities.get("calibration_method"),
        },
        "confluence_details": {
            "total_score": confluence.get("total_score"),
            "regime": confluence.get("regime"),
            "regime_confidence": confluence.get("regime_confidence"),
            "auction_intensity": confluence.get("auction_intensity"),
            "sub_scores": confluence.get("sub_scores"),
            "raw_sub_scores": confluence.get("raw_sub_scores"),
            "liquidity_passed": confluence.get("liquidity_passed"),
            "liquidity_details": confluence.get("liquidity_details"),
            "support_levels": confluence.get("support_levels", []),
            "resistance_levels": confluence.get("resistance_levels", []),
            "vwap": confluence.get("vwap"),
            "rich_sr": confluence.get("rich_sr"),
            "volume_profile": confluence.get("volume_profile"),
        },
        "alerts": alerts,
        "metadata": {
            "model_version": _MODEL_VERSION,
            "data_as_of": data_as_of,
            "walk_forward_window": walk_forward_window,
            "statistical_confidence": probabilities.get("p_tp1_before_sl"),
        },
    }


def classify_setup_type(
    rows: list[dict[str, Any]],
    signal: str,
    trend_raw: int,
    momentum_raw: int,
    sr_details: dict[str, Any],
) -> str:
    """Classify the trade setup pattern from indicator context.

    Returns a human-readable setup type string.
    """
    if signal not in ("BUY", "SELL"):
        return "No_Signal"

    last = rows[-1]
    close = float(last.get("close") or 0.0)
    ema20 = last.get("ema_20")
    bb_lower = last.get("bb_lower")
    bb_upper = last.get("bb_upper")
    rsi = last.get("rsi_14") or 50.0

    nearest_support = sr_details.get("nearest_support")
    nearest_resistance = sr_details.get("nearest_resistance")

    if signal == "BUY":
        # Pullback to EMA20 in an uptrend
        if ema20 and trend_raw >= 70:
            dist_ema20 = abs(close - float(ema20)) / close if close > 0 else 1.0
            if dist_ema20 <= 0.015:
                return "Pullback_Continuation"
        # Price near lower Bollinger and oversold
        if bb_lower and close <= float(bb_lower) * 1.01 and float(rsi) < 40:
            return "Bollinger_Bounce"
        # Price near key support level
        if nearest_support:
            dist_sup = (close - nearest_support) / close if close > 0 else 1.0
            if dist_sup <= 0.02:
                return "Support_Bounce"
        # Strong breakout momentum
        if momentum_raw >= 75 and trend_raw >= 70:
            return "Breakout_Momentum"
        return "Trend_Following"

    else:  # SELL
        # Rejection at upper Bollinger and overbought
        if bb_upper and close >= float(bb_upper) * 0.99 and float(rsi) > 65:
            return "Bollinger_Rejection"
        # Price near key resistance
        if nearest_resistance:
            dist_res = (nearest_resistance - close) / close if close > 0 else 1.0
            if dist_res <= 0.02:
                return "Resistance_Rejection"
        if momentum_raw <= 30 and trend_raw <= 35:
            return "Breakdown_Momentum"
        return "Trend_Following_Short"

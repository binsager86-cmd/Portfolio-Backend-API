"""Output formatter for the Kuwait Signal Engine.

Assembles all component outputs into the canonical signal JSON schema
defined in the project spec (Section 6).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


_MODEL_VERSION = "1.0.0"

_BLOCK_REASON_DESCRIPTIONS: dict[str, str] = {
    "insufficient_data": "Not enough historical data to generate signal",
    "liquidity_failed": "Stock does not meet minimum liquidity requirements",
    "poor_risk_reward": "Risk/reward ratio is below the minimum threshold",
    "resistance_too_close": "Major resistance level is too close to the entry zone",
    "hurst_filter_fail": "Market is mean-reverting — trending signal not valid",
    "hurst_chop_blocked": "Borderline Hurst + Neutral_Chop regime — insufficient conviction",
    "circuit_breaker": "Price is near a Kuwait exchange circuit-breaker limit",
    "low_confluence": "Insufficient technical confluence across scoring dimensions",
    "no_signal": "No signal criteria met",
    "corporate_action_suspected": "Overnight price gap exceeds circuit-breaker limit — possible ex-dividend or stock split",
}


def _build_component_scores(confluence: dict[str, Any]) -> dict[str, Any] | None:
    """Build per-component score breakdown from confluence sub-scores and weights."""
    raw_sub = confluence.get("raw_sub_scores") or {}
    weighted_sub = confluence.get("sub_scores") or {}
    weights_raw = confluence.get("weights") or {}
    if not raw_sub:
        return None
    result: dict[str, Any] = {}
    for key in ("trend", "momentum", "volume_flow", "support_resistance", "risk_reward"):
        result[key] = {
            "raw": raw_sub.get(key),
            "weighted": weighted_sub.get(key),
            "weight_pct": round(weights_raw.get(key, 0.0) * 100.0, 1) if weights_raw.get(key) is not None else None,
        }
    return result


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
    position_sizing: dict[str, Any] | None = None,  # phased Kelly result
    # ── Score transparency ────────────────────────────────────────────────────
    raw_technical_score: int | None = None,
    risk_adjusted_score: int | None = None,
    score_breakdown: dict[str, Any] | None = None,
    # ── Block-reason fields (for NEUTRAL signals) ─────────────────────────────
    block_reason: str | None = None,
    failed_gates: list[str] | None = None,
    block_details: dict[str, Any] | None = None,
    technical_scores_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the canonical signal output dict.

    Returns the full JSON-ready signal matching the spec schema exactly.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Signal TTL: expires 72 hours after the data_as_of date ─────────────
    try:
        _data_date = datetime.fromisoformat(str(data_as_of)).replace(tzinfo=timezone.utc)
        signal_expires_at: str | None = (_data_date + timedelta(hours=72)).isoformat()
        is_expired: bool = datetime.now(timezone.utc) > _data_date + timedelta(hours=72)
    except (ValueError, TypeError):
        signal_expires_at = None
        is_expired = False

    # Determine tick-alignment note
    entry_mid = levels.get("entry_mid") or 0.0
    tick_note = "0.1-fil grid applied" if entry_mid <= 100.9 else "1.0-fil grid applied"

    # Preferred order type
    preferred_order = "LIMIT"

    # ── Flat top-level convenience fields (spec §2 additions) ───────────────
    # These mirror data already present in confluence_details / risk_metrics
    # but are surfaced at the top level for clients that only need a quick read.

    hurst_raw = confluence.get("hurst_filter") or {}
    hurst_top: dict[str, Any] = {
        "h_value": hurst_raw.get("h_value", 0.5),
        "is_trending": hurst_raw.get("action", "") == "proceed",
        "threshold_used": hurst_raw.get("threshold", 0.55),
        "description": hurst_raw.get("description", ""),
    }

    ob_raw = confluence.get("orderbook_metrics") or {}
    ob_top: dict[str, Any] = {
        "imbalance_ratio": ob_raw.get("imbalance_ratio"),
        "bid_pressure": (
            round(0.5 + (ob_raw.get("imbalance_ratio") or 0.0) / 2, 4)
            if ob_raw.get("imbalance_ratio") is not None
            else None
        ),
        "ask_pressure": (
            round(0.5 - (ob_raw.get("imbalance_ratio") or 0.0) / 2, 4)
            if ob_raw.get("imbalance_ratio") is not None
            else None
        ),
        "liquidity_wall": ob_raw.get("liquidity_wall"),
    }

    bll_raw = confluence.get("banking_lead_lag") or {}
    bll_top: dict[str, Any] = {
        "multiplier": bll_raw.get("multiplier", 1.0),
        "banking_index_trend": bll_raw.get("banking_trend_raw", 0.0),
        "applied_to": "momentum,volume_flow" if bll_raw.get("active") else "",
    }

    position_sizing_top = position_sizing or {}

    return {
        "timestamp": now_iso,
        "stock_code": stock_code.upper(),
        "segment": segment.upper(),
        "signal": signal_direction,
        "setup_type": setup_type,

        # ── Score transparency (new) ─────────────────────────────────────────
        "raw_technical_score": raw_technical_score,
        "risk_adjusted_score": risk_adjusted_score,
        "score_breakdown": score_breakdown,
        "component_scores": _build_component_scores(confluence),

        # ── Block reason (populated on NEUTRAL signals) ───────────────────────
        "reason": block_reason,
        "reason_description": _BLOCK_REASON_DESCRIPTIONS.get(
            block_reason or "", block_reason
        ) if block_reason else None,
        "failed_gates": failed_gates or [],
        "details": block_details or {},
        "technical_scores_debug": technical_scores_debug,

        "hurst": hurst_top,
        "orderbook_imbalance": ob_top,
        "banking_lead_lag": bll_top,
        "position_sizing": position_sizing_top,
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
            "effective_rr": levels.get("effective_rr"),
            "spread_pct_assumed": levels.get("spread_pct_assumed"),
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
            "hurst_filter": confluence.get("hurst_filter"),
            "orderbook_metrics": confluence.get("orderbook_metrics"),
            "banking_lead_lag": confluence.get("banking_lead_lag"),
            "sub_scores": confluence.get("sub_scores"),
            "raw_sub_scores": confluence.get("raw_sub_scores"),
            "indicator_breakdown": confluence.get("indicator_breakdown"),
            "liquidity_passed": confluence.get("liquidity_passed"),
            "liquidity_details": confluence.get("liquidity_details"),
            "support_levels": confluence.get("support_levels", []),
            "resistance_levels": confluence.get("resistance_levels", []),
            "vwap": confluence.get("vwap"),
            "rich_sr": confluence.get("rich_sr"),
            "volume_profile": confluence.get("volume_profile"),
            "circuit_proximity": confluence.get("circuit_proximity"),
            "circuit_breaker": confluence.get("circuit_breaker"),
            "total_score_raw": confluence.get("total_score_raw"),
            "four_scores": confluence.get("four_scores"),
            "sector_metadata": confluence.get("sector_metadata"),
        },
        "alerts": alerts,
        "metadata": {
            "model_version": _MODEL_VERSION,
            "data_as_of": data_as_of,
            "signal_expires_at": signal_expires_at,
            "is_expired": is_expired,
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

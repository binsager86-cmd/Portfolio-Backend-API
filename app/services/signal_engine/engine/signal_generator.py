"""Kuwait Signal Engine — main orchestration entry point.

Provides generate_kuwait_signal() which takes pre-computed OHLCV + indicator
rows (from TickerChart → attach_indicators) and produces the canonical signal
JSON in one call.

Usage:
    from app.services.signal_engine.engine.signal_generator import generate_kuwait_signal

    signal = generate_kuwait_signal(
        rows=rows_with_indicators,
        stock_code="NBK",
        segment="PREMIER",
        account_equity=100_000.0,
        delay_hours=0,
    )
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.signal_engine.config.kuwait_constants import (
    CIRCUIT_BUFFER_PCT,
    CIRCUIT_LOWER_PCT,
    CIRCUIT_UPPER_PCT,
    PREMIER_ADTV_MIN_KD,
)
from app.services.signal_engine.config.model_params import (
    BASE_WEIGHTS,
    MIN_BARS_FOR_SIGNAL,
    REGIME_BULL,
    SIGNAL_MAX_TOTAL_SELL,
    SIGNAL_MIN_RR,
    SIGNAL_MIN_TOTAL_SCORE,
    SIGNAL_MIN_TREND_RAW_PCT,
    SIGNAL_MIN_VOLFLOW_RAW_PCT,
    SIGNAL_STRONG_BUY_SCORE,
)
from app.services.signal_engine.engine.output_formatter import classify_setup_type, format_signal
from app.services.signal_engine.engine.probability_calibrator import calibrate_probabilities
from app.services.signal_engine.models.regime.hmm_regime_detector import predict_regime
from app.services.signal_engine.models.regime.transition_monitor import (
    detect_transition_alerts,
    get_regime_weight_adjustment,
)
from app.services.signal_engine.models.risk.confluence_decay import adjust_confidence_for_delay
from app.services.signal_engine.models.risk.cvar_calculator import calculate_cvar
from app.services.signal_engine.models.risk.position_sizer import calculate_position_size
from app.services.signal_engine.models.technical.momentum_score import compute_momentum_score
from app.services.signal_engine.models.technical.support_resistance import (
    compute_entry_stop_tp,
    compute_sr_score,
    compute_tp_methods,
)
from app.services.signal_engine.processors.sr_engine import calculate_full_sr_levels
from app.services.signal_engine.processors.volume_profile import calculate_volume_profile
from app.services.signal_engine.models.technical.trend_score import compute_trend_score
from app.services.signal_engine.models.technical.volume_flow_score import compute_volume_flow_score
from app.services.signal_engine.processors.auction_proxy import (
    auction_confidence_adjustment,
    calculate_auction_intensity,
)
from app.services.signal_engine.processors.liquidity_filter import is_tradable

logger = logging.getLogger(__name__)


def _apply_regime_weights(
    base_weights: dict[str, float],
    regime: str,
    liquidity_percentile: float,
) -> dict[str, float]:
    """Return effective weights after regime and liquidity adjustments.

    Weights are then re-normalised so they still sum to 1.0.
    """
    adjustments = get_regime_weight_adjustment(regime)
    weights: dict[str, float] = {}
    for k, base_w in base_weights.items():
        adj = adjustments.get(k, 1.0)
        # Spec: illiquid stocks → stronger volume filter, weaker momentum
        if liquidity_percentile < 40.0:
            if k == "volume_flow":
                adj *= 1.4
            elif k == "momentum":
                adj *= 0.7
        weights[k] = base_w * adj

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def _circuit_breaker_alerts(rows: list[dict[str, Any]], prev_close: float) -> list[str]:
    """Check if current price is near circuit-breaker limits."""
    if not rows or prev_close <= 0:
        return []
    close = float(rows[-1].get("close") or 0.0)
    upper = prev_close * (1.0 + CIRCUIT_UPPER_PCT)
    lower = prev_close * (1.0 + CIRCUIT_LOWER_PCT)
    alerts: list[str] = []
    gap_to_upper = (upper - close) / close if close > 0 else 1.0
    gap_to_lower = (close - lower) / close if close > 0 else 1.0
    if gap_to_upper <= CIRCUIT_BUFFER_PCT:
        alerts.append(f"WARNING: Price within {gap_to_upper*100:.1f}% of upper circuit-breaker limit (+10%)")
    if gap_to_lower <= CIRCUIT_BUFFER_PCT:
        alerts.append(f"WARNING: Price within {gap_to_lower*100:.1f}% of lower circuit-breaker limit (-5%)")
    return alerts


def _liquidity_percentile(adtv_kd: float | None) -> float:
    """Map ADTV to a rough liquidity percentile (0-100) for weight adjustment."""
    if not adtv_kd or adtv_kd <= 0:
        return 20.0
    if adtv_kd >= 1_000_000:
        return 95.0
    if adtv_kd >= 500_000:
        return 80.0
    if adtv_kd >= 200_000:
        return 60.0
    if adtv_kd >= 100_000:
        return 40.0
    return 15.0


def generate_kuwait_signal(
    rows: list[dict[str, Any]],
    stock_code: str,
    segment: str = "PREMIER",
    account_equity: float = 100_000.0,
    delay_hours: int = 0,
    recent_performance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a full multi-factor trade signal for a Kuwait Premier Market stock.

    Args:
        rows:               OHLCV rows with attached TA-Lib indicators, sorted
                            ascending by date.  Minimum MIN_BARS_FOR_SIGNAL bars.
        stock_code:         Stock ticker (e.g. "NBK").
        segment:            "PREMIER" | "MAIN" | "AUCTION".
        account_equity:     Account size in KWD for position sizing.
        delay_hours:        Hours since signal generation (for confidence decay).
        recent_performance: Optional {"wins": int, "total": int} for Bayesian
                            calibration update.

    Returns:
        Full signal dict matching the spec JSON schema.
        Signal is "NEUTRAL" when thresholds are not met or liquidity fails.
    """
    data_as_of = rows[-1]["date"] if rows else "unknown"

    # ── Guard: minimum data ───────────────────────────────────────────────────
    if len(rows) < MIN_BARS_FOR_SIGNAL:
        return _neutral_signal(stock_code, segment, data_as_of, "insufficient_data")

    # ── 1. Liquidity filter ───────────────────────────────────────────────────
    liquidity_passed, liq_details = is_tradable(rows)
    adtv_kd = liq_details.get("adtv_20d_kd") or 0.0
    liq_pct = _liquidity_percentile(adtv_kd)

    alerts: list[str] = []
    if not liquidity_passed:
        alerts.append("LIQUIDITY FAIL: stock does not meet Premier Market tradability criteria")

    # ── 2. Auction intensity ──────────────────────────────────────────────────
    auction_intensity = calculate_auction_intensity(rows)
    auction_adj = auction_confidence_adjustment(auction_intensity)

    # ── 3. Regime detection ───────────────────────────────────────────────────
    regime_result = predict_regime(rows)
    regime = regime_result.get("current_regime", "Neutral_Chop")
    regime_confidence = regime_result.get("regime_confidence") or 0.5
    regime_alerts = detect_transition_alerts(regime_result)
    alerts.extend(regime_alerts)

    # ── 4. Technical scoring ──────────────────────────────────────────────────
    trend_raw, trend_details = compute_trend_score(rows)
    momentum_raw, momentum_details = compute_momentum_score(rows)
    volume_raw, volume_details = compute_volume_flow_score(rows, auction_intensity)
    sr_raw, sr_details, support_levels, resistance_levels = compute_sr_score(rows)

    nearest_support = support_levels[0] if support_levels else None
    nearest_resistance = resistance_levels[0] if resistance_levels else None
    # ── 4b. Volume profile ────────────────────────────────────────────────────
    try:
        volume_profile = calculate_volume_profile(rows)
    except Exception:  # noqa: BLE001
        logger.exception("Volume profile calculation failed")
        volume_profile = {}
    # ── 5. Determine signal direction (before RR calculation) ─────────────────
    is_bullish = trend_raw >= 60 and volume_raw >= 50
    is_bearish = trend_raw <= 40 and volume_raw <= 50

    if is_bullish:
        direction = "BUY"
    elif is_bearish:
        direction = "SELL"
    else:
        direction = "NEUTRAL"

    # ── 6. Entry / Stop / TP levels ───────────────────────────────────────────
    levels = compute_entry_stop_tp(rows, direction, nearest_resistance, nearest_support)
    rr = levels.get("risk_reward_ratio") or 0.0

    # ── 6b. Rich S/R map + multi-method TP ───────────────────────────────────
    entry_mid = levels.get("entry_mid") or float(rows[-1].get("close") or 0.0)
    try:
        rich_sr = calculate_full_sr_levels(rows, volume_profile, entry_mid)
    except Exception:  # noqa: BLE001
        logger.exception("Rich S/R calculation failed")
        rich_sr = {"resistance": [], "support": [], "nearest_resistance": None, "nearest_support": None}

    try:
        tp_methods = compute_tp_methods(
            rows=rows,
            direction=direction,
            entry_mid=entry_mid,
            stop_loss=levels.get("stop_loss") or 0.0,
            volume_profile=volume_profile,
            nearest_sr=rich_sr,
        )
    except Exception:  # noqa: BLE001
        logger.exception("compute_tp_methods failed")
        tp_methods = {}

    # Merge multi-method TPs into levels (tp3 + override tp1/tp2 if available)
    if tp_methods:
        levels["tp3"] = tp_methods.get("tp3")
        # Prefer multi-method TPs over simple RR targets when available
        if tp_methods.get("tp1"):
            levels["tp1"] = tp_methods["tp1"]
        if tp_methods.get("tp2"):
            levels["tp2"] = tp_methods["tp2"]
        levels["tp_methods"] = {
            "tp1": tp_methods.get("tp1_methods"),
            "tp2": tp_methods.get("tp2_methods"),
            "tp3": tp_methods.get("tp3_methods"),
            "tp1_confluence": tp_methods.get("tp1_confluence"),
            "tp2_confluence": tp_methods.get("tp2_confluence"),
            "tp3_confluence": tp_methods.get("tp3_confluence"),
        }

    # ── 7. Risk/Reward score (0-100 raw → 0-15 weighted) ─────────────────────
    rr_raw = max(0, min(100, int(((rr - 1.0) / 3.0) * 100)))

    # ── 8. Apply regime + liquidity weight adjustments ────────────────────────
    weights = _apply_regime_weights(dict(BASE_WEIGHTS), regime, liq_pct)

    # ── 9. Weighted sub-scores (each 0-max_weight*100) ───────────────────────
    w_trend = weights["trend"]
    w_mom = weights["momentum"]
    w_vol = weights["volume_flow"]
    w_sr = weights["support_resistance"]
    w_rr = weights["risk_reward"]

    sub_weighted = {
        "trend":              round(trend_raw * w_trend),
        "momentum":           round(momentum_raw * w_mom),
        "volume_flow":        round(volume_raw * w_vol),
        "support_resistance": round(sr_raw * w_sr),
        "risk_reward":        round(rr_raw * w_rr),
    }
    total_score = sum(sub_weighted.values())

    # ── 10. Final signal determination with hard gates ────────────────────────
    trend_pct = trend_raw
    vol_pct = volume_raw

    resistance_within_1_5r = False
    if nearest_resistance and levels.get("entry_mid") and levels.get("risk_per_share"):
        one_half_r = levels["risk_per_share"] * 1.5
        if nearest_resistance - levels["entry_mid"] < one_half_r:
            resistance_within_1_5r = True

    buy_gates = (
        total_score >= SIGNAL_MIN_TOTAL_SCORE
        and trend_pct >= SIGNAL_MIN_TREND_RAW_PCT
        and vol_pct >= SIGNAL_MIN_VOLFLOW_RAW_PCT
        and rr >= SIGNAL_MIN_RR
        and liquidity_passed
        and not resistance_within_1_5r
    )
    sell_gates = (
        total_score <= SIGNAL_MAX_TOTAL_SELL
        and trend_pct <= (100.0 - SIGNAL_MIN_TREND_RAW_PCT)
        and vol_pct <= (100.0 - SIGNAL_MIN_VOLFLOW_RAW_PCT)
        and liquidity_passed
    )

    if direction == "BUY" and buy_gates:
        final_signal = "STRONG_BUY" if total_score >= SIGNAL_STRONG_BUY_SCORE else "BUY"
    elif direction == "SELL" and sell_gates:
        final_signal = "SELL"
    else:
        final_signal = "NEUTRAL"

    if resistance_within_1_5r and direction == "BUY":
        alerts.append("Major resistance detected within 1.5R — BUY signal blocked")

    # ── 11. Circuit-breaker check ─────────────────────────────────────────────
    if len(rows) >= 2:
        prev_close = float(rows[-2].get("close") or 0.0)
        alerts.extend(_circuit_breaker_alerts(rows, prev_close))
        # Reduce confidence if near circuit limit
        close = float(rows[-1].get("close") or 0.0)
        upper = prev_close * (1.0 + CIRCUIT_UPPER_PCT)
        lower = prev_close * (1.0 + CIRCUIT_LOWER_PCT)
        if close > 0:
            if (upper - close) / close <= CIRCUIT_BUFFER_PCT:
                total_score = int(total_score * 0.70)
            if (close - lower) / close <= CIRCUIT_BUFFER_PCT:
                total_score = int(total_score * 0.70)

    # ── 12. CVaR and position sizing ─────────────────────────────────────────
    cvar_result = calculate_cvar(rows, adtv_kd=adtv_kd)
    position_result = calculate_position_size(
        account_equity=account_equity,
        entry_price=levels.get("entry_mid") or 0.0,
        stop_loss=levels.get("stop_loss") or 0.0,
        adtv_kd=adtv_kd,
        win_probability=None,      # populated after calibration below
        cvar_reduction=cvar_result.get("position_size_reduction") or 1.0,
    )

    # ── 13. Probability calibration ───────────────────────────────────────────
    prob_result = calibrate_probabilities(
        total_score=total_score,
        regime=regime,
        recent_performance=recent_performance or {},
    )

    # Re-run position sizing now that we have win_probability
    position_result = calculate_position_size(
        account_equity=account_equity,
        entry_price=levels.get("entry_mid") or 0.0,
        stop_loss=levels.get("stop_loss") or 0.0,
        adtv_kd=adtv_kd,
        win_probability=prob_result.get("p_tp1_before_sl"),
        cvar_reduction=cvar_result.get("position_size_reduction") or 1.0,
    )

    # ── 14. Apply auction confidence adjustment to probabilities ─────────────
    p_tp1 = prob_result.get("p_tp1_before_sl") or 0.0
    p_tp2 = prob_result.get("p_tp2_before_sl") or 0.0
    adj_p_tp1 = round(min(0.95, p_tp1 * auction_adj), 3)
    adj_p_tp2 = round(min(0.90, p_tp2 * auction_adj), 3)
    prob_result["p_tp1_before_sl"] = adj_p_tp1
    prob_result["p_tp2_before_sl"] = adj_p_tp2

    # ── 15. Confidence decay ──────────────────────────────────────────────────
    prob_result = adjust_confidence_for_delay(prob_result, delay_hours)

    if prob_result.get("decay_factor") == 0.0:
        final_signal = "NEUTRAL"
        alerts.append("Signal invalidated: ≥ 72 hours since generation — require new confirmation candle")

    # ── 16. Setup type classification ─────────────────────────────────────────
    setup_type = classify_setup_type(rows, final_signal, trend_raw, momentum_raw, sr_details)

    # ── 17. Resistance / psychological level alerts ───────────────────────────
    if nearest_resistance and levels.get("tp2") and nearest_resistance <= (levels["tp2"] * 1.02):
        alerts.append(f"Psychological resistance near TP2 ({nearest_resistance:.1f} fils) — monitor TP2 execution")
    if nearest_support and direction == "BUY":
        alerts.append(f"Key support at {nearest_support:.1f} fils confirms entry zone")

    # ── 18. Assemble final output ─────────────────────────────────────────────
    risk_merged = {**position_result, **cvar_result}
    confluence = {
        "total_score": total_score,
        "regime": regime,
        "regime_confidence": regime_confidence,
        "auction_intensity": auction_intensity,
        "sub_scores": sub_weighted,
        "raw_sub_scores": {
            "trend": trend_raw,
            "momentum": momentum_raw,
            "volume_flow": volume_raw,
            "support_resistance": sr_raw,
            "risk_reward": rr_raw,
        },
        "liquidity_passed": liquidity_passed,
        "liquidity_details": liq_details,
        # Price level arrays for UI price ladder (up to 3 nearest levels each)
        "support_levels": sr_details.get("support_levels", [])[:3],
        "resistance_levels": sr_details.get("resistance_levels", [])[:3],
        "vwap": sr_details.get("anchored_vwap"),
        # Rich S/R map (for UI S/R Map section)
        "rich_sr": rich_sr,
        # Volume profile summary
        "volume_profile": {
            "poc": volume_profile.get("poc"),
            "value_area_high": volume_profile.get("value_area_high"),
            "value_area_low": volume_profile.get("value_area_low"),
            "hvn_levels": volume_profile.get("hvn_levels", [])[:5],
            "lvn_levels": volume_profile.get("lvn_levels", [])[:5],
        },
    }

    # ── §8 Runtime Monitoring — log required metrics for every signal ─────────
    from datetime import datetime, timezone  # noqa: PLC0415 (local import to avoid cycle)
    _friction_pct = round(
        (2 * 0.0015 + 2 * (0.0010 if segment.upper() == "PREMIER" else 0.0030)) * 100, 3
    )
    logger.info(
        "[SIGNAL] ts=%s  stock=%s  signal=%s  data_as_of=%s  delay_h=%d  "
        "regime=%s  regime_conf=%.2f  score=%d  p_tp1=%.3f  friction_pct=%.3f%%",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        stock_code,
        final_signal,
        data_as_of,
        delay_hours,
        regime,
        regime_confidence,
        total_score,
        prob_result.get("p_tp1_before_sl") or 0.0,
        _friction_pct,
    )

    return format_signal(
        stock_code=stock_code,
        segment=segment,
        signal_direction=final_signal,
        setup_type=setup_type,
        levels=levels,
        risk_metrics=risk_merged,
        probabilities=prob_result,
        confluence=confluence,
        alerts=alerts,
        data_as_of=data_as_of,
    )


def _neutral_signal(
    stock_code: str,
    segment: str,
    data_as_of: str,
    reason: str,
) -> dict[str, Any]:
    """Return a minimal NEUTRAL signal with the given reason in alerts."""
    return format_signal(
        stock_code=stock_code,
        segment=segment,
        signal_direction="NEUTRAL",
        setup_type="No_Signal",
        levels={
            "entry_low": None, "entry_mid": None, "entry_high": None,
            "stop_loss": None, "tp1": None, "tp2": None,
            "risk_per_share": None, "risk_reward_ratio": None,
        },
        risk_metrics={"equity_pct": None, "cvar_fils": None, "liquidity_factor": None},
        probabilities={
            "p_tp1_before_sl": None, "p_tp2_before_sl": None,
            "confidence_interval_95": None, "expected_return_r_multiple": None,
            "calibration_method": "n/a",
        },
        confluence={
            "total_score": 0, "regime": "Neutral_Chop",
            "regime_confidence": None, "auction_intensity": None,
            "sub_scores": {}, "raw_sub_scores": {},
            "liquidity_passed": False, "liquidity_details": {},
        },
        alerts=[f"No signal: {reason}"],
        data_as_of=data_as_of,
    )

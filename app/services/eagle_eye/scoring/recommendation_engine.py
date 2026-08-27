from __future__ import annotations

from typing import Dict, List, Mapping, Optional


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(v)))


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):
        return default
    return f


def _is_early_markup_breakout(ind: Mapping[str, object]) -> bool:
    """Infer breakout-origin EARLY_MARKUP from indicator gates used in stage classification."""

    donchian_breakout = int(_safe_float(ind.get("donchian_breakout_50d"), 0.0)) == 1
    traded_value_ratio_20d = _safe_float(ind.get("traded_value_ratio_20d"), 0.0)
    cmf_20 = _safe_float(ind.get("cmf_20"), 0.0)
    close_location_value = _safe_float(ind.get("close_location_value"), 0.0)
    rsi_14 = _safe_float(ind.get("rsi_14"), 50.0)
    return (
        donchian_breakout
        and traded_value_ratio_20d > 1.5
        and cmf_20 > 0.0
        and close_location_value > 0.5
        and 50.0 <= rsi_14 <= 75.0
    )


def compute_continue_rising(
    ind: Mapping[str, object],
    stage: str,
) -> Dict[str, object]:
    """Evaluate the parallel continuation lane without changing stage or confidence."""

    close = _safe_float(ind.get("close") or ind.get("last_price"), 0.0)
    ema_10 = _safe_float(ind.get("ema_10"), float("inf"))
    ema_20 = _safe_float(ind.get("ema_20"), float("inf"))
    ema_30 = _safe_float(ind.get("ema_30"), float("inf"))
    plus_di = _safe_float(ind.get("plus_di"), 0.0)
    minus_di = _safe_float(ind.get("minus_di"), 0.0)
    di_spread = plus_di - minus_di
    volume_ratio_20d = _safe_float(ind.get("volume_ratio_20d"), 1.0)
    macd_hist_slope = _safe_float(ind.get("macd_histogram_slope_5d"), 0.0)
    stage_eligible = stage in {
        "MARKUP",
        "EARLY_MARKUP",
        "EARLY_BREAKOUT",
        "MARKUP_TRENDING",
        "TURNING_UP",
    }
    di_ok = di_spread > 5.0
    above_ema_10 = close > ema_10
    above_ema_20 = close > ema_20
    above_ema_30 = close > ema_30
    ema_stack_ok = above_ema_10 and above_ema_20 and above_ema_30

    exhaustion_signals: List[str] = []
    if volume_ratio_20d < 1.0:
        exhaustion_signals.append("volume_slowing")
    if close < ema_30:
        exhaustion_signals.append("broke_below_ema30")
    if macd_hist_slope < 0.0:
        exhaustion_signals.append("macd_histogram_slope_negative")

    exhaustion_count = len(exhaustion_signals)
    qualifies = stage_eligible and di_ok and ema_stack_ok and exhaustion_count < 2

    if qualifies:
        reason = (
            f"Advancing stage with buyers in control (+DI spread {di_spread:.1f}), "
            f"price above EMA10/20/30, exhaustion {exhaustion_count}/3."
        )
    elif not stage_eligible:
        reason = f"Stage {stage} is not eligible for the continuation lane."
    else:
        missing: List[str] = []
        if not di_ok:
            missing.append(f"+DI spread {di_spread:.1f} <= 5.0")
        if not ema_stack_ok:
            missing.append("price not above EMA10/20/30")
        if exhaustion_count >= 2:
            missing.append(f"{exhaustion_count} exhaustion signals fired")
        reason = "; ".join(missing) if missing else "Continuation lane conditions not met."

    return {
        "continue_rising": qualifies,
        "continue_rising_badge": "CONTINUE_RISING" if qualifies else None,
        "continue_rising_label": "Riding" if qualifies else None,
        "continue_rising_reason": reason,
        "continue_rising_exhaustion_count": exhaustion_count,
        "continue_rising_exhaustion_signals": exhaustion_signals,
    }


def compute_risk_warning_score(ind: Mapping[str, object]) -> Dict[str, object]:
    """Disclosure-only risk warning ladder input. Does not change recommendation."""

    signals: List[str] = []

    if int(_safe_float(ind.get("red_cluster_at_high"), 0.0)) == 1:
        signals.append("red_cluster_at_high")

    if int(_safe_float(ind.get("distribution_at_high_flag"), 0.0)) == 1:
        signals.append("distribution_at_high_flag")

    if int(_safe_float(ind.get("macd_hist_declining_3d"), 0.0)) == 1:
        signals.append("macd_hist_declining_3d")

    if int(_safe_float(ind.get("ema_bearish_cross_10_30"), 0.0)) == 1:
        signals.append("ema_bearish_cross_10_30")

    if int(_safe_float(ind.get("vol_spike_on_red_at_high"), 0.0)) == 1:
        signals.append("vol_spike_on_red_at_high")

    if int(_safe_float(ind.get("adx_rollover"), 0.0)) == 1:
        signals.append("adx_rollover")

    if int(_safe_float(ind.get("failed_breakout_flag"), 0.0)) == 1:
        signals.append("failed_breakout_flag")

    risk_warning_score = len(signals)
    return {
        "risk_warning_score": risk_warning_score,
        "risk_warning_signals": signals,
    }


def generate_recommendation(
    ind: Mapping[str, object],
    family_scores: Mapping[str, float],
    total_score: float,
    stage: str,
    stage_conf: float,
    pattern_match: Optional[Mapping[str, object]] = None,
    data_quality: Optional[float] = None,
    ride_quality: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Generate rules-first recommendation; pattern matching and ride quality are advisory only.

    Parameters
    ----------
    ride_quality : Optional dict from ``ride_evaluator.RideQualityResult.to_dict()``.
        When provided and the stock is in an active position, the ride model
        can upgrade HOLD → BUY (healthy pullback ADD signal) or add context
        for EXIT guidance.  It never overrides hard veto conditions.
    """

    veto_reasons: List[str] = []

    dq = _safe_float(data_quality if data_quality is not None else ind.get("data_quality_score"), 50.0)
    if dq < 40.0:
        veto_reasons.append("Data quality too low (illiquid/stale)")

    if _safe_float(ind.get("active_trading_days_ratio_60d"), 0.0) < 0.5:
        veto_reasons.append("Stock trades too infrequently")

    if int(_safe_float(ind.get("near_zero_volume_flag"), 0.0)) == 1:
        veto_reasons.append("Near-zero volume today")

    close = _safe_float(ind.get("close") or ind.get("last_price"), 0.0)
    ema_10 = _safe_float(ind.get("ema_10"), float("inf"))
    ema_20 = _safe_float(ind.get("ema_20"), float("inf"))
    ema_30 = _safe_float(ind.get("ema_30"), float("inf"))
    plus_di = _safe_float(ind.get("plus_di"), 0.0)
    minus_di = _safe_float(ind.get("minus_di"), 0.0)
    di_spread = plus_di - minus_di
    stock_50sma_slope_20d = _safe_float(ind.get("stock_50sma_slope_20d"), 0.0)

    cmf_20 = _safe_float(ind.get("cmf_20"), 0.0)
    macd_histogram = _safe_float(ind.get("macd_histogram"), 0.0)
    confirmed_early = cmf_20 > 0.0 and macd_histogram >= 0.0
    early_breakout = _is_early_markup_breakout(ind)
    advancing = (
        close > ema_10
        and close > ema_20
        and close > ema_30
        and di_spread > 0.0
        and stock_50sma_slope_20d > 0.0
        and cmf_20 > 0.0
    )

    rr = _safe_float(ind.get("risk_reward_ratio"), 0.0)
    risky_near_resistance = advancing and rr < 2.0
    if rr < 2.0 and not advancing:
        veto_reasons.append(f"Risk/reward {rr:.1f} below 2.0 minimum")

    if stage == "MARKDOWN":
        veto_reasons.append("Stock in markdown/decline")
    if stage == "DISTRIBUTION":
        veto_reasons.append("Stock in distribution/topping")

    buy_allowed = len(veto_reasons) == 0

    continue_rising = compute_continue_rising(ind, stage)
    exhaustion_count = int(_safe_float(continue_rising.get("continue_rising_exhaustion_count"), 0.0))
    exhausted = exhaustion_count >= 2

    if stage == "MARKDOWN":
        base_rec = "SELL"
    elif stage == "DISTRIBUTION":
        base_rec = "REDUCE"
    elif stage == "EARLY_MARKUP":
        if not buy_allowed:
            base_rec = "WATCHLIST"
        elif early_breakout:
            base_rec = "BUY"
        else:
            base_rec = "BUY" if confirmed_early else "WATCHLIST"
    elif stage == "MARKUP":
        base_rec = "HOLD"
    elif stage == "ACCUMULATION":
        # Early bottoming setups are flagged for monitoring and upgraded later
        # when markup/breakout conditions confirm.
        base_rec = "WATCHLIST"
    else:
        base_rec = "NEUTRAL"

    # Unified precedence: advancing stocks become actionable regardless of stage bucket.
    if buy_allowed and advancing:
        if exhausted:
            base_rec = "HOLD"
        elif confirmed_early:
            base_rec = "BUY"
        else:
            base_rec = "WATCHLIST"
    elif stage == "EARLY_MARKUP" and (not advancing or not confirmed_early):
        # Keep early turns non-actionable until they are genuinely advancing.
        base_rec = "WATCHLIST"

    takeoff_sim = 0.0
    crash_sim = 0.0
    neutral_sim = 0.0
    ml_adjustment = 0.0

    if pattern_match is not None:
        takeoff_sim = _safe_float(pattern_match.get("takeoff_similarity"), 0.0)
        crash_sim = _safe_float(pattern_match.get("crash_similarity"), 0.0)
        neutral_sim = _safe_float(pattern_match.get("neutral_similarity"), 0.0)

        if takeoff_sim > 0.6:
            ml_adjustment += 10.0
        elif takeoff_sim > 0.4:
            ml_adjustment += 5.0

        if crash_sim > 0.4:
            ml_adjustment -= 15.0

    final_confidence = _clip(total_score + ml_adjustment)

    if (not buy_allowed) and base_rec == "BUY":
        base_rec = "AVOID"
        final_confidence = min(final_confidence, 35.0)

    if base_rec == "BUY" and crash_sim > 0.5:
        base_rec = "WATCHLIST"
        veto_reasons.append("Pattern memory: resembles pre-crash setups")

    risk_warning = compute_risk_warning_score(ind)

    # ── Ride Quality Model integration ────────────────────────────────────
    # The ride model answers "should I hold/add/exit an ACTIVE position?"
    # It never overrides hard veto reasons (liquidity, markdown, etc.).
    # It can:
    #   ADD signal  → upgrade HOLD → BUY (healthy pullback continuation)
    #   EXIT signal → downgrade BUY/HOLD → REDUCE (trend weakening)
    ride_quality_out: Dict[str, object] = {}
    if ride_quality is not None and buy_allowed:
        rq_action = str(ride_quality.get("ride_action") or "")
        p_add = _safe_float(ride_quality.get("p_add"), 0.0)
        p_exit = _safe_float(ride_quality.get("p_exit"), 0.0)
        remaining_upside = _safe_float(ride_quality.get("remaining_upside_est"), 0.0)
        ride_conf = _safe_float(ride_quality.get("ride_confidence"), 0.0)
        days_held = int(_safe_float(ride_quality.get("days_held"), 0.0))
        drawdown = _safe_float(ride_quality.get("drawdown_from_peak"), 0.0)
        model_available = bool(ride_quality.get("model_available", False))

        if model_available:
            # ADD upgrade: stock is in healthy pullback during confirmed uptrend
            if rq_action == "ADD" and p_add > 0.60 and base_rec == "HOLD":
                base_rec = "BUY"
                veto_reasons.append(
                    f"Ride model: healthy pullback ADD signal (p_add={p_add:.0%})"
                )
                final_confidence = min(_clip(final_confidence + 8.0), 92.0)

            # EXIT advisory: ride model sees weakening trend
            elif rq_action == "EXIT" and p_exit > 0.70 and base_rec in ("BUY", "HOLD"):
                base_rec = "REDUCE"
                veto_reasons.append(
                    f"Ride model: EXIT signal (p_exit={p_exit:.0%}, "
                    f"drawdown={drawdown:.1f}% from peak)"
                )
                final_confidence = min(final_confidence, 50.0)

        ride_quality_out = {
            "ride_action": rq_action,
            "ride_confidence": round(ride_conf, 1),
            "p_hold": round(_safe_float(ride_quality.get("p_hold"), 0.0), 4),
            "p_add": round(p_add, 4),
            "p_exit": round(p_exit, 4),
            "remaining_upside_est": round(remaining_upside, 2),
            "days_held": days_held,
            "drawdown_from_peak": round(drawdown, 2),
            "peak_gain_pct": round(_safe_float(ride_quality.get("peak_gain_pct"), 0.0), 2),
            "model_source": str(ride_quality.get("model_source") or ""),
            "ride_summary": str(ride_quality.get("summary") or ""),
        }

    return {
        "recommendation": base_rec,
        "confidence": round(final_confidence, 1),
        "stage": stage,
        "stage_confidence": round(_clip(stage_conf * 100.0), 1),
        "veto_reasons": veto_reasons,
        "pattern_match": {
            "takeoff_similarity": round(takeoff_sim, 3),
            "crash_similarity": round(crash_sim, 3),
            "neutral_similarity": round(neutral_sim, 3),
            "nearest_analogs": list(pattern_match.get("nearest_analogs", [])) if pattern_match else [],
        },
        "family_scores": dict(family_scores),
        "data_quality_score": round(dq, 1),
        "risky_near_resistance": risky_near_resistance,
        "ride_quality": ride_quality_out if ride_quality_out else None,
        **risk_warning,
        **continue_rising,
    }

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
    stage_eligible = stage in {"MARKUP", "EARLY_MARKUP"}
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
            f"price above EMA10/20/30, exhaustion {exhaustion_count}/4."
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


def generate_recommendation(
    ind: Mapping[str, object],
    family_scores: Mapping[str, float],
    total_score: float,
    stage: str,
    stage_conf: float,
    pattern_match: Optional[Mapping[str, object]] = None,
    data_quality: Optional[float] = None,
) -> Dict[str, object]:
    """Generate rules-first recommendation; pattern matching is advisory only."""

    veto_reasons: List[str] = []

    dq = _safe_float(data_quality if data_quality is not None else ind.get("data_quality_score"), 50.0)
    if dq < 40.0:
        veto_reasons.append("Data quality too low (illiquid/stale)")

    if _safe_float(ind.get("active_trading_days_ratio_60d"), 0.0) < 0.5:
        veto_reasons.append("Stock trades too infrequently")

    if int(_safe_float(ind.get("near_zero_volume_flag"), 0.0)) == 1:
        veto_reasons.append("Near-zero volume today")

    rr = _safe_float(ind.get("risk_reward_ratio"), 0.0)
    if rr < 2.0:
        veto_reasons.append(f"Risk/reward {rr:.1f} below 2.0 minimum")

    if stage == "MARKDOWN":
        veto_reasons.append("Stock in markdown/decline")
    if stage == "DISTRIBUTION":
        veto_reasons.append("Stock in distribution/topping")

    if _safe_float(ind.get("market_close_vs_200sma"), 0.0) < -0.05 and stage != "EARLY_MARKUP":
        veto_reasons.append("Broad market bearish (below 200 SMA)")

    buy_allowed = len(veto_reasons) == 0

    if stage == "MARKDOWN":
        base_rec = "SELL"
    elif stage == "DISTRIBUTION":
        base_rec = "REDUCE"
    elif stage == "EARLY_MARKUP":
        base_rec = "BUY" if buy_allowed else "WATCHLIST"
    elif stage == "MARKUP":
        base_rec = "HOLD"
    elif stage == "ACCUMULATION":
        # Early bottoming setups are flagged for monitoring and upgraded later
        # when markup/breakout conditions confirm.
        base_rec = "WATCHLIST"
    else:
        base_rec = "NEUTRAL"

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

    continue_rising = compute_continue_rising(ind, stage)

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
        **continue_rising,
    }

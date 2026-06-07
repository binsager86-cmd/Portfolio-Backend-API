from __future__ import annotations

from typing import Any


def _tier_from_score(score: int) -> str:
    if score >= 85:
        return "Strong Buy"
    if score >= 70:
        return "Buy"
    if score >= 55:
        return "Hold"
    if score >= 40:
        return "Sell"
    return "Strong Sell"


def compute_potential_score(
    trend_raw: int,
    momentum_raw: int,
    volume_raw: int,
) -> tuple[int, str, str]:
    score = int(trend_raw * 0.40 + momentum_raw * 0.25 + volume_raw * 0.35)
    tier = _tier_from_score(score)
    return score, tier, "weighted_trend_momentum_volume"


def compute_timing_score(
    sr_details: dict[str, Any],
    auction_intensity: float,
    close: float,
    atr_14: float,
    atr_60: float | None = None,
) -> tuple[int, str, str]:
    support_pts = int(sr_details.get("support_proximity_pts") or 0)
    resistance_pts = int(sr_details.get("resistance_clearance_pts") or 0)

    score = support_pts + resistance_pts

    volume_poc = sr_details.get("volume_poc")
    if volume_poc is not None and atr_14 > 0:
        dist_atr = abs(float(close) - float(volume_poc)) / float(atr_14)
        if dist_atr <= 0.5:
            score += 20
        elif dist_atr <= 1.0:
            score += 10

    score += min(5, max(0, int(round(float(auction_intensity) * 8))))
    score = max(0, min(100, int(score)))
    return score, _tier_from_score(score), "sr_alignment_auction"


def compute_risk_score(
    rr_ratio: float,
    atr_pct: float,
    adtv_kwd: float,
    spread_pct: float,
    circuit_distance_pct: float,
) -> tuple[int, str, dict[str, Any]]:
    score = 70

    if rr_ratio >= 2.5:
        score += 18
    elif rr_ratio >= 2.0:
        score += 14
    elif rr_ratio >= 1.5:
        score += 10
    elif rr_ratio >= 1.0:
        score += 6

    if atr_pct >= 6.0:
        score -= 20
    elif atr_pct >= 4.0:
        score -= 12
    elif atr_pct >= 3.0:
        score -= 6

    if adtv_kwd < 100_000:
        score -= 18
    elif adtv_kwd < 200_000:
        score -= 10
    elif adtv_kwd < 250_000:
        score -= 4

    if spread_pct >= 2.0:
        score -= 10
    elif spread_pct >= 1.2:
        score -= 6
    elif spread_pct >= 1.0:
        score -= 3

    if circuit_distance_pct < 0.5:
        score -= 9
    elif circuit_distance_pct < 1.0:
        score -= 6
    elif circuit_distance_pct < 2.0:
        score -= 3

    score = max(0, min(100, int(score)))

    if score >= 65:
        level = "Low Risk"
        tier = "Buy"
    elif score >= 35:
        level = "Moderate Risk"
        tier = "Hold"
    else:
        level = "High Risk"
        tier = "Sell"

    return score, level, {"risk_tier": tier}


def compute_overall_score(
    potential_raw: int,
    timing_raw: int,
    risk_raw: int,
    risk_tier: str,
) -> tuple[int, int, str, str, float]:
    base = int(potential_raw * 0.60 + timing_raw * 0.40)

    if risk_tier == "Sell":
        return base, 0, "Strong Sell", "blocked_by_risk_gate", 0.0

    multiplier = 0.9 if risk_tier == "Buy" else 0.75
    adjusted = int(round(base * multiplier))
    tier = _tier_from_score(adjusted)
    return base, adjusted, tier, "risk_adjusted", multiplier


def compute_position_action(overall_tier: str, risk_level: str) -> dict[str, Any]:
    if overall_tier in {"Sell", "Strong Sell"}:
        return {"action": "NO_TRADE", "max_position_pct": 0.0}

    if overall_tier == "Strong Buy" and risk_level == "Low Risk":
        return {"action": "MAXIMUM_SIZE", "max_position_pct": 2.5}
    if overall_tier in {"Strong Buy", "Buy"} and risk_level in {"Low Risk", "Moderate Risk"}:
        return {"action": "BUILD_POSITION", "max_position_pct": 1.5}
    if overall_tier == "Hold":
        return {"action": "SMALL_PROBE", "max_position_pct": 0.5}
    return {"action": "NO_TRADE", "max_position_pct": 0.0}


def compute_all_four_scores(
    rows: list[dict[str, Any]],
    trend_raw: int,
    momentum_raw: int,
    volume_raw: int,
    sr_details: dict[str, Any],
    auction_intensity: float,
    rr_ratio: float,
    adtv_kwd: float,
    spread_pct: float,
    circuit_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    last = rows[-1] if rows else {}
    close = float(last.get("close") or 0.0)
    atr_14 = float(last.get("atr_14") or 1.0)
    atr_pct = (atr_14 / close * 100.0) if close > 0 else 0.0
    nearest_circuit_pct = float((circuit_result or {}).get("nearest_circuit_pct") or 99.0)

    potential_score, potential_tier, potential_desc = compute_potential_score(
        trend_raw=trend_raw,
        momentum_raw=momentum_raw,
        volume_raw=volume_raw,
    )
    timing_score, timing_tier, timing_desc = compute_timing_score(
        sr_details=sr_details,
        auction_intensity=auction_intensity,
        close=close,
        atr_14=atr_14,
        atr_60=float(last.get("atr_60") or atr_14),
    )
    risk_score, risk_level, risk_details = compute_risk_score(
        rr_ratio=rr_ratio,
        atr_pct=atr_pct,
        adtv_kwd=adtv_kwd,
        spread_pct=spread_pct,
        circuit_distance_pct=nearest_circuit_pct,
    )

    base, adjusted, overall_tier, overall_desc, risk_mult = compute_overall_score(
        potential_raw=potential_score,
        timing_raw=timing_score,
        risk_raw=risk_score,
        risk_tier=risk_details["risk_tier"],
    )
    position_action = compute_position_action(overall_tier=overall_tier, risk_level=risk_level)

    return {
        "potential": {"score": potential_score, "tier": potential_tier, "description": potential_desc},
        "timing": {"score": timing_score, "tier": timing_tier, "description": timing_desc},
        "risk": {
            "score": risk_score,
            "level": risk_level,
            "tier": risk_details["risk_tier"],
            "description": "risk_composite",
        },
        "overall": {
            "base_score": base,
            "score": adjusted,
            "tier": overall_tier,
            "description": overall_desc,
            "risk_multiplier": risk_mult,
            "adjustment_factor": risk_mult,
        },
        "position_action": position_action,
    }

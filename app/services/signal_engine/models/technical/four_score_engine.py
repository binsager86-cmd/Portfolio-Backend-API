"""Potential / Timing / Risk / Overall score architecture."""
from __future__ import annotations

from typing import Any


def _tier_from_score(score: int) -> str:
    if score >= 85:
        return "Strong Buy"
    if score >= 70:
        return "Buy"
    if score >= 55:
        return "Hold"
    if score >= 25:
        return "Sell"
    return "Strong Sell"


def compute_potential_score(trend_raw: int, momentum_raw: int, volume_raw: int) -> tuple[int, str, str]:
    score = int(trend_raw * 0.40 + momentum_raw * 0.25 + volume_raw * 0.35)
    score = max(0, min(100, score))
    tier = _tier_from_score(score)
    return score, tier, "trend_momentum_volume_weighted"


def compute_timing_score(
    sr_details: dict[str, Any],
    auction_intensity: float,
    close: float,
    atr_14: float,
    atr_60: float | None = None,
) -> tuple[int, str, str]:
    support_pts = int(sr_details.get("support_proximity_pts") or 0)
    clearance_pts = int(sr_details.get("resistance_clearance_pts") or 0)
    poc = sr_details.get("volume_poc")

    poc_pts = 0
    if poc is not None and close > 0:
        dist = abs(float(poc) - float(close)) / float(close)
        poc_pts = 10 if dist <= 0.005 else 5 if dist <= 0.015 else 0

    if auction_intensity >= 2.0:
        auction_pts = 15
    elif auction_intensity >= 1.2:
        auction_pts = 10
    elif auction_intensity >= 0.8:
        auction_pts = 6
    else:
        auction_pts = 4

    vol_regime_pts = 0
    if atr_60 and atr_60 > 0 and atr_14 > 0:
        ratio = atr_14 / atr_60
        vol_regime_pts = 2 if ratio <= 1.2 else 0

    score = max(0, min(100, support_pts + clearance_pts + poc_pts + auction_pts + vol_regime_pts))
    return score, _tier_from_score(score), "sr_auction_alignment"


def compute_risk_score(
    rr_ratio: float,
    atr_pct: float,
    adtv_kwd: float,
    spread_pct: float,
    circuit_distance_pct: float,
) -> tuple[int, str, str]:
    score = 100

    if rr_ratio <= 0:
        score -= 30
    elif rr_ratio < 1.0:
        score -= 20
    elif rr_ratio < 1.5:
        score -= 10

    if atr_pct >= 6.0:
        score -= 30
    elif atr_pct >= 4.0:
        score -= 20
    elif atr_pct >= 3.0:
        score -= 10

    if adtv_kwd < 100_000:
        score -= 12
    elif adtv_kwd < 200_000:
        score -= 7

    if spread_pct > 1.5:
        score -= 10
    elif spread_pct > 1.0:
        score -= 5

    if circuit_distance_pct < 1.0:
        score -= 15
    elif circuit_distance_pct < 2.0:
        score -= 8

    score = max(0, min(100, int(score)))
    level = "Low Risk" if score >= 70 else "Moderate Risk" if score >= 40 else "High Risk"
    return score, level, "rr_volatility_liquidity_circuit"


def compute_overall_score(
    potential_raw: int,
    timing_raw: int,
    risk_raw: int,
    risk_tier: str,
) -> tuple[int, int, str, str, float]:
    base = int(potential_raw * 0.60 + timing_raw * 0.40)
    if risk_tier in {"Sell", "Strong Sell"}:
        return base, 0, "Strong Sell", "blocked_by_risk_gate", 0.0

    risk_multiplier = {
        "Strong Buy": 1.0,
        "Buy": 0.9,
        "Hold": 0.8,
    }.get(risk_tier, 0.8)
    adjusted = int(max(0, min(100, base * risk_multiplier)))
    return base, adjusted, _tier_from_score(adjusted), "risk_adjusted", risk_multiplier


def compute_position_action(overall_tier: str, risk_level: str) -> dict[str, Any]:
    if overall_tier == "Strong Buy" and risk_level == "Low Risk":
        return {"action": "MAXIMUM_SIZE", "max_position_pct": 2.5}
    if overall_tier in {"Strong Buy", "Buy"}:
        return {"action": "STANDARD_SIZE", "max_position_pct": 1.5}
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
    atr_14 = float(last.get("atr_14") or 0.0)
    atr_60 = None
    if rows:
        vals = [float(r.get("atr_14") or 0.0) for r in rows[-60:] if float(r.get("atr_14") or 0.0) > 0]
        if vals:
            atr_60 = sum(vals) / len(vals)
    atr_pct = (atr_14 / close * 100.0) if close > 0 and atr_14 > 0 else 0.0

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
        atr_60=atr_60,
    )
    risk_score, risk_level, risk_desc = compute_risk_score(
        rr_ratio=rr_ratio,
        atr_pct=atr_pct,
        adtv_kwd=adtv_kwd,
        spread_pct=spread_pct,
        circuit_distance_pct=float((circuit_result or {}).get("nearest_circuit_pct") or 99.0),
    )
    risk_tier = "Buy" if risk_level == "Low Risk" else "Hold" if risk_level == "Moderate Risk" else "Sell"

    base, adjusted, overall_tier, overall_desc, risk_multiplier = compute_overall_score(
        potential_raw=potential_score,
        timing_raw=timing_score,
        risk_raw=risk_score,
        risk_tier=risk_tier,
    )
    position_action = compute_position_action(overall_tier=overall_tier, risk_level=risk_level)

    return {
        "potential": {"score": potential_score, "tier": potential_tier, "description": potential_desc},
        "timing": {"score": timing_score, "tier": timing_tier, "description": timing_desc},
        "risk": {"score": risk_score, "level": risk_level, "tier": risk_tier, "description": risk_desc},
        "overall": {
            "base_score": base,
            "score": adjusted,
            "tier": overall_tier,
            "description": overall_desc,
            "risk_multiplier": risk_multiplier,
            "adjustment_factor": risk_multiplier,
            "risk_raw": risk_score,
        },
        "position_action": position_action,
    }

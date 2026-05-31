from __future__ import annotations

from typing import Dict, List, Mapping, Optional


def _safe_float(v: object) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _invalidation_conditions(rec: Mapping[str, object], ind: Mapping[str, object]) -> List[str]:
    recommendation = str(rec.get("recommendation") or "NEUTRAL").upper()
    stage = str(rec.get("stage") or "NEUTRAL_AMBIGUOUS").upper()
    items: List[str] = []

    if recommendation in ("BUY", "WATCHLIST"):
        items.append("CMF turns negative for several sessions")
        items.append("OBV slope rolls over and stays below zero")
        items.append("Price closes back below 50-day moving average")
        items.append("Breakout fails and price falls under breakout level")

    if stage in ("MARKUP", "EARLY_MARKUP"):
        items.append("MACD histogram slope flips negative with rising traded value")

    if recommendation in ("REDUCE", "SELL", "AVOID"):
        items.append("CMF recovers above +0.05 and OBV slope turns positive")
        items.append("Fresh 50-day Donchian breakout with strong close location")

    return items


def explain(
    rec: Mapping[str, object],
    ind: Mapping[str, object],
    pattern_match: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    supporting: List[str] = []
    conflicting: List[str] = []

    tvr = _safe_float(ind.get("traded_value_ratio_20d"))
    if tvr is not None and tvr > 1.5:
        supporting.append(f"Traded value {tvr:.1f}x average")
    elif tvr is not None and tvr < 0.6:
        conflicting.append(f"Traded value muted ({tvr:.1f}x average)")

    cmf = _safe_float(ind.get("cmf_20"))
    if cmf is not None and cmf > 0.05:
        supporting.append(f"Money flowing in (CMF +{cmf:.2f})")
    elif cmf is not None and cmf < -0.05:
        conflicting.append(f"Money flowing out (CMF {cmf:.2f})")

    obv_slope = _safe_float(ind.get("obv_slope_20d"))
    if obv_slope is not None and obv_slope > 0:
        supporting.append("OBV rising (accumulation)")
    elif obv_slope is not None and obv_slope < 0:
        conflicting.append("OBV falling (distribution)")

    if int(_safe_float(ind.get("failed_breakout_flag")) or 0) == 1:
        conflicting.append("Recent breakout failed")

    rr = _safe_float(ind.get("risk_reward_ratio"))
    if rr is not None and rr >= 2.0:
        supporting.append(f"Risk/reward is favorable ({rr:.2f})")
    elif rr is not None and rr < 2.0:
        conflicting.append(f"Risk/reward below threshold ({rr:.2f})")

    if int(_safe_float(ind.get("high_volume_weak_close_flag")) or 0) == 1:
        conflicting.append("High-volume weak close suggests distribution")

    analog_text: List[str] = []
    source = pattern_match or rec.get("pattern_match") or {}
    analogs = list(source.get("nearest_analogs") or [])
    for a in analogs[:3]:
        ticker = str(a.get("ticker") or "?")
        dt = str(a.get("date") or "?")
        tpe = str(a.get("type") or "neutral")
        outcome = "rose" if tpe == "takeoff" else ("fell" if tpe == "crash" else "stayed flat")
        analog_text.append(f"{ticker} on {dt} (later {outcome})")

    return {
        "why_supporting": supporting,
        "why_conflicting": conflicting,
        "historical_analogs": analog_text,
        "what_invalidates": _invalidation_conditions(rec, ind),
    }

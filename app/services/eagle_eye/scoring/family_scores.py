from __future__ import annotations

from typing import Dict, Mapping, Optional


def _safe_float(v: object, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):
        return default
    return f


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(v)))


def _indicator(ind: Mapping[str, object], key: str, default: float) -> float:
    v = _safe_float(ind.get(key), default)
    return float(v if v is not None else default)


def liquidity_score(ind: Mapping[str, object]) -> float:
    score = 50.0

    tvr = _indicator(ind, "traded_value_ratio_20d", 1.0)
    if tvr >= 2.0:
        score += 20.0
    elif tvr >= 1.5:
        score += 12.0
    elif tvr < 0.5:
        score -= 15.0

    cmf_20 = _indicator(ind, "cmf_20", 0.0)
    if cmf_20 > 0.05:
        score += 12.0
    elif cmf_20 < -0.05:
        score -= 15.0

    if _indicator(ind, "obv_slope_20d", 0.0) > 0:
        score += 12.0
    else:
        score -= 10.0

    ud_ratio = _indicator(ind, "up_down_volume_ratio_20d", 1.0)
    if ud_ratio > 1.2:
        score += 10.0
    elif ud_ratio < 0.8:
        score -= 12.0

    if int(_indicator(ind, "high_volume_weak_close_flag", 0.0)) == 1:
        score -= 15.0

    return _clip(score)


def trend_score(ind: Mapping[str, object]) -> float:
    score = 50.0

    if _indicator(ind, "stock_close_vs_200sma", 0.0) > 0:
        score += 15.0
    else:
        score -= 15.0

    if _indicator(ind, "stock_close_vs_50sma", 0.0) > 0:
        score += 10.0
    else:
        score -= 8.0

    if _indicator(ind, "stock_50sma_slope_20d", 0.0) > 0:
        score += 15.0
    else:
        score -= 12.0

    if _indicator(ind, "market_close_vs_200sma", 0.0) > 0:
        score += 10.0
    else:
        score -= 10.0

    return _clip(score)


def _rsi_to_score(rsi: float) -> float:
    if 50 <= rsi <= 65:
        return 80.0
    if 65 < rsi <= 75:
        return 90.0
    if 40 <= rsi < 50:
        return 60.0
    if rsi > 75:
        return 70.0
    if 30 <= rsi < 40:
        return 35.0
    return 20.0


def _cci_to_score(cci: float) -> float:
    if cci > 200:
        return 70.0
    if cci > 100:
        return 90.0
    if cci > 0:
        return 70.0
    if cci > -100:
        return 40.0
    return 25.0


def momentum_score(ind: Mapping[str, object]) -> float:
    rsi_score = _rsi_to_score(_indicator(ind, "rsi_14", _indicator(ind, "rsi", 50.0)))
    cci_score = _cci_to_score(_indicator(ind, "cci_20", _indicator(ind, "cci", 0.0)))

    oscillator_subscore = 0.60 * rsi_score + 0.40 * cci_score

    score = 0.0
    score += 0.40 * oscillator_subscore
    score += 0.25 * (70.0 if _indicator(ind, "relative_strength_3m", 0.0) > 0 else 35.0)
    score += 0.20 * (80.0 if _indicator(ind, "macd_histogram_slope_5d", 0.0) > 0 else 30.0)
    score += 0.15 * (75.0 if _indicator(ind, "return_3m", 0.0) > 0 else 35.0)
    return _clip(score)


def geometry_score(ind: Mapping[str, object]) -> float:
    score = 50.0

    if _indicator(ind, "bb_width_percentile_252d", 0.5) < 0.20:
        score += 15.0

    if int(_indicator(ind, "donchian_breakout_50d", 0.0)) == 1:
        if _indicator(ind, "close_location_value", 0.0) > 0.5:
            score += 20.0
        else:
            score += 5.0

    if int(_indicator(ind, "failed_breakout_flag", 0.0)) == 1:
        score -= 25.0

    dist_res = _indicator(ind, "distance_to_major_resistance", 0.0)
    if dist_res > 0.10:
        score += 10.0
    elif dist_res < 0.03:
        score -= 10.0

    return _clip(score)


def risk_reward_score(ind: Mapping[str, object]) -> float:
    rr = _indicator(ind, "risk_reward_ratio", 0.0)
    if rr >= 3.0:
        score = 90.0
    elif rr >= 2.0:
        score = 70.0
    elif rr >= 1.5:
        score = 50.0
    else:
        score = 25.0

    ext = _indicator(ind, "price_extension_from_50sma", 0.0)
    if ext > 0.15:
        score -= 25.0
    elif ext > 0.10:
        score -= 12.0

    # Cheap-vs-buying gate: high R:R only counts when buyers are actually showing up.
    cmf = _indicator(ind, "cmf_20", 0.0)
    obv_slope = _indicator(ind, "obv_slope_20d", 0.0)
    plus_di = _indicator(ind, "plus_di", 0.0)
    minus_di = _indicator(ind, "minus_di", 0.0)

    support = 0
    if cmf > 0:
        support += 1
    if obv_slope > 0:
        support += 1
    if plus_di > minus_di:
        support += 1

    if support == 0:
        score = min(score, 30.0)
    elif support == 1:
        score = min(score, 55.0)

    return _clip(score)


def compute_family_scores(indicators: Mapping[str, object]) -> Dict[str, float]:
    liq = liquidity_score(indicators)
    trd = trend_score(indicators)
    mom = momentum_score(indicators)
    geo = geometry_score(indicators)
    rr = risk_reward_score(indicators)

    total = (
        0.30 * liq
        + 0.20 * trd
        + 0.20 * mom
        + 0.15 * geo
        + 0.15 * rr
    )

    return {
        "liquidity": round(liq, 2),
        "trend": round(trd, 2),
        "momentum": round(mom, 2),
        "geometry": round(geo, 2),
        "risk_reward": round(rr, 2),
        "total_score": round(_clip(total), 2),
    }

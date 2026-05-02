"""Whale Flow Proxy Engine.

Pure-Python, stateless scoring engine that estimates an *estimated whale flow proxy*
from EOD OHLCV, liquidity, VWAP behavior, volume behavior and multi-timeframe
confirmation. Standard library only.

NOTE: This engine does NOT identify actual institutions. It estimates a proxy
based on observable market behavior.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------------------

def safe_divide(n: float, d: float, default: float = 0.0) -> float:
    """Return n/d, or `default` when |d| is effectively zero."""
    return default if abs(d) < 1e-10 else n / d


def clip(v: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp `v` into the inclusive [low, high] range."""
    return max(low, min(high, v))


def percentile_rank(raw: float, history: List[float]) -> float:
    """Empirical percentile rank of `raw` within `history`, clipped to [0.05, 0.95].

    Returns 0.5 when history is empty, has fewer than 2 entries, or is constant.
    """
    if not history or len(history) < 2 or max(history) - min(history) < 1e-10:
        return 0.5
    less = sum(1 for x in history if x < raw)
    eq = sum(1 for x in history if abs(x - raw) < 1e-10)
    return clip((less + 0.5 * eq) / len(history), 0.05, 0.95)


def normalize_factor(raw: float, history: List[float]) -> float:
    """Normalize `raw` against `history` using percentile_rank."""
    return percentile_rank(raw, history)


def select_history(
    h90: Dict[str, List[float]],
    h252: Optional[Dict[str, List[float]]],
    horizon: str,
    regime: str,
    key: str,
) -> List[float]:
    """Pick the history series to use based on horizon and regime."""
    if horizon == "252d" and h252 and key in h252 and len(h252[key]) >= 180:
        return h252[key]
    if (
        horizon == "auto"
        and regime in ("base_building", "choppy")
        and h252
        and key in h252
        and len(h252[key]) >= 180
    ):
        return h252[key]
    return h90.get(key, [])


# ---------------------------------------------------------------------------
# 2. Factor calculations
# ---------------------------------------------------------------------------

def calculate_factors(inp: Dict) -> Dict[str, float]:
    """Compute the seven normalized factors N, V, W, A, C, R, D."""
    h90 = inp["history_90d"]
    h252 = inp.get("history_252d")
    horizon = inp.get("normalization_horizon", "90d")
    regime = inp.get("long_term_regime", "unknown")
    vwap = inp["anchored_vwap"]
    price = inp["current_price"]

    factors: Dict[str, float] = {}

    factors["N"] = normalize_factor(
        inp["net_liquidity_3d_avg"],
        select_history(h90, h252, horizon, regime, "net_liquidity"),
    )

    rel_vol = safe_divide(inp["volume_today"], inp["volume_20d_avg"], default=1.0)
    factors["V"] = normalize_factor(
        rel_vol,
        select_history(h90, h252, horizon, regime, "rel_volume"),
    )

    # VWAP Strength: 30% Position + 40% Time-Above + 30% Slope (Refinement B)
    pos = normalize_factor(
        safe_divide(price - vwap["value"], vwap["value"], 0.0),
        select_history(h90, h252, horizon, regime, "vwap_position"),
    )
    time_above = normalize_factor(
        vwap["time_above_vwap_ratio_10d"],
        select_history(h90, h252, horizon, regime, "time_above_vwap"),
    )
    slp = normalize_factor(
        vwap["slope_5d"],
        select_history(h90, h252, horizon, regime, "vwap_slope"),
    )
    factors["W"] = clip(0.30 * pos + 0.40 * time_above + 0.30 * slp, 0.05, 0.95)

    factors["A"] = normalize_factor(
        inp["ad_line_slope_5d"],
        select_history(h90, h252, horizon, regime, "ad_slope"),
    )
    factors["C"] = clip(
        safe_divide(inp["cmf_10d"] + 0.30, 0.60, default=0.5),
        0.05,
        0.95,
    )

    comp_raw = 1.0 / (
        safe_divide(inp["price_range_today"], inp["atr_20d"], default=1.0) + 0.01
    )
    factors["R"] = normalize_factor(
        comp_raw,
        select_history(h90, h252, horizon, regime, "range_compression"),
    )

    daily_range = inp["high_price"] - inp["low_price"]
    close_loc = clip(
        safe_divide(inp["close_price"] - inp["low_price"], daily_range, 0.5),
        0.0,
        1.0,
    )
    close_near_low = 1.0 - close_loc
    range_exp = normalize_factor(
        safe_divide(inp["price_range_today"], inp["atr_20d"], 1.0),
        select_history(h90, h252, horizon, regime, "range_expansion"),
    )
    downside_raw = (
        close_near_low * range_exp * clip(safe_divide(factors["V"], 0.95), 0.0, 1.0)
    )
    hist_d = select_history(h90, h252, horizon, regime, "downside_pressure")
    factors["D"] = (
        normalize_factor(downside_raw, hist_d)
        if len(hist_d) >= 2
        else clip(1.0 - factors["R"], 0.05, 0.95)
    )

    return factors


# ---------------------------------------------------------------------------
# 3. Scoring & bias
# ---------------------------------------------------------------------------

def calculate_scores(factors: Dict[str, float]) -> Tuple[int, int]:
    """Compute (accumulation_score, distribution_score) as integers in [0, 100]."""
    N, V, W, A, C, R, D = (factors[k] for k in "NVWACRD")
    acc = 25 * N + 20 * V + 20 * W + 15 * A + 10 * C + 10 * R
    dist = 25 * (1 - N) + 20 * V + 20 * (1 - W) + 15 * (1 - A) + 10 * (1 - C) + 10 * D
    return int(round(clip(acc, 0, 100))), int(round(clip(dist, 0, 100)))


def determine_bias(acc: int, dist: int) -> str:
    """Return 'accumulation' | 'distribution' | 'neutral' based on score gap."""
    diff = acc - dist
    if diff > 20:
        return "accumulation"
    if diff < -20:
        return "distribution"
    return "neutral"


# ---------------------------------------------------------------------------
# 4. Persistence bonus
# ---------------------------------------------------------------------------

def calculate_persistence_bonus(
    hist_30d: List[float],
    threshold: float = 0.60,
    min_consec: int = 8,
    max_bonus: float = 0.10,
) -> float:
    """Bonus when the most recent values stay above `threshold` for >= min_consec sessions."""
    if not hist_30d or len(hist_30d) < min_consec:
        return 0.0
    consec = 0
    for v in reversed(hist_30d):
        if v >= threshold:
            consec += 1
        else:
            break
    if consec < min_consec:
        return 0.0
    return min(max_bonus, (consec - min_consec + 1) * 0.01)


# ---------------------------------------------------------------------------
# 5. Dynamic sigmoid & flow estimation
# ---------------------------------------------------------------------------

def get_dynamic_midpoint(market_bias: str, active_bias: str) -> float:
    """Sigmoid midpoint adjusted for market regime alignment."""
    if market_bias == "bullish":
        return 60.0 if active_bias == "accumulation" else 72.0
    if market_bias == "bearish":
        return 72.0 if active_bias == "accumulation" else 55.0
    return 65.0


def estimate_participation_ratio(
    score: float, k: float = 0.12, midpoint: float = 65.0
) -> float:
    """Logistic sigmoid mapping a 0-100 score to a participation ratio in (0, 1)."""
    return 1.0 / (1.0 + math.exp(-k * (score - midpoint)))


def estimate_whale_flow_proxy(
    total_traded_value: float,
    accum_score: int,
    dist_score: int,
    confidence: float,
    market_bias: str,
    active_bias: str,
) -> List[float]:
    """Estimate a [min, max] flow proxy band as a fraction of total traded value."""
    active_score = max(accum_score, dist_score)
    dynamic_mid = get_dynamic_midpoint(market_bias, active_bias)
    participation = estimate_participation_ratio(active_score, k=0.12, midpoint=dynamic_mid)
    base_flow = total_traded_value * participation * confidence
    return [round(base_flow * 0.85, 2), round(base_flow * 1.15, 2)]


# ---------------------------------------------------------------------------
# 6. Liquidity gate
# ---------------------------------------------------------------------------

def check_liquidity_gate(total_traded_value: float, minimum_threshold: float) -> bool:
    """Return True when traded value meets the minimum threshold."""
    return total_traded_value >= minimum_threshold


# ---------------------------------------------------------------------------
# 7. Action engine (with volume-exhaustion guard)
# ---------------------------------------------------------------------------

def determine_action(
    bias: str,
    accum_score: int,
    dist_score: int,
    price: float,
    vwap_value: float,
    ad_slope: float,
    alignment: str,
    liquidity_passed: bool,
    factors: Dict[str, float],
) -> str:
    """Return 'BUY' | 'SELL' | 'WAIT'. Liquidity gate and exhaustion guards apply."""
    if not liquidity_passed:
        return "WAIT"
    if (
        bias == "accumulation"
        and accum_score >= 70
        and price > vwap_value
        and ad_slope > 0
        and alignment != "conflicting"
    ):
        return "BUY"
    if (
        bias == "distribution"
        and dist_score >= 70
        and price < vwap_value
        and ad_slope < 0
        and alignment != "conflicting"
    ):
        if factors.get("V", 0.0) > 0.92:
            # Volume exhaustion / capitulation guard.
            return "WAIT"
        return "SELL"
    return "WAIT"


# ---------------------------------------------------------------------------
# 8. Alignment & confidence
# ---------------------------------------------------------------------------

def determine_alignment(active_bias: str, higher_tf_bias: str) -> str:
    """Compare active bias against higher-timeframe bias."""
    if active_bias == "neutral" or higher_tf_bias == "neutral":
        return "mixed"
    return "aligned" if active_bias == higher_tf_bias else "conflicting"


def calculate_confidence_factor(
    data_quality: str,
    alignment: str,
    market_bias: str,
    active_bias: str,
) -> float:
    """Combine data quality, alignment, and market regime into [0.60, 1.00] confidence."""
    base = 1.00 if data_quality == "direct" else 0.80
    align_mult = {"aligned": 1.00, "mixed": 0.85, "conflicting": 0.70}.get(alignment, 0.85)
    if active_bias == "accumulation":
        market_mult = {"bullish": 1.00, "neutral": 0.90, "bearish": 0.75}.get(market_bias, 0.90)
    elif active_bias == "distribution":
        market_mult = {"bearish": 1.00, "neutral": 0.90, "bullish": 0.75}.get(market_bias, 0.90)
    else:
        market_mult = 0.90
    return clip(base * align_mult * market_mult, 0.60, 1.00)


# ---------------------------------------------------------------------------
# 9. Contribution breakdown & alert
# ---------------------------------------------------------------------------

def calculate_contribution_breakdown(
    factors: Dict[str, float], bias: str
) -> Dict[str, float]:
    """Per-factor weighted contribution to the active score (rounded)."""
    N, V, W, A, C, R, D = (factors[k] for k in "NVWACRD")
    if bias == "accumulation":
        return {
            "net_liquidity": round(25 * N, 2),
            "relative_volume": round(20 * V, 2),
            "vwap_strength": round(20 * W, 2),
            "ad_trend": round(15 * A, 2),
            "cmf": round(10 * C, 2),
            "range_compression": round(10 * R, 2),
        }
    return {
        "negative_net_liquidity": round(25 * (1 - N), 2),
        "relative_volume": round(20 * V, 2),
        "weak_vwap": round(20 * (1 - W), 2),
        "falling_ad_trend": round(15 * (1 - A), 2),
        "negative_cmf": round(10 * (1 - C), 2),
        "downside_pressure": round(10 * D, 2),
    }


def generate_alert(
    ticker: str,
    price: float,
    accum_score: int,
    dist_score: int,
    bias: str,
    action: str,
    factors: Dict[str, float],
    breakdown: Dict[str, float],
    flow_range: List[float],
    confidence: float,
    alignment: str,
    vwap_value: float,
    liquidity_passed: bool,
    persistence_bonus: float,
) -> Dict:
    """Assemble the human-facing alert payload."""
    active_score = accum_score if bias == "accumulation" else dist_score
    if active_score >= 80 and liquidity_passed:
        alert_level = "STRONG"
    elif active_score >= 70 and liquidity_passed:
        alert_level = "MODERATE"
    else:
        alert_level = "WEAK"

    primary = max(breakdown, key=lambda k: breakdown[k])
    conf_signals = [
        k for k, _ in sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[1:4]
    ]

    if bias == "accumulation":
        key = f"Break above {round(price * 1.02, 3)} on volume > 2x confirms"
        inv = f"Close below {round(vwap_value * 0.99, 3)} negates setup"
    elif bias == "distribution":
        key = f"Break below {round(price * 0.98, 3)} on volume > 2x confirms"
        inv = f"Close above {round(vwap_value * 1.01, 3)} weakens bearish setup"
    else:
        key = "Monitor VWAP and volume for directional confirmation"
        inv = "N/A"

    act_msg = {
        "BUY": "Consider staged entry only after confirmation; use VWAP as invalidation level.",
        "SELL": "Consider reducing exposure or avoiding new entry until pressure fades.",
        "WAIT": "No high-confidence action; monitor VWAP, volume, and liquidity confirmation.",
    }.get(action, "Monitor setup.")

    if persistence_bonus > 0.05 and active_score >= 60:
        phase = f"silent_{bias}"
    elif active_score >= 75:
        phase = "active"
    else:
        phase = "building"

    return {
        "ticker": ticker,
        "alert_level": alert_level,
        "bias": bias.capitalize(),
        "action": action,
        "accumulation_score": accum_score,
        "distribution_score": dist_score,
        "primary_driver": primary,
        "confirmation_signals": conf_signals,
        "estimated_whale_flow_proxy": (
            f"KD {flow_range[0] / 1e6:.2f}M – KD {flow_range[1] / 1e6:.2f}M"
        ),
        "confidence": f"{round(confidence * 100):.0f}%",
        "timeframe_alignment": alignment,
        "key_level": key,
        "invalidation": inv,
        "liquidity_status": "PASSED" if liquidity_passed else "FAILED",
        "suggested_action": act_msg,
        "phase": phase,
    }


# ---------------------------------------------------------------------------
# 10. Main orchestrator
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = (
    "ticker",
    "current_price",
    "total_traded_value",
    "minimum_total_traded_value",
    "history_90d",
    "anchored_vwap",
    "data_quality",
    "higher_timeframe_bias",
    "market_context",
)


def run_whale_engine(input_dict: Dict) -> Dict:
    """Run the full estimated-whale-flow-proxy pipeline. Pure & JSON-safe."""
    for f in REQUIRED_FIELDS:
        if f not in input_dict:
            raise ValueError(f"Missing required field: {f}")

    factors = calculate_factors(input_dict)
    acc, dist = calculate_scores(factors)

    hist_n_30 = input_dict["history_90d"].get("net_liquidity", [])[-30:]
    persist_bonus = calculate_persistence_bonus(hist_n_30)
    if persist_bonus > 0:
        acc = int(round(clip(acc + persist_bonus * 10, 0, 100)))

    bias = determine_bias(acc, dist)
    alignment = determine_alignment(bias, input_dict["higher_timeframe_bias"])
    confidence = calculate_confidence_factor(
        input_dict["data_quality"],
        alignment,
        input_dict["market_context"]["market_bias"],
        bias,
    )
    flow_range = estimate_whale_flow_proxy(
        input_dict["total_traded_value"],
        acc,
        dist,
        confidence,
        input_dict["market_context"]["market_bias"],
        bias,
    )

    liquidity_passed = check_liquidity_gate(
        input_dict["total_traded_value"], input_dict["minimum_total_traded_value"]
    )
    action = determine_action(
        bias,
        acc,
        dist,
        input_dict["current_price"],
        input_dict["anchored_vwap"]["value"],
        input_dict["ad_line_slope_5d"],
        alignment,
        liquidity_passed,
        factors,
    )

    breakdown = calculate_contribution_breakdown(factors, bias)
    alert = generate_alert(
        input_dict["ticker"],
        input_dict["current_price"],
        acc,
        dist,
        bias,
        action,
        factors,
        breakdown,
        flow_range,
        confidence,
        alignment,
        input_dict["anchored_vwap"]["value"],
        liquidity_passed,
        persist_bonus,
    )

    return {
        "ticker": input_dict["ticker"],
        "timeframe": input_dict.get("timeframe", "daily"),
        "as_of_date": input_dict.get("as_of_date", ""),
        "accumulation_score": acc,
        "distribution_score": dist,
        "bias": bias,
        "action": action,
        "estimated_whale_flow_proxy_range": flow_range,
        "confidence": round(confidence, 2),
        "alignment": alignment,
        "liquidity_gate_passed": liquidity_passed,
        "persistence_bonus": round(persist_bonus, 3),
        "factors": {k: round(v, 4) for k, v in factors.items()},
        "contribution_breakdown": breakdown,
        "alert": alert,
    }

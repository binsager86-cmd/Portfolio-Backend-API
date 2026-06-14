from __future__ import annotations

from typing import Any

from app.services.signal_engine.config.kuwait_constants import (
    CIRCUIT_BUFFER_PCT,
    CIRCUIT_UPPER_PCT,
    align_to_tick,
)


def _compute_momentum_exhaustion(
    rows: list[dict[str, Any]],
    current_price: float,
) -> tuple[int, list[str]]:
    """Compute momentum exhaustion score for trim/exit handling."""
    if not rows:
        return 0, ["insufficient_data"]

    last = rows[-1]
    rsi = float(last.get("rsi_14") or 50.0)
    adx = float(last.get("adx_14") or 20.0)
    ema_20 = float(last.get("ema_20") or current_price)
    atr = float(last.get("atr_14") or 1.0)

    score = 0
    reasons: list[str] = []

    if rsi >= 80.0:
        if adx < 25.0:
            score += 62
            reasons.append("RSI extreme while trend strength weakens")
        elif adx >= 30.0:
            score += 38
            reasons.append("RSI elevated but ADX still strong")
        else:
            score += 48
            reasons.append("RSI elevated above 80")
    elif rsi >= 75.0:
        score += 28
        reasons.append("RSI elevated above 75")
    elif rsi >= 70.0:
        score += 12
        reasons.append("RSI elevated above 70")

    if len(rows) >= 3:
        m2 = float(rows[-3].get("macd_hist") or 0.0)
        m1 = float(rows[-2].get("macd_hist") or 0.0)
        m0 = float(rows[-1].get("macd_hist") or 0.0)
        if m0 < 0 and m1 > 0:
            score += 8
            reasons.append("MACD histogram minor pullback")
        elif m0 < m1 < m2 and m0 < 0:
            score += 10
            reasons.append("MACD momentum rollover")

    if atr > 0:
        stretch_atr = (current_price - ema_20) / atr
        if stretch_atr >= 3.5:
            score += 10
            reasons.append("Price stretched above EMA20")

    if float(last.get("cmf_20") or 0.0) < -0.08:
        score += 6
        reasons.append("Negative money flow detected")

    return max(0, min(100, int(score))), reasons


def _detect_distribution(rows: list[dict[str, Any]], pnl_pct: float) -> bool:
    if len(rows) < 6 or pnl_pct < 12.0:
        return False

    last = rows[-1]
    cmf = float(last.get("cmf_20") or 0.0)
    is_bearish_candle = float(last.get("close") or 0.0) < float(last.get("open") or 0.0)

    obv_values = [float(r.get("obv") or 0.0) for r in rows[-6:]]
    obv_down = obv_values[-1] < obv_values[0]

    return cmf <= -0.08 and is_bearish_candle and obv_down


def generate_exit_signal(
    stock_code: str,
    rows: list[dict[str, Any]],
    entry_price: float,
    bars_held: int,
) -> dict[str, Any]:
    """Generate trim/exit recommendation for an existing long position."""
    if not rows:
        return {
            "stock_code": stock_code,
            "action": "HOLD",
            "urgency": "LOW",
            "reasons": ["insufficient_data"],
            "suggested_trim_pct": 0,
            "pnl_pct": 0.0,
            "trailing_stop": None,
            "near_circuit": False,
            "momentum_exhaustion_score": 0,
            "distribution_detected": False,
        }

    last = rows[-1]
    close = float(last.get("close") or 0.0)
    low = float(last.get("low") or close)
    atr = float(last.get("atr_14") or max(close * 0.01, 1.0))

    pnl_pct = ((close - entry_price) / entry_price * 100.0) if entry_price > 0 else 0.0
    trailing_stop = align_to_tick(close - 1.8 * atr)

    if bars_held < 5:
        return {
            "stock_code": stock_code,
            "action": "HOLD",
            "urgency": "LOW",
            "reasons": ["insufficient_data"],
            "suggested_trim_pct": 0,
            "pnl_pct": round(pnl_pct, 2),
            "trailing_stop": trailing_stop,
            "near_circuit": False,
            "momentum_exhaustion_score": 0,
            "distribution_detected": False,
        }

    stop_hit = low <= trailing_stop

    near_circuit = False
    if len(rows) >= 2:
        prev_close = float(rows[-2].get("close") or 0.0)
        if prev_close > 0 and close > 0:
            upper = prev_close * (1.0 + CIRCUIT_UPPER_PCT)
            gap_to_upper = (upper - close) / close
            near_circuit = gap_to_upper <= max(CIRCUIT_BUFFER_PCT, 0.015)

    exhaustion_score, exhaustion_reasons = _compute_momentum_exhaustion(rows, current_price=close)
    distribution = _detect_distribution(rows, pnl_pct)

    action = "HOLD"
    urgency = "LOW"
    suggested_trim = 0
    reasons = exhaustion_reasons[:] if exhaustion_reasons else []

    if stop_hit:
        action = "EXIT"
        urgency = "CRITICAL"
        reasons = ["trailing_stop_breached"]
    elif near_circuit and pnl_pct >= 12.0:
        action = "TRIM"
        urgency = "HIGH"
        suggested_trim = 50
        reasons.append("near_circuit_with_large_gain")
    elif distribution and exhaustion_score >= 75:
        action = "TRIM"
        urgency = "HIGH"
        suggested_trim = 50
        reasons.append("distribution_with_high_exhaustion")
    elif distribution:
        action = "TRIM"
        urgency = "MEDIUM"
        suggested_trim = 50
        reasons.append("distribution_detected")
    elif bars_held >= 20 and pnl_pct > 0:
        action = "TRIM"
        urgency = "MEDIUM"
        suggested_trim = 50
        reasons.append("time_stop")
    elif exhaustion_score >= 80:
        action = "TRIM"
        urgency = "MEDIUM"
        suggested_trim = 50
        reasons.append("very_high_exhaustion")
    elif exhaustion_score >= 70 and pnl_pct >= 12.0:
        action = "TRIM"
        urgency = "MEDIUM"
        suggested_trim = 50
        reasons.append("exhaustion_with_profit")

    return {
        "stock_code": stock_code,
        "action": action,
        "urgency": urgency,
        "reasons": reasons or ["none"],
        "suggested_trim_pct": suggested_trim,
        "pnl_pct": round(pnl_pct, 2),
        "trailing_stop": trailing_stop,
        "near_circuit": near_circuit,
        "momentum_exhaustion_score": exhaustion_score,
        "distribution_detected": distribution,
    }

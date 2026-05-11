"""Exit signal heuristics for Kuwait signal lifecycle management."""
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
    score = 0
    reasons: list[str] = []
    if not rows:
        return score, reasons

    last = rows[-1]
    rsi = float(last.get("rsi_14") or 50.0)
    adx = float(last.get("adx_14") or 20.0)
    macd_now = float(last.get("macd_hist") or 0.0)
    macd_prev = float(rows[-2].get("macd_hist") or 0.0) if len(rows) >= 2 else macd_now
    macd_prev2 = float(rows[-3].get("macd_hist") or 0.0) if len(rows) >= 3 else macd_prev

    if rsi >= 80:
        score += 40
        reasons.append("RSI extreme overbought")
    elif rsi >= 70:
        score += 12
        reasons.append("RSI elevated")

    if macd_prev > 0 and macd_now < 0:
        if macd_prev2 > 0:
            score += 20
            reasons.append("MACD bearish reversal")
        else:
            score += 8
            reasons.append("MACD minor pullback")

    atr = float(last.get("atr_14") or 0.0)
    ema20 = float(last.get("ema_20") or current_price)
    if atr > 0 and current_price > ema20:
        stretch = (current_price - ema20) / atr
        if stretch >= 3.5:
            score += 30
            reasons.append("price severely stretched above EMA20")
        elif stretch >= 2.5:
            score += 10
            reasons.append("price moderately stretched above EMA20")
        elif stretch >= 1.5:
            score += 5
            reasons.append("price mildly stretched above EMA20")

    if rsi >= 80 and adx >= 30:
        score = min(score, 45 if score < 50 else score)

    return min(100, score), reasons


def generate_exit_signal(
    stock_code: str,
    rows: list[dict[str, Any]],
    entry_price: float,
    bars_held: int,
) -> dict[str, Any]:
    if not rows:
        return {
            "stock_code": stock_code,
            "action": "HOLD",
            "urgency": "LOW",
            "reasons": ["no_data"],
            "momentum_exhaustion_score": 0,
            "distribution_detected": False,
            "near_circuit": False,
            "pnl_pct": 0.0,
            "suggested_trim_pct": 0,
            "trailing_stop": align_to_tick(entry_price),
        }

    last = rows[-1]
    close = float(last.get("close") or 0.0)
    low = float(last.get("low") or close)
    atr = float(last.get("atr_14") or 0.0)
    cmf = float(last.get("cmf_20") or 0.0)
    obv_now = float(last.get("obv") or 0.0)
    obv_prev = float(rows[-2].get("obv") or obv_now) if len(rows) >= 2 else obv_now
    pnl_pct = ((close - entry_price) / entry_price * 100.0) if entry_price > 0 else 0.0

    trailing_stop = entry_price - (1.5 * atr if atr > 0 else 0.0)
    if atr > 0:
        trailing_stop = max(trailing_stop, close - (2.0 * atr))
    trailing_stop = align_to_tick(trailing_stop)

    score, score_reasons = _compute_momentum_exhaustion(rows, current_price=close)
    distribution_detected = pnl_pct >= 12.0 and (cmf <= -0.08 or obv_now < obv_prev * 0.99)

    near_circuit = False
    if len(rows) >= 2:
        prev_close = float(rows[-2].get("close") or 0.0)
        if prev_close > 0 and close > 0:
            upper = prev_close * (1.0 + CIRCUIT_UPPER_PCT)
            gap_to_upper = (upper - close) / close
            near_circuit = gap_to_upper <= max(CIRCUIT_BUFFER_PCT * 2.0, 0.015)

    if bars_held < 5:
        return {
            "stock_code": stock_code,
            "action": "HOLD",
            "urgency": "LOW",
            "reasons": ["insufficient_data"],
            "momentum_exhaustion_score": score,
            "distribution_detected": distribution_detected,
            "near_circuit": near_circuit,
            "pnl_pct": round(pnl_pct, 2),
            "suggested_trim_pct": 0,
            "trailing_stop": trailing_stop,
        }

    if low <= trailing_stop:
        return {
            "stock_code": stock_code,
            "action": "EXIT",
            "urgency": "CRITICAL",
            "reasons": ["trailing_stop_breached"],
            "momentum_exhaustion_score": score,
            "distribution_detected": distribution_detected,
            "near_circuit": near_circuit,
            "pnl_pct": round(pnl_pct, 2),
            "suggested_trim_pct": 100,
            "trailing_stop": trailing_stop,
        }

    if bars_held >= 20:
        return {
            "stock_code": stock_code,
            "action": "TRIM",
            "urgency": "MEDIUM",
            "reasons": ["time_stop"],
            "momentum_exhaustion_score": score,
            "distribution_detected": distribution_detected,
            "near_circuit": near_circuit,
            "pnl_pct": round(pnl_pct, 2),
            "suggested_trim_pct": 50,
            "trailing_stop": trailing_stop,
        }

    should_trim = score >= 75 or distribution_detected or near_circuit
    urgency = "HIGH" if score >= 75 or near_circuit else "MEDIUM"
    action = "TRIM" if should_trim else "HOLD"
    suggested = 50 if action == "TRIM" else 0
    reasons = score_reasons[:] if score_reasons else []
    if distribution_detected:
        reasons.append("distribution_detected")
    if near_circuit:
        reasons.append("near_upper_circuit")
    if not reasons:
        reasons = ["no_exit_condition"]

    return {
        "stock_code": stock_code,
        "action": action,
        "urgency": urgency if action == "TRIM" else "LOW",
        "reasons": reasons,
        "momentum_exhaustion_score": score,
        "distribution_detected": distribution_detected,
        "near_circuit": near_circuit,
        "pnl_pct": round(pnl_pct, 2),
        "suggested_trim_pct": suggested,
        "trailing_stop": trailing_stop,
    }

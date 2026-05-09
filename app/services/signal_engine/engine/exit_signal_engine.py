"""Exit Signal Engine for Kuwait Signal Engine.
Monitors open positions for:
- ATR-adaptive trailing stops (triggers on intraday low)
- Regime-aware momentum exhaustion (ADX-weighted RSI, 2-bar MACD, swing divergence)
- Smart money distribution (12% P&L threshold, OBV/CMF/volume spike)
- Time-stop (capital efficiency guard)
- Dynamic trim sizing & conflict resolution
Returns actionable HOLD / TRIM / EXIT signals with urgency levels.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.services.signal_engine.config.kuwait_constants import CIRCUIT_UPPER_PCT, align_to_tick


# -- 1. Trailing Stop Calculator ------------------------------------------------
def _compute_trailing_stop(
    rows: list[dict[str, Any]],
    entry_price: float,
    current_price: float,
    highest_since_entry: float,
    atr: float,
) -> float:
    """ATR-adaptive trailing stop. Triggers on intraday low, not close."""
    del rows
    if entry_price <= 0:
        return entry_price - (2.0 * atr)

    pnl_pct = (current_price - entry_price) / entry_price

    if pnl_pct < 0.05:
        return align_to_tick(max(entry_price - (1.5 * atr), entry_price * 0.95))
    if pnl_pct < 0.10:
        return align_to_tick(max(entry_price, entry_price - (1.0 * atr)))
    if pnl_pct < 0.20:
        return align_to_tick(max(highest_since_entry - (1.5 * atr), entry_price))
    return align_to_tick(max(highest_since_entry - (2.0 * atr), entry_price * 1.05))


# -- 2. Momentum Exhaustion Detector -------------------------------------------
def _compute_momentum_exhaustion(rows: list[dict[str, Any]], current_price: float) -> tuple[int, list[str]]:
    """ADX-weighted exhaustion. Requires confirmation, not single-bar noise."""
    if len(rows) < 15:
        return 20, ["insufficient_data"]

    last = rows[-1]
    prev = rows[-2] if len(rows) > 1 else last
    prev2 = rows[-3] if len(rows) > 2 else prev
    reasons: list[str] = []
    score = 0

    adx = float(last.get("adx_14") or 20.0)
    rsi = float(last.get("rsi_14") or 50.0)

    # RSI penalty scaled by trend strength
    if rsi > 80:
        score += 30 if adx < 25 else 15
        reasons.append(f"RSI extremely overbought ({rsi:.0f})")
    elif rsi > 75:
        score += 20 if adx < 25 else 10
        reasons.append(f"RSI overbought ({rsi:.0f})")
    elif rsi > 70:
        score += 12
        reasons.append(f"RSI elevated ({rsi:.0f})")

    # MACD: Require 2-bar confirmation OR RSI>70 confluence
    macd_hist = float(last.get("macd_hist") or 0.0)
    macd_hist_prev = float(prev.get("macd_hist") or 0.0)
    macd_hist_prev2 = float(prev2.get("macd_hist") or 0.0)

    if macd_hist_prev > 0 and macd_hist <= 0:
        if macd_hist_prev2 > 0 or rsi > 70:
            score += 25
            reasons.append("MACD histogram confirmed rollover")
        else:
            score += 8
            reasons.append("MACD minor pullback (monitor)")

    # EMA Stretch
    ema20 = float(last.get("ema_20") or current_price)
    atr = float(last.get("atr_14") or (current_price * 0.015))
    stretch = (current_price - ema20) / atr if atr > 0 else 0.0
    if stretch > 3.0:
        score += 25
        reasons.append(f"Parabolic extension ({stretch:.1f}xATR)")
    elif stretch > 2.5:
        score += 15
        reasons.append(f"Extended ({stretch:.1f}xATR)")

    # Swing-based divergence
    if len(rows) >= 20:
        highs = [float(r.get("high") or 0.0) for r in rows[-20:]]
        rsis = [float(r.get("rsi_14") or 50.0) for r in rows[-20:]]
        pivots: list[tuple[int, float, float]] = []
        for i in range(2, len(highs) - 2):
            if highs[i] > max(highs[i - 2 : i]) and highs[i] > max(highs[i + 1 : i + 3]):
                pivots.append((i, highs[i], rsis[i]))
        if len(pivots) >= 2:
            p1, p2 = pivots[-2], pivots[-1]
            if p2[1] > p1[1] and p2[2] < p1[2]:
                score += 20
                reasons.append("Bearish RSI divergence at swing pivots")

    return min(100, score), reasons


# -- 3. Distribution Detector ---------------------------------------------------
def _detect_distribution(rows: list[dict[str, Any]], current_price: float, entry_price: float) -> tuple[bool, list[str]]:
    """Smart money exit detection. 12% threshold for Kuwait cost structure."""
    if len(rows) < 10 or entry_price <= 0:
        return False, []

    pnl_pct = (current_price - entry_price) / entry_price
    if pnl_pct < 0.12:
        return False, []

    last = rows[-1]
    reasons: list[str] = []
    detected = False

    obv_vals = [float(r.get("obv") or 0.0) for r in rows[-10:] if r.get("obv") is not None]
    if len(obv_vals) >= 5:
        obv_slope = (obv_vals[-1] - obv_vals[0]) / max(abs(obv_vals[0]), 1.0)
        if obv_slope < -0.05:
            reasons.append("OBV declining while price rising")
            detected = True

    cmf = float(last.get("cmf_20") or 0.0)
    if cmf < -0.05:
        reasons.append(f"CMF negative ({cmf:.3f}) - distribution")
        detected = True

    close = float(last.get("close") or 0.0)
    open_ = float(last.get("open") or close)
    volume = float(last.get("volume") or 0.0)
    median_vol = float(np.median([float(r.get("volume") or 0.0) for r in rows[-20:]]))
    if close < open_ and median_vol > 0 and volume > median_vol * 1.5:
        reasons.append(f"High volume down day ({volume / median_vol:.1f}x)")
        detected = True

    return detected, reasons


# -- 4. Time-Stop & Dynamic Trim -----------------------------------------------
def _check_time_stop(bars_held: int, pnl_pct: float) -> tuple[bool, str]:
    if bars_held >= 20 and pnl_pct < 3.0:
        return True, f"Time-stop: {bars_held} bars held, only +{pnl_pct:.1f}% P&L"
    return False, ""


def _compute_dynamic_trim(urgency: str, pnl_pct: float, exhaustion_score: int) -> int:
    base = 30 if urgency == "MEDIUM" else 50 if urgency == "HIGH" else 70
    pnl_scale = min(1.5, max(0.5, pnl_pct / 15.0))
    conviction_scale = 1.0 if exhaustion_score >= 70 else 0.7
    return min(80, max(20, int(base * pnl_scale * conviction_scale)))


# -- 5. Main Exit Signal Generator ---------------------------------------------
def generate_exit_signal(
    symbol: str,
    rows: list[dict[str, Any]],
    entry_price: float,
    bars_held: int = 0,
    position_size_pct: float = 0.0,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del position_size_pct, meta
    if not rows or entry_price <= 0:
        return _default_hold(symbol, "invalid_input")

    last = rows[-1]
    current_price = float(last.get("close") or 0.0)
    if current_price <= 0:
        return _default_hold(symbol, "invalid_price")

    pnl_pct = (current_price - entry_price) / entry_price * 100.0
    entry_idx = max(0, len(rows) - 60)
    highest_since_entry = max(float(r.get("high") or 0.0) for r in rows[entry_idx:])
    atr = float(last.get("atr_14") or (current_price * 0.015))

    trailing_stop = _compute_trailing_stop(rows, entry_price, current_price, highest_since_entry, atr)
    dist_to_stop_pct = ((current_price - trailing_stop) / current_price * 100.0) if current_price > 0 else 0.0

    exhaustion_score, exhaustion_reasons = _compute_momentum_exhaustion(rows, current_price)
    distribution_detected, distribution_reasons = _detect_distribution(rows, current_price, entry_price)

    prev_close = float(rows[-2].get("close") or current_price) if len(rows) > 1 else current_price
    upper_limit = prev_close * (1.0 + CIRCUIT_UPPER_PCT)
    circuit_dist_pct = ((upper_limit - current_price) / prev_close * 100.0) if prev_close > 0 else 99.0
    near_circuit = circuit_dist_pct < 1.5

    time_stop_hit, time_stop_reason = _check_time_stop(bars_held, pnl_pct)
    stop_hit = float(last.get("low") or current_price) <= trailing_stop

    action = "HOLD"
    urgency = "LOW"
    reasons: list[str] = []
    suggested_trim_pct = 0

    if bars_held < 5:
        action, urgency = "HOLD", "LOW"
        reasons = ["insufficient_data"]
        exhaustion_reasons = []
        distribution_reasons = []
    elif stop_hit:
        action, urgency = "EXIT", "CRITICAL"
        reasons.append(f"Stop hit at {trailing_stop:.1f} (intraday low)")
    elif time_stop_hit:
        action, urgency = "TRIM", "MEDIUM"
        reasons.append(time_stop_reason)
        suggested_trim_pct = 50
    elif exhaustion_score >= 75 and distribution_detected:
        action, urgency = "TRIM", "HIGH"
        reasons.extend(exhaustion_reasons[:2] + distribution_reasons[:1])
        suggested_trim_pct = _compute_dynamic_trim("HIGH", pnl_pct, exhaustion_score)
    elif exhaustion_score >= 80:
        action, urgency = "TRIM", "MEDIUM"
        reasons.extend(exhaustion_reasons[:2])
        suggested_trim_pct = _compute_dynamic_trim("MEDIUM", pnl_pct, exhaustion_score)
    elif distribution_detected and pnl_pct > 15:
        action, urgency = "TRIM", "MEDIUM"
        reasons.extend(distribution_reasons[:2])
        suggested_trim_pct = _compute_dynamic_trim("MEDIUM", pnl_pct, 60)
    elif near_circuit and pnl_pct > 12:
        action, urgency = "TRIM", "HIGH"
        reasons.append(f"Near +10% circuit ({circuit_dist_pct:.1f}% away)")
        suggested_trim_pct = 40
    else:
        reasons.append("Trend intact - hold position")

    all_reasons = list(dict.fromkeys(reasons + exhaustion_reasons + distribution_reasons))[:4]

    return {
        "symbol": symbol,
        "action": action,
        "urgency": urgency,
        "reasons": all_reasons,
        "current_price": round(current_price, 1),
        "entry_price": round(entry_price, 1),
        "pnl_pct": round(pnl_pct, 2),
        "trailing_stop": round(trailing_stop, 1),
        "distance_to_stop_pct": round(dist_to_stop_pct, 2),
        "momentum_exhaustion_score": exhaustion_score,
        "distribution_detected": distribution_detected,
        "parabolic_extension": exhaustion_score >= 70 and any("Parabolic" in r for r in exhaustion_reasons),
        "near_circuit": near_circuit,
        "suggested_trim_pct": suggested_trim_pct,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _default_hold(symbol: str, reason: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "action": "HOLD",
        "urgency": "LOW",
        "reasons": [reason],
        "current_price": 0.0,
        "entry_price": 0.0,
        "pnl_pct": 0.0,
        "trailing_stop": 0.0,
        "distance_to_stop_pct": 0.0,
        "momentum_exhaustion_score": 0,
        "distribution_detected": False,
        "parabolic_extension": False,
        "near_circuit": False,
        "suggested_trim_pct": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

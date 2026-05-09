"""Entry-timing trigger detectors for BUY-tier setups.

High confluence scores validate setup quality, but this module validates
bar-level timing before entry.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.services.signal_engine.config.model_params import (
    ACCUMULATION_CMF_MIN,
    ACCUMULATION_OBV_MIN_SLOPE_PCT,
    BREAKOUT_RANGE_ATR_MULT_MAX,
    BREAKOUT_RANGE_BARS,
    BREAKOUT_VOLUME_AVG_BARS,
    BREAKOUT_VOLUME_MULT_MIN,
    OBV_SLOPE_BARS,
    PULLBACK_EMA_PROXIMITY_PCT,
    PULLBACK_LOOKBACK_BARS,
    PULLBACK_STOCH_MAX,
)


def _detect_pullback_trigger(rows: list[dict[str, Any]]) -> tuple[bool, int, dict[str, Any]]:
    """Detect pullback continuation trigger near a rising EMA-20."""
    details: dict[str, Any] = {"trigger": "pullback"}

    if len(rows) < max(PULLBACK_LOOKBACK_BARS + 1, 6):
        details["fail"] = "insufficient_history"
        return False, 0, details

    last = rows[-1]
    prev = rows[-2]

    ema_now = last.get("ema_20")
    ema_5_ago = rows[-6].get("ema_20") if len(rows) >= 6 else None
    if ema_now is None or ema_5_ago is None:
        details["fail"] = "ema_20_missing"
        return False, 0, details

    ema_rising = float(ema_now) > float(ema_5_ago)
    details["ema_rising"] = ema_rising
    if not ema_rising:
        details["fail"] = "ema_20_not_rising"
        return False, 0, details

    pullback_window = rows[-PULLBACK_LOOKBACK_BARS:]
    touched = False
    min_dist_pct = 1.0
    for row in pullback_window:
        ema_val = row.get("ema_20")
        low = row.get("low")
        if ema_val is None or low is None:
            continue

        dist_pct = (float(low) - float(ema_val)) / float(ema_val)
        if dist_pct <= PULLBACK_EMA_PROXIMITY_PCT:
            touched = True
        if abs(dist_pct) < min_dist_pct:
            min_dist_pct = abs(dist_pct)

    details["pullback_touched_ema"] = touched
    details["min_dist_to_ema_pct"] = round(min_dist_pct * 100, 2)
    if not touched:
        details["fail"] = "no_pullback_to_ema"
        return False, 0, details

    close_now = float(last.get("close") or 0.0)
    open_now = float(last.get("open") or 0.0)
    high_prev = float(prev.get("high") or 0.0)
    bar_bullish = close_now > open_now
    closes_above_prior_high = close_now > high_prev

    details["bar_bullish"] = bar_bullish
    details["closes_above_prior_high"] = closes_above_prior_high
    if not (bar_bullish and closes_above_prior_high):
        details["fail"] = "no_bullish_confirmation_candle"
        return False, 0, details

    stoch_k = last.get("stoch_k")
    stoch_d = last.get("stoch_d")
    if stoch_k is None or stoch_d is None:
        details["fail"] = "stoch_missing"
        return False, 0, details

    k, d = float(stoch_k), float(stoch_d)
    stoch_ok = (k < PULLBACK_STOCH_MAX) and (k > d)
    details["stoch_k"] = round(k, 1)
    details["stoch_d"] = round(d, 1)
    details["stoch_recovering"] = stoch_ok
    if not stoch_ok:
        details["fail"] = "stoch_not_recovering"
        return False, 0, details

    closeness_pts = int(max(0, 40 - (min_dist_pct * 100 * 20)))
    confirm_strength = (close_now - high_prev) / high_prev if high_prev > 0 else 0
    confirm_pts = int(min(30, confirm_strength * 1500))
    stoch_room_pts = int(max(0, 30 - (k / PULLBACK_STOCH_MAX * 30)))
    strength = max(0, min(100, closeness_pts + confirm_pts + stoch_room_pts))

    details["strength_breakdown"] = {
        "closeness_pts": closeness_pts,
        "confirm_pts": confirm_pts,
        "stoch_room_pts": stoch_room_pts,
    }
    return True, strength, details


def _detect_breakout_trigger(rows: list[dict[str, Any]]) -> tuple[bool, int, dict[str, Any]]:
    """Detect close-confirmed breakout from a tight range with volume expansion."""
    details: dict[str, Any] = {"trigger": "breakout"}

    min_bars = max(BREAKOUT_RANGE_BARS + 1, BREAKOUT_VOLUME_AVG_BARS + 1)
    if len(rows) < min_bars:
        details["fail"] = "insufficient_history"
        return False, 0, details

    last = rows[-1]
    range_window = rows[-(BREAKOUT_RANGE_BARS + 1):-1]

    highs = [float(r.get("high") or 0.0) for r in range_window]
    lows = [float(r.get("low") or 0.0) for r in range_window]
    if not highs or not lows or min(lows) <= 0:
        details["fail"] = "bad_range_data"
        return False, 0, details

    range_high = max(highs)
    range_low = min(lows)
    range_size = range_high - range_low

    atr_raw = last.get("atr_14")
    if atr_raw is None or float(atr_raw) <= 0:
        details["fail"] = "atr_missing"
        return False, 0, details
    atr = float(atr_raw)

    range_atr_mult = range_size / atr
    tight = range_atr_mult <= BREAKOUT_RANGE_ATR_MULT_MAX
    details["range_high"] = round(range_high, 1)
    details["range_low"] = round(range_low, 1)
    details["range_atr_mult"] = round(range_atr_mult, 2)
    details["range_tight"] = tight
    if not tight:
        details["fail"] = "range_not_tight"
        return False, 0, details

    close_now = float(last.get("close") or 0.0)
    closes_above = close_now > range_high
    details["close_above_range"] = closes_above
    details["close_above_pct"] = round(((close_now - range_high) / range_high) * 100, 2) if range_high > 0 else 0
    if not closes_above:
        details["fail"] = "no_close_above_range"
        return False, 0, details

    vol_window = rows[-(BREAKOUT_VOLUME_AVG_BARS + 1):-1]
    volumes = [float(r.get("volume") or 0.0) for r in vol_window]
    if not volumes or sum(volumes) == 0:
        details["fail"] = "volume_data_missing"
        return False, 0, details

    avg_vol = sum(volumes) / len(volumes)
    cur_vol = float(last.get("volume") or 0.0)
    vol_mult = cur_vol / avg_vol if avg_vol > 0 else 0
    vol_ok = vol_mult >= BREAKOUT_VOLUME_MULT_MIN

    details["volume_mult"] = round(vol_mult, 2)
    details["volume_ok"] = vol_ok
    if not vol_ok:
        details["fail"] = "insufficient_volume_expansion"
        return False, 0, details

    tight_pts = int(max(0, 40 - ((range_atr_mult - 1.0) * 40)))
    above_pct = (close_now - range_high) / range_high if range_high > 0 else 0
    above_pts = int(min(30, above_pct * 1500))
    vol_pts = int(min(30, (vol_mult - BREAKOUT_VOLUME_MULT_MIN) * 30 + 15))
    strength = max(0, min(100, tight_pts + above_pts + vol_pts))

    details["strength_breakdown"] = {
        "tight_pts": tight_pts,
        "above_pts": above_pts,
        "vol_pts": vol_pts,
    }
    return True, strength, details


def _detect_accumulation(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Classify accumulation state to split WATCH from HOLD."""
    details: dict[str, Any] = {}

    if len(rows) < OBV_SLOPE_BARS + 1:
        return "absent", {"reason": "insufficient_history"}

    recent = rows[-(OBV_SLOPE_BARS + 1):]
    obvs = [row.get("obv") for row in recent]

    obv_rising = False
    if all(v is not None for v in obvs):
        vals = np.array([float(v) for v in obvs])
        x = np.arange(len(vals), dtype=float)
        x_mean, y_mean = x.mean(), vals.mean()
        if y_mean != 0:
            denom = float(np.sum((x - x_mean) ** 2))
            if denom > 0:
                slope = float(np.sum((x - x_mean) * (vals - y_mean)) / denom)
                slope_pct = slope / abs(y_mean) * 100.0
                obv_rising = slope_pct > ACCUMULATION_OBV_MIN_SLOPE_PCT
                details["obv_slope_pct"] = round(slope_pct, 2)

    cmf = rows[-1].get("cmf_20")
    cmf_positive = False
    if cmf is not None:
        cmf_val = float(cmf)
        cmf_positive = cmf_val > ACCUMULATION_CMF_MIN
        details["cmf"] = round(cmf_val, 3)

    details["obv_rising"] = obv_rising
    details["cmf_positive"] = cmf_positive

    if obv_rising and cmf_positive:
        return "active", details
    if obv_rising or cmf_positive:
        return "building", details
    return "absent", details


def evaluate_entry_trigger(rows: list[dict[str, Any]], score_tier: str) -> dict[str, Any]:
    """Evaluate entry timing and return action + detector breakdown."""
    if score_tier not in ("Buy", "Strong Buy"):
        return {
            "action": "HOLD",
            "trigger": "none",
            "pullback": {"triggered": False, "reason": "signal_not_buy", "strength": 0},
            "breakout": {"triggered": False, "reason": "signal_not_buy", "strength": 0},
            "accumulation": {"state": "absent", "obv_slope_pct": None, "cmf": None},
            "triggered": False,
            "trigger_type": None,
            "trigger_strength": 0,
            "accumulation_state": "absent",
            "recommended_state": score_tier.upper().replace(" ", "_"),
            "details": {"skipped": "non_buy_tier"},
        }

    pullback_fired, pullback_strength, pullback_details = _detect_pullback_trigger(rows)
    breakout_fired, breakout_strength, breakout_details = False, 0, {"skipped": "buy_tier_breakout_disabled"}
    if score_tier == "Strong Buy":
        breakout_fired, breakout_strength, breakout_details = _detect_breakout_trigger(rows)

    triggered = pullback_fired or breakout_fired
    if pullback_fired and breakout_fired:
        if pullback_strength >= breakout_strength:
            trigger_type, trigger_strength = "pullback", pullback_strength
        else:
            trigger_type, trigger_strength = "breakout", breakout_strength
    elif pullback_fired:
        trigger_type, trigger_strength = "pullback", pullback_strength
    elif breakout_fired:
        trigger_type, trigger_strength = "breakout", breakout_strength
    else:
        trigger_type, trigger_strength = None, 0

    accumulation_state, accumulation_details = _detect_accumulation(rows)

    if triggered:
        action = "ENTER"
        trigger = trigger_type or "none"
        recommended_state = "BUY"
    elif accumulation_state in ("active", "building"):
        action = "WATCH"
        trigger = "accumulation_only"
        recommended_state = "WATCH"
    else:
        action = "HOLD"
        trigger = "none"
        recommended_state = "HOLD"

    details: dict[str, Any] = {
        "pullback_eval": pullback_details,
        "breakout_eval": breakout_details,
        "accumulation": accumulation_details,
    }

    if pullback_fired and breakout_fired:
        details["trigger_tiebreak"] = "pullback_preferred_on_equal_strength"

    pullback_reason = "triggered" if pullback_fired else str(pullback_details.get("fail") or "not_triggered")
    breakout_reason = "triggered" if breakout_fired else str(breakout_details.get("fail") or breakout_details.get("skipped") or "not_triggered")

    return {
        "action": action,
        "trigger": trigger,
        "pullback": {
            "triggered": pullback_fired,
            "reason": pullback_reason,
            "strength": pullback_strength,
        },
        "breakout": {
            "triggered": breakout_fired,
            "reason": breakout_reason,
            "strength": breakout_strength,
        },
        "accumulation": {
            "state": accumulation_state,
            "obv_slope_pct": accumulation_details.get("obv_slope_pct"),
            "cmf": accumulation_details.get("cmf"),
        },
        "triggered": triggered,
        "trigger_type": trigger_type,
        "trigger_strength": trigger_strength,
        "accumulation_state": accumulation_state,
        "recommended_state": recommended_state,
        "details": details,
    }

"""Entry-timing detectors for the Kuwait Signal Engine.

Three micro-structure filters that run *after* the multi-factor confluence
score is computed.  They answer: "The score says BUY, but is **right now**
actually a good candle to enter on?"

Detectors
─────────
1. Pullback   — price pulled back to rising EMA-20 with a bullish
                confirmation candle and stochastic recovering from oversold.
2. Breakout   — price breaks above a tight consolidation range on
                above-average volume.
3. Accumulate — OBV slope and CMF both positive → institutional buying
                pressure is building.

evaluate_entry_trigger() orchestrates all three and returns a summary dict
that the signal generator embeds in the output JSON.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.services.signal_engine.config.model_params import (
    BREAKOUT_RANGE_ATR_MULT_MAX,
    BREAKOUT_RANGE_BARS,
    BREAKOUT_VOLUME_AVG_BARS,
    BREAKOUT_VOLUME_MULT_MIN,
    PULLBACK_EMA_PROXIMITY_PCT,
    PULLBACK_LOOKBACK_BARS,
    PULLBACK_STOCH_MAX,
    ACCUMULATION_CMF_MIN,
    ACCUMULATION_OBV_MIN_SLOPE_PCT,
)


def _detect_pullback_trigger(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect a pullback-to-EMA entry with bullish confirmation.

    Criteria (all must be true):
      1. EMA-20 is rising over the lookback window.
      2. Price dipped to within PULLBACK_EMA_PROXIMITY_PCT of EMA-20.
      3. Latest candle is bullish (close > open).
      4. Stochastic %K is recovering (below PULLBACK_STOCH_MAX and K > D).
    """
    n = PULLBACK_LOOKBACK_BARS
    if len(rows) < n + 1:
        return {"triggered": False, "reason": "insufficient_data"}

    window = rows[-(n + 1):]
    last = window[-1]

    ema_vals = [r.get("ema_20") for r in window]
    if any(v is None for v in ema_vals):
        return {"triggered": False, "reason": "ema_missing"}

    ema_floats = [float(v) for v in ema_vals]
    ema_rising = ema_floats[-1] > ema_floats[0]
    if not ema_rising:
        return {"triggered": False, "reason": "ema_not_rising"}

    close = float(last.get("close") or 0.0)
    ema_now = ema_floats[-1]
    if ema_now <= 0:
        return {"triggered": False, "reason": "ema_zero"}

    proximity = abs(close - ema_now) / ema_now
    near_ema = proximity <= PULLBACK_EMA_PROXIMITY_PCT
    if not near_ema:
        return {"triggered": False, "reason": f"price_not_near_ema_{proximity:.4f}"}

    open_price = float(last.get("open") or 0.0)
    bullish_candle = close > open_price
    if not bullish_candle:
        return {"triggered": False, "reason": "bearish_candle"}

    stoch_k = last.get("stoch_k")
    stoch_d = last.get("stoch_d")
    if stoch_k is None or stoch_d is None:
        return {"triggered": False, "reason": "stoch_missing"}

    k, d = float(stoch_k), float(stoch_d)
    stoch_ok = k <= PULLBACK_STOCH_MAX and k > d
    if not stoch_ok:
        return {"triggered": False, "reason": f"stoch_not_recovering_k{k:.0f}_d{d:.0f}"}

    return {"triggered": True, "reason": "pullback_confirmed"}


def _detect_breakout_trigger(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect a breakout from a tight consolidation range.

    Criteria (all must be true):
      1. The high-low range of the prior BREAKOUT_RANGE_BARS bars is
         "tight" — less than BREAKOUT_RANGE_ATR_MULT_MAX × ATR-14.
      2. Latest close breaks above the range high.
      3. Latest volume ≥ BREAKOUT_VOLUME_MULT_MIN × 20-bar average volume.
    """
    needed = max(BREAKOUT_RANGE_BARS, BREAKOUT_VOLUME_AVG_BARS) + 1
    if len(rows) < needed:
        return {"triggered": False, "reason": "insufficient_data"}

    last = rows[-1]
    range_window = rows[-(BREAKOUT_RANGE_BARS + 1):-1]

    atr = last.get("atr_14")
    if atr is None or float(atr) <= 0:
        return {"triggered": False, "reason": "atr_missing"}

    atr_val = float(atr)

    highs = [float(r.get("high") or 0.0) for r in range_window]
    lows = [float(r.get("low") or 0.0) for r in range_window]
    range_high = max(highs)
    range_low = min(lows)
    range_size = range_high - range_low

    tight = range_size <= atr_val * BREAKOUT_RANGE_ATR_MULT_MAX
    if not tight:
        return {"triggered": False, "reason": f"range_not_tight_{range_size:.1f}_vs_{atr_val * BREAKOUT_RANGE_ATR_MULT_MAX:.1f}"}

    close = float(last.get("close") or 0.0)
    if close <= range_high:
        return {"triggered": False, "reason": "no_breakout_above_range"}

    vol_window = rows[-(BREAKOUT_VOLUME_AVG_BARS + 1):-1]
    volumes = [float(r.get("volume") or 0.0) for r in vol_window]
    avg_vol = float(np.mean(volumes)) if volumes else 0.0
    cur_vol = float(last.get("volume") or 0.0)

    if avg_vol <= 0:
        return {"triggered": False, "reason": "avg_volume_zero"}

    vol_mult = cur_vol / avg_vol
    if vol_mult < BREAKOUT_VOLUME_MULT_MIN:
        return {"triggered": False, "reason": f"volume_expansion_weak_{vol_mult:.2f}x"}

    return {"triggered": True, "reason": f"breakout_confirmed_vol_{vol_mult:.1f}x"}


def _detect_accumulation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect institutional accumulation via OBV slope + CMF.

    Returns accumulation state:
      - "active"   — both OBV slope and CMF pass thresholds
      - "building" — one of the two passes
      - "absent"   — neither passes
    """
    if len(rows) < 6:
        return {"state": "absent", "obv_slope_pct": None, "cmf": None}

    recent = rows[-6:]
    obvs = [r.get("obv") for r in recent]
    if any(v is None for v in obvs):
        return {"state": "absent", "obv_slope_pct": None, "cmf": None}

    vals = np.array([float(v) for v in obvs])
    x = np.arange(len(vals), dtype=float)
    x_mean = x.mean()
    y_mean = vals.mean()
    if y_mean == 0:
        obv_slope_pct = 0.0
    else:
        slope = np.sum((x - x_mean) * (vals - y_mean)) / np.sum((x - x_mean) ** 2)
        obv_slope_pct = slope / abs(y_mean) * 100.0

    cmf = float(rows[-1].get("cmf_20") or 0.0)

    obv_pass = obv_slope_pct >= ACCUMULATION_OBV_MIN_SLOPE_PCT
    cmf_pass = cmf >= ACCUMULATION_CMF_MIN

    if obv_pass and cmf_pass:
        state = "active"
    elif obv_pass or cmf_pass:
        state = "building"
    else:
        state = "absent"

    return {
        "state": state,
        "obv_slope_pct": round(obv_slope_pct, 3),
        "cmf": round(cmf, 4),
    }


def evaluate_entry_trigger(
    rows: list[dict[str, Any]],
    score_tier: str,
) -> dict[str, Any]:
    """Run all entry-timing detectors and return a summary.

    Args:
        rows:       OHLCV + indicator rows sorted ascending.
        score_tier: "Strong Buy" | "Buy" | other — controls which triggers
                    are eligible.  Only STRONG_BUY tier can use breakout.

    Returns:
        {
            "action": "ENTER" | "WATCH" | "HOLD",
            "trigger": "pullback" | "breakout" | "accumulation_only" | "none",
            "pullback": { ... },
            "breakout": { ... },
            "accumulation": { ... },
        }
    """
    pullback = _detect_pullback_trigger(rows)
    breakout = _detect_breakout_trigger(rows)
    accumulation = _detect_accumulation(rows)

    result: dict[str, Any] = {
        "pullback": pullback,
        "breakout": breakout,
        "accumulation": accumulation,
    }

    if pullback["triggered"]:
        result["action"] = "ENTER"
        result["trigger"] = "pullback"
        return result

    if breakout["triggered"] and score_tier == "Strong Buy":
        result["action"] = "ENTER"
        result["trigger"] = "breakout"
        return result

    if accumulation["state"] == "active":
        result["action"] = "WATCH"
        result["trigger"] = "accumulation_only"
        return result

    result["action"] = "HOLD"
    result["trigger"] = "none"
    return result

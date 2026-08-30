"""
Rating Engine — produces the 8 Eagle Eye outputs for a single stock.

Given an indicator snapshot and optional behavioral DNA, computes:
  1. classify_stage       → one of 8 lifecycle stages
  2. compute_support_resistance → SR levels
  3. compute_entry_stop_targets → entry/stop/TP levels
  4. compute_position_size      → Kelly-based position sizing
  5. compute_confidence         → weighted composite 0-100
  6. compute_rating             → STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
  7. generate_thesis            → one-sentence plain-English explanation
"""
from __future__ import annotations

from datetime import date, datetime
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.services.eagle_eye.config import CONFIG, RATINGS
from app.services.eagle_eye.stage_classifier import classify_stage


# Re-export for convenience — callers may import classify_stage from here
__all__ = [
    "classify_stage",
    "compute_support_resistance",
    "compute_entry_stop_targets",
    "compute_position_size",
    "is_stock_active",
    "validate_and_adjust_ml_score",
    "compute_final_confidence",
    "compute_confidence",
    "compute_confidence_from_phase",
    "compute_rating_from_phase_score",
    "compute_rating",
    "generate_thesis",
    "compute_volume_context",
]

IndicatorsRow = Dict[str, Any]


# ---------------------------------------------------------------------------
# 0. Volume Context
# ---------------------------------------------------------------------------

def compute_volume_context(df: pd.DataFrame, stage: str) -> Dict[str, Any]:
    """
    Compute volume context for today's bar.

    Returns a dict with relative volume, liquidity tier, confirmation flag,
    volume character, and trend — used to gate confidence and simulator entries.
    """
    if df is None or len(df) < 2:
        return {
            "today_volume": 0,
            "today_turnover_kwd": 0.0,
            "avg_20d_volume": 0,
            "avg_20d_turnover_kwd": 0.0,
            "relative_volume": 1.0,
            "relative_volume_percentile": 50.0,
            "volume_trend_5d": "NEUTRAL",
            "volume_trend_20d": "NEUTRAL",
            "liquidity_tier": "WATCH_ONLY",
            "is_volume_confirmed": True,
            "volume_character": "NEUTRAL",
            "institutional_volume_flag": False,
        }

    today = df.iloc[-1]

    avg_20d_vol = df["volume"].tail(20).mean()
    avg_20d_turnover = df["turnover_kwd"].tail(20).mean()

    relative_volume = float(today["volume"]) / avg_20d_vol if avg_20d_vol > 0 else 1.0

    # Percentile in 252-day volume history
    vol_252 = df["volume"].tail(252)
    rv_percentile = float((vol_252 < today["volume"]).sum() / len(vol_252) * 100)

    # Volume trends
    avg_5d = df["volume"].tail(5).mean()
    avg_40d = df["volume"].tail(40).mean()
    vol_trend_5d = (
        "EXPANDING" if avg_5d > avg_20d_vol * 1.1
        else ("CONTRACTING" if avg_5d < avg_20d_vol * 0.9 else "NEUTRAL")
    )
    vol_trend_20d = (
        "EXPANDING" if avg_20d_vol > avg_40d * 1.1
        else ("CONTRACTING" if avg_20d_vol < avg_40d * 0.9 else "NEUTRAL")
    )

    # Volume character: up-day vs down-day volume ratio over last 10 bars
    last_10 = df.tail(10).copy()
    last_10["prev_close"] = last_10["close"].shift(1)
    up_vol = float(last_10.loc[last_10["close"] > last_10["prev_close"], "volume"].sum())
    down_vol = float(last_10.loc[last_10["close"] < last_10["prev_close"], "volume"].sum())
    total_vol = up_vol + down_vol
    if total_vol > 0:
        up_ratio = up_vol / total_vol
        character = (
            "ACCUMULATION" if up_ratio > 0.6
            else ("DISTRIBUTION" if up_ratio < 0.4 else "NEUTRAL")
        )
    else:
        character = "NEUTRAL"

    # Liquidity tier
    if avg_20d_turnover >= 10_000:
        tier = "TRADEABLE"
    elif avg_20d_turnover >= 2_000:
        tier = "WATCH_ONLY"
    else:
        tier = "ILLIQUID"

    # Signal confirmation: EARLY_MARKUP needs 1.5x volume; others 0.8x.
    if stage in ("EARLY_MARKUP", "EARLY_BREAKOUT"):
        is_confirmed = bool(relative_volume >= 1.5)
    else:
        is_confirmed = bool(relative_volume >= 0.8)

    institutional_flag = bool(relative_volume > 3.0)

    return {
        "today_volume": int(today["volume"]),
        "today_turnover_kwd": float(today["turnover_kwd"]),
        "avg_20d_volume": int(avg_20d_vol),
        "avg_20d_turnover_kwd": float(avg_20d_turnover),
        "relative_volume": round(float(relative_volume), 2),
        "relative_volume_percentile": round(rv_percentile, 1),
        "volume_trend_5d": vol_trend_5d,
        "volume_trend_20d": vol_trend_20d,
        "liquidity_tier": tier,
        "is_volume_confirmed": is_confirmed,
        "volume_character": character,
        "institutional_volume_flag": institutional_flag,
    }
# ---------------------------------------------------------------------------

def _safe(v, default=None):
    """Return v unless it is None or NaN."""
    if v is None:
        return default
    try:
        if math.isnan(float(v)):
            return default
    except (TypeError, ValueError):
        pass
    return v


def _safe_numeric(v: Any) -> Optional[float]:
    """Coerce a value to finite float; return None for non-numeric/NaN/Inf."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def is_stock_active(
    ticker: str,
    df: Optional[pd.DataFrame],
    *,
    as_of: Optional[date] = None,
) -> bool:
    """
    Return False for suspended/delisted/dead symbols.

    Activity checks are deliberately conservative and focus on recency,
    executable price, and non-zero recent participation.
    """
    if df is None or len(df) < 5:
        return False
    if "close" not in df.columns or "volume" not in df.columns:
        return False

    try:
        last_bar_date = pd.Timestamp(df.index[-1]).date()
    except Exception:
        return False

    today = as_of or datetime.utcnow().date()
    if (today - last_bar_date).days > 20:
        return False

    try:
        recent_volume = float(pd.to_numeric(df["volume"], errors="coerce").tail(10).fillna(0.0).sum())
    except Exception:
        recent_volume = 0.0
    if recent_volume <= 0.0:
        return False

    last_close = _safe_numeric(df["close"].iloc[-1])
    if last_close is None or last_close <= 0.0:
        return False

    return True


def validate_and_adjust_ml_score(
    ml_score: float,
    indicators: IndicatorsRow,
    df: pd.DataFrame,
    ticker: str,
) -> float:
    """
    Reality-check ML opportunity score against current market state.

    This guards against late-entry chasing and stale/dead symbols while
    preserving the existing gain->confidence mapping in compute_confidence().
    """
    adjusted = _safe_numeric(ml_score)
    if adjusted is None:
        return 0.0

    if df is None or len(df) < 10:
        return 0.0
    if not is_stock_active(ticker, df):
        return 0.0

    close_now = _safe_numeric(df["close"].iloc[-1]) if "close" in df.columns else None

    def _indicator_float(name: str) -> Optional[float]:
        try:
            return _safe_numeric(indicators.get(name))
        except Exception:
            return None

    # Check 1: Extension from 20d base.
    price_ext_20d = _indicator_float("price_extension_from_20d_low_pct")
    if price_ext_20d is None and close_now is not None and len(df) >= 20 and "low" in df.columns:
        low_20d = _safe_numeric(pd.to_numeric(df["low"], errors="coerce").tail(20).min())
        if low_20d is not None and low_20d > 0:
            price_ext_20d = (close_now / low_20d - 1.0) * 100.0

    if price_ext_20d is not None:
        if price_ext_20d >= 50.0:
            adjusted = min(adjusted, 10.0)
        elif price_ext_20d >= 30.0:
            adjusted = min(adjusted, 25.0)
        elif price_ext_20d >= 20.0:
            adjusted = min(adjusted, 40.0)
        elif price_ext_20d >= 15.0:
            adjusted = min(adjusted, 55.0)

    # Check 2: Position in 60d range (top-of-range chasing filter).
    range_position_60d = _indicator_float("position_in_60d_range_pct")
    if (
        range_position_60d is None
        and close_now is not None
        and len(df) >= 60
        and "low" in df.columns
        and "high" in df.columns
    ):
        low_60d = _safe_numeric(pd.to_numeric(df["low"], errors="coerce").tail(60).min())
        high_60d = _safe_numeric(pd.to_numeric(df["high"], errors="coerce").tail(60).max())
        if low_60d is not None and high_60d is not None and high_60d > low_60d:
            range_position_60d = ((close_now - low_60d) / (high_60d - low_60d)) * 100.0

    if range_position_60d is not None:
        if range_position_60d >= 90.0:
            adjusted = min(adjusted, 20.0)
        elif range_position_60d >= 75.0:
            adjusted = min(adjusted, 45.0)

    # Check 3: Extension from 120d low (major move exhaustion filter).
    ext_120d = _indicator_float("price_extension_from_120d_low_pct")
    if ext_120d is None and close_now is not None and len(df) >= 120 and "low" in df.columns:
        low_120d = _safe_numeric(pd.to_numeric(df["low"], errors="coerce").tail(120).min())
        if low_120d is not None and low_120d > 0:
            ext_120d = (close_now / low_120d - 1.0) * 100.0

    if ext_120d is not None:
        if ext_120d >= 100.0:
            adjusted = min(adjusted, 10.0)
        elif ext_120d >= 60.0:
            adjusted = min(adjusted, 30.0)

    # Check 4: RSI extreme overbought.
    rsi = _indicator_float("rsi")
    if rsi is not None:
        if rsi >= 80.0:
            adjusted = min(adjusted, 25.0)
        elif rsi >= 75.0:
            adjusted = min(adjusted, 45.0)

    return float(np.clip(adjusted, 0.0, 100.0))


def _apply_universal_safety_clamp(confidence: float, indicators: IndicatorsRow) -> float:
    """Apply caps only for extreme extension/overbought regimes."""
    clamped = _safe_numeric(confidence)
    if clamped is None:
        clamped = 0.0

    def _get(key: str) -> Optional[float]:
        try:
            return _safe_numeric(indicators.get(key))
        except Exception:
            return None

    ext_20 = _get("price_extension_from_20d_low_pct")
    if ext_20 is not None:
        if ext_20 >= 80.0:
            clamped = min(clamped, 25.0)
        elif ext_20 >= 60.0:
            clamped = min(clamped, 35.0)

    ext_60 = _get("price_extension_from_60d_low_pct")
    if ext_60 is not None:
        if ext_60 >= 120.0:
            clamped = min(clamped, 20.0)
        elif ext_60 >= 90.0:
            clamped = min(clamped, 35.0)

    ext_120 = _get("price_extension_from_120d_low_pct")
    if ext_120 is not None:
        if ext_120 >= 170.0:
            clamped = min(clamped, 20.0)
        elif ext_120 >= 120.0:
            clamped = min(clamped, 35.0)

    rsi = _get("rsi")
    if rsi is not None:
        if rsi >= 88.0:
            clamped = min(clamped, 30.0)
        elif rsi >= 82.0:
            clamped = min(clamped, 45.0)

    return float(np.clip(clamped, 0.0, 100.0))


def compute_final_confidence(
    ml_score: float,
    indicators: IndicatorsRow,
    stage: str,
) -> Tuple[float, str]:
    """
    v14 confidence path: model score -> safety clamp -> stage cap -> rating.
    """
    conf = _safe_numeric(ml_score)
    confidence = float(conf if conf is not None else 0.0)

    confidence = _apply_universal_safety_clamp(confidence, indicators)

    # Volume reality gate: enforce hard caps for dead/illiquid names.
    volume_cap = 100.0
    vol_ratio = _safe_numeric(indicators.get("volume_ratio_20d"))
    if vol_ratio is None:
        vol_ratio = _safe_numeric(indicators.get("rel_volume"))
    if vol_ratio is not None:
        if vol_ratio < 0.3:
            volume_cap = min(volume_cap, 35.0)
        elif vol_ratio < 0.5:
            volume_cap = min(volume_cap, 50.0)

    avg_turnover = _safe_numeric(indicators.get("avg_20d_turnover_kwd"))
    if avg_turnover is not None and avg_turnover < 2000.0:
        volume_cap = min(volume_cap, 40.0)

    if stage in ("MARKDOWN", "MARKDOWN_DECLINE", "DISTRIBUTION", "DISTRIBUTION_TOPPING"):
        confidence = min(confidence, 40.0)

    # Dormant-stage sanity rails keep weak basing names from overstating conviction,
    # while preserving a minimum floor for rare strong-confluence accumulation setups.
    if stage in ("DORMANT", "NEUTRAL_AMBIGUOUS"):
        trend_conf = _safe_numeric(indicators.get("trend_confluence"))
        momentum_conf = _safe_numeric(indicators.get("momentum_confluence"))
        overall_conf = _safe_numeric(indicators.get("overall_confluence"))
        volume_flow_conf = _safe_numeric(indicators.get("volume_flow_confluence"))
        range_pos = _safe_numeric(indicators.get("position_in_60d_range_pct"))

        if (
            trend_conf is not None
            and momentum_conf is not None
            and overall_conf is not None
            and trend_conf <= 0.05
            and momentum_conf <= 0.05
            and overall_conf <= 0.55
        ):
            confidence = min(confidence, 34.0)

        if range_pos is not None and range_pos >= 85.0:
            confidence = min(confidence, 65.0)

        if (
            overall_conf is not None
            and volume_flow_conf is not None
            and momentum_conf is not None
            and trend_conf is not None
            and overall_conf >= 0.85
            and volume_flow_conf >= 0.90
            and momentum_conf >= 0.80
            and trend_conf >= 0.40
        ):
            confidence = max(confidence, 62.0)

    confidence = min(confidence, volume_cap)

    confidence = round(float(np.clip(confidence, 0.0, 100.0)), 1)

    if confidence >= 80.0:
        rating = "STRONG_BUY"
    elif confidence >= 60.0:
        rating = "BUY"
    elif confidence >= 40.0:
        rating = "HOLD"
    elif confidence >= 25.0:
        rating = "SELL"
    else:
        rating = "STRONG_SELL"

    return confidence, rating


# ---------------------------------------------------------------------------
# 1. Support / Resistance
# ---------------------------------------------------------------------------

def compute_support_resistance(
    df: pd.DataFrame,
    indicators: IndicatorsRow,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Multi-method SR detection.

    Methods:
      - Swing highs/lows from last 252 bars (weighted by recency + touch count)
      - Volume Profile POC, VAH, VAL from last 90 bars
      - Fibonacci levels from most significant 252-bar swing
      - VWAP ± 1σ and ± 2σ

    Returns:
      {
        "supports":    [{"price": float, "strength": 0-100, "method": str}, ...],  # top 3
        "resistances": [{"price": float, "strength": 0-100, "method": str}, ...],  # top 3
      }
    """
    if len(df) < 20:
        return {"supports": [], "resistances": []}

    current_close = float(df["close"].iloc[-1])
    supports_raw: List[Tuple[float, float, str]] = []   # (price, raw_strength, method)
    resistances_raw: List[Tuple[float, float, str]] = []

    # --- Swing points ---
    window_df = df.tail(252)
    window_highs = window_df["high"]
    window_lows = window_df["low"]
    n = len(window_df)

    def _swing_strength(idx: int, total: int, touch_count: int) -> float:
        """Higher = more recent, more touches."""
        recency = idx / max(total, 1)        # 0=old, 1=recent
        touch_bonus = min(touch_count, 5) / 5
        return (0.6 * recency + 0.4 * touch_bonus) * 100

    # Detect swing highs/lows with a 5-bar fractal window
    sw_window = 5
    for i in range(sw_window, n - sw_window):
        high_i = window_highs.iloc[i]
        low_i = window_lows.iloc[i]
        # Swing high
        if high_i == window_highs.iloc[i-sw_window:i+sw_window+1].max():
            touches = int((window_highs.between(high_i * 0.99, high_i * 1.01)).sum())
            strength = _swing_strength(i, n, touches)
            if high_i > current_close:
                resistances_raw.append((float(high_i), strength, "swing_high"))
            else:
                supports_raw.append((float(high_i), strength * 0.7, "prior_swing_high"))
        # Swing low
        if low_i == window_lows.iloc[i-sw_window:i+sw_window+1].min():
            touches = int((window_lows.between(low_i * 0.99, low_i * 1.01)).sum())
            strength = _swing_strength(i, n, touches)
            if low_i < current_close:
                supports_raw.append((float(low_i), strength, "swing_low"))
            else:
                resistances_raw.append((float(low_i), strength * 0.7, "prior_swing_low"))

    # --- Volume Profile ---
    try:
        from app.services.eagle_eye.indicators import volume_profile
        vp = volume_profile(df, lookback=90)
        poc = _safe(vp.get("poc"))
        vah = _safe(vp.get("vah"))
        val_ = _safe(vp.get("val"))
        if poc is not None:
            lst = resistances_raw if poc > current_close else supports_raw
            lst.append((poc, 75.0, "vp_poc"))
        if vah is not None:
            (resistances_raw if vah > current_close else supports_raw).append((vah, 65.0, "vp_vah"))
        if val_ is not None:
            (supports_raw if val_ < current_close else resistances_raw).append((val_, 65.0, "vp_val"))
    except Exception:
        pass

    # --- Fibonacci ---
    try:
        from app.services.eagle_eye.indicators import fibonacci_levels
        fibs = fibonacci_levels(df, lookback=252)
        for label, price in fibs.items():
            if price is None or price <= 0:
                continue
            fib_pct = label.replace("fib_", "")
            if float(fib_pct) in (38.2, 50.0, 61.8):
                strength = 70.0
            elif float(fib_pct) in (23.6, 78.6):
                strength = 55.0
            else:
                strength = 40.0
            if price < current_close:
                supports_raw.append((price, strength, f"fib_{fib_pct}"))
            elif price > current_close:
                resistances_raw.append((price, strength, f"fib_{fib_pct}"))
    except Exception:
        pass

    # --- VWAP bands ---
    vwap_v = _safe(indicators.get("vwap"))
    vwap_sigma = _safe(indicators.get("vwap_distance_sigma"))
    atr_v = _safe(indicators.get("atr"), 0.01 * current_close)
    if vwap_v and atr_v:
        # Approximate sigma using ATR as std proxy
        sigma_est = atr_v * 1.2
        for mult, strength in [(1.0, 70.0), (2.0, 55.0)]:
            up = vwap_v + mult * sigma_est
            dn = vwap_v - mult * sigma_est
            (resistances_raw if up > current_close else supports_raw).append(
                (up, strength, f"vwap_+{mult}sigma")
            )
            (supports_raw if dn < current_close else resistances_raw).append(
                (dn, strength, f"vwap_-{mult}sigma")
            )

    # --- Cluster confluence: boost strength when 2+ methods agree within 1% ---
    def _apply_confluence(raw: List[Tuple[float, float, str]]) -> List[Dict[str, Any]]:
        results = []
        for price, strength, method in raw:
            cluster_count = sum(
                1 for p2, _, _ in raw
                if p2 != price and abs(p2 - price) / max(price, 1e-9) < 0.01
            )
            boosted = min(100.0, strength + cluster_count * 10.0)
            results.append({"price": round(price, 4), "strength": round(boosted, 1), "method": method})
        # Deduplicate by proximity (keep highest strength per cluster)
        deduped: List[Dict[str, Any]] = []
        used = set()
        for item in sorted(results, key=lambda x: -x["strength"]):
            key = round(item["price"] / max(current_close, 1e-9) / 0.01)
            if key not in used:
                deduped.append(item)
                used.add(key)
        return deduped

    final_supports = sorted(
        _apply_confluence(supports_raw),
        key=lambda x: -x["strength"],
    )[:3]
    final_resistances = sorted(
        _apply_confluence(resistances_raw),
        key=lambda x: -x["strength"],
    )[:3]

    return {"supports": final_supports, "resistances": final_resistances}


# ---------------------------------------------------------------------------
# 2. Entry / Stop / Targets
# ---------------------------------------------------------------------------

def compute_entry_stop_targets(
    df: pd.DataFrame,
    indicators: IndicatorsRow,
    support_resistance: Dict[str, List[Dict[str, Any]]],
    dna: Optional[Any] = None,
    stage: str = "UNKNOWN",
) -> Dict[str, Any]:
    """
    Compute entry zone, stop loss, and TP1/TP2/TP3 with probabilities.

    Returns
    -------
    dict with keys:
      entry_primary, entry_aggressive, entry_conservative,
      stop_loss,
      tp1, tp1_probability,
      tp2, tp2_probability,
      tp3, tp3_probability
    """
    current_close = float(df["close"].iloc[-1])
    atr_v = _safe(indicators.get("atr"), current_close * 0.02)
    supports = support_resistance.get("supports", [])
    resistances = support_resistance.get("resistances", [])
    min_favorable_rr = 1.8
    max_conditional_pullback = max(2.5 * atr_v, current_close * 0.12)

    def _nearest_support_below(price: float) -> Optional[float]:
        candidates = [float(item["price"]) for item in supports if float(item["price"]) < price]
        return max(candidates) if candidates else None

    def _nearest_resistance_above(price: float, floor: float, cap: float) -> Optional[float]:
        for res in resistances:
            res_price = float(res["price"])
            if price + floor <= res_price <= price + cap:
                return res_price
        return None

    def _sanitize_tp1(entry_price: float, stop_price: float, candidate_tp1: float) -> Tuple[Optional[float], List[str]]:
        if entry_price <= 0 or stop_price <= 0 or stop_price >= entry_price:
            return None, ["invalid_risk"]
        risk = entry_price - stop_price
        rr_cap_price = entry_price + risk * 10.0
        gain_cap_price = entry_price * 1.60
        capped_tp1 = min(candidate_tp1, rr_cap_price, gain_cap_price)
        reasons: List[str] = []
        if candidate_tp1 > rr_cap_price + 1e-9:
            reasons.append("rr_cap")
        if candidate_tp1 > gain_cap_price + 1e-9:
            reasons.append("gain_cap")
        if capped_tp1 <= entry_price:
            reasons.append("non_positive_reward")
            return None, reasons
        return capped_tp1, reasons

    def _compute_stop(entry_price: float) -> Optional[float]:
        stop_from_atr = entry_price - 1.75 * atr_v
        nearest_support_below_entry = _nearest_support_below(entry_price)
        stop_candidates = [stop_from_atr]
        if nearest_support_below_entry is not None:
            stop_candidates.append(nearest_support_below_entry * 0.99)
        valid_candidates = [candidate for candidate in stop_candidates if candidate < entry_price]
        if not valid_candidates:
            return None
        stop_price = max(valid_candidates)
        min_stop_gap = max(0.75 * atr_v, entry_price * 0.02)
        if entry_price - stop_price < min_stop_gap:
            stop_price = entry_price - min_stop_gap
        return stop_price

    def _build_plan(
        entry_price: float,
        *,
        tp1_override: Optional[float] = None,
        plan_state: str = "ACTIVE",
    ) -> Optional[Dict[str, Any]]:
        stop_price = _compute_stop(entry_price)
        if stop_price is None or stop_price >= entry_price:
            return None

        tp1_floor_mult = STAGE_TP1_ATR_FLOORS.get(stage, 1.5)
        tp1_cap_mult = STAGE_TP1_ATR_CAPS.get(stage, 3.0)
        min_tp1 = entry_price + tp1_floor_mult * atr_v
        max_tp1 = entry_price + tp1_cap_mult * atr_v

        raw_tp1_source = "override"
        if tp1_override is not None and tp1_override > entry_price:
            raw_tp1 = tp1_override
        else:
            raw_tp1 = _nearest_resistance_above(entry_price, tp1_floor_mult * atr_v, tp1_cap_mult * atr_v)
            if raw_tp1 is None:
                raw_tp1 = min_tp1
                raw_tp1_source = "atr_floor"
            else:
                raw_tp1_source = "resistance"

        tp1_price, sanitize_reasons = _sanitize_tp1(entry_price, stop_price, float(raw_tp1))
        if tp1_price is None:
            return None

        risk = entry_price - stop_price
        reward = tp1_price - entry_price
        if risk <= 0 or reward <= 0:
            return None

        rr_value = reward / risk
        gain_pct = (reward / entry_price) * 100.0 if entry_price > 0 else None

        tp2_price = None
        for res in resistances:
            res_price = float(res["price"])
            if res_price > tp1_price * 1.005:
                tp2_price = res_price
                break
        if tp2_price is None:
            tp2_price = tp1_price + atr_v
        tp2_price = max(tp2_price, tp1_price + 0.75 * atr_v)

        if len(df) >= 60:
            recent_range = df["high"].tail(60).max() - df["low"].tail(60).min()
            breakout_base = df["high"].tail(60).max()
            tp3_price = breakout_base + recent_range
        else:
            tp3_price = entry_price * 1.20
        tp3_price = max(float(tp3_price), tp2_price + atr_v)

        if "rr_cap" in sanitize_reasons or "gain_cap" in sanitize_reasons:
            tp2_price = min(tp2_price, entry_price * 2.00)
            tp3_price = min(tp3_price, entry_price * 2.50)

        entry_aggressive_price = current_close if plan_state == "ACTIVE" and momentum_firing else None
        conservative_support = _nearest_support_below(entry_price)
        conservative_pullback = entry_price - 0.75 * atr_v
        if conservative_support is not None:
            entry_conservative_price = max(conservative_support * 1.002, conservative_pullback)
        else:
            entry_conservative_price = conservative_pullback
        entry_conservative_price = min(entry_price, max(entry_conservative_price, entry_price - 1.5 * atr_v))

        return {
            "entry_primary": round(entry_price, 4),
            "entry_aggressive": round(entry_aggressive_price, 4) if entry_aggressive_price is not None else None,
            "entry_conservative": round(entry_conservative_price, 4),
            "stop_loss": round(stop_price, 4),
            "tp1": round(tp1_price, 4),
            "tp2": round(tp2_price, 4),
            "tp3": round(tp3_price, 4),
            "risk_reward_ratio": round(rr_value, 4),
            "gain_pct_to_tp1": round(gain_pct, 4) if gain_pct is not None else None,
            "tp1_sanitize_reasons": sanitize_reasons,
            "raw_tp1": round(float(raw_tp1), 4),
            "raw_tp1_source": raw_tp1_source,
        }

    def _entry_status(entry_price: float) -> str:
        """
        Classify how far current price is from the setup anchor.

        AT_ANCHOR      → price is still within 3% of the 20-day low — ideal entry
        PULLBACK_ZONE  → price has pulled back to within 5% of anchor after a run
        EXTENDED       → price is 8-15% above anchor — risk/reward deteriorated
        WAIT_FOR_PULLBACK → price is >15% above anchor — wait for a reset
        """
        price_ext = _safe(indicators.get("price_extension_from_20d_low_pct"), None)
        if price_ext is None:
            return "UNKNOWN"
        ext = float(price_ext)
        if ext <= 3.0:
            return "AT_ANCHOR"
        if ext <= 5.0:
            return "PULLBACK_ZONE"
        if ext <= 15.0:
            return "EXTENDED"
        return "WAIT_FOR_PULLBACK"

    def _declined_plan(reason: str) -> Dict[str, Any]:
        return {
            "plan_state":       "DECLINED",
            "plan_reason":      reason,
            "entry_status":     _entry_status(current_close),
            "conditional_entry": None,
            "tp1_sanitize_reasons": [],
            "entry_primary":    None,
            "entry_aggressive": None,
            "entry_conservative": None,
            "stop_loss":        None,
            "tp1":              None,
            "tp1_probability":  None,
            "tp2":              None,
            "tp2_probability":  None,
            "tp3":              None,
            "tp3_probability":  None,
            "risk_reward_ratio": None,
            "gain_pct_to_tp1":  None,
        }

    def _dna_target_prices(
        entry_price: float,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Derive TP1/TP2/TP3 from the stock's historical target clusters when
        available.  Falls back to the ATR-based plan values if DNA is absent
        or has no clusters.

        DNA clusters are informative milestones — they show WHERE the stock
        has historically paused.  The exit decision system runs separately
        and never forces an exit when a cluster is reached.
        """
        if dna is None:
            return None, None, None
        clusters = getattr(dna, "historical_target_clusters", [])
        if not clusters:
            return None, None, None
        sorted_clusters = sorted(clusters, key=lambda c: c.gain_pct_from_entry)
        def _cluster_price(idx: int) -> Optional[float]:
            if idx < len(sorted_clusters):
                gain = sorted_clusters[idx].gain_pct_from_entry / 100.0
                return round(entry_price * (1.0 + gain), 4)
            return None
        return _cluster_price(0), _cluster_price(1), _cluster_price(2)

    def _finalize_plan(
        *,
        state: str,
        reason: Optional[str],
        plan: Dict[str, Any],
        conditional_entry: Optional[float] = None,
    ) -> Dict[str, Any]:
        tp1_prob = _dna_tp_prob(0, 0.55)
        tp2_prob = _dna_tp_prob(1, 0.35)
        tp3_prob = _dna_tp_prob(2, 0.15)

        # Use DNA-derived TP levels when available; fall back to ATR plan.
        # A DNA cluster's gain_pct_from_entry is historical and isn't
        # guaranteed to sit above entry for every cluster/entry combination --
        # only accept dna_tp1 as this plan's TP1 if it's actually an upside
        # target, otherwise keep the ATR/resistance-based plan tp1.
        entry_for_dna  = plan["entry_primary"] or current_close
        dna_tp1, dna_tp2, dna_tp3 = _dna_target_prices(entry_for_dna)
        tp1_final = dna_tp1 if dna_tp1 is not None and dna_tp1 > entry_for_dna else plan["tp1"]
        tp2_final = dna_tp2 if dna_tp2 is not None else plan["tp2"]
        tp3_final = dna_tp3 if dna_tp3 is not None else plan["tp3"]

        # risk_reward_ratio / gain_pct_to_tp1 were validated against
        # min_favorable_rr using plan["tp1"] (the ATR/resistance target)
        # BEFORE this DNA swap. If tp1_final ends up different from
        # plan["tp1"] -- which happens whenever DNA clusters are available --
        # the validated ratio no longer describes the TP1 actually returned
        # below, silently breaking the "ACTIVE/CONDITIONAL plans always meet
        # min_favorable_rr" guarantee the rest of this function relies on.
        # Recompute both from the actual final entry/stop/tp1 so the numbers
        # shown are internally consistent.
        entry_price = float(plan["entry_primary"])
        stop_price = float(plan["stop_loss"])
        risk = entry_price - stop_price
        reward = float(tp1_final) - entry_price
        risk_reward_ratio = round(reward / risk, 4) if risk > 0 else plan["risk_reward_ratio"]
        gain_pct_to_tp1 = round((reward / entry_price) * 100.0, 4) if entry_price > 0 else None

        # Hit-rate probabilities from DNA clusters when available
        clusters = getattr(dna, "historical_target_clusters", []) if dna else []
        sorted_c = sorted(clusters, key=lambda c: c.gain_pct_from_entry)
        if sorted_c:
            tp1_prob = sorted_c[0].hit_rate if len(sorted_c) > 0 else tp1_prob
            tp2_prob = sorted_c[1].hit_rate if len(sorted_c) > 1 else tp2_prob
            tp3_prob = sorted_c[2].hit_rate if len(sorted_c) > 2 else tp3_prob

        return {
            "plan_state":         state,
            "plan_reason":        reason,
            "entry_status":       _entry_status(entry_for_dna),
            "conditional_entry":  round(conditional_entry, 4) if conditional_entry is not None else None,
            "tp1_sanitize_reasons": plan.get("tp1_sanitize_reasons", []),
            "entry_primary":      plan["entry_primary"],
            "entry_aggressive":   plan["entry_aggressive"],
            "entry_conservative": plan["entry_conservative"],
            "stop_loss":          plan["stop_loss"],
            "tp1":                tp1_final,
            "tp1_probability":    round(float(tp1_prob), 3),
            "tp2":                tp2_final,
            "tp2_probability":    round(float(tp2_prob), 3),
            "tp3":                tp3_final,
            "tp3_probability":    round(float(tp3_prob), 3),
            "tp_source":          "dna_clusters" if tp1_final == dna_tp1 else "atr_based",
            "risk_reward_ratio":  risk_reward_ratio,
            "gain_pct_to_tp1":    gain_pct_to_tp1,
            # Informational: optimal hold window from DNA
            "optimal_hold_window_days": getattr(dna, "optimal_hold_window_days", None),
        }

    # Entry zones: keep entry near current price and reject deep historical anchors.
    max_entry_distance = 2.0 * atr_v
    nearby_supports = [
        float(s["price"])
        for s in supports
        if float(s["price"]) < current_close
        and (current_close - float(s["price"])) <= max_entry_distance
    ]
    nearby_supports = sorted(nearby_supports, reverse=True)

    if nearby_supports:
        entry_primary = nearby_supports[0] * 1.005
    else:
        entry_primary = current_close - 0.5 * atr_v

    # Hard guardrail: never let entry drift far below spot.
    entry_primary = max(entry_primary, current_close - max_entry_distance)

    # Check if momentum signals are firing
    rsi_v = _safe(indicators.get("rsi"), 50.0)
    macd_h = _safe(indicators.get("macd_histogram"), 0.0)
    momentum_firing = (rsi_v > 50 and macd_h > 0)

    # TP levels — stage-aware minimum ATR multiple to avoid noise triggering TP1
    # on random days. Without this, nearest resistance is often within 0.3-0.5%
    # which any intraday noise can reach, inflating the baseline hit rate to ~48%.
    # TP1 floor: minimum distance from current_close to avoid noise triggering TP1.
    # TP1 cap: maximum distance so TP1 is achievable within the 20-day horizon.
    # Resistance is used only if it falls within [floor, cap]; otherwise the floor
    # is used as a synthetic ATR-distance profit target.
    STAGE_TP1_ATR_FLOORS = {
        "ACCUMULATION":          1.8,
        "EARLY_MARKUP":          1.0,
        "MARKUP":                1.0,
        "DISTRIBUTION":          1.5,
        "MARKDOWN":              1.5,
        "NEUTRAL_AMBIGUOUS":     1.5,
        "DORMANT":                 2.0,
        "STEALTH_ACCUMULATION":    1.8,
        "EARLY_BREAKOUT":          1.0,
        "MARKUP_TRENDING":         1.0,
        "ACCELERATION_CLIMAX":     0.8,
        "DISTRIBUTION_TOPPING":    1.5,
        "MARKDOWN_DECLINE":        1.5,
        "CAPITULATION_EXHAUSTION": 1.0,
    }
    STAGE_TP1_ATR_CAPS = {
        "EARLY_MARKUP":          2.5,
        "MARKUP":                2.5,
        "ACCUMULATION":          8.0,
        "DISTRIBUTION":          20.0,
        "MARKDOWN":              20.0,
        "NEUTRAL_AMBIGUOUS":     20.0,
        # Tight caps only for active bullish stages — these have confidence 50-90+
        # and need TP1 to be achievable within the 20-day horizon.
        "EARLY_BREAKOUT":          2.5,   # floor=1.0, cap=2.5  → max ~2.5% from close
        "MARKUP_TRENDING":         2.5,   # floor=1.0, cap=2.5  → max ~2.5% from close
        "ACCELERATION_CLIMAX":     2.0,   # floor=0.8, cap=2.0  → max ~1.5% from close
        # Very high caps for passive/bearish stages — effectively no cap (preserves old
        # first-resistance-above-floor behaviour for DORMANT/MARKDOWN/DISTRIBUTION).
        # These stages dominate the 00-49 band and should keep TP1 at the natural
        # resistance level so the baseline stays ~30-35%, not inflated by floor fallback.
        "DORMANT":                 20.0,
        "STEALTH_ACCUMULATION":    8.0,
        "DISTRIBUTION_TOPPING":    20.0,
        "MARKDOWN_DECLINE":        20.0,
        "CAPITULATION_EXHAUSTION": 8.0,
    }
    def _dna_tp_prob(threshold_index: int, default: float) -> float:
        if dna is None:
            return default
        try:
            profiles = getattr(dna, "profiles_by_threshold", [])
            if profiles:
                # Use the lowest threshold profile's success rate as TP1 base
                p = profiles[min(threshold_index, len(profiles) - 1)]
                rate = p.success_rate if hasattr(p, "success_rate") else default * 100
                return round(rate / 100, 2)
        except Exception:
            pass
        return default

    current_plan = _build_plan(entry_primary)
    if current_plan is None:
        return _declined_plan("No favorable entry at current price — reward does not justify risk here.")

    if current_plan["risk_reward_ratio"] >= min_favorable_rr:
        return _finalize_plan(
            state="ACTIVE",
            reason=None,
            plan=current_plan,
        )

    candidate_entries: List[float] = []
    conservative_candidate = min(entry_primary, max(current_close - 0.75 * atr_v, current_close * 0.97))
    if conservative_candidate < current_close:
        candidate_entries.append(conservative_candidate)
    for support in supports:
        support_price = float(support["price"]) * 1.005
        if support_price < current_close:
            candidate_entries.append(support_price)

    conditional_plan = None
    seen_candidates = set()
    for candidate_entry in sorted(candidate_entries, reverse=True):
        if current_close - candidate_entry > max_conditional_pullback:
            continue
        key = round(candidate_entry, 4)
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        candidate_plan = _build_plan(
            candidate_entry,
            tp1_override=current_plan["raw_tp1"],
            plan_state="CONDITIONAL",
        )
        if candidate_plan is None:
            continue
        if candidate_plan["risk_reward_ratio"] >= min_favorable_rr:
            conditional_plan = candidate_plan
            break

    if conditional_plan is not None:
        conditional_plan["entry_aggressive"] = None
        return _finalize_plan(
            state="CONDITIONAL",
            reason="Setup forms on a pullback — current price is extended relative to the first target.",
            plan=conditional_plan,
            conditional_entry=conditional_plan["entry_primary"],
        )

    return _declined_plan(
        "No favorable entry at current price — reward does not justify risk here. Waiting for a better setup.",
    )


# ---------------------------------------------------------------------------
# 3. Position Sizing
# ---------------------------------------------------------------------------

def compute_position_size(
    confidence: float,
    entry: float,
    stop: float,
    portfolio_kwd: float,
    avg_daily_turnover_kwd: float,
    dna: Optional[Any] = None,
    regime_multiplier: float = 1.0,
    tp1_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Kelly-based (half-Kelly) position sizing with hard liquidity cap.

    The liquidity cap (10% of avg daily turnover / portfolio) is
    non-negotiable and cannot be toggled off.

    Returns
    -------
    dict with keys:
      size_pct, liquidity_capped, requires_confirmation, suggested_kwd
    """
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0 or entry <= 0:
        return {
            "size_pct": 0.0,
            "liquidity_capped": False,
            "requires_confirmation": False,
            "suggested_kwd": 0.0,
        }

    # Win rate from DNA or default
    win_rate = 0.55
    if dna is not None:
        try:
            profiles = getattr(dna, "profiles_by_threshold", [])
            if profiles:
                win_rate = profiles[0].success_rate / 100
        except Exception:
            pass

    # Average win in R multiples
    if tp1_price is not None and tp1_price > entry:
        avg_win_r = (tp1_price - entry) / risk_per_share
    else:
        avg_win_r = 1.5  # default 1.5R target

    # Kelly fraction (half-Kelly)
    if avg_win_r > 0:
        kelly = (win_rate * (avg_win_r + 1) - 1) / avg_win_r
    else:
        kelly = 0.0
    kelly = max(kelly, 0.0)
    half_kelly = kelly * CONFIG.HALF_KELLY_MULTIPLIER

    # Confidence multiplier — non-linear scaling
    conf_mult = (confidence / 100.0) ** 1.5

    # Raw size as percentage of portfolio
    raw_size_pct = half_kelly * conf_mult * regime_multiplier * 100.0

    # Hard liquidity cap — ALWAYS applied, cannot be toggled off
    if avg_daily_turnover_kwd > 0 and portfolio_kwd > 0:
        cap_pct = (CONFIG.LIQUIDITY_CAP_PCT_OF_DAILY_TURNOVER / 100.0 *
                   avg_daily_turnover_kwd / portfolio_kwd * 100.0)
    else:
        cap_pct = 100.0  # no cap if we can't compute turnover

    final_pct = min(raw_size_pct, cap_pct)
    liquidity_capped = final_pct < raw_size_pct

    requires_confirmation = final_pct > CONFIG.CONFIRMATION_MODAL_THRESHOLD_PCT

    suggested_kwd = round(portfolio_kwd * final_pct / 100.0, 2)

    return {
        "size_pct": round(final_pct, 2),
        "liquidity_capped": liquidity_capped,
        "requires_confirmation": requires_confirmation,
        "suggested_kwd": suggested_kwd,
    }


# ---------------------------------------------------------------------------
# 4. Confidence Score
# ---------------------------------------------------------------------------

def compute_confidence_from_phase(
    phase_score: float,
    indicators: IndicatorsRow,
) -> Tuple[float, str]:
    """
    Translate a phase-regression score to confidence and rating.

    Phase score interpretation:
      4.0 = STRONG_ACCUMULATION
      3.0 = ACCUMULATION
      2.0 = EARLY_MARKUP
      1.0 = HOLD_NEUTRAL
      0.0 = DISTRIBUTION
     -1.0 = STRONG_DISTRIBUTION
    """
    p = _safe_numeric(phase_score)
    if p is None:
        p = 1.0

    if p >= 3.5:
        confidence = 85.0 + (p - 3.5) * 20.0
        rating = "STRONG_BUY"
    elif p >= 2.5:
        confidence = 70.0 + (p - 2.5) * 15.0
        rating = "BUY"
    elif p >= 1.8:
        confidence = 55.0 + (p - 1.8) * 21.0
        rating = "BUY"
    elif p >= 1.2:
        confidence = 40.0 + (p - 1.2) * 25.0
        rating = "HOLD"
    elif p >= 0.5:
        confidence = 30.0 + (p - 0.5) * 14.0
        rating = "HOLD"
    elif p >= -0.3:
        confidence = 20.0 + (p + 0.3) * 12.0
        rating = "SELL"
    else:
        confidence = max(5.0, 20.0 + p * 10.0)
        rating = "STRONG_SELL"

    confidence = _apply_universal_safety_clamp(confidence, indicators)
    return round(float(np.clip(confidence, 0.0, 100.0)), 1), rating


def compute_rating_from_phase_score(
    phase_score: float,
    confidence: Optional[float] = None,
) -> str:
    """Return rating tier from phase score (confidence kept for API symmetry)."""
    del confidence
    _, rating = compute_confidence_from_phase(phase_score, indicators={})
    return rating

def compute_confidence(
    indicators: IndicatorsRow,
    stage: str,
    dna: Optional[Any],
    regime: str = "NEUTRAL",
    ml_score: Optional[float] = None,
    ml_proba: Optional[Dict[str, float]] = None,
) -> float:
    """
    Weighted composite confidence score 0-100.

    Weights:
      0.20 volume_flow_score   (extracted separately — highest single weight)
      0.18 confluence_score    (excl. volume_flow: trend, momentum, volatility, structure, institutional, statistical, regime)
      0.18 historical_base_rate
      0.15 accumulation_score
      0.13 risk_reward_score
      0.08 regime_alignment
      0.06 stage_score
      0.02 dna_pattern_match
    """
    if ml_proba is not None:
        try:
            phase_score = _safe_numeric(ml_proba.get("phase_score"))
            if phase_score is not None:
                conf, _ = compute_confidence_from_phase(phase_score, indicators)
                return conf

            p_buy = float(ml_proba.get("buy", 0.0))
            p_sell = float(ml_proba.get("sell", 0.0))
            p_hold = float(ml_proba.get("hold", 0.0))
            del p_hold

            buy_confidence = p_buy * 100.0

            if p_sell > p_buy and p_sell > 0.40:
                buy_confidence = min(buy_confidence, 20.0)

            if stage in ("MARKDOWN", "MARKDOWN_DECLINE", "DISTRIBUTION", "DISTRIBUTION_TOPPING"):
                buy_confidence = min(buy_confidence, 30.0)

            regime_u = str(regime or "NEUTRAL").upper()
            if regime_u == "RISK_OFF":
                buy_confidence -= 3.0
            elif regime_u == "RISK_ON":
                buy_confidence += 2.0

            buy_confidence = _apply_universal_safety_clamp(buy_confidence, indicators)
            return round(float(np.clip(buy_confidence, 0.0, 100.0)), 1)
        except (TypeError, ValueError, AttributeError):
            pass

    if ml_score is not None:
        try:
            predicted_gain = float(ml_score)
            if not math.isnan(predicted_gain):
                # Translate predicted gain (0-100%) to user confidence.
                if predicted_gain <= 0.0:
                    base_confidence = 5.0
                elif predicted_gain <= 2.0:
                    base_confidence = 10.0 + predicted_gain * 7.5
                elif predicted_gain <= 5.0:
                    base_confidence = 25.0 + (predicted_gain - 2.0) * 6.67
                elif predicted_gain <= 10.0:
                    base_confidence = 45.0 + (predicted_gain - 5.0) * 3.0
                elif predicted_gain <= 20.0:
                    base_confidence = 60.0 + (predicted_gain - 10.0) * 1.8
                elif predicted_gain <= 40.0:
                    base_confidence = 78.0 + (predicted_gain - 20.0) * 0.6
                else:
                    base_confidence = 90.0 + min(8.0, (predicted_gain - 40.0) * 0.13)

                # Stage hard cap for clearly bearish lifecycle states.
                if stage in ("MARKDOWN", "MARKDOWN_DECLINE", "DISTRIBUTION", "DISTRIBUTION_TOPPING"):
                    base_confidence = min(base_confidence, 35.0)

                # Regime nudge.
                if regime == "RISK_OFF":
                    base_confidence -= 4.0
                elif regime == "RISK_ON":
                    base_confidence += 3.0

                # Volume confirmation bonus.
                vol_ratio = None
                try:
                    vol_ratio = indicators.get("volume_ratio_20d")
                    if vol_ratio is None:
                        vol_ratio = indicators.get("rel_volume")
                except Exception:
                    vol_ratio = None
                if vol_ratio is not None:
                    try:
                        vr = float(vol_ratio)
                        if vr > 2.0 and base_confidence > 55.0:
                            base_confidence += 4.0
                    except (TypeError, ValueError):
                        pass

                base_confidence = _apply_universal_safety_clamp(base_confidence, indicators)
                return round(float(np.clip(base_confidence, 0.0, 100.0)), 1)
        except (TypeError, ValueError):
            pass

    # --- 1. Category scores: how many indicators in each category have bullish signals ---
    from app.services.eagle_eye.config import INDICATOR_CATEGORIES
    category_bullish = {}
    for cat, ind_list in INDICATOR_CATEGORIES.items():
        bullish_count = 0
        checked = 0
        for ind in ind_list:
            v = _safe(indicators.get(ind))
            if v is None:
                continue
            checked += 1
            # Simple heuristic: positive for trend/momentum/flow is bullish
            if cat in ("trend", "momentum"):
                if isinstance(v, (int, float)) and v > 0:
                    bullish_count += 1
            elif cat == "volume_flow":
                if isinstance(v, (int, float)) and v > 0:
                    bullish_count += 1
        if checked > 0:
            category_bullish[cat] = bullish_count / checked

    # Volume flow is extracted first and scored independently so it carries
    # its own explicit weight in the final formula rather than being diluted
    # as 1-of-8 inside a flat average.
    volume_flow_score = category_bullish.pop("volume_flow", 0.5) * 100

    # Confluence: average of all remaining (non-volume) categories
    confluence_score = (sum(category_bullish.values()) / max(len(category_bullish), 1)) * 100

    # --- 2. Historical base rate from DNA ---
    historical_base_rate = 0.5
    if dna is not None:
        try:
            profiles = getattr(dna, "profiles_by_threshold", [])
            if profiles:
                rates = [p.success_rate / 100 for p in profiles if hasattr(p, "success_rate")]
                if rates:
                    historical_base_rate = float(np.mean(rates))
        except Exception:
            pass

    # --- 3. Accumulation score (normalized 0-1) ---
    acc_score_raw = _safe(indicators.get("accumulation_score"), 50.0)
    acc_norm = float(acc_score_raw) / 100.0

    # --- 4. Risk-reward score (needs entry/stop/tp context — use placeholder 0.6 here) ---
    # The caller may inject a precomputed R:R; default to neutral
    rr_score = _safe(indicators.get("_risk_reward_ratio"), None)
    if rr_score is not None:
        rr_norm = min(float(rr_score) / 2.0, 1.0)
    else:
        rr_norm = 0.6  # conservative default assuming ~1.5R

    # --- 5. Regime alignment ---
    regime_map = {"RISK_ON": 1.0, "NEUTRAL": 0.6, "RISK_OFF": 0.3}
    regime_align = regime_map.get(regime.upper(), 0.6)

    # --- 6. Stage score ---
    # Keyed by both the raw stage names returned by classify_stage_with_confidence
    # (MARKUP, ACCUMULATION, DISTRIBUTION, MARKDOWN, EARLY_MARKUP, NEUTRAL_AMBIGUOUS)
    # and the legacy alias names, so this cap applies regardless of which
    # naming convention the caller passes.
    stage_scores = {
        "EARLY_BREAKOUT":         1.0,
        "EARLY_MARKUP":           1.0,
        "STEALTH_ACCUMULATION":   1.0,
        "ACCUMULATION":           1.0,
        "MARKUP_TRENDING":        0.8,
        "MARKUP":                 0.8,
        "DORMANT":                0.5,
        "NEUTRAL_AMBIGUOUS":      0.5,
        "CAPITULATION_EXHAUSTION": 0.5,  # contrarian opportunity
        "ACCELERATION_CLIMAX":    0.3,
        "DISTRIBUTION_TOPPING":   0.1,
        "DISTRIBUTION":           0.1,
        "MARKDOWN_DECLINE":       0.1,
        "MARKDOWN":               0.1,
    }
    stage_sc = stage_scores.get(stage, 0.5)

    # --- 7. DNA pattern match ---
    dna_match = 0.5
    if dna is not None:
        try:
            most_reliable = getattr(dna, "most_reliable_signals_overall", [])
            if most_reliable:
                fired_count = 0
                for sig_rel in most_reliable[:5]:
                    sig_name = getattr(sig_rel, "signal", None) or sig_rel.get("signal")
                    if sig_name and _safe(indicators.get(sig_name)):
                        fired_count += 1
                dna_match = fired_count / min(len(most_reliable), 5)
        except Exception:
            pass

    score = (
        0.20 * volume_flow_score                  # volume gets highest single weight
        + 0.18 * confluence_score                 # trend/momentum/structure/etc (excl. volume)
        + 0.18 * historical_base_rate * 100
        + 0.15 * acc_norm * 100
        + 0.13 * rr_norm * 100
        + 0.08 * regime_align * 100
        + 0.06 * stage_sc * 100
        + 0.02 * dna_match * 100
    )
    raw_confidence = float(np.clip(score, 0.0, 100.0))

    # ── Entry extension penalty ────────────────────────────────────────────
    # If price has already moved significantly from the setup anchor before the
    # signal fires, the risk/reward has deteriorated.  A stock that is 15%+
    # above its 20-day low is NOT a "cheap entry" regardless of how bullish the
    # indicators look.  Penalise confidence so the rating correctly reflects
    # that the easy money has already been made.
    #
    # Penalty table (applied to raw_confidence):
    #   < 5%   extension → no penalty   (still near the base)
    #   5-10%  extension → −5 points    (slightly stretched)
    #   10-15% extension → −12 points   (extended)
    #   15-20% extension → −20 points   (chasing)
    #   > 20%  extension → −30 points   (dangerous late entry)
    price_ext = _safe(indicators.get("price_extension_from_20d_low_pct"), None)
    if price_ext is not None:
        ext = float(price_ext)
        if ext >= 20.0:
            raw_confidence -= 30.0
        elif ext >= 15.0:
            raw_confidence -= 20.0
        elif ext >= 10.0:
            raw_confidence -= 12.0
        elif ext >= 5.0:
            raw_confidence -= 5.0
    raw_confidence = float(np.clip(raw_confidence, 0.0, 100.0))

    # --- TASK 1: Stage-gated confidence caps ---
    # A DORMANT stock structurally cannot hit TP1 in 20 days.
    # These caps prevent the composite score from misleading callers.
    # Keyed by both the raw stage names returned by classify_stage_with_confidence
    # and the legacy alias names — see stage_scores above for why both are needed.
    STAGE_CONFIDENCE_CAPS = {
        "DORMANT":                 40,  # quiet stock — should NEVER be high-conf BUY
        "NEUTRAL_AMBIGUOUS":       40,
        "STEALTH_ACCUMULATION":    75,  # institutional accumulation — can be high
        "ACCUMULATION":            75,
        "EARLY_BREAKOUT":         100,  # the IDEAL buy stage — no cap
        "EARLY_MARKUP":           100,
        "MARKUP_TRENDING":         90,  # trending up — high but not max
        "MARKUP":                  90,
        "ACCELERATION_CLIMAX":     55,  # late stage — risk rising
        "DISTRIBUTION_TOPPING":    30,  # actively topping — almost never BUY
        "DISTRIBUTION":            30,
        "MARKDOWN_DECLINE":        20,  # declining — should be SELL/HOLD only
        "MARKDOWN":                20,
        "CAPITULATION_EXHAUSTION": 50,  # potential reversal — moderate ceiling
    }
    stage_cap = STAGE_CONFIDENCE_CAPS.get(stage, 70)
    capped_confidence = min(raw_confidence, stage_cap)

    # --- TASK 2: Structural readiness multiplier ---
    # Even outside DORMANT, a stock with dead volume and tiny ATR cannot
    # realistically move enough to hit TP1 within the 20-day horizon.
    atr_pct_252 = _safe(indicators.get("atr_percentile_252"), 50.0)
    rel_vol = _safe(indicators.get("rel_volume"), 1.0)
    rsi_v = _safe(indicators.get("rsi"), 50.0)
    close_v = _safe(indicators.get("close"), None)
    ema50_v = _safe(indicators.get("ema_50"), None)

    price_above_50ma = (
        close_v is not None and ema50_v is not None and float(close_v) > float(ema50_v)
    )
    is_structurally_ready = (
        float(atr_pct_252) > 30
        and float(rel_vol) > 0.7
        and (price_above_50ma or float(rsi_v) > 40)
    )
    if not is_structurally_ready:
        capped_confidence = min(capped_confidence, 55)

    capped_confidence = _apply_universal_safety_clamp(capped_confidence, indicators)
    return round(float(np.clip(capped_confidence, 0.0, 100.0)), 1)


# ---------------------------------------------------------------------------
# 5. Rating
# ---------------------------------------------------------------------------

def compute_rating_from_proba(
    ml_proba: Optional[Dict[str, float]],
    confidence: float,
    dna: Optional[Any] = None,
) -> str:
    """Map ML class probabilities to rating tiers.

    Falls back to the legacy confidence-only mapping when probability payload
    is unavailable.
    """
    if ml_proba is None:
        return compute_rating(confidence, dna=dna)

    try:
        p_buy = float(ml_proba.get("buy", 0.0))
        p_sell = float(ml_proba.get("sell", 0.0))
    except (TypeError, ValueError, AttributeError):
        return compute_rating(confidence, dna=dna)

    safe_conf = _safe_numeric(confidence)
    if safe_conf is None:
        safe_conf = 0.0

    if p_sell > 0.60:
        return "STRONG_SELL"
    if safe_conf <= 25.0:
        return "SELL" if p_sell > 0.40 else "HOLD"
    if p_sell > 0.40:
        return "SELL"
    if safe_conf <= 40.0:
        return "HOLD"
    if p_buy > 0.75:
        return "STRONG_BUY"
    if p_buy > 0.55:
        return "BUY"
    if p_buy > 0.40:
        return "HOLD"
    return "HOLD"

def compute_rating(confidence: float, dna: Optional[Any] = None) -> str:
    """
    Map a confidence score to a rating string.

    Returns INSUFFICIENT_DATA when DNA is missing and history is too short.
    """
    if dna is None:
        # Without DNA we can still issue a rating, but flag insufficient data
        # only if confidence is very low (ambiguous)
        if confidence < 30:
            return "INSUFFICIENT_DATA"

    if confidence >= CONFIG.STRONG_BUY_CONFIDENCE:
        return "STRONG_BUY"
    elif confidence >= CONFIG.BUY_CONFIDENCE:
        return "BUY"
    elif confidence >= CONFIG.HOLD_CONFIDENCE:
        return "HOLD"
    elif confidence >= CONFIG.SELL_CONFIDENCE:
        return "SELL"
    else:
        return "STRONG_SELL"


# ---------------------------------------------------------------------------
# 6. Thesis Generator
# ---------------------------------------------------------------------------

# Template sentences by stage
_STAGE_INTRO: Dict[str, str] = {
    "ACCUMULATION":         "{ticker} is basing in accumulation",
    "EARLY_MARKUP":         "{ticker} is attempting an early markup breakout",
    "MARKUP":               "{ticker} is in an established markup trend",
    "DISTRIBUTION":         "{ticker} is showing distribution/topping behaviour",
    "MARKDOWN":             "{ticker} is in a markdown decline regime",
    "NEUTRAL_AMBIGUOUS":    "{ticker} has a mixed/ambiguous structure",
    # Legacy stage labels preserved for backward compatibility.
    "EARLY_BREAKOUT":          "{ticker} is staging an early breakout",
    "STEALTH_ACCUMULATION":    "{ticker} is in stealth accumulation",
    "MARKUP_TRENDING":         "{ticker} is in an established uptrend",
    "DORMANT":                 "{ticker} is dormant",
    "ACCELERATION_CLIMAX":     "{ticker} is approaching climax conditions",
    "DISTRIBUTION_TOPPING":    "{ticker} shows distribution/topping signals",
    "MARKDOWN_DECLINE":        "{ticker} is in a confirmed downtrend",
    "CAPITULATION_EXHAUSTION": "{ticker} is at capitulation/exhaustion levels",
}

_RATING_TAIL: Dict[str, str] = {
    "STRONG_BUY":        "presenting a high-conviction opportunity.",
    "BUY":               "presenting a favourable risk/reward setup.",
    "HOLD":              "warranting a hold stance.",
    "SELL":              "suggesting reducing exposure.",
    "STRONG_SELL":       "indicating a strong sell signal.",
    "REDUCE":            "suggesting exposure should be reduced.",
    "WATCHLIST":         "worth monitoring for confirmation before entry.",
    "AVOID":             "best avoided until structure improves.",
    "NEUTRAL":           "not offering a strong directional edge.",
    "INSUFFICIENT_DATA": "but data is insufficient for a firm recommendation.",
}


def generate_thesis(
    ticker: str,
    rating: str,
    stage: str,
    indicators: IndicatorsRow,
    dna: Optional[Any],
    top_signals_fired: List[str],
) -> str:
    """
    Build a one- to two-sentence plain-English thesis from templates.
    No AI generation -- deterministic and fast.
    """
    intro_tmpl = _STAGE_INTRO.get(stage, "{ticker} shows mixed signals")
    intro = intro_tmpl.format(ticker=ticker)

    detail_parts: List[str] = []

    # Volume context
    rel_vol = _safe(indicators.get("rel_volume"), 1.0)
    if rel_vol and rel_vol > 1.5:
        detail_parts.append(f"volume {rel_vol:.1f}x average")

    # OBV trend
    obv_slope = _safe(indicators.get("obv_slope_20"), 0.0)
    if obv_slope and obv_slope > 0:
        detail_parts.append("OBV trending up")

    # Accumulation
    acc = _safe(indicators.get("accumulation_score"), 50.0)
    if acc and acc > 60:
        detail_parts.append(f"accumulation score {acc:.0f}")

    # Entry status -- inform the user if price is extended
    price_ext = _safe(indicators.get("price_extension_from_20d_low_pct"), None)
    if price_ext is not None and float(price_ext) >= 10.0:
        detail_parts.append(f"price {float(price_ext):.0f}% above base (wait for pullback)")

    # DNA base rate
    dna_note = ""
    if dna is not None:
        try:
            profiles = getattr(dna, "profiles_by_threshold", [])
            if profiles:
                sr = profiles[0].success_rate
                dna_note = f" Setup matched TP1 in {sr:.0f}% of prior similar conditions."
        except Exception:
            pass

    # Top signals
    if top_signals_fired:
        signals_str = ", ".join(top_signals_fired[:3])
        detail_parts.append(signals_str)

    rating_tail = _RATING_TAIL.get(rating, "")

    if detail_parts:
        detail = " with " + ", ".join(detail_parts) + " -- " + rating_tail
    else:
        detail = " -- " + rating_tail

    return f"{intro}{detail}{dna_note}".strip()

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
    "compute_confidence",
    "compute_rating",
    "generate_thesis",
]

IndicatorsRow = Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
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

    # Entry zones
    if supports:
        primary_support = supports[0]["price"]
        entry_primary = primary_support * 1.005  # slightly above nearest support
    else:
        entry_primary = current_close

    # Check if momentum signals are firing
    rsi_v = _safe(indicators.get("rsi"), 50.0)
    macd_h = _safe(indicators.get("macd_histogram"), 0.0)
    momentum_firing = (rsi_v > 50 and macd_h > 0)

    entry_aggressive = current_close if momentum_firing else entry_primary
    entry_conservative = (supports[1]["price"] if len(supports) > 1
                          else entry_primary * 0.97)

    # Stop loss: max(entry - 2 * ATR, below nearest strong support)
    stop_from_atr = entry_primary - 2.0 * atr_v
    if supports:
        stop_from_support = supports[0]["price"] * 0.99
        stop_loss = max(stop_from_atr, stop_from_support)
    else:
        stop_loss = stop_from_atr

    # TP levels — stage-aware minimum ATR multiple to avoid noise triggering TP1
    # on random days. Without this, nearest resistance is often within 0.3-0.5%
    # which any intraday noise can reach, inflating the baseline hit rate to ~48%.
    # TP1 floor: minimum distance from current_close to avoid noise triggering TP1.
    # TP1 cap: maximum distance so TP1 is achievable within the 20-day horizon.
    # Resistance is used only if it falls within [floor, cap]; otherwise the floor
    # is used as a synthetic ATR-distance profit target.
    STAGE_TP1_ATR_FLOORS = {
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
    tp1_floor_mult = STAGE_TP1_ATR_FLOORS.get(stage, 1.5)
    tp1_cap_mult   = STAGE_TP1_ATR_CAPS.get(stage, 3.0)
    min_tp1 = current_close + tp1_floor_mult * atr_v
    max_tp1 = current_close + tp1_cap_mult   * atr_v

    # Use nearest resistance within [floor, cap]; fall back to floor if none found.
    tp1 = None
    for res in resistances:
        if min_tp1 <= res["price"] <= max_tp1:
            tp1 = res["price"]
            break
    if tp1 is None:
        tp1 = min_tp1  # ATR-floor fallback

    # TP2: next resistance beyond TP1, or TP1 + 1×ATR
    tp2 = None
    for res in resistances:
        if res["price"] > tp1 * 1.005:  # at least 0.5% beyond TP1
            tp2 = res["price"]
            break
    if tp2 is None:
        tp2 = tp1 + atr_v

    # TP3: measured-move target — range projected from breakout
    if len(df) >= 60:
        recent_range = df["high"].tail(60).max() - df["low"].tail(60).min()
        breakout_base = df["high"].tail(60).max()
        tp3 = breakout_base + recent_range
    else:
        tp3 = entry_primary * 1.20

    # Probabilities — from DNA if available, else conservative defaults
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

    tp1_prob = _dna_tp_prob(0, 0.55)
    tp2_prob = _dna_tp_prob(1, 0.35)
    tp3_prob = _dna_tp_prob(2, 0.15)

    return {
        "entry_primary":    round(entry_primary, 4),
        "entry_aggressive": round(entry_aggressive, 4),
        "entry_conservative": round(entry_conservative, 4),
        "stop_loss":        round(stop_loss, 4),
        "tp1":              round(tp1, 4),
        "tp1_probability":  tp1_prob,
        "tp2":              round(tp2, 4),
        "tp2_probability":  tp2_prob,
        "tp3":              round(tp3, 4),
        "tp3_probability":  tp3_prob,
    }


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

def compute_confidence(
    indicators: IndicatorsRow,
    stage: str,
    dna: Optional[Any],
    regime: str = "NEUTRAL",
) -> float:
    """
    Weighted composite confidence score 0-100.

    Weights:
      0.25 confluence_score
      0.20 historical_base_rate
      0.15 accumulation_score
      0.15 risk_reward_score
      0.10 regime_alignment
      0.10 stage_score
      0.05 dna_pattern_match
    """
    # --- 1. Confluence score: how many of 8 categories have bullish signals ---
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
    stage_scores = {
        "EARLY_BREAKOUT":         1.0,
        "STEALTH_ACCUMULATION":   1.0,
        "MARKUP_TRENDING":        0.8,
        "DORMANT":                0.5,
        "CAPITULATION_EXHAUSTION": 0.5,  # contrarian opportunity
        "ACCELERATION_CLIMAX":    0.3,
        "DISTRIBUTION_TOPPING":   0.1,
        "MARKDOWN_DECLINE":       0.1,
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
        0.25 * confluence_score
        + 0.20 * historical_base_rate * 100
        + 0.15 * acc_norm * 100
        + 0.15 * rr_norm * 100
        + 0.10 * regime_align * 100
        + 0.10 * stage_sc * 100
        + 0.05 * dna_match * 100
    )
    raw_confidence = float(np.clip(score, 0.0, 100.0))

    # --- TASK 1: Stage-gated confidence caps ---
    # A DORMANT stock structurally cannot hit TP1 in 20 days.
    # These caps prevent the composite score from misleading callers.
    STAGE_CONFIDENCE_CAPS = {
        "DORMANT":                 40,  # quiet stock — should NEVER be high-conf BUY
        "STEALTH_ACCUMULATION":    75,  # institutional accumulation — can be high
        "EARLY_BREAKOUT":         100,  # the IDEAL buy stage — no cap
        "MARKUP_TRENDING":         90,  # trending up — high but not max
        "ACCELERATION_CLIMAX":     55,  # late stage — risk rising
        "DISTRIBUTION_TOPPING":    30,  # actively topping — almost never BUY
        "MARKDOWN_DECLINE":        20,  # declining — should be SELL/HOLD only
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

    return round(capped_confidence, 2)


# ---------------------------------------------------------------------------
# 5. Rating
# ---------------------------------------------------------------------------

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
    "STRONG_BUY":       "presenting a high-conviction opportunity.",
    "BUY":              "presenting a favourable risk/reward setup.",
    "HOLD":             "warranting a hold stance.",
    "SELL":             "suggesting reducing exposure.",
    "STRONG_SELL":      "indicating a strong sell signal.",
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
    No AI generation — deterministic and fast.
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
        detail = " with " + ", ".join(detail_parts) + " — " + rating_tail
    else:
        detail = " — " + rating_tail

    return f"{intro}{detail}{dna_note}".strip()

"""
Stage Classifier — classifies a stock into one of 8 lifecycle stages
using rule-based logic derived from the EngineConfig thresholds.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.services.eagle_eye.config import CONFIG

# Type alias — an "indicators row" is any mapping of str->value
IndicatorsRow = Dict[str, Any]


def _get(row: IndicatorsRow, key: str, default=None):
    """Safe getter that returns default when key is missing or NaN."""
    import math
    v = row.get(key, default)
    if v is None:
        return default
    try:
        if math.isnan(v):
            return default
    except (TypeError, ValueError):
        pass
    return v


def classify_stage(
    indicators_row: IndicatorsRow,
    dna: Optional[Any] = None,
) -> str:
    """
    Classify a stock into one of 8 lifecycle stages based on indicator values.

    Parameters
    ----------
    indicators_row : dict
        The most recent row from compute_all_indicators().
    dna : BehavioralDNA or None
        Optional behavioral DNA for personality-aware adjustments (unused in
        rule-based tier; reserved for future ML tier).

    Returns
    -------
    str — one of the 8 STAGES values.
    """
    atr_pct = _get(indicators_row, "atr_percentile_252", 50.0)
    bb_bw_pct = _get(indicators_row, "bb_bandwidth", None)
    acc_score = _get(indicators_row, "accumulation_score", 50.0)
    rsi_val = _get(indicators_row, "rsi", 50.0)
    zscore_20 = _get(indicators_row, "zscore_20", 0.0)
    adx_val = _get(indicators_row, "adx", 15.0)
    plus_di = _get(indicators_row, "plus_di", 20.0)
    minus_di = _get(indicators_row, "minus_di", 20.0)
    ema_ribbon = _get(indicators_row, "ema_ribbon_aligned", 0)
    obv_slope_20 = _get(indicators_row, "obv_slope_20", 0.0)
    cmf_val = _get(indicators_row, "cmf", 0.0)
    rel_vol = _get(indicators_row, "rel_volume", 1.0)

    # Derive BB bandwidth percentile as a proxy if raw bb_bandwidth is given
    # (atr_percentile_252 already normalizes ATR to 252-bar percentile)
    bb_bw_pctile = _get(indicators_row, "bb_squeeze", None)
    # bb_squeeze == 1 means bandwidth is in bottom quintile (<20th percentile)
    is_bb_compressed = (bb_bw_pctile == 1) or (
        bb_bw_pct is not None and bb_bw_pct < 0.05
    )

    # --- DORMANT ---
    # Very low ATR + very compressed Bollinger Bands
    if (
        atr_pct is not None and atr_pct < CONFIG.DORMANT_ATR_PCTILE_MAX
        and is_bb_compressed
    ):
        return "DORMANT"

    # --- CAPITULATION_EXHAUSTION ---
    # Extreme oversold: RSI < 25 AND price > 2 stds below 20MA
    if rsi_val is not None and rsi_val < CONFIG.CAPITULATION_RSI_THRESHOLD:
        if zscore_20 is not None and zscore_20 < -2.0:
            return "CAPITULATION_EXHAUSTION"

    # --- ACCELERATION_CLIMAX ---
    # RSI > 80 AND price > 2 stds above 20MA AND volume spike
    if rsi_val is not None and rsi_val > CONFIG.CLIMAX_RSI_THRESHOLD:
        if zscore_20 is not None and zscore_20 > 2.0:
            if rel_vol is not None and rel_vol > 1.5:
                return "ACCELERATION_CLIMAX"

    # --- DISTRIBUTION_TOPPING ---
    # OBV diverging down AND CMF negative
    obv_falling = obv_slope_20 is not None and obv_slope_20 < 0
    cmf_negative = cmf_val is not None and cmf_val < 0
    rsi_high = rsi_val is not None and rsi_val > 60
    if obv_falling and cmf_negative and rsi_high:
        return "DISTRIBUTION_TOPPING"

    # --- MARKDOWN_DECLINE ---
    # EMA ribbon bearish AND ADX > threshold with -DI dominant
    if ema_ribbon is not None and ema_ribbon == -1:
        if adx_val is not None and adx_val > CONFIG.TRENDING_ADX_THRESHOLD:
            if minus_di is not None and plus_di is not None and minus_di > plus_di:
                return "MARKDOWN_DECLINE"

    # --- MARKUP_TRENDING ---
    # EMA ribbon bullish AND ADX > threshold
    if ema_ribbon is not None and ema_ribbon == 1:
        if adx_val is not None and adx_val > CONFIG.TRENDING_ADX_THRESHOLD:
            return "MARKUP_TRENDING"

    # --- EARLY_BREAKOUT ---
    # Price breaks recent highs with volume > 2x avg AND ADX rising
    is_volume_breakout = rel_vol is not None and rel_vol > CONFIG.BREAKOUT_VOLUME_MULTIPLIER
    adx_rising = adx_val is not None and adx_val > 18  # ADX > 18 signals trending onset
    if is_volume_breakout and adx_rising:
        if zscore_20 is not None and zscore_20 > 0.5:  # price above 20MA
            return "EARLY_BREAKOUT"

    # --- STEALTH_ACCUMULATION ---
    # Accumulation score > threshold AND price relatively flat AND OBV rising
    if acc_score is not None and acc_score > CONFIG.ACCUMULATION_SCORE_THRESHOLD:
        if obv_slope_20 is not None and obv_slope_20 > 0:
            if zscore_20 is not None and abs(zscore_20) < 1.0:  # price near MA = flat
                return "STEALTH_ACCUMULATION"

    # Default fallback
    return "DORMANT"

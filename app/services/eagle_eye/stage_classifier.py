"""Wyckoff stage classifier (Phase 1: rules only)."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple


IndicatorsRow = Dict[str, Any]


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):
        return default
    return f


def classify_stage_rules(
    indicators_row: Mapping[str, object],
    family_scores: Optional[Mapping[str, float]] = None,
) -> Tuple[str, float]:
    """Return (stage, confidence 0-1) using rules-only Wyckoff logic."""
    ind = indicators_row

    def g(key: str, default: float = 0.0) -> float:
        return _safe_float(ind.get(key), default)

    if (
        g("traded_value_ratio_20d") > 1.2
        and (
            int(g("high_volume_weak_close_flag")) == 1
            or g("cmf_20") < 0.0
            or g("up_down_volume_ratio_20d", 1.0) < 0.9
            or g("close_location_value") < -0.3
        )
        and g("macd_histogram_slope_5d") < 0.0
    ):
        return "DISTRIBUTION", 0.85

    cmf_turning = g("cmf_20_change_5d") > 0.0 and g("cmf_20_change_10d") > 0.0
    obv_turning = g("obv_slope_change_10d") > 0.0
    rsi_recovering = g("rsi_14_change_5d") > 2.0 and g("rsi_14", 50.0) < 50.0
    green_bars = g("consecutive_up_closes") >= 2.0
    near_bottom = g("pct_above_60d_low") < 12.0
    deeply_sold = g("rsi_14", 50.0) < 45.0

    turn_signals = sum([cmf_turning, obv_turning, rsi_recovering, green_bars])

    # Early accumulation / bottoming: negative flow can still qualify if the
    # flow inflection is improving from depressed levels.
    if (
        near_bottom
        and deeply_sold
        and turn_signals >= 2
        and int(g("high_volume_weak_close_flag")) == 0
    ):
        conf = 0.55 + 0.08 * turn_signals
        return "ACCUMULATION", min(conf, 0.85)

    if (
        g("stock_close_vs_50sma") < 0.0
        and g("stock_50sma_slope_20d") < 0.0
        and g("cmf_20") < 0.0
        and g("obv_slope_20d") < 0.0
        and not (cmf_turning and obv_turning)
    ):
        return "MARKDOWN", 0.90

    di_spread = g("plus_di") - g("minus_di")
    breakout_path = (
        int(g("donchian_breakout_50d")) == 1
        and g("traded_value_ratio_20d") > 1.5
        and g("cmf_20") > 0.0
        and g("close_location_value") > 0.5
        and 50.0 <= g("rsi_14", 50.0) <= 75.0
    )
    trend_turn_path = (
        di_spread > 0.0
        and g("cmf_20") > -0.02
        and g("stock_50sma_slope_20d") > 0.0
        and 50.0 <= g("rsi_14", 50.0) <= 72.0
        and g("macd_histogram_slope_5d") > -0.08
        and g("pct_above_60d_low", 100.0) < 20.0
        and g("stock_close_vs_200sma") < 0.05
        and g("stock_close_vs_50sma") < 0.10
    )

    if (
        breakout_path
        or trend_turn_path
    ):
        return "EARLY_MARKUP", 0.85 if breakout_path else 0.72

    if (
        g("stock_close_vs_50sma") > 0.0
        and g("stock_50sma_slope_20d") > 0.0
        and g("stock_close_vs_200sma") > 0.0
        and g("cmf_20") > 0.0
    ):
        return "MARKUP", 0.80

    if (
        g("bb_width_percentile_252d", 0.5) < 0.25
        and g("obv_slope_20d") > 0.0
        and g("cmf_20") > -0.05
        and g("rsi_14", 50.0) > 40.0
        and g("price_extension_from_50sma") < 0.05
    ):
        return "ACCUMULATION", 0.75

    # Secondary nudge: if family view clearly risk-off, mark markdown instead of ambiguous.
    if family_scores is not None:
        liq = float(family_scores.get("liquidity", 50.0))
        trd = float(family_scores.get("trend", 50.0))
        if (
            liq < 45.0
            and trd < 45.0
            and g("cmf_20") < 0.05
            and g("stock_close_vs_50sma") < 0.0
        ):
            return "MARKDOWN", 0.65

    return "NEUTRAL_AMBIGUOUS", 0.40


def classify_stage(
    indicators_row: IndicatorsRow,
    dna: Optional[Any] = None,
    family_scores: Optional[Mapping[str, float]] = None,
) -> str:
    """Compatibility helper returning only the stage label."""
    del dna
    stage, _ = classify_stage_rules(indicators_row, family_scores=family_scores)
    return stage


def classify_stage_with_confidence(
    indicators_row: IndicatorsRow,
    family_scores: Optional[Mapping[str, float]] = None,
) -> Tuple[str, float]:
    """Explicit stage API for the new rules-first recommendation engine."""
    return classify_stage_rules(indicators_row, family_scores=family_scores)

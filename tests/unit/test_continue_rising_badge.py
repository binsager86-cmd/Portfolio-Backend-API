from __future__ import annotations

from app.services.eagle_eye.scoring.recommendation_engine import compute_continue_rising


def _base_indicators() -> dict[str, object]:
    return {
        "close": 105.0,
        "ema_10": 100.0,
        "ema_20": 98.0,
        "ema_30": 95.0,
        "plus_di": 28.0,
        "minus_di": 18.0,
        "volume_ratio_20d": 1.3,
        "macd_histogram_slope_5d": 0.2,
    }


def test_continue_rising_qualifies_for_early_breakout_stage() -> None:
    result = compute_continue_rising(_base_indicators(), "EARLY_BREAKOUT")

    assert result["continue_rising"] is True
    assert result["continue_rising_badge"] == "CONTINUE_RISING"
    assert result["continue_rising_label"] == "Riding"


def test_continue_rising_qualifies_for_markup_trending_stage() -> None:
    result = compute_continue_rising(_base_indicators(), "MARKUP_TRENDING")

    assert result["continue_rising"] is True
    assert result["continue_rising_badge"] == "CONTINUE_RISING"


def test_continue_rising_rejects_neutral_stage() -> None:
    result = compute_continue_rising(_base_indicators(), "NEUTRAL_AMBIGUOUS")

    assert result["continue_rising"] is False
    assert result["continue_rising_badge"] is None
    assert result["continue_rising_label"] is None
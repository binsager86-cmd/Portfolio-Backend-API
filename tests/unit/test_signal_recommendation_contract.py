from __future__ import annotations

from app.services.signal_engine.engine.signal_generator import (
    _compute_data_quality_score,
    _compute_recommendation_contract,
)


def test_contract_marks_insufficient_data_when_data_quality_too_low() -> None:
    contract = _compute_recommendation_contract(
        final_signal="NEUTRAL",
        direction_score=5.0,
        setup_quality_score=20.0,
        timing_score=10.0,
        data_quality_score=25.0,
        expected_value_r=None,
        entry_trigger_action="HOLD",
        neutral_reason="insufficient_data",
    )
    assert contract["recommendation"] == "INSUFFICIENT_DATA"
    assert contract["actionable"] is False


def test_contract_maps_high_quality_long_to_strong_buy() -> None:
    contract = _compute_recommendation_contract(
        final_signal="STRONG_BUY",
        direction_score=68.0,
        setup_quality_score=86.0,
        timing_score=81.0,
        data_quality_score=92.0,
        expected_value_r=0.62,
        entry_trigger_action="ENTER",
        neutral_reason="",
    )
    assert contract["direction"] == "LONG"
    assert contract["recommendation"] == "STRONG_BUY"
    assert contract["actionable"] is True


def test_contract_maps_long_without_trigger_to_watch_long() -> None:
    contract = _compute_recommendation_contract(
        final_signal="BUY",
        direction_score=42.0,
        setup_quality_score=70.0,
        timing_score=40.0,
        data_quality_score=90.0,
        expected_value_r=0.21,
        entry_trigger_action="HOLD",
        neutral_reason="",
    )
    assert contract["direction"] == "LONG"
    assert contract["recommendation"] == "WATCH_LONG"
    assert contract["actionable"] is False


def test_contract_does_not_convert_weak_bullish_to_short() -> None:
    contract = _compute_recommendation_contract(
        final_signal="NEUTRAL",
        direction_score=6.0,
        setup_quality_score=51.0,
        timing_score=50.0,
        data_quality_score=88.0,
        expected_value_r=0.02,
        entry_trigger_action="HOLD",
        neutral_reason="",
    )
    assert contract["direction"] == "NEUTRAL"
    assert contract["recommendation"] == "HOLD"


def test_contract_maps_high_quality_short_to_sell() -> None:
    contract = _compute_recommendation_contract(
        final_signal="SELL",
        direction_score=-55.0,
        setup_quality_score=72.0,
        timing_score=66.0,
        data_quality_score=85.0,
        expected_value_r=0.34,
        entry_trigger_action="WATCH",
        neutral_reason="",
    )
    assert contract["direction"] == "SHORT"
    assert contract["recommendation"] == "SELL"
    assert contract["actionable"] is True


def test_data_quality_score_falls_with_missing_market_and_indicator_fields() -> None:
    rows = [
        {
            "date": "2026-01-01",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
            "value": 100.0,
        }
        for _ in range(80)
    ]
    rows[-1].pop("volume", None)
    rows[-1].pop("ema_20", None)

    score, reasons = _compute_data_quality_score(rows, min_bars_required=60)
    assert score < 100.0
    assert "missing_market_fields" in reasons

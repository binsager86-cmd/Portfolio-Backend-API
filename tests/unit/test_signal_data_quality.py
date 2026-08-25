from __future__ import annotations

from app.services.signal_engine.engine.signal_generator import _compute_data_quality_score


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "close": 100.0,
        "volume": 1_000_000.0,
        "value": 100_000.0,
        "ema_20": 100.0,
        "ema_50": 99.0,
        "sma_200": 98.0,
        "adx_14": 25.0,
        "rsi_14": 55.0,
        "macd": 1.0,
        "macd_signal": 0.5,
        "atr_14": 2.0,
        "cmf_20": 0.1,
        "corporate_actions": [],
        "market": "KSE",
    }
    row.update(overrides)
    return row


def test_complete_data_has_full_quality() -> None:
    score, reasons = _compute_data_quality_score([_row() for _ in range(60)], min_bars_required=60)
    assert score == 100.0
    assert reasons == []


def test_missing_indicator_is_named_and_reduces_quality() -> None:
    score, reasons = _compute_data_quality_score(
        [_row() for _ in range(59)] + [_row(cmf_20=None)], min_bars_required=60
    )
    assert score < 100.0
    assert "missing_indicator_cmf_20" in reasons


def test_invalid_ohlc_is_named_and_reduces_quality() -> None:
    score, reasons = _compute_data_quality_score(
        [_row() for _ in range(59)] + [_row(high=90.0)], min_bars_required=60
    )
    assert score < 100.0
    assert any(reason.startswith("invalid_ohlc_bars_") for reason in reasons)


def test_low_indicator_coverage_is_below_required_threshold() -> None:
    score, reasons = _compute_data_quality_score(
        [_row() for _ in range(59)] + [_row(ema_20=None, ema_50=None, sma_200=None, adx_14=None, rsi_14=None)],
        min_bars_required=60,
    )
    assert score < 35.0
    assert "indicator_coverage_below_75%" in reasons

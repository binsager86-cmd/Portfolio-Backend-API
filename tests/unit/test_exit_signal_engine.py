"""Unit tests for exit signal engine scenario checklist."""

from __future__ import annotations

from app.services.signal_engine.config.kuwait_constants import align_to_tick
from app.services.signal_engine.engine.exit_signal_engine import (
    _compute_momentum_exhaustion,
    generate_exit_signal,
)


def _row(
    close: float,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 120_000.0,
    atr_14: float = 2.0,
    adx_14: float = 20.0,
    rsi_14: float = 55.0,
    macd_hist: float = 0.2,
    ema_20: float | None = None,
    obv: float = 1_000_000.0,
    cmf_20: float = 0.10,
) -> dict:
    return {
        "date": "2026-05-09",
        "open": open_ if open_ is not None else close - 0.5,
        "high": high if high is not None else close + 0.8,
        "low": low if low is not None else close - 0.8,
        "close": close,
        "volume": volume,
        "atr_14": atr_14,
        "adx_14": adx_14,
        "rsi_14": rsi_14,
        "macd_hist": macd_hist,
        "ema_20": ema_20 if ema_20 is not None else close - 0.5,
        "obv": obv,
        "cmf_20": cmf_20,
    }


def _rows(count: int = 20, start: float = 100.0, step: float = 0.2, **kwargs) -> list[dict]:
    return [_row(close=start + (i * step), **kwargs) for i in range(count)]


def test_gap_down_stop_triggers_exit_on_intraday_low() -> None:
    rows = _rows(count=20, start=98.0, step=0.1, atr_14=2.0, rsi_14=50.0)
    rows[-1] = _row(
        close=99.0,
        open_=96.5,
        high=99.4,
        low=95.4,
        atr_14=2.0,
        rsi_14=52.0,
        macd_hist=0.1,
        ema_20=98.8,
    )

    signal = generate_exit_signal("NBK", rows, entry_price=100.0, bars_held=6)

    assert signal["action"] == "EXIT"
    assert signal["urgency"] == "CRITICAL"
    assert rows[-1]["close"] > signal["trailing_stop"]
    assert rows[-1]["low"] <= signal["trailing_stop"]


def test_new_position_under_five_bars_returns_hold_low_with_insufficient_data() -> None:
    rows = _rows(count=20, start=98.0, step=0.1, atr_14=2.0, rsi_14=50.0)
    rows[-1] = _row(
        close=99.0,
        open_=96.5,
        high=99.4,
        low=95.4,
        atr_14=2.0,
        rsi_14=52.0,
        macd_hist=0.1,
        ema_20=98.8,
    )

    signal = generate_exit_signal("NBK", rows, entry_price=100.0, bars_held=2)

    assert signal["action"] == "HOLD"
    assert signal["urgency"] == "LOW"
    assert signal["reasons"] == ["insufficient_data"]


def test_adx_plus_rsi_guard_avoids_premature_trim() -> None:
    rows = _rows(
        count=20,
        start=120.0,
        step=0.0,
        adx_14=32.0,
        rsi_14=82.0,
        macd_hist=0.4,
        atr_14=4.0,
        ema_20=119.5,
    )

    score, _ = _compute_momentum_exhaustion(rows, current_price=float(rows[-1]["close"]))
    assert score < 50


def test_rsi_seventy_tier_adds_twelve_points() -> None:
    rows = _rows(
        count=20,
        start=100.0,
        step=0.0,
        adx_14=30.0,
        rsi_14=72.0,
        macd_hist=0.3,
        atr_14=2.0,
        ema_20=99.9,
    )

    score, reasons = _compute_momentum_exhaustion(rows, current_price=float(rows[-1]["close"]))

    assert score == 12
    assert any("RSI elevated" in reason for reason in reasons)


def test_runtime_scenario_two_rsi_seventy_eight_is_trim_medium() -> None:
    rows = _rows(
        count=20,
        start=112.0,
        step=0.2,
        rsi_14=78.0,
        adx_14=50.0,
        macd_hist=0.5,
        atr_14=2.0,
        ema_20=108.0,
        cmf_20=0.08,
    )
    rows[-1] = _row(
        close=116.0,
        open_=117.2,
        high=117.5,
        low=115.6,
        volume=300_000.0,
        atr_14=2.0,
        adx_14=50.0,
        rsi_14=78.0,
        macd_hist=0.6,
        ema_20=108.4,
        obv=980_000.0,
        cmf_20=-0.10,
    )

    signal = generate_exit_signal("ALIMTIAZ", rows, entry_price=100.0, bars_held=10)

    assert signal["action"] == "TRIM"
    assert signal["urgency"] == "MEDIUM"
    assert signal["momentum_exhaustion_score"] < 75
    assert signal["distribution_detected"] is True


def test_runtime_scenario_two_b_true_high_confluence_is_trim_high() -> None:
    rows = _rows(
        count=20,
        start=112.0,
        step=0.2,
        rsi_14=82.0,
        adx_14=20.0,
        macd_hist=0.5,
        atr_14=2.0,
        ema_20=108.0,
        cmf_20=0.05,
    )
    rows[-3]["macd_hist"] = 0.5
    rows[-2]["macd_hist"] = 0.3
    rows[-1] = _row(
        close=116.0,
        open_=117.2,
        high=117.6,
        low=115.4,
        volume=320_000.0,
        atr_14=2.0,
        adx_14=20.0,
        rsi_14=82.0,
        macd_hist=-0.2,
        ema_20=108.4,
        obv=960_000.0,
        cmf_20=-0.11,
    )

    signal = generate_exit_signal("ALIMTIAZ", rows, entry_price=100.0, bars_held=10)

    assert signal["action"] == "TRIM"
    assert signal["urgency"] == "HIGH"
    assert signal["momentum_exhaustion_score"] >= 75
    assert signal["distribution_detected"] is True


def test_macd_one_bar_noise_adds_only_eight_points() -> None:
    rows = _rows(count=20, start=101.0, step=0.0, rsi_14=60.0, ema_20=100.8, atr_14=3.0)
    rows[-3]["macd_hist"] = -0.2
    rows[-2]["macd_hist"] = 0.3
    rows[-1]["macd_hist"] = -0.1
    rows[-1]["rsi_14"] = 60.0

    score, reasons = _compute_momentum_exhaustion(rows, current_price=float(rows[-1]["close"]))

    assert score == 8
    assert any("minor pullback" in reason for reason in reasons)


def test_distribution_threshold_blocks_detection_below_twelve_percent() -> None:
    rows = _rows(count=20, start=103.0, step=0.3)
    for i, row in enumerate(rows):
        row["obv"] = 1_000_000.0 - (i * 20_000.0)
        row["cmf_20"] = -0.12
    rows[-1]["close"] = 109.0
    rows[-1]["open"] = 111.0
    rows[-1]["high"] = 111.2
    rows[-1]["low"] = 108.4

    signal = generate_exit_signal("ZAIN", rows, entry_price=100.0, bars_held=8)

    assert signal["pnl_pct"] < 12.0
    assert signal["distribution_detected"] is False


def test_time_stop_triggers_medium_trim_with_fifty_percent_suggestion() -> None:
    rows = _rows(count=20, start=100.5, step=0.08, rsi_14=56.0, atr_14=1.2, ema_20=100.9)
    rows[-1]["close"] = 102.0
    rows[-1]["low"] = 101.4

    signal = generate_exit_signal("KFH", rows, entry_price=100.0, bars_held=22)

    assert signal["action"] == "TRIM"
    assert signal["urgency"] == "MEDIUM"
    assert signal["suggested_trim_pct"] == 50


def test_conflict_resolution_holds_when_exhaustion_high_but_below_trim_floor() -> None:
    rows = _rows(count=19, start=106.0, step=0.25, rsi_14=82.0, adx_14=20.0, atr_14=1.0)
    rows[-2]["macd_hist"] = 0.3
    rows[-1]["macd_hist"] = -0.1
    rows[-3]["macd_hist"] = 0.2
    rows[-1]["ema_20"] = rows[-1]["close"] - 2.6
    rows[-1]["low"] = rows[-1]["close"] - 0.2
    rows[-1]["cmf_20"] = 0.2

    signal = generate_exit_signal("MABANEE", rows, entry_price=100.0, bars_held=5)

    assert signal["momentum_exhaustion_score"] == 70
    assert signal["distribution_detected"] is False
    assert signal["action"] == "HOLD"


def test_conflict_resolution_promotes_trim_at_exhaustion_floor() -> None:
    rows = _rows(count=19, start=106.0, step=0.25, rsi_14=82.0, adx_14=20.0, atr_14=1.0)
    rows[-2]["macd_hist"] = 0.3
    rows[-1]["macd_hist"] = -0.1
    rows[-3]["macd_hist"] = 0.2
    rows[-1]["ema_20"] = rows[-1]["close"] - 3.8
    rows[-1]["low"] = rows[-1]["close"] - 0.2
    rows[-1]["cmf_20"] = 0.2

    signal = generate_exit_signal("MABANEE", rows, entry_price=100.0, bars_held=5)

    assert signal["momentum_exhaustion_score"] >= 80
    assert signal["distribution_detected"] is False
    assert signal["action"] == "TRIM"


def test_near_circuit_with_large_gain_triggers_high_urgency_trim() -> None:
    rows = _rows(count=20, start=100.0, step=0.1, rsi_14=58.0, atr_14=2.0, cmf_20=0.20)
    rows[-2]["close"] = 100.0
    rows[-1] = _row(
        close=108.8,
        open_=108.2,
        high=109.2,
        low=107.9,
        atr_14=2.0,
        rsi_14=60.0,
        adx_14=24.0,
        macd_hist=0.2,
        ema_20=108.0,
        obv=1_020_000.0,
        cmf_20=0.10,
    )

    signal = generate_exit_signal("AGILITY", rows, entry_price=93.8, bars_held=9)

    assert signal["pnl_pct"] > 12.0
    assert signal["near_circuit"] is True
    assert signal["action"] == "TRIM"
    assert signal["urgency"] == "HIGH"


def test_trailing_stop_is_aligned_to_kuwait_tick_grid() -> None:
    rows = _rows(count=20, start=97.0, step=0.2, atr_14=1.7, rsi_14=57.0)
    signal = generate_exit_signal("BAYAN", rows, entry_price=95.0, bars_held=7)

    assert signal["trailing_stop"] == align_to_tick(signal["trailing_stop"])

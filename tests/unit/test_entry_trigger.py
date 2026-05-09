"""Unit tests for entry-trigger timing logic.

Covers pullback, breakout, accumulation states, and orchestrator actions.
"""
from __future__ import annotations

from app.services.signal_engine.models.technical.entry_trigger import (
    _detect_pullback_trigger,
    evaluate_entry_trigger,
)


def _row(
    close: float,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 100_000.0,
    ema_20: float | None = None,
    stoch_k: float | None = None,
    stoch_d: float | None = None,
    atr_14: float | None = None,
    obv: float | None = None,
    cmf_20: float | None = None,
) -> dict:
    return {
        "date": "2026-05-09",
        "close": close,
        "open": open_ if open_ is not None else close - 1.0,
        "high": high if high is not None else close + 2.0,
        "low": low if low is not None else close - 2.0,
        "volume": volume,
        "ema_20": ema_20,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "atr_14": atr_14,
        "obv": obv,
        "cmf_20": cmf_20,
    }


def _pullback_rows() -> list[dict]:
    return [
        _row(close=100.0, ema_20=99.0, obv=1_000_000, cmf_20=0.00),
        _row(close=101.0, ema_20=99.5, obv=1_005_000, cmf_20=0.01),
        _row(close=102.0, ema_20=100.0, obv=1_010_000, cmf_20=0.01),
        _row(close=101.3, ema_20=100.5, low=100.2, obv=1_015_000, cmf_20=0.01),
        _row(close=100.8, ema_20=100.9, high=101.2, low=100.3, obv=1_020_000, cmf_20=0.01),
        _row(
            close=101.6,
            open_=100.9,
            high=102.0,
            low=100.8,
            ema_20=101.1,
            stoch_k=42.0,
            stoch_d=35.0,
            obv=1_025_000,
            cmf_20=0.02,
            atr_14=4.0,
            volume=120_000,
        ),
    ]


def _breakout_rows() -> list[dict]:
    rows: list[dict] = []
    for _ in range(20):
        rows.append(
            _row(
                close=100.0,
                open_=99.8,
                high=101.0,
                low=99.0,
                volume=50_000,
                atr_14=5.0,
                ema_20=95.0,
                stoch_k=62.0,
                stoch_d=58.0,
                obv=1_000_000,
                cmf_20=0.00,
            )
        )

    rows.append(
        _row(
            close=103.2,
            open_=102.0,
            high=103.5,
            low=101.5,
            volume=150_000,
            atr_14=5.0,
            ema_20=95.2,
            stoch_k=64.0,
            stoch_d=60.0,
            obv=1_001_000,
            cmf_20=0.00,
        )
    )
    return rows


def test_pullback_fires_for_buy_tier() -> None:
    result = evaluate_entry_trigger(_pullback_rows(), "Buy")
    assert result["action"] == "ENTER"
    assert result["trigger"] == "pullback"
    assert result["trigger_strength"] > 0


def test_breakout_fires_for_strong_buy() -> None:
    result = evaluate_entry_trigger(_breakout_rows(), "Strong Buy")
    assert result["action"] == "ENTER"
    assert result["trigger"] == "breakout"


def test_buy_tier_rejects_breakout_only_setup() -> None:
    result = evaluate_entry_trigger(_breakout_rows(), "Buy")
    assert not (result["action"] == "ENTER" and result["trigger"] == "breakout")


def test_watch_when_accumulation_active_no_trigger() -> None:
    rows = [
        _row(close=110.0, ema_20=100.0, obv=1_000_000, cmf_20=0.10),
        _row(close=111.0, ema_20=100.2, obv=1_010_000, cmf_20=0.10),
        _row(close=112.0, ema_20=100.4, obv=1_020_000, cmf_20=0.10),
        _row(close=113.0, ema_20=100.6, obv=1_030_000, cmf_20=0.10),
        _row(close=114.0, ema_20=100.8, obv=1_040_000, cmf_20=0.10),
        _row(
            close=115.0,
            open_=114.0,
            high=115.3,
            low=114.1,
            ema_20=101.0,
            stoch_k=55.0,
            stoch_d=50.0,
            atr_14=4.5,
            volume=90_000,
            obv=1_050_000,
            cmf_20=0.10,
        ),
    ]

    result = evaluate_entry_trigger(rows, "Buy")
    assert result["action"] == "WATCH"
    assert result["trigger"] == "accumulation_only"


def test_hold_when_no_accumulation_no_trigger() -> None:
    rows = [
        _row(close=100.0, ema_20=95.0, obv=1_000_000, cmf_20=-0.05),
        _row(close=100.0, ema_20=95.0, obv=1_000_000, cmf_20=-0.05),
        _row(close=100.0, ema_20=95.0, obv=1_000_000, cmf_20=-0.05),
        _row(close=100.0, ema_20=95.0, obv=1_000_000, cmf_20=-0.05),
        _row(close=100.0, ema_20=95.0, obv=1_000_000, cmf_20=-0.05),
        _row(
            close=100.0,
            open_=100.7,
            high=100.8,
            low=99.7,
            ema_20=95.0,
            stoch_k=60.0,
            stoch_d=58.0,
            atr_14=4.0,
            volume=80_000,
            obv=1_000_000,
            cmf_20=-0.05,
        ),
    ]

    result = evaluate_entry_trigger(rows, "Buy")
    assert result["action"] == "HOLD"
    assert result["trigger"] == "none"


def test_non_buy_tier_passes_through() -> None:
    result = evaluate_entry_trigger(_pullback_rows(), "Sell")
    assert result["action"] == "HOLD"
    assert result["recommended_state"] == "SELL"
    assert result["details"].get("skipped") == "non_buy_tier"


def test_falling_ema_rejects_pullback() -> None:
    rows = [
        _row(close=105.0, ema_20=106.0),
        _row(close=104.0, ema_20=105.5),
        _row(close=103.0, ema_20=105.0),
        _row(close=102.0, ema_20=104.5),
        _row(close=101.5, ema_20=104.0),
        _row(close=102.2, open_=101.8, high=102.4, low=101.4, ema_20=103.5, stoch_k=40.0, stoch_d=35.0),
    ]

    fired, _, details = _detect_pullback_trigger(rows)
    assert fired is False
    assert details.get("fail") == "ema_20_not_rising"

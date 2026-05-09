"""Entry trigger smoke test.

Runs a focused set of behavior checks for pullback/breakout timing actions.
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


def _watch_rows() -> list[dict]:
    return [
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


def _hold_rows() -> list[dict]:
    return [
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


def _falling_ema_rows() -> list[dict]:
    return [
        _row(close=105.0, ema_20=106.0),
        _row(close=104.0, ema_20=105.5),
        _row(close=103.0, ema_20=105.0),
        _row(close=102.0, ema_20=104.5),
        _row(close=101.5, ema_20=104.0),
        _row(close=102.2, open_=101.8, high=102.4, low=101.4, ema_20=103.5, stoch_k=40.0, stoch_d=35.0),
    ]


def _assert(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(f"FAIL: {name}")
    print(f"OK: {name}")


def main() -> None:
    pullback = evaluate_entry_trigger(_pullback_rows(), "Buy")
    _assert("Pullback fires", pullback["action"] == "ENTER" and pullback["trigger"] == "pullback" and pullback["trigger_strength"] > 0)

    breakout = evaluate_entry_trigger(_breakout_rows(), "Strong Buy")
    _assert("Breakout fires", breakout["action"] == "ENTER" and breakout["trigger"] == "breakout")

    breakout_buy = evaluate_entry_trigger(_breakout_rows(), "Buy")
    _assert("Buy tier rejects breakout-only", not (breakout_buy["action"] == "ENTER" and breakout_buy["trigger"] == "breakout"))

    _assert("Strong Buy accepts breakout", breakout["action"] == "ENTER" and breakout["trigger"] == "breakout")

    watch = evaluate_entry_trigger(_watch_rows(), "Buy")
    _assert("WATCH when accumulation active", watch["action"] == "WATCH")

    hold = evaluate_entry_trigger(_hold_rows(), "Buy")
    _assert("HOLD when no accumulation", hold["action"] == "HOLD")

    non_buy = evaluate_entry_trigger(_pullback_rows(), "Sell")
    _assert("Non-buy tier passes unchanged", non_buy["recommended_state"] == "SELL")

    fired, _, details = _detect_pullback_trigger(_falling_ema_rows())
    _assert("Falling EMA rejects pullback", (not fired) and details.get("fail") == "ema_20_not_rising")


if __name__ == "__main__":
    main()

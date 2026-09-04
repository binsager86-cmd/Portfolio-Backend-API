"""
Unit tests for the Trend-Hold Book's gate-snapshot capture (trend_hold_engine)
and post-trade forward-look (trend_hold_lessons).

Regression coverage for a module that had none before: replay_symbol()'s
gate_snapshot/entry_gate fields feed the Lessons Learned report's "what
triggered the buy/sell" detail, and compute_forward_look() answers "did the
stock have more room to run after this system exited it." Both are pure
functions of synthetic OHLCV -- no network calls, no DB.
"""
from __future__ import annotations

import os
import sys

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

import numpy as np
import pandas as pd
import pytest

from app.services.eagle_eye_v2.trend_hold_engine import (
    CHANDELIER_ATR_MULT,
    DONCHIAN_LOOKBACK_SESSIONS,
    compute_daily_features,
    replay_symbol,
)
from app.services.eagle_eye_v2.trend_hold_lessons import analyze_trade, compute_forward_look


def _make_breakout_ohlcv(n: int = 260, seed: int = 7) -> pd.DataFrame:
    """
    A synthetic series engineered to fire a clean Donchian breakout roughly
    two-thirds through the window: flat/choppy base, then a sustained
    up-trend, then a sharp reversal (to also exercise the chandelier stop).
    """
    rng = np.random.default_rng(seed)
    base_n = int(n * 0.55)
    trend_n = int(n * 0.30)
    reversal_n = n - base_n - trend_n

    base = 100 + rng.normal(0, 0.6, base_n).cumsum() * 0.05
    base = np.clip(base, 95, 105)
    trend = base[-1] + np.linspace(0, 40, trend_n) + rng.normal(0, 0.4, trend_n).cumsum() * 0.05
    reversal = trend[-1] - np.linspace(0, 25, reversal_n) + rng.normal(0, 0.3, reversal_n).cumsum() * 0.05
    close = np.concatenate([base, trend, reversal])

    high = close * 1.01
    low = close * 0.99
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(200_000, 500_000, n).astype(float)

    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y-%m-%d"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "value_kwd": close * volume,
        }
    )


@pytest.fixture(scope="module")
def replayed():
    raw = _make_breakout_ohlcv()
    features = compute_daily_features(raw)
    rows = replay_symbol(features)
    return rows


def test_buy_row_has_full_entry_gate_snapshot(replayed):
    buys = [r for r in replayed if r["decision"] == "BUY"]
    assert buys, "synthetic series should fire at least one BUY"

    gate = buys[0]["gate_snapshot"]
    assert gate is not None
    assert gate["entry_path"] in ("DONCHIAN", "EMA_CROSS")
    assert "confidence" in gate and 0.0 <= gate["confidence"] <= 100.0
    assert "confidence_breakdown" in gate
    assert set(gate["confidence_breakdown"]) == {"breakout_score", "volume_score", "flow_score"}
    # Every declared gate input is present (even if null for a given path)
    for key in (
        "donchian_high", "ema10", "ema30", "ema50", "rel_volume", "rel_volume_floor",
        "cmf10", "cmf_floor", "obv_slope40", "flow_pass_via", "adx14", "sma200",
        "sma200_slope", "atr14", "breakout_margin_pct",
    ):
        assert key in gate

    # The BUY row's own entry_gate mirrors gate_snapshot exactly.
    assert buys[0]["entry_gate"] == gate


def test_entry_gate_persists_across_hold_days_until_exit(replayed):
    buy_idx = next(i for i, r in enumerate(replayed) if r["decision"] == "BUY")
    entry_gate = replayed[buy_idx]["entry_gate"]
    assert entry_gate is not None

    # Every HOLD day between entry and exit carries the SAME entry_gate
    # forward (needed so a still-open position can show "what triggered
    # the buy" without waiting for the exit).
    i = buy_idx + 1
    while i < len(replayed) and replayed[i]["position_state"] == "IN_POSITION" and replayed[i]["decision"] == "HOLD":
        assert replayed[i]["entry_gate"] == entry_gate
        i += 1


def test_exit_row_has_exit_gate_with_chandelier_math(replayed):
    exits = [r for r in replayed if r["decision"] == "SELL_SIGNAL"]
    assert exits, "synthetic series' sharp reversal should trigger a chandelier stop"

    exit_gate = exits[0]["gate_snapshot"]
    assert exit_gate is not None
    assert exit_gate["trigger"] == "CHANDELIER_STOP"
    assert exit_gate["structural_stop"] == pytest.approx(exits[0]["structural_stop"])
    assert exit_gate["days_since_peak"] is None or exit_gate["days_since_peak"] >= 0
    # A BUY row's gate_snapshot is its entry_gate; an EXIT row's must NOT be
    # (they're different dicts describing different moments).
    assert exit_gate is not exits[0]["entry_gate"]


def test_donchian_breakout_margin_matches_close_vs_ceiling(replayed):
    donchian_buys = [r for r in replayed if r["decision"] == "BUY" and r["gate_snapshot"]["entry_path"] == "DONCHIAN"]
    if not donchian_buys:
        pytest.skip("this synthetic seed didn't fire via the Donchian path")
    row = donchian_buys[0]
    gate = row["gate_snapshot"]
    expected_margin = (row["close"] - gate["donchian_high"]) / gate["donchian_high"] * 100.0
    assert gate["breakout_margin_pct"] == pytest.approx(expected_margin, abs=0.01)


# ---------------------------------------------------------------------------
# compute_forward_look
# ---------------------------------------------------------------------------

def _make_flat_then_rally_ohlcv(n: int = 40) -> pd.DataFrame:
    """First half flat at 100, second half rallies 10% -- lets a test assert
    an exact, known forward return regardless of exactly which index it lands on."""
    close = np.concatenate([np.full(n // 2, 100.0), np.linspace(100.0, 110.0, n - n // 2)])
    dates = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    return df


def test_forward_look_unavailable_before_five_sessions_exist():
    ohlcv = _make_flat_then_rally_ohlcv(n=10)
    exit_date = ohlcv.index[6].strftime("%Y-%m-%d")  # only 3 sessions remain after this
    result = compute_forward_look("TEST", exit_date, exit_price=100.0, ohlcv=ohlcv, sessions=5)
    assert result is not None
    assert result["available"] is False
    assert result["sessions_available"] == 3


def test_forward_look_available_reports_return_and_peak():
    ohlcv = _make_flat_then_rally_ohlcv(n=40)
    exit_date = ohlcv.index[10].strftime("%Y-%m-%d")
    exit_price = float(ohlcv.loc[ohlcv.index == pd.Timestamp(exit_date), "close"].iloc[0])
    result = compute_forward_look("TEST", exit_date, exit_price, ohlcv, sessions=5)
    assert result is not None
    assert result["available"] is True
    assert result["sessions_available"] >= 5
    # Flat region immediately after this exit_date -> ~0% return at the 1wk mark
    assert result["return_1w_pct"] == pytest.approx(0.0, abs=0.5)
    # But the extended window reaches into the rally -> materially positive peak
    assert result["peak_20d_pct"] > 5.0


def test_forward_look_none_for_missing_inputs():
    ohlcv = _make_flat_then_rally_ohlcv(n=40)
    assert compute_forward_look("TEST", None, 100.0, ohlcv) is None
    assert compute_forward_look("TEST", "2024-01-01", 0.0, ohlcv) is None
    assert compute_forward_look("TEST", "2024-01-01", 100.0, None) is None


# ---------------------------------------------------------------------------
# analyze_trade's entry_gate-aware QUICK_STOP enhancement text
# ---------------------------------------------------------------------------

def _quick_stop_ohlcv() -> pd.DataFrame:
    """Entry, one session held, then a sharp drop -- classifies as QUICK_STOP."""
    dates = pd.bdate_range("2024-01-01", periods=5)
    close = [100.0, 100.0, 85.0, 84.0, 83.0]
    df = pd.DataFrame(
        {"open": close, "high": [c * 1.01 for c in close], "low": [c * 0.99 for c in close], "close": close, "volume": 1.0},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    return df


def test_quick_stop_enhancement_cites_gate_numbers_when_available():
    ohlcv = _quick_stop_ohlcv()
    entry_gate = {
        "rel_volume": 3.5, "rel_volume_floor": 0.8,
        "cmf10": 0.2, "cmf_floor": -0.05,
        "flow_pass_via": "CMF",
    }
    lesson = analyze_trade(
        side="EXIT", entry_date="2024-01-01", entry_price=100.0,
        exit_date="2024-01-03", exit_price=85.0, ohlcv=ohlcv, entry_gate=entry_gate,
    )
    assert lesson.classification == "QUICK_STOP"
    assert "3.50" in lesson.enhancement  # cites the actual rel-volume value
    assert "well clear" in lesson.enhancement  # margin is large, not "barely cleared"


def test_quick_stop_enhancement_falls_back_without_gate():
    ohlcv = _quick_stop_ohlcv()
    lesson = analyze_trade(
        side="EXIT", entry_date="2024-01-01", entry_price=100.0,
        exit_date="2024-01-03", exit_price=85.0, ohlcv=ohlcv, entry_gate=None,
    )
    assert lesson.classification == "QUICK_STOP"
    assert "MIN_REL_VOLUME" in lesson.enhancement

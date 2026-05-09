"""Unit tests for the entry_trigger module.

Covers all three detectors (pullback, breakout, accumulation) and the
evaluate_entry_trigger orchestrator with 8 scenarios.
"""
from __future__ import annotations

import pytest

from app.services.signal_engine.models.technical.entry_trigger import (
    _detect_accumulation,
    _detect_breakout_trigger,
    _detect_pullback_trigger,
    evaluate_entry_trigger,
)


def _make_row(
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
        "date": "2025-05-01",
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


# ── Pullback detector tests ─────────────────────────────────────────────

class TestPullbackTrigger:
    def test_pullback_confirmed(self):
        """All conditions met: EMA rising, close near EMA, bullish candle, stoch recovering."""
        rows = [
            _make_row(close=100.0, ema_20=99.0),
            _make_row(close=101.0, ema_20=99.5),
            _make_row(close=102.0, ema_20=100.0),
            _make_row(close=101.5, ema_20=100.5),
            _make_row(close=100.8, ema_20=100.8),
            # Last candle: close > open, near EMA, stoch recovering
            _make_row(close=101.0, open_=100.5, ema_20=101.0, stoch_k=40.0, stoch_d=35.0),
        ]
        result = _detect_pullback_trigger(rows)
        assert result["triggered"] is True
        assert result["reason"] == "pullback_confirmed"

    def test_pullback_ema_not_rising(self):
        """EMA is falling — no pullback."""
        rows = [
            _make_row(close=102.0, ema_20=103.0),
            _make_row(close=101.0, ema_20=102.5),
            _make_row(close=100.0, ema_20=102.0),
            _make_row(close=99.5, ema_20=101.5),
            _make_row(close=99.0, ema_20=101.0),
            _make_row(close=100.5, open_=100.0, ema_20=100.5, stoch_k=35.0, stoch_d=30.0),
        ]
        result = _detect_pullback_trigger(rows)
        assert result["triggered"] is False
        assert "ema_not_rising" in result["reason"]

    def test_pullback_bearish_candle(self):
        """Candle is bearish (close < open) — no pullback."""
        rows = [
            _make_row(close=100.0, ema_20=99.0),
            _make_row(close=101.0, ema_20=99.5),
            _make_row(close=102.0, ema_20=100.0),
            _make_row(close=101.5, ema_20=100.5),
            _make_row(close=100.8, ema_20=100.8),
            # close < open = bearish
            _make_row(close=100.5, open_=101.5, ema_20=101.0, stoch_k=40.0, stoch_d=35.0),
        ]
        result = _detect_pullback_trigger(rows)
        assert result["triggered"] is False
        assert "bearish_candle" in result["reason"]

    def test_pullback_insufficient_data(self):
        rows = [_make_row(close=100.0)]
        result = _detect_pullback_trigger(rows)
        assert result["triggered"] is False
        assert "insufficient_data" in result["reason"]


# ── Breakout detector tests ──────────────────────────────────────────────

class TestBreakoutTrigger:
    def test_breakout_confirmed(self):
        """Tight range, close above range high, volume expansion."""
        # Build a consolidation range of 8 bars + volume history of 20 bars
        base_rows = [_make_row(close=100.0, high=101.0, low=99.0, volume=50_000, atr_14=5.0)
                      for _ in range(12)]
        # 8 tight range bars (range = 101 - 99 = 2, ATR*1.8 = 9 → passes)
        range_bars = [_make_row(close=100.0, high=101.0, low=99.0, volume=50_000, atr_14=5.0)
                      for _ in range(8)]
        # Breakout bar: close above 101 range high, volume spike
        breakout = _make_row(close=103.0, high=103.5, low=100.5, volume=150_000, atr_14=5.0)
        rows = base_rows + range_bars + [breakout]
        result = _detect_breakout_trigger(rows)
        assert result["triggered"] is True
        assert "breakout_confirmed" in result["reason"]

    def test_breakout_range_too_wide(self):
        """Range is wider than ATR*1.8 — not tight enough."""
        base_rows = [_make_row(close=100.0, high=110.0, low=90.0, volume=50_000, atr_14=5.0)
                      for _ in range(12)]
        # Wide range bars (range = 110 - 90 = 20, ATR*1.8 = 9 → fails)
        range_bars = [_make_row(close=100.0, high=110.0, low=90.0, volume=50_000, atr_14=5.0)
                      for _ in range(8)]
        breakout = _make_row(close=112.0, high=113.0, low=100.0, volume=150_000, atr_14=5.0)
        rows = base_rows + range_bars + [breakout]
        result = _detect_breakout_trigger(rows)
        assert result["triggered"] is False
        assert "range_not_tight" in result["reason"]

    def test_breakout_no_volume(self):
        """Tight range and breakout but volume is not expanded enough."""
        base_rows = [_make_row(close=100.0, high=101.0, low=99.0, volume=100_000, atr_14=5.0)
                      for _ in range(12)]
        range_bars = [_make_row(close=100.0, high=101.0, low=99.0, volume=100_000, atr_14=5.0)
                      for _ in range(8)]
        # Volume is same as average — not a spike
        breakout = _make_row(close=103.0, high=103.5, low=100.5, volume=100_000, atr_14=5.0)
        rows = base_rows + range_bars + [breakout]
        result = _detect_breakout_trigger(rows)
        assert result["triggered"] is False
        assert "volume_expansion_weak" in result["reason"]


# ── Accumulation detector tests ──────────────────────────────────────────

class TestAccumulation:
    def test_active_accumulation(self):
        """Both OBV slope and CMF above thresholds → active."""
        rows = [
            _make_row(close=100.0, obv=1_000_000, cmf_20=0.10),
            _make_row(close=101.0, obv=1_010_000, cmf_20=0.10),
            _make_row(close=102.0, obv=1_020_000, cmf_20=0.10),
            _make_row(close=103.0, obv=1_030_000, cmf_20=0.10),
            _make_row(close=104.0, obv=1_040_000, cmf_20=0.10),
            _make_row(close=105.0, obv=1_050_000, cmf_20=0.10),
        ]
        result = _detect_accumulation(rows)
        assert result["state"] == "active"
        assert result["obv_slope_pct"] > 0.3
        assert result["cmf"] >= 0.05

    def test_absent_accumulation(self):
        """OBV flat and CMF negative → absent."""
        rows = [
            _make_row(close=100.0, obv=1_000_000, cmf_20=-0.05),
            _make_row(close=100.0, obv=1_000_000, cmf_20=-0.05),
            _make_row(close=100.0, obv=1_000_000, cmf_20=-0.05),
            _make_row(close=100.0, obv=1_000_000, cmf_20=-0.05),
            _make_row(close=100.0, obv=1_000_000, cmf_20=-0.05),
            _make_row(close=100.0, obv=1_000_000, cmf_20=-0.05),
        ]
        result = _detect_accumulation(rows)
        assert result["state"] == "absent"


# ── Orchestrator tests ───────────────────────────────────────────────────

class TestEvaluateEntryTrigger:
    def test_pullback_enter(self):
        """Pullback detected → action ENTER."""
        rows = [
            _make_row(close=100.0, ema_20=99.0, obv=1_000_000, cmf_20=0.0),
            _make_row(close=101.0, ema_20=99.5, obv=1_000_000, cmf_20=0.0),
            _make_row(close=102.0, ema_20=100.0, obv=1_000_000, cmf_20=0.0),
            _make_row(close=101.5, ema_20=100.5, obv=1_000_000, cmf_20=0.0),
            _make_row(close=100.8, ema_20=100.8, obv=1_000_000, cmf_20=0.0),
            _make_row(close=101.0, open_=100.5, ema_20=101.0,
                      stoch_k=40.0, stoch_d=35.0, obv=1_000_000, cmf_20=0.0),
        ]
        result = evaluate_entry_trigger(rows, "Buy")
        assert result["action"] == "ENTER"
        assert result["trigger"] == "pullback"

    def test_breakout_only_for_strong_buy(self):
        """Breakout with tier=Buy → not allowed; tier=Strong Buy → allowed."""
        base_rows = [_make_row(close=100.0, high=101.0, low=99.0, volume=50_000,
                                atr_14=5.0, ema_20=90.0, obv=1_000_000, cmf_20=0.0)
                      for _ in range(20)]
        breakout = _make_row(close=103.0, high=103.5, low=100.5, volume=150_000,
                              atr_14=5.0, ema_20=90.0,
                              stoch_k=60.0, stoch_d=55.0, obv=1_000_000, cmf_20=0.0)
        rows = base_rows + [breakout]

        result_buy = evaluate_entry_trigger(rows, "Buy")
        assert result_buy["action"] != "ENTER" or result_buy["trigger"] != "breakout"

        result_strong = evaluate_entry_trigger(rows, "Strong Buy")
        assert result_strong["action"] == "ENTER"
        assert result_strong["trigger"] == "breakout"

    def test_accumulation_watch(self):
        """No pullback or breakout, but accumulation active → WATCH."""
        rows = [
            _make_row(close=100.0, ema_20=95.0, obv=1_000_000, cmf_20=0.10),
            _make_row(close=101.0, ema_20=95.5, obv=1_010_000, cmf_20=0.10),
            _make_row(close=102.0, ema_20=96.0, obv=1_020_000, cmf_20=0.10),
            _make_row(close=103.0, ema_20=96.5, obv=1_030_000, cmf_20=0.10),
            _make_row(close=104.0, ema_20=97.0, obv=1_040_000, cmf_20=0.10),
            # Not near EMA → no pullback; no breakout setup either
            _make_row(close=105.0, open_=104.0, ema_20=97.5,
                      stoch_k=40.0, stoch_d=35.0, obv=1_050_000, cmf_20=0.10),
        ]
        result = evaluate_entry_trigger(rows, "Buy")
        assert result["action"] == "WATCH"
        assert result["trigger"] == "accumulation_only"

    def test_hold_nothing_triggered(self):
        """No detectors fire → HOLD."""
        rows = [
            _make_row(close=100.0, ema_20=95.0, obv=1_000_000, cmf_20=-0.05),
            _make_row(close=100.0, ema_20=95.0, obv=1_000_000, cmf_20=-0.05),
            _make_row(close=100.0, ema_20=95.0, obv=1_000_000, cmf_20=-0.05),
            _make_row(close=100.0, ema_20=95.0, obv=1_000_000, cmf_20=-0.05),
            _make_row(close=100.0, ema_20=95.0, obv=1_000_000, cmf_20=-0.05),
            _make_row(close=100.0, open_=100.5, ema_20=95.0,
                      stoch_k=60.0, stoch_d=55.0, obv=1_000_000, cmf_20=-0.05),
        ]
        result = evaluate_entry_trigger(rows, "Buy")
        assert result["action"] == "HOLD"
        assert result["trigger"] == "none"

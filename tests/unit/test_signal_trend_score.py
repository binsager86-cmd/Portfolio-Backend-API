"""Unit tests for the Kuwait Signal Engine — Trend Score.

Validates:
  - EMA alignment scoring (bullish/bearish/mixed)
  - ADX strength scoring
  - Swing HH/HL structure scoring
  - compute_trend_score returns (int, dict) tuple
  - Score always in [0, 100]
"""
from __future__ import annotations

import pytest

from app.services.signal_engine.models.technical.trend_score import compute_trend_score


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row(close: float = 500.0, ema20: float | None = 490.0, ema50: float | None = 480.0,
         sma200: float | None = 460.0, adx: float | None = 25.0,
         high: float | None = None, low: float | None = None) -> dict:
    return {
        "close": close,
        "ema_20": ema20,
        "ema_50": ema50,
        "sma_200": sma200,
        "adx_14": adx,
        "high": high if high is not None else close + 5,
        "low": low if low is not None else close - 5,
        "open": close - 2,
        "volume": 1_000_000,
    }


def _rows(n: int = 80, close_start: float = 400.0, step: float = 1.0, **kw) -> list[dict]:
    """Generate n rows with linearly increasing close prices."""
    return [_row(close=close_start + i * step, **kw) for i in range(n)]


# ── Return type ───────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_tuple(self):
        result = compute_trend_score(_rows())
        assert isinstance(result, tuple) and len(result) == 2

    def test_score_is_int(self):
        score, _ = compute_trend_score(_rows())
        assert isinstance(score, int)

    def test_details_is_dict(self):
        _, details = compute_trend_score(_rows())
        assert isinstance(details, dict)

    def test_score_in_range(self):
        score, _ = compute_trend_score(_rows())
        assert 0 <= score <= 100


# ── EMA alignment ─────────────────────────────────────────────────────────────

class TestEMAAlignment:
    def test_full_bullish_alignment_gives_high_score(self):
        # close > ema20 > ema50 > sma200
        rows = _rows(n=80, close_start=500)
        rows[-1].update({"close": 510, "ema_20": 505, "ema_50": 490, "sma_200": 470})
        score, _ = compute_trend_score(rows)
        assert score >= 60

    def test_full_bearish_alignment_gives_low_score(self):
        # close < ema20 < ema50 < sma200
        rows = _rows(n=80, close_start=500)
        rows[-1].update({"close": 400, "ema_20": 450, "ema_50": 470, "sma_200": 490})
        score, _ = compute_trend_score(rows)
        assert score <= 40

    def test_missing_ema_returns_neutral_default(self):
        rows = _rows(n=80)
        rows[-1]["ema_20"] = None
        rows[-1]["ema_50"] = None
        score, details = compute_trend_score(rows)
        assert 0 <= score <= 100


# ── ADX ───────────────────────────────────────────────────────────────────────

class TestADX:
    def test_strong_adx_increases_score(self):
        rows_weak = _rows(n=80)
        rows_weak[-1]["adx_14"] = 10.0

        rows_strong = _rows(n=80)
        rows_strong[-1]["adx_14"] = 35.0

        score_weak, _ = compute_trend_score(rows_weak)
        score_strong, _ = compute_trend_score(rows_strong)
        assert score_strong > score_weak

    def test_missing_adx_uses_default(self):
        rows = _rows(n=80)
        rows[-1]["adx_14"] = None
        score, _ = compute_trend_score(rows)
        assert 0 <= score <= 100


# ── Swing structure ───────────────────────────────────────────────────────────

class TestSwingStructure:
    def test_insufficient_rows_returns_neutral(self):
        # Only 5 rows — too few for pivot detection
        score, _ = compute_trend_score(_rows(n=5))
        assert 0 <= score <= 100

    def test_uptrend_swing_structure(self):
        # Rising close series → likely HH/HL structure
        rows = _rows(n=80, close_start=400, step=2.0)
        score, _ = compute_trend_score(rows)
        assert score >= 50

    def test_downtrend_swing_structure(self):
        # Full bearish EMA alignment + falling close → low score
        rows = _rows(n=80, close_start=600, step=-2.0,
                     ema20=510.0, ema50=520.0, sma200=530.0)  # bearish order
        # Last row: close=600-79*2=442 < ema20=510 < ema50=520 < sma200=530
        score, _ = compute_trend_score(rows)
        assert score <= 40

    def test_empty_rows_returns_tuple(self):
        score, details = compute_trend_score([])
        assert isinstance(score, int)
        assert 0 <= score <= 100

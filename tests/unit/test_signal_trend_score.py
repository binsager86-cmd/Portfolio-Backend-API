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

from app.services.signal_engine.models.technical.trend_score import (
    _bars_since_ema20_50_cross,
    _kaufman_er,
    compute_trend_score,
)


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
        # Note: age_mult (0.65 floor for old trends) reduces the raw 94 → ~56.
        # Score >= 50 confirms bullish > neutral (≈40) > bearish (≈7).
        rows = _rows(n=80, close_start=500)
        rows[-1].update({"close": 510, "ema_20": 505, "ema_50": 490, "sma_200": 470})
        score, _ = compute_trend_score(rows)
        assert score >= 50

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


# ── Kaufman ER ────────────────────────────────────────────────────────────────

class TestKaufmanER:
    def test_perfectly_trending_returns_one(self):
        # All bars move the same direction → ER = 1.0
        rows = [{"close": float(i)} for i in range(1, 16)]
        assert _kaufman_er(rows) == pytest.approx(1.0)

    def test_flat_returns_zero(self):
        # All closes identical → noise = 0 denominator → returns 0.5 (neutral)
        rows = [{"close": 100.0}] * 15
        er = _kaufman_er(rows)
        assert er == pytest.approx(0.5)

    def test_insufficient_data_returns_neutral(self):
        rows = [{"close": float(i)} for i in range(5)]
        assert _kaufman_er(rows, period=14) == pytest.approx(0.5)

    def test_noisy_series_below_clean(self):
        # Zig-zag (high noise, low net direction) vs straight trend
        zigzag = [{"close": 100.0 + (5.0 if i % 2 == 0 else -4.5)} for i in range(15)]
        straight = [{"close": 100.0 + i * 0.5} for i in range(15)]
        assert _kaufman_er(zigzag) < _kaufman_er(straight)


# ── Trend age ────────────────────────────────────────────────────────────────

class TestTrendAge:
    def _cross_rows(self, bars_before_cross: int = 5) -> list[dict]:
        """Build rows where EMA20 crosses above EMA50 at position -bars_before_cross."""
        rows: list[dict] = []
        total = 40
        cross_at = total - bars_before_cross   # index where cross happens
        for i in range(total):
            if i < cross_at:
                # Bearish: ema20 < ema50
                rows.append({"close": 100.0, "ema_20": 95.0, "ema_50": 100.0,
                              "sma_200": 105.0, "adx_14": 28.0,
                              "high": 102.0, "low": 98.0, "atr_14": 4.0})
            else:
                # Bullish: ema20 > ema50
                rows.append({"close": 110.0, "ema_20": 108.0, "ema_50": 100.0,
                              "sma_200": 95.0, "adx_14": 28.0,
                              "high": 112.0, "low": 108.0, "atr_14": 4.0})
        return rows

    def test_fresh_cross_detected_as_zero_bars(self):
        rows = self._cross_rows(bars_before_cross=1)
        assert _bars_since_ema20_50_cross(rows) == 0

    def test_old_cross_returns_correct_age(self):
        rows = self._cross_rows(bars_before_cross=10)
        assert _bars_since_ema20_50_cross(rows) == 9

    def test_no_bullish_cross_returns_n(self):
        # Bearish throughout
        rows = [{"close": 100.0, "ema_20": 95.0, "ema_50": 100.0} for _ in range(20)]
        assert _bars_since_ema20_50_cross(rows) == len(rows)

    def test_fresh_cross_scores_higher_than_old(self):
        fresh = self._cross_rows(bars_before_cross=1)
        old   = self._cross_rows(bars_before_cross=30)
        score_fresh, _ = compute_trend_score(fresh)
        score_old,   _ = compute_trend_score(old)
        assert score_fresh > score_old

    def test_modifier_keys_in_details(self):
        rows = self._cross_rows(bars_before_cross=5)
        _, d = compute_trend_score(rows)
        for key in ("er_value", "er_mult", "age_mult", "bars_since_ema_cross",
                    "stretch_mult", "sector_mult", "combined_mult", "base_raw"):
            assert key in d, f"Missing key: {key}"


# ── EMA stretch guard ────────────────────────────────────────────────────────

class TestStretchGuard:
    def _row_with_stretch(self, stretch_atr: float) -> dict:
        atr = 10.0
        ema20 = 500.0
        close = ema20 + stretch_atr * atr  # price above ema20
        return {"close": close, "ema_20": ema20, "ema_50": ema20 * 0.96,
                "sma_200": ema20 * 0.92, "adx_14": 28.0,
                "high": close + 5, "low": close - 5, "atr_14": atr}

    def test_within_bounds_no_penalty(self):
        rows = _rows(n=40) + [self._row_with_stretch(1.0)]
        _, d = compute_trend_score(rows)
        assert d["stretch_mult"] == 1.0

    def test_moderate_extension_penalty(self):
        rows = _rows(n=40) + [self._row_with_stretch(2.0)]
        _, d = compute_trend_score(rows)
        assert d["stretch_mult"] == 0.75

    def test_severe_extension_penalty(self):
        rows = _rows(n=40) + [self._row_with_stretch(2.5)]
        _, d = compute_trend_score(rows)
        assert d["stretch_mult"] == 0.45

    def test_stretch_reduces_final_score(self):
        rows_tight = _rows(n=40) + [self._row_with_stretch(0.5)]
        rows_stretched = _rows(n=40) + [self._row_with_stretch(2.5)]
        score_tight, _ = compute_trend_score(rows_tight)
        score_stretched, _ = compute_trend_score(rows_stretched)
        assert score_tight > score_stretched

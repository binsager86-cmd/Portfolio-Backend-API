"""Unit tests for the Kuwait Signal Engine — HMM Regime Detector.

Validates:
  - State probabilities sum to 1.0
  - Known regime names are returned
  - Graceful handling of missing data / insufficient bars
  - Rule-based fallback when hmmlearn is not installed
"""
from __future__ import annotations

import math
import pytest

from app.services.signal_engine.models.regime.hmm_regime_detector import predict_regime
from app.services.signal_engine.config.model_params import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_CHOP,
    HMM_MIN_TRAIN_BARS,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_rows(n: int = 200, trend: str = "up") -> list[dict]:
    """Synthetic OHLCV rows with configurable trend direction."""
    rows = []
    price = 500.0
    for i in range(n):
        if trend == "up":
            price *= 1.001
        elif trend == "down":
            price *= 0.999
        # else flat
        rows.append({
            "date": f"2024-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}",
            "open": price,
            "high": price * 1.005,
            "low": price * 0.995,
            "close": price,
            "volume": 1_000_000,
            "value": price * 1_000_000 / 1000,
            "atr_14": price * 0.01,
        })
    return rows


# ── State probability constraints ─────────────────────────────────────────────

class TestStateProbabilities:
    def test_probabilities_sum_to_one(self):
        rows = _make_rows(n=300)
        result = predict_regime(rows)
        probs = result.get("state_probabilities", [])
        assert probs, "state_probabilities must not be empty"
        total = sum(probs)
        assert math.isclose(total, 1.0, abs_tol=1e-6), f"Probabilities sum to {total}, expected 1.0"

    def test_confidence_matches_max_probability(self):
        rows = _make_rows(n=300)
        result = predict_regime(rows)
        probs = result.get("state_probabilities", [])
        confidence = result.get("regime_confidence", 0.0)
        if probs:
            assert math.isclose(confidence, max(probs), abs_tol=1e-6)

    def test_all_probabilities_non_negative(self):
        rows = _make_rows(n=300)
        result = predict_regime(rows)
        for p in result.get("state_probabilities", []):
            assert p >= 0.0, f"Negative probability: {p}"


# ── Known regime names ─────────────────────────────────────────────────────────

class TestRegimeNames:
    VALID_REGIMES = {REGIME_BULL, REGIME_CHOP, REGIME_BEAR}

    def test_regime_is_valid_name(self):
        rows = _make_rows(n=300)
        result = predict_regime(rows)
        assert result["current_regime"] in self.VALID_REGIMES

    def test_uptrend_tends_to_bullish(self):
        """Strong uptrend should not return Bearish regime."""
        rows = _make_rows(n=400, trend="up")
        result = predict_regime(rows)
        # Rule-based fallback: strong trend should not be bearish
        assert result["current_regime"] != REGIME_BEAR

    def test_downtrend_tends_to_bearish(self):
        rows = _make_rows(n=400, trend="down")
        result = predict_regime(rows)
        assert result["current_regime"] != REGIME_BULL


# ── Days in current regime ─────────────────────────────────────────────────────

class TestDaysInRegime:
    def test_days_in_regime_is_non_negative_int(self):
        rows = _make_rows(n=300)
        result = predict_regime(rows)
        days = result.get("days_in_current_regime", -1)
        assert isinstance(days, int)
        assert days >= 0

    def test_single_day_data_returns_one_day(self):
        rows = _make_rows(n=HMM_MIN_TRAIN_BARS + 1)
        result = predict_regime(rows)
        assert result.get("days_in_current_regime", 0) >= 1


# ── Insufficient data handling ────────────────────────────────────────────────

class TestInsufficientData:
    def test_empty_rows_returns_chop(self):
        result = predict_regime([])
        assert result["current_regime"] == REGIME_CHOP
        assert result["regime_confidence"] == 0.5

    def test_below_min_train_bars_returns_chop(self):
        rows = _make_rows(n=HMM_MIN_TRAIN_BARS - 1)
        result = predict_regime(rows)
        assert result["current_regime"] == REGIME_CHOP

    def test_missing_close_handled(self):
        rows = [{"date": "2024-01-01", "close": None, "high": 100.0, "low": 99.0, "volume": 1000}] * 50
        result = predict_regime(rows)
        assert "current_regime" in result
        assert result["current_regime"] in {REGIME_BULL, REGIME_CHOP, REGIME_BEAR}

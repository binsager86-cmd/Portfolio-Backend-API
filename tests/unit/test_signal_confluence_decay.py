"""Unit tests for the Kuwait Signal Engine — Confluence Decay.

Validates:
  - Decay schedule applied at T+0, T+24, T+48, T+72 hours
  - Factor is non-increasing as delay increases
  - adjust_confidence_for_delay multiplies probability dict keys correctly
  - At T+72+ the signal is invalidated (factor = 0.0)
"""
from __future__ import annotations

import pytest

from app.services.signal_engine.models.risk.confluence_decay import (
    get_decay_factor,
    adjust_confidence_for_delay,
)
from app.services.signal_engine.config.risk_config import DECAY_SCHEDULE


# ── Decay schedule boundary values ────────────────────────────────────────────

class TestDecayFactor:
    def test_t_plus_0_is_1(self):
        assert get_decay_factor(0) == pytest.approx(1.0, abs=0.001)

    def test_t_plus_24_matches_schedule(self):
        assert get_decay_factor(24) == pytest.approx(DECAY_SCHEDULE[24], rel=0.01)

    def test_t_plus_48_matches_schedule(self):
        assert get_decay_factor(48) == pytest.approx(DECAY_SCHEDULE[48], rel=0.01)

    def test_t_plus_72_invalidated(self):
        """At T+72 the signal is fully invalidated — multiplier = 0.0."""
        assert get_decay_factor(72) == pytest.approx(0.0, abs=0.001)

    def test_factor_is_non_increasing(self):
        """Decay factor must monotonically decrease (or stay equal) as delay increases."""
        delays = [0, 6, 12, 24, 36, 48, 60, 71]
        factors = [get_decay_factor(d) for d in delays]
        for i in range(len(factors) - 1):
            assert factors[i] >= factors[i + 1] - 1e-9, (
                f"Non-monotone decay: delay {delays[i]}h → {factors[i]:.4f} "
                f"but delay {delays[i+1]}h → {factors[i+1]:.4f}"
            )

    def test_beyond_schedule_is_zero(self):
        """Any delay beyond the max schedule key returns 0.0 (invalidated)."""
        assert get_decay_factor(9999) == pytest.approx(0.0)

    def test_interpolation_at_12h(self):
        """12h should sit between T+0 and T+24 factor values."""
        f0, f12, f24 = get_decay_factor(0), get_decay_factor(12), get_decay_factor(24)
        assert f24 <= f12 <= f0


# ── adjust_confidence_for_delay ────────────────────────────────────────────────

class TestAdjustConfidence:
    def _probs(self, p1: float = 0.80, p2: float = 0.55) -> dict:
        return {
            "p_tp1_before_sl": p1,
            "p_tp2_before_sl": p2,
            "calibration_method": "lookup_table",
        }

    def test_immediate_signal_unchanged(self):
        result = adjust_confidence_for_delay(self._probs(0.80), 0)
        assert result["p_tp1_before_sl"] == pytest.approx(0.80, abs=0.001)

    def test_adds_decay_factor_key(self):
        result = adjust_confidence_for_delay(self._probs(), 0)
        assert "decay_factor" in result

    def test_24h_reduces_probability(self):
        base_p = 0.80
        result = adjust_confidence_for_delay(self._probs(base_p), 24)
        expected = round(base_p * DECAY_SCHEDULE[24], 3)
        assert result["p_tp1_before_sl"] == pytest.approx(expected, abs=0.002)

    def test_72h_gives_zero_probability(self):
        result = adjust_confidence_for_delay(self._probs(0.85), 72)
        assert result["p_tp1_before_sl"] == pytest.approx(0.0, abs=0.001)

    def test_48h_lower_than_24h(self):
        base = self._probs(0.80)
        adj_24 = adjust_confidence_for_delay(base, 24)
        adj_48 = adjust_confidence_for_delay(base, 48)
        assert adj_48["p_tp1_before_sl"] <= adj_24["p_tp1_before_sl"]

    def test_does_not_mutate_input(self):
        probs = self._probs(0.75)
        original_p = probs["p_tp1_before_sl"]
        adjust_confidence_for_delay(probs, 24)
        assert probs["p_tp1_before_sl"] == original_p

    def test_hours_since_generation_added(self):
        result = adjust_confidence_for_delay(self._probs(), 24)
        assert result["hours_since_generation"] == 24

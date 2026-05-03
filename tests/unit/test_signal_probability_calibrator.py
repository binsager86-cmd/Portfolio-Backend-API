"""Unit tests for the Kuwait Signal Engine — Probability Calibrator.

Validates:
  - Isotonic regression monotonicity constraint
  - Bayesian update moves estimate toward new evidence
  - Confidence intervals are valid [0, 1] and lo ≤ hi
  - Calibration method label is included in output
  - Lookup-table fallback produces monotone estimates
"""
from __future__ import annotations

import pytest

from app.services.signal_engine.engine.probability_calibrator import calibrate_probabilities
from app.services.signal_engine.config.model_params import REGIME_BULL, REGIME_CHOP, REGIME_BEAR
from app.services.signal_engine.config.risk_config import SCORE_TO_WIN_RATE


# ── Monotonicity ───────────────────────────────────────────────────────────────

class TestMonotonicity:
    def test_higher_score_gives_higher_or_equal_tp1_prob(self):
        """Calibration must be monotone: higher score → higher win probability."""
        scores = [60, 65, 70, 75, 80, 85, 90, 95]
        probs = []
        for s in scores:
            out = calibrate_probabilities(s, REGIME_BULL)
            probs.append(out["p_tp1_before_sl"])
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1] + 1e-9, (
                f"Monotonicity violated: score {scores[i]} → p={probs[i]:.4f} "
                f"but score {scores[i+1]} → p={probs[i+1]:.4f}"
            )

    def test_tp2_prob_lte_tp1_prob(self):
        """P(TP2) must always ≤ P(TP1) — TP2 is harder to hit."""
        for score in range(50, 101, 5):
            out = calibrate_probabilities(score, REGIME_BULL)
            assert out["p_tp2_before_sl"] <= out["p_tp1_before_sl"] + 1e-9


# ── Output schema ──────────────────────────────────────────────────────────────

class TestOutputSchema:
    def test_required_keys_present(self):
        out = calibrate_probabilities(80, REGIME_BULL)
        required = [
            "p_tp1_before_sl", "p_tp2_before_sl",
            "confidence_interval_95", "expected_return_r_multiple",
            "calibration_method",
        ]
        for k in required:
            assert k in out, f"Missing key: {k}"

    def test_probabilities_in_unit_interval(self):
        out = calibrate_probabilities(80, REGIME_BULL)
        assert 0.0 <= out["p_tp1_before_sl"] <= 1.0
        assert 0.0 <= out["p_tp2_before_sl"] <= 1.0

    def test_confidence_interval_valid(self):
        out = calibrate_probabilities(80, REGIME_BULL)
        ci = out["confidence_interval_95"]
        assert ci is not None
        lo, hi = ci
        assert 0.0 <= lo <= hi <= 1.0

    def test_calibration_method_is_string(self):
        out = calibrate_probabilities(80, REGIME_BULL)
        assert isinstance(out["calibration_method"], str)
        assert len(out["calibration_method"]) > 0


# ── Bayesian update ────────────────────────────────────────────────────────────

class TestBayesianUpdate:
    def test_perfect_track_record_raises_probability(self):
        """100 % recent win rate should push calibrated prob above prior."""
        prior = calibrate_probabilities(80, REGIME_BULL)
        updated = calibrate_probabilities(
            80, REGIME_BULL,
            recent_performance={"wins": 20, "total": 20},
        )
        assert updated["p_tp1_before_sl"] >= prior["p_tp1_before_sl"]

    def test_zero_win_rate_lowers_probability(self):
        prior = calibrate_probabilities(80, REGIME_BULL)
        updated = calibrate_probabilities(
            80, REGIME_BULL,
            recent_performance={"wins": 0, "total": 20},
        )
        assert updated["p_tp1_before_sl"] <= prior["p_tp1_before_sl"]

    def test_empty_performance_unchanged(self):
        base = calibrate_probabilities(80, REGIME_BULL)
        with_empty = calibrate_probabilities(
            80, REGIME_BULL,
            recent_performance={"wins": 0, "total": 0},
        )
        assert abs(base["p_tp1_before_sl"] - with_empty["p_tp1_before_sl"]) < 0.01


# ── Regime-specific calibration ───────────────────────────────────────────────

class TestRegimeCalibration:
    def test_all_regimes_return_valid_output(self):
        for regime in (REGIME_BULL, REGIME_CHOP, REGIME_BEAR):
            out = calibrate_probabilities(80, regime)
            assert 0.0 <= out["p_tp1_before_sl"] <= 1.0

    def test_chop_regime_lower_than_bull(self):
        """Chop regime should have lower or equal win rate than bull."""
        bull = calibrate_probabilities(80, REGIME_BULL)
        chop = calibrate_probabilities(80, REGIME_CHOP)
        # Allow small tolerance for lookup table rounding
        assert chop["p_tp1_before_sl"] <= bull["p_tp1_before_sl"] + 0.05


# ── Score boundary cases ───────────────────────────────────────────────────────

class TestScoreBoundaries:
    def test_score_zero_returns_low_probability(self):
        out = calibrate_probabilities(0, REGIME_CHOP)
        assert out["p_tp1_before_sl"] < 0.55

    def test_score_100_returns_high_probability(self):
        out = calibrate_probabilities(100, REGIME_BULL)
        assert out["p_tp1_before_sl"] >= 0.68

    def test_score_75_meets_spec_target(self):
        """Spec: ≥ 68 % win rate at score ≥ 75."""
        out = calibrate_probabilities(75, REGIME_BULL)
        assert out["p_tp1_before_sl"] >= 0.68

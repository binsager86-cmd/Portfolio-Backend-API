"""Probability calibrator for the Kuwait Signal Engine.

Two-stage calibration:
  Stage 1 — Isotonic regression mapping from score buckets to win rates
             (requires scikit-learn; falls back to pre-seeded lookup table).
  Stage 2 — Bayesian update with recent live-trade performance.

The pre-seeded lookup table reflects the spec target (≥ 68 % win rate at
score ≥ 75) and is refined once enough live trades are recorded.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from app.services.signal_engine.config.risk_config import (
    BAYES_PRIOR_PSEUDO_OBS,
    ISO_MIN_SAMPLES,
    REGIME_WIN_RATE_MULTIPLIERS,
)

logger = logging.getLogger(__name__)

# ── Optional scikit-learn import ─────────────────────────────────────────────
try:
    from sklearn.isotonic import IsotonicRegression  # type: ignore
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.info("scikit-learn not installed — using lookup-table probability calibration")


def _bayesian_update(prior_win_rate: float, recent_performance: dict[str, Any]) -> float:
    """Apply Bayesian update given recent trade outcomes.

    Args:
        prior_win_rate: Win rate from isotonic regression / lookup [0, 1].
        recent_performance: Dict with keys 'wins' and 'total' (recent trade counts).

    Returns:
        Posterior win rate.
    """
    wins = int(recent_performance.get("wins") or 0)
    total = int(recent_performance.get("total") or 0)
    if total <= 0:
        return prior_win_rate

    # Prior expressed as pseudo-observations centred on prior win rate
    alpha = prior_win_rate * BAYES_PRIOR_PSEUDO_OBS
    beta = (1.0 - prior_win_rate) * BAYES_PRIOR_PSEUDO_OBS

    posterior = (alpha + wins) / (alpha + beta + total)
    return round(max(0.01, min(0.99, posterior)), 4)


def calibrate_probabilities(
    total_score: int,
    regime: str,
    recent_performance: dict[str, Any] | None = None,
    historical_scores: list[float] | None = None,
    historical_outcomes: list[int] | None = None,
    historical_scores_tp2: list[float] | None = None,
    historical_outcomes_tp2: list[int] | None = None,
) -> dict[str, Any]:
    """Map a raw confluence score to calibrated win probabilities.

    Args:
        total_score:          Weighted total score [0, 100].
        regime:               Current HMM regime name.
        recent_performance:   Dict {wins: int, total: int} for Bayesian update.
        historical_scores:    Optional list of past signal scores for isotonic fit.
        historical_outcomes:  Corresponding 0/1 outcomes (1 = hit TP1 before SL).

    Returns:
        Dict with p_tp1_before_sl, p_tp2_before_sl, confidence_interval_95,
        expected_return_r_multiple, calibration_method.
    """
    def unavailable(status: str, sample_size: int = 0) -> dict[str, Any]:
        return {
            "p_tp1_before_sl": None,
            "p_tp2_before_sl": None,
            "confidence_interval_95": None,
            "expected_return_r_multiple": None,
            "calibration_method": None,
            "probability_status": status,
            "sample_size": sample_size,
            "calibrated_as_of": None,
            "brier_score": None,
            "log_loss": None,
            "calibration_curve": None,
        }

    # Live inference must never present the seeded score table as observed probability.
    if historical_scores is None or historical_outcomes is None:
        return unavailable("UNVALIDATED")
    if len(historical_scores) != len(historical_outcomes):
        return unavailable("INSUFFICIENT_SAMPLE", len(historical_outcomes))

    # ── Stage 1: point-in-time isotonic calibration only ─────────────────────
    use_iso = (
        _SKLEARN_AVAILABLE
        and len(historical_scores) >= ISO_MIN_SAMPLES
    )
    if not use_iso:
        return unavailable("INSUFFICIENT_SAMPLE", len(historical_outcomes))

    if use_iso:
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(historical_scores, historical_outcomes)
            raw_p = float(iso.predict([total_score])[0])
            method = "isotonic_regression"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Isotonic regression failed (%s)", exc)
            return unavailable("INSUFFICIENT_SAMPLE", len(historical_outcomes))

    # ── Regime adjustment ────────────────────────────────────────────────────
    regime_mult = REGIME_WIN_RATE_MULTIPLIERS.get(regime, 1.0)
    raw_p = min(0.95, raw_p * regime_mult)

    # ── Stage 2: Bayesian update only after empirical calibration ─────────────
    if recent_performance and int(recent_performance.get("total") or 0) > 0:
        p_tp1 = _bayesian_update(raw_p, recent_performance)
        method += "+bayesian_update"
    else:
        p_tp1 = raw_p

    p_tp1 = round(min(0.95, max(0.05, p_tp1)), 3)
    p_tp2 = None
    if historical_scores_tp2 is not None and historical_outcomes_tp2 is not None:
        if len(historical_scores_tp2) == len(historical_outcomes_tp2) and len(historical_outcomes_tp2) >= ISO_MIN_SAMPLES:
            iso_tp2 = IsotonicRegression(out_of_bounds="clip")
            iso_tp2.fit(historical_scores_tp2, historical_outcomes_tp2)
            p_tp2 = round(min(0.95, max(0.05, float(iso_tp2.predict([total_score])[0]))), 3)

    # Wilson interval from observed outcomes only; no pseudo-observations.
    n_trades = int((recent_performance or {}).get("total") or 0)
    n = len(historical_outcomes)
    observed_wins = sum(int(value) for value in historical_outcomes)
    observed_p = observed_wins / n if n else 0.0
    z = 1.96
    denominator = 1.0 + z * z / n
    centre = observed_p + z * z / (2.0 * n)
    margin = z * math.sqrt((observed_p * (1.0 - observed_p) + z * z / (4.0 * n)) / n)
    ci_low = round(max(0.0, (centre - margin) / denominator), 3)
    ci_high = round(min(1.0, (centre + margin) / denominator), 3)

    brier = sum((float(outcome) - raw_prediction) ** 2 for outcome, raw_prediction in zip(historical_outcomes, iso.predict(historical_scores))) / n
    log_loss = -sum(
        outcome * math.log(max(1e-6, min(1.0 - 1e-6, prediction)))
        + (1 - outcome) * math.log(max(1e-6, min(1.0 - 1e-6, 1.0 - prediction)))
        for outcome, prediction in zip(historical_outcomes, iso.predict(historical_scores))
    ) / n

    return {
        "p_tp1_before_sl": p_tp1,
        "p_tp2_before_sl": p_tp2,
        "confidence_interval_95": [ci_low, ci_high],
        "expected_return_r_multiple": None,
        "calibration_method": method,
        "probability_status": "CALIBRATED",
        "sample_size": n,
        "calibrated_as_of": None,
        "brier_score": round(brier, 4),
        "log_loss": round(log_loss, 4),
        "calibration_curve": None,
    }

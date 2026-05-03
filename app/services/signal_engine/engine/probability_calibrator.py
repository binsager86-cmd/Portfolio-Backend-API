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
from typing import Any

from app.services.signal_engine.config.risk_config import (
    BAYES_PRIOR_PSEUDO_OBS,
    ISO_MIN_SAMPLES,
    SCORE_TO_WIN_RATE,
    TP2_WIN_RATE_FRACTION,
)

logger = logging.getLogger(__name__)

# ── Optional scikit-learn import ─────────────────────────────────────────────
try:
    from sklearn.isotonic import IsotonicRegression  # type: ignore
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.info("scikit-learn not installed — using lookup-table probability calibration")


def _lookup_win_rate(total_score: int) -> float:
    """Interpolate win rate from the pre-seeded score-to-win-rate table."""
    breakpoints = sorted(SCORE_TO_WIN_RATE.keys())
    if total_score <= breakpoints[0]:
        return SCORE_TO_WIN_RATE[breakpoints[0]]
    if total_score >= breakpoints[-1]:
        return SCORE_TO_WIN_RATE[breakpoints[-1]]
    for i in range(len(breakpoints) - 1):
        lo, hi = breakpoints[i], breakpoints[i + 1]
        if lo <= total_score < hi:
            frac = (total_score - lo) / (hi - lo)
            return SCORE_TO_WIN_RATE[lo] + (SCORE_TO_WIN_RATE[hi] - SCORE_TO_WIN_RATE[lo]) * frac
    return 0.50


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
    # ── Stage 1: isotonic regression or lookup table ──────────────────────────
    use_iso = (
        _SKLEARN_AVAILABLE
        and historical_scores is not None
        and historical_outcomes is not None
        and len(historical_scores) >= ISO_MIN_SAMPLES
    )

    if use_iso:
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(historical_scores, historical_outcomes)
            raw_p = float(iso.predict([[total_score]])[0])
            method = "isotonic_regression"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Isotonic regression failed (%s) — using lookup", exc)
            raw_p = _lookup_win_rate(total_score)
            method = "lookup_table_fallback"
    else:
        raw_p = _lookup_win_rate(total_score)
        method = "lookup_table"

    # ── Stage 2: Bayesian update ──────────────────────────────────────────────
    if recent_performance and int(recent_performance.get("total") or 0) > 0:
        p_tp1 = _bayesian_update(raw_p, recent_performance)
        method += "+bayesian_update"
    else:
        p_tp1 = raw_p

    p_tp1 = round(min(0.95, max(0.05, p_tp1)), 3)
    p_tp2 = round(p_tp1 * TP2_WIN_RATE_FRACTION, 3)

    # 95 % confidence interval via normal approximation (Wilson interval)
    n_trades = int((recent_performance or {}).get("total") or 0)
    n = max(10, n_trades + BAYES_PRIOR_PSEUDO_OBS)
    z = 1.96
    se = (p_tp1 * (1 - p_tp1) / n) ** 0.5
    ci_low = round(max(0.0, p_tp1 - z * se), 3)
    ci_high = round(min(1.0, p_tp1 + z * se), 3)

    # Expected return in R-multiples (TP1 probability × reward_multiple − loss_prob × 1.0)
    reward_r = 1.5   # TP1 is set at 1.5R
    expected_r = round(p_tp1 * reward_r - (1.0 - p_tp1) * 1.0, 3)

    return {
        "p_tp1_before_sl": p_tp1,
        "p_tp2_before_sl": p_tp2,
        "confidence_interval_95": [ci_low, ci_high],
        "expected_return_r_multiple": expected_r,
        "calibration_method": method,
    }

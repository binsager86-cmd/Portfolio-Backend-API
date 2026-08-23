"""Tests for honest point-in-time probability calibration."""
from __future__ import annotations

from app.services.signal_engine.engine.probability_calibrator import calibrate_probabilities
from app.services.signal_engine.config.model_params import REGIME_BULL


def _calibration_data() -> tuple[list[float], list[int]]:
    scores = [float(index) for index in range(30, 90)]
    outcomes = [1 if index % 3 else 0 for index in range(len(scores))]
    return scores, outcomes


def test_no_point_in_time_labels_are_unvalidated_and_null() -> None:
    result = calibrate_probabilities(80, REGIME_BULL)
    assert result["probability_status"] == "UNVALIDATED"
    assert result["p_tp1_before_sl"] is None
    assert result["p_tp2_before_sl"] is None
    assert result["confidence_interval_95"] is None
    assert result["calibration_method"] is None
    assert result["sample_size"] == 0


def test_insufficient_labels_are_null() -> None:
    result = calibrate_probabilities(
        80,
        REGIME_BULL,
        historical_scores=[1, 2],
        historical_outcomes=[1, 0],
    )
    assert result["probability_status"] == "INSUFFICIENT_SAMPLE"
    assert result["p_tp1_before_sl"] is None


def test_calibrated_prediction_is_monotone_and_reports_metrics() -> None:
    scores, outcomes = _calibration_data()
    predictions = [
        calibrate_probabilities(
            score,
            REGIME_BULL,
            historical_scores=scores,
            historical_outcomes=outcomes,
        )["p_tp1_before_sl"]
        for score in (40, 50, 60, 70, 80)
    ]
    assert predictions == sorted(predictions)
    result = calibrate_probabilities(
        80,
        REGIME_BULL,
        historical_scores=scores,
        historical_outcomes=outcomes,
    )
    assert result["probability_status"] == "CALIBRATED"
    assert result["sample_size"] == len(outcomes)
    assert result["brier_score"] is not None
    assert result["log_loss"] is not None
    assert result["confidence_interval_95"][0] <= result["confidence_interval_95"][1]


def test_tp2_is_not_derived_from_tp1() -> None:
    scores, outcomes = _calibration_data()
    tp2_outcomes = [1 if index % 5 == 0 else 0 for index in range(len(scores))]
    result = calibrate_probabilities(
        80,
        REGIME_BULL,
        historical_scores=scores,
        historical_outcomes=outcomes,
        historical_scores_tp2=scores,
        historical_outcomes_tp2=tp2_outcomes,
    )
    assert result["p_tp2_before_sl"] is not None
    assert result["p_tp2_before_sl"] != round(result["p_tp1_before_sl"] * 0.65, 3)

from app.services.technical_batch_service import (
    _resolve_combined_scores_from_signal,
    _serialize_score_row,
)


def test_resolve_combined_scores_prefers_explicit_signal_fields() -> None:
    signal = {
        "combined_score_adjusted_directional": 81,
        "combined_score_unadjusted_directional": 74,
        "score_breakdown": {
            "combined_adjusted_directional": 10,
            "combined_unadjusted_directional": 9,
        },
        "raw_technical_score": 65,
        "confluence_details": {
            "total_score_raw": 64,
            "total_score": 63,
            "four_scores": {
                "overall": {
                    "base_score": 62,
                    "score": 61,
                }
            },
        },
    }

    adjusted, unadjusted = _resolve_combined_scores_from_signal(signal)

    assert adjusted == 81
    assert unadjusted == 74


def test_resolve_combined_scores_uses_score_breakdown_fallback() -> None:
    signal = {
        "score_breakdown": {
            "combined_adjusted_directional": 70,
            "combined_unadjusted_directional": 66,
        },
        "confluence_details": {
            "total_score_raw": 58,
            "total_score": 55,
        },
    }

    adjusted, unadjusted = _resolve_combined_scores_from_signal(signal)

    assert adjusted == 70
    assert unadjusted == 66


def test_resolve_combined_scores_uses_raw_and_total_fallbacks() -> None:
    signal = {
        "raw_technical_score": 73,
        "confluence_details": {
            "total_score_raw": 73,
            "total_score": 69,
            "four_scores": {
                "overall": {
                    "base_score": 68,
                    "score": 67,
                }
            },
        },
    }

    adjusted, unadjusted = _resolve_combined_scores_from_signal(signal)

    assert adjusted == 73
    assert unadjusted == 68


def test_resolve_combined_scores_backfills_unadjusted_from_adjusted() -> None:
    signal = {
        "combined_score_adjusted_directional": 77,
    }

    adjusted, unadjusted = _resolve_combined_scores_from_signal(signal)

    assert adjusted == 77
    assert unadjusted == 77


def test_serialize_score_row_keeps_overall_null() -> None:
    row = {
        "symbol": "NBK",
        "company_name": "National Bank of Kuwait",
        "segment": "PREMIER",
        "signal": "BUY",
        "reason": None,
        "trend_score": None,
        "momentum_score": None,
        "buying_pressure_score": None,
        "key_price_level_score": None,
        "overall_score": None,
        "raw_technical_score": 71,
        "risk_adjusted_score": 74,
        "error": None,
    }

    serialized = _serialize_score_row(row)

    assert serialized["overall_score"] is None
    assert serialized["raw_technical_score"] == 71
    assert serialized["risk_adjusted_score"] == 74


def _base_row_for_action_tests() -> dict:
    return {
        "symbol": "NBK",
        "company_name": "National Bank of Kuwait",
        "segment": "PREMIER",
        "signal": "BUY",
        "reason": None,
        "trend_score": 60,
        "momentum_score": 60,
        "buying_pressure_score": 60,
        "key_price_level_score": 60,
        "overall_score": None,
        "raw_technical_score": 70,
        "risk_adjusted_score": 70,
        "error": None,
    }


def test_negative_gap_execute_requires_trend_and_adjusted_gate() -> None:
    row = _base_row_for_action_tests()
    row["raw_technical_score"] = 63
    row["risk_adjusted_score"] = 66
    row["trend_score"] = 49

    serialized = _serialize_score_row(row)

    assert serialized["score_gap"] == -3
    assert serialized["action_recommendation"] == "HOLD"

    row["trend_score"] = 50
    serialized = _serialize_score_row(row)

    assert serialized["score_gap"] == -3
    assert serialized["action_recommendation"] == "EXECUTE"


def test_negative_gap_adjusted_below_55_is_avoid() -> None:
    row = _base_row_for_action_tests()
    row["raw_technical_score"] = 50
    row["risk_adjusted_score"] = 54
    row["trend_score"] = 80

    serialized = _serialize_score_row(row)

    assert serialized["score_gap"] == -4
    assert serialized["action_recommendation"] == "AVOID"


def test_negative_gap_trend_below_30_is_flag() -> None:
    row = _base_row_for_action_tests()
    row["raw_technical_score"] = 62
    row["risk_adjusted_score"] = 66
    row["trend_score"] = 29

    serialized = _serialize_score_row(row)

    assert serialized["score_gap"] == -4
    assert serialized["action_recommendation"] == "FLAG"


def test_positive_gap_execute_band_remains_unchanged() -> None:
    row = _base_row_for_action_tests()
    row["raw_technical_score"] = 73
    row["risk_adjusted_score"] = 69
    row["trend_score"] = 40

    serialized = _serialize_score_row(row)

    assert serialized["score_gap"] == 4
    assert serialized["action_recommendation"] == "EXECUTE"

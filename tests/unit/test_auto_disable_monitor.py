from __future__ import annotations

from app.services.eagle_eye.ml import auto_disable_monitor


def test_check_trigger_a_skips_mce_on_cold_start() -> None:
    seven_day_query_called = False

    def fake_query_one(sql: str, params: tuple[object, ...]) -> dict[str, int]:
        assert "COUNT(*) AS n" in sql
        assert params == ()
        return {"n": 13}

    def fake_query_all(sql: str, params: tuple[object, ...]) -> list[dict[str, float]]:
        nonlocal seven_day_query_called
        seven_day_query_called = True
        return [{"calibrated_prob": 0.95, "rule_confidence": 0.10}]

    trigger, reason = auto_disable_monitor._check_trigger_a(
        "2026-05-19",
        fake_query_one,
        fake_query_all,
    )

    assert (trigger, reason) == (None, None)
    assert seven_day_query_called is False

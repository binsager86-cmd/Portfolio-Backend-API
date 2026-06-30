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


def test_check_trigger_d_uses_kuwait_trading_calendar() -> None:
    # Tuesday: previous days are Mon (trading), Sun (trading), Sat/Fri (closed).
    # If Sunday has rows, this must NOT trigger even when Friday is empty.
    counts_by_date = {
        "2026-06-29": 0,   # Monday (trading)
        "2026-06-28": 12,  # Sunday (trading) -> streak broken
        "2026-06-26": 0,   # Friday (closed, should be ignored)
    }

    def fake_query_all(sql: str, params: tuple[object, ...]) -> list[dict[str, int]]:
        if "FROM ml_models" in sql:
            return [{"n": 14}]
        assert "FROM ml_shadow_log" in sql
        ds = str(params[0])
        return [{"n": counts_by_date.get(ds, 0)}]

    trigger, reason = auto_disable_monitor._check_trigger_d("2026-06-30", fake_query_all)

    assert (trigger, reason) == (None, None)


def test_check_trigger_d_triggers_on_two_consecutive_trading_failures() -> None:
    counts_by_date = {
        "2026-06-29": 0,  # Monday (trading)
        "2026-06-28": 0,  # Sunday (trading)
    }

    def fake_query_all(sql: str, params: tuple[object, ...]) -> list[dict[str, int]]:
        if "FROM ml_models" in sql:
            return [{"n": 14}]
        assert "FROM ml_shadow_log" in sql
        ds = str(params[0])
        return [{"n": counts_by_date.get(ds, 0)}]

    trigger, reason = auto_disable_monitor._check_trigger_d("2026-06-30", fake_query_all)

    assert trigger == "SCORING_FAILURE"
    assert reason is not None
    assert "consecutive days with no shadow rows" in reason


def test_check_trigger_d_skips_when_no_shadow_models() -> None:
    def fake_query_all(sql: str, params: tuple[object, ...]) -> list[dict[str, int]]:
        if "FROM ml_models" in sql:
            return [{"n": 0}]
        raise AssertionError("ml_shadow_log should not be queried when there are no SHADOW models")

    trigger, reason = auto_disable_monitor._check_trigger_d("2026-06-30", fake_query_all)

    assert (trigger, reason) == (None, None)

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.core.database import exec_sql, query_one
from app.services.eagle_eye.audit_service import ensure_schema as ensure_audit_schema
from app.services.eagle_eye.market_data_service import ensure_schema, ingest_tickerchart, repair_value_units
from app.services.eagle_eye.risk_service import liquidity_filter_at


@pytest.fixture(autouse=True)
def _reset_r9_tables():
    ensure_schema()
    ensure_audit_schema()
    for table in [
        "ee_ohlcv",
        "ee_data_quality_quarantine",
        "ee_audit_events",
        "ee_change_requests",
        "ee_change_status_history",
    ]:
        try:
            exec_sql(f"DELETE FROM {table}", ())
        except Exception:
            pass
    yield


def _make_row(day: date, close: float, value: float, volume: float = 1_000.0) -> dict:
    return {
        "date": day.isoformat(),
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": volume,
        "value": value,
    }


def test_ingest_normalizes_value_kwd_and_liquidity_gate(monkeypatch):
    dates = [date(2021, 1, 1).fromordinal(date(2021, 1, 1).toordinal() + i) for i in range(20)]
    rows_by_symbol = {
        "HIGH": [_make_row(day, 100.0, 200_000_000.0) for day in dates],
        "LOW": [_make_row(day, 100.0, 50_000_000.0) for day in dates],
    }

    async def fake_fetch_ohlcv(base_symbol, market_abb, from_d=None, to_d=None, interval="day"):
        return rows_by_symbol[base_symbol]

    monkeypatch.setattr("app.services.tickerchart_service.fetch_ohlcv", fake_fetch_ohlcv)

    result = ingest_tickerchart(["HIGH", "LOW"])
    assert result["rows_upserted"] == 40

    high_row = query_one("SELECT value_kwd, value_unit FROM ee_ohlcv WHERE symbol = ? ORDER BY trade_date LIMIT 1", ("HIGH",))
    low_row = query_one("SELECT value_kwd, value_unit FROM ee_ohlcv WHERE symbol = ? ORDER BY trade_date LIMIT 1", ("LOW",))

    assert float(high_row["value_kwd"]) == pytest.approx(200_000.0)
    assert float(low_row["value_kwd"]) == pytest.approx(50_000.0)
    assert high_row["value_unit"] == "kwd"
    assert low_row["value_unit"] == "kwd"

    high_ok, high_details = liquidity_filter_at("HIGH", None, 100_000.0)
    low_ok, low_details = liquidity_filter_at("LOW", None, 100_000.0)

    assert high_ok is True
    assert low_ok is False
    assert high_details["median_daily_value_kwd_20"] == pytest.approx(200_000.0)
    assert low_details["median_daily_value_kwd_20"] == pytest.approx(50_000.0)


def test_ingest_value_repair_is_idempotent_on_rerun(monkeypatch):
    dates = [date(2021, 1, 1).fromordinal(date(2021, 1, 1).toordinal() + i) for i in range(3)]
    rows = [_make_row(day, 200.0, 200_000_000.0) for day in dates]

    async def fake_fetch_ohlcv(base_symbol, market_abb, from_d=None, to_d=None, interval="day"):
        return rows

    monkeypatch.setattr("app.services.tickerchart_service.fetch_ohlcv", fake_fetch_ohlcv)

    ingest_tickerchart(["IDEMP"])
    first = query_one(
        "SELECT value_kwd FROM ee_ohlcv WHERE symbol = ? ORDER BY trade_date LIMIT 1",
        ("IDEMP",),
    )
    assert first is not None
    first_value = float(first["value_kwd"])

    ingest_tickerchart(["IDEMP"])
    second = query_one(
        "SELECT value_kwd FROM ee_ohlcv WHERE symbol = ? ORDER BY trade_date LIMIT 1",
        ("IDEMP",),
    )
    assert second is not None
    second_value = float(second["value_kwd"])

    assert first_value == pytest.approx(second_value)
    assert second_value == pytest.approx(200_000.0)


def test_repair_value_units_sweep_is_idempotent():
    td = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp())
    exec_sql(
        """
        INSERT INTO ee_ohlcv (symbol, trade_date, open, high, low, close, volume, value_kwd, value_unit, source, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, trade_date) DO UPDATE SET
            value_kwd = excluded.value_kwd,
            value_unit = excluded.value_unit
        """,
        ("LEGACY", td, 100.0, 101.0, 99.0, 100.0, 1_000.0, 200_000_000.0, "fils", "test", td),
    )

    first = repair_value_units(source="test-repair")
    second = repair_value_units(source="test-repair")
    row = query_one("SELECT value_kwd, value_unit FROM ee_ohlcv WHERE symbol = ? AND trade_date = ?", ("LEGACY", td))

    assert first["rows_touched"] == 1
    assert second["rows_touched"] == 0
    assert float(row["value_kwd"]) == pytest.approx(200_000.0)
    assert row["value_unit"] == "kwd"


def test_ingest_ignores_historical_session_gaps(monkeypatch):
    rows = [
        _make_row(date(2021, 1, 1), 100.0, 100_000_000.0),
        _make_row(date(2021, 1, 2), 100.0, 100_000_000.0),
        _make_row(date(2021, 3, 1), 100.0, 100_000_000.0),
        _make_row(date(2021, 3, 2), 100.0, 100_000_000.0),
    ]

    async def fake_fetch_ohlcv(base_symbol, market_abb, from_d=None, to_d=None, interval="day"):
        return rows

    monkeypatch.setattr("app.services.tickerchart_service.fetch_ohlcv", fake_fetch_ohlcv)

    result = ingest_tickerchart(["GAP"])
    assert result["rows_upserted"] == 4
    assert query_one("SELECT status FROM ee_data_quality_quarantine WHERE symbol = ?", ("GAP",)) is None
    assert int(query_one("SELECT COUNT(1) AS n FROM ee_ohlcv WHERE symbol = ?", ("GAP",))["n"]) == 4


def test_ingest_quarantines_recent_jump(monkeypatch):
    rows = [
        _make_row(date(2021, 1, 1), 100.0, 100_000_000.0),
        _make_row(date(2021, 1, 2), 130.0, 130_000_000.0),
    ]

    async def fake_fetch_ohlcv(base_symbol, market_abb, from_d=None, to_d=None, interval="day"):
        return rows

    monkeypatch.setattr("app.services.tickerchart_service.fetch_ohlcv", fake_fetch_ohlcv)

    result = ingest_tickerchart(["JUMP"])
    assert result["rows_upserted"] == 0

    row = query_one("SELECT status, reason_json FROM ee_data_quality_quarantine WHERE symbol = ?", ("JUMP",))
    assert row is not None
    assert row["status"] == "quarantined"
    payload = str(row["reason_json"])
    assert "price_jump_gt_25pct" in payload
    assert "rejected_close" in payload
    assert "prior_close" in payload
    assert "pct" in payload
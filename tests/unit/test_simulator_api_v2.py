from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.eagle_eye_v2.simulator.ledger import SimulatorLedger


def _fixture_ledger(tmp_path: Path) -> Path:
    path = tmp_path / "ee_sim_ledger.db"
    ledger = SimulatorLedger(path)
    with ledger.connect() as conn:
        conn.execute(
            """
            INSERT INTO transactions (
                created_at, portfolio, transaction_type, symbol, quantity, price,
                gross_value_kwd, commission_kwd, net_cash_delta_kwd, decision_session,
                fill_session, source_event_id, reason, status, voids_transaction_id,
                suspension_gap_sessions, data_ingested_at, decision_close_ts, state_snapshot_json
            ) VALUES
            ('2026-08-03T10:00:00Z', 'BUY', 'BUY', 'KFH', 100, 2.000, 200, 0.65, -200.65, '2026-08-02', '2026-08-03', 'E1', 'BASE_CONFIRMED_DIRECT', 'POSTED', NULL, 0, '2026-08-03T12:00:00+00:00', '2026-08-03T13:00:00+00:00', '{"lifecycle_state":"BASE_VALID","avoid_tier":"NONE"}'),
            ('2026-08-04T10:00:00Z', 'BUY', 'VOID', 'KFH', 0, 0, 0, 0, 0, '2026-08-02', '2026-08-04', 'E1', 'test correction', 'VOID', 1, 0, '2026-08-04T12:00:00+00:00', '2026-08-04T13:00:00+00:00', '{}'),
            ('2026-08-03T10:00:00Z', 'WATCHLIST', 'BUY', 'ZAIN', 50, 1.000, 50, 0.1625, -50.1625, '2026-08-02', '2026-08-03', 'V1', 'MARKUP_CONFIRMED_DIRECT', 'POSTED', NULL, 0, '2026-08-03T12:00:00+00:00', '2026-08-03T13:00:00+00:00', '{"lifecycle_state":"MARKUP_ACTIVE","avoid_tier":"AVOID_SOFT"}')
            """
        )
        conn.execute(
            """
            INSERT INTO daily_valuations (created_at, portfolio, symbol, session, quantity, close_price, market_value_kwd, cash_kwd, nav_kwd, state_snapshot_json)
            VALUES
            ('2026-08-03T13:00:00Z', 'BUY', 'KFH', '2026-08-03', 100, 2.20, 220, 99799.35, 100019.35, '{"lifecycle_state":"BASE_VALID","avoid_tier":"NONE"}'),
            ('2026-08-03T13:00:00Z', 'WATCHLIST', 'ZAIN', '2026-08-03', 50, 1.10, 55, 99949.8375, 100004.8375, '{"lifecycle_state":"MARKUP_ACTIVE","avoid_tier":"AVOID_SOFT"}')
            """
        )
        conn.execute(
            """
            INSERT INTO decision_log (created_at, symbol, decision_session, kind, reason, portfolio, frozen_action_json, state_snapshot_json, veto_tier, would_have_entry_reason, data_ingested_at, decision_close_ts)
            VALUES
            ('2026-08-02T13:00:00Z', 'KFH', '2026-08-02', 'ENTRY', 'BASE_CONFIRMED_DIRECT', 'BUY', '{"type":"OPEN_POSITION"}', '{"lifecycle_state":"BASE_VALID","avoid_tier":"NONE"}', NULL, NULL, '2026-08-02T12:00:00+00:00', '2026-08-02T13:00:00+00:00'),
            ('2026-08-02T13:00:00Z', 'ZAIN', '2026-08-02', 'VETOED_ENTRY', 'AVOID_SOFT', 'WATCHLIST', '{"type":"VETOED_ENTRY"}', '{"lifecycle_state":"MARKUP_ACTIVE","avoid_tier":"AVOID_SOFT"}', 'AVOID_SOFT', 'M3', '2026-08-02T12:00:00+00:00', '2026-08-02T13:00:00+00:00')
            """
        )
        conn.commit()
    return path


def _client(monkeypatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("SIMULATOR_LEDGER_PATH", str(_fixture_ledger(tmp_path)))
    return TestClient(app)


def test_v2_simulator_router_exposes_only_get_routes():
    methods = {
        method
        for route in app.routes
        if getattr(route, "path", "").startswith("/api/v2/simulator")
        for method in getattr(route, "methods", set())
    }

    assert methods <= {"GET", "HEAD"}


def test_portfolios_positions_nav_transactions_decisions_and_state(monkeypatch, tmp_path: Path):
    client = _client(monkeypatch, tmp_path)

    portfolios = client.get("/api/v2/simulator/portfolios").json()["portfolios"]
    assert {row["book"] for row in portfolios} == {"BUY", "WATCHLIST"}
    assert next(row for row in portfolios if row["book"] == "BUY")["open_position_count"] == 1

    positions = client.get("/api/v2/simulator/portfolios/BUY/positions").json()["positions"]
    assert positions[0]["symbol"] == "KFH"
    assert positions[0]["unrealized_pnl_pct"] == pytest.approx(10)

    nav = client.get("/api/v2/simulator/portfolios/BUY/nav", params={"days": 5}).json()["series"]
    assert nav[-1]["nav_kwd"] == 100019.35

    transactions = client.get("/api/v2/simulator/transactions", params={"book": "BUY"}).json()["transactions"]
    assert [row["transaction_type"] for row in transactions] == ["VOID", "BUY"]

    decisions = client.get("/api/v2/simulator/decisions", params={"symbol": "ZAIN"}).json()["decisions"]
    assert decisions[0]["disposition"] == "VETOED_ENTRY"
    assert decisions[0]["tier"] == "AVOID_SOFT"

    states = client.get("/api/v2/simulator/symbols/state").json()["symbols"]
    assert states["ZAIN"]["lifecycle"] == "MARKUP_ACTIVE"


def test_symbols_state_uses_day_zero_fallback_for_genesis(monkeypatch, tmp_path: Path):
    path = tmp_path / "empty.db"
    SimulatorLedger(path)
    monkeypatch.setenv("SIMULATOR_LEDGER_PATH", str(path))
    client = TestClient(app)

    states = client.get("/api/v2/simulator/symbols/state").json()["symbols"]

    assert "KFH" in states
    assert states["KFH"]["source"] == "day_zero_inventory"


def test_integrity_reports_row_counts_hash_and_no_cache(monkeypatch, tmp_path: Path):
    client = _client(monkeypatch, tmp_path)

    response = client.get("/api/v2/simulator/system/integrity")
    body = response.json()

    assert response.headers["Cache-Control"] == "no-store"
    assert body["row_counts"]["transactions"] == 3
    assert body["guard_trips_count"] == 0
    assert len(body["ledger_sha256"]) == 64
    assert isinstance(body["seal_verification"]["pass"], bool)


def test_sql_map_contains_endpoint_queries(monkeypatch, tmp_path: Path):
    client = _client(monkeypatch, tmp_path)

    sql_map = client.get("/api/v2/simulator/sql-map").json()

    assert "GET /api/v2/simulator/portfolios" in sql_map
    assert "decision_log" in sql_map["GET /api/v2/simulator/decisions"]
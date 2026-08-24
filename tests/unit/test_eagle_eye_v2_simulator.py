from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.services.eagle_eye_v2.simulator.accounting import PaperPortfolioEngine
from app.services.eagle_eye_v2.simulator.ledger import BackfillGuardError, SimulatorLedger
from app.services.eagle_eye_v2.simulator.models import DecisionKind, FrozenEvent, MarketSession, TransactionType
from app.services.eagle_eye_v2.simulator.runner import SimulatorRunner, UI_REQUIRED_SNAPSHOT_FIELDS, build_day_zero_inventory
from app.services.eagle_eye_v2.simulator.sealed_imports import verify_frozen_imports


def _market(symbol: str, session: str, open_price: float = 1.0, close_price: float = 1.1) -> MarketSession:
    return MarketSession(
        symbol=symbol,
        session=session,
        open_price=open_price,
        close_price=close_price,
        ingestion_ts=f"{session}T12:00:00+00:00",
        decision_close_ts=f"{session}T13:00:00+00:00",
    )


def _entry(symbol: str, session: str, reason: str = "BASE_CONFIRMED_DIRECT") -> FrozenEvent:
    return FrozenEvent(
        symbol=symbol,
        decision_session=session,
        kind=DecisionKind.ENTRY,
        reason=reason,
        action={"type": "OPEN_POSITION", "entry_reason": reason, "position_id": "POS0001"},
        state_snapshot={"lifecycle_state": "BASE_VALID", "avoid_tier": "NONE"},
    )


def _write_sealed_segment_map(db_path: Path | str, rows: list[tuple[str, str, int, int, int, int]]) -> Path:
    target_path = Path(db_path)
    conn = sqlite3.connect(str(target_path))
    conn.execute(
        "CREATE TABLE ee_symbol_segment_map (original_symbol TEXT, segment_symbol TEXT, segment_id INTEGER, bars_count INTEGER, start_trade_date INTEGER, end_trade_date INTEGER)"
    )

    real_rows = list(rows)
    while len(real_rows) < 309:
        idx = len(real_rows) + 1
        real_rows.append((f"FILLER{idx}", f"FILLER{idx}__SEG0001", idx, 1, 1767398400, 1786924800))

    conn.executemany(
        "INSERT INTO ee_symbol_segment_map (original_symbol, segment_symbol, segment_id, bars_count, start_trade_date, end_trade_date) VALUES (?, ?, ?, ?, ?, ?)",
        real_rows,
    )
    conn.commit()
    conn.close()
    return target_path


def test_verify_frozen_imports_hashes_sealed_files():
    hashes = verify_frozen_imports()

    assert hashes["state_machine"] == "d16afb2ffa7faf80dfe2ad3d64034403589c7a21ed35b0fd09bd958954cf2eeb"
    assert hashes["harness"] == "968625754efd1deb35259bc749ad583e2514e33efe46205186351a9692be1eee"


def test_transactions_are_append_only(tmp_path: Path):
    ledger = SimulatorLedger(tmp_path / "ee_sim_ledger.db")
    market = _market("KFH", "2026-08-02")
    tx_id = ledger.append_transaction(
        portfolio="BUY",
        transaction_type=TransactionType.BUY,
        symbol="KFH",
        quantity=100.0,
        price=1.0,
        gross_value_kwd=100.0,
        commission_kwd=0.325,
        net_cash_delta_kwd=-100.325,
        decision_session="2026-07-30",
        fill_session="2026-08-02",
        reason="BASE_CONFIRMED_DIRECT",
        market_session=market,
        state_snapshot={},
    )

    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        with ledger.connect() as conn:
            conn.execute("UPDATE transactions SET price = 2 WHERE id = ?", (tx_id,))

    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        with ledger.connect() as conn:
            conn.execute("DELETE FROM transactions WHERE id = ?", (tx_id,))


def test_fill_session_must_be_after_decision_session(tmp_path: Path):
    ledger = SimulatorLedger(tmp_path / "ee_sim_ledger.db")

    with pytest.raises(BackfillGuardError):
        ledger.append_transaction(
            portfolio="BUY",
            transaction_type=TransactionType.BUY,
            symbol="KFH",
            quantity=1.0,
            price=1.0,
            gross_value_kwd=1.0,
            commission_kwd=0.00325,
            net_cash_delta_kwd=-1.00325,
            decision_session="2026-08-02",
            fill_session="2026-08-02",
            reason="BASE_CONFIRMED_DIRECT",
            market_session=_market("KFH", "2026-08-02"),
            state_snapshot={},
        )


def test_no_decision_can_consume_late_ingested_data(tmp_path: Path):
    ledger = SimulatorLedger(tmp_path / "ee_sim_ledger.db")
    late_market = MarketSession(
        symbol="KFH",
        session="2026-08-02",
        open_price=1.0,
        close_price=1.1,
        ingestion_ts="2026-08-02T14:00:00+00:00",
        decision_close_ts="2026-08-02T13:00:00+00:00",
    )

    with pytest.raises(BackfillGuardError):
        ledger.append_decision(_entry("KFH", "2026-08-02"), late_market)


def test_buy_entry_fills_next_session_and_commission(tmp_path: Path):
    ledger = SimulatorLedger(tmp_path / "ee_sim_ledger.db")
    engine = PaperPortfolioEngine(ledger)

    result_1 = engine.process_session("2026-08-02", {"KFH": _market("KFH", "2026-08-02")}, [_entry("KFH", "2026-08-02")])
    result_2 = engine.process_session("2026-08-03", {"KFH": _market("KFH", "2026-08-03", open_price=2.0, close_price=2.2)}, [])

    assert result_1.transactions == []
    assert len(result_2.transactions) == 1
    with ledger.connect() as conn:
        row = conn.execute("SELECT * FROM transactions").fetchone()
    assert row["portfolio"] == "BUY"
    assert row["transaction_type"] == "BUY"
    assert row["decision_session"] == "2026-08-02"
    assert row["fill_session"] == "2026-08-03"
    assert row["commission_kwd"] == pytest.approx(32.5)


def test_suspended_next_session_fills_first_available_open_with_gap(tmp_path: Path):
    ledger = SimulatorLedger(tmp_path / "ee_sim_ledger.db")
    engine = PaperPortfolioEngine(ledger)

    engine.process_session("2026-08-02", {"KFH": _market("KFH", "2026-08-02")}, [_entry("KFH", "2026-08-02")])
    engine.process_session("2026-08-03", {"KFH": _market("KFH", "2026-08-03", open_price=0.0)}, [])
    engine.process_session("2026-08-04", {"KFH": _market("KFH", "2026-08-04", open_price=2.0)}, [])

    with ledger.connect() as conn:
        row = conn.execute("SELECT fill_session, suspension_gap_sessions FROM transactions").fetchone()
    assert row["fill_session"] == "2026-08-04"
    assert row["suspension_gap_sessions"] == 1


def test_watchlist_tracks_soft_vetoed_entry(tmp_path: Path):
    ledger = SimulatorLedger(tmp_path / "ee_sim_ledger.db")
    runner = SimulatorRunner(ledger)
    veto = runner.veto_event(
        symbol="KFH",
        decision_session="2026-08-02",
        would_have_entry_reason="M3",
        veto_tier="AVOID_SOFT",
        state_snapshot={"lifecycle_state": "MARKUP_ACTIVE", "avoid_tier": "AVOID_SOFT"},
    )

    runner.ingest_session(session="2026-08-02", market_sessions={"KFH": _market("KFH", "2026-08-02")}, frozen_events=[veto])
    runner.ingest_session(session="2026-08-03", market_sessions={"KFH": _market("KFH", "2026-08-03", open_price=2.0)}, frozen_events=[])

    with ledger.connect() as conn:
        row = conn.execute("SELECT portfolio, reason FROM transactions").fetchone()
    assert row["portfolio"] == "WATCHLIST"
    assert row["reason"] == "MARKUP_CONFIRMED_DIRECT"


def test_watchlist_exits_from_daily_state_avoid_hard_tier(tmp_path: Path):
    ledger = SimulatorLedger(tmp_path / "ee_sim_ledger.db")
    runner = SimulatorRunner(ledger)
    veto = runner.veto_event(
        symbol="KFH",
        decision_session="2026-08-02",
        would_have_entry_reason="BASE_CONFIRMED_DIRECT",
        veto_tier="AVOID_SOFT",
        state_snapshot={"lifecycle_state": "BASE_VALID", "avoid_tier": "AVOID_SOFT"},
    )
    daily_state = FrozenEvent(
        symbol="KFH",
        decision_session="2026-08-04",
        kind=DecisionKind.DAILY_STATE,
        reason="AVOID_HARD",
        action={"type": "DAILY_STATE", "avoid_tier": "AVOID_HARD"},
        state_snapshot={"lifecycle_state": "MARKDOWN", "avoid_tier": "AVOID_HARD"},
    )

    runner.ingest_session(session="2026-08-02", market_sessions={"KFH": _market("KFH", "2026-08-02")}, frozen_events=[veto])
    runner.ingest_session(session="2026-08-03", market_sessions={"KFH": _market("KFH", "2026-08-03", open_price=2.0)}, frozen_events=[])
    runner.ingest_session(session="2026-08-04", market_sessions={"KFH": _market("KFH", "2026-08-04", open_price=2.2)}, frozen_events=[daily_state])
    runner.ingest_session(session="2026-08-05", market_sessions={"KFH": _market("KFH", "2026-08-05", open_price=2.1)}, frozen_events=[])

    with ledger.connect() as conn:
        rows = conn.execute("SELECT portfolio, transaction_type, reason FROM transactions ORDER BY id").fetchall()
    assert [(row["portfolio"], row["transaction_type"], row["reason"]) for row in rows] == [
        ("WATCHLIST", "BUY", "BASE_CONFIRMED_DIRECT"),
        ("WATCHLIST", "SELL", "EXIT_AVOID_HARD"),
    ]


def test_watchlist_exits_from_daily_state_published_structural_reason(tmp_path: Path):
    ledger = SimulatorLedger(tmp_path / "ee_sim_ledger.db")
    runner = SimulatorRunner(ledger)
    veto = runner.veto_event(
        symbol="KFH",
        decision_session="2026-08-02",
        would_have_entry_reason="M3",
        veto_tier="AVOID_SOFT",
        state_snapshot={"lifecycle_state": "MARKUP_ACTIVE", "avoid_tier": "AVOID_SOFT"},
    )
    daily_state = FrozenEvent(
        symbol="KFH",
        decision_session="2026-08-04",
        kind=DecisionKind.DAILY_STATE,
        reason="DAILY_STATE",
        action={"type": "DAILY_STATE", "avoid_tier": "NONE", "structural_exit_reason": "EXIT_STRUCTURAL_EMA30_2C"},
        state_snapshot={"lifecycle_state": "MARKUP_ACTIVE", "avoid_tier": "NONE"},
    )

    runner.ingest_session(session="2026-08-02", market_sessions={"KFH": _market("KFH", "2026-08-02")}, frozen_events=[veto])
    runner.ingest_session(session="2026-08-03", market_sessions={"KFH": _market("KFH", "2026-08-03", open_price=2.0)}, frozen_events=[])
    runner.ingest_session(session="2026-08-04", market_sessions={"KFH": _market("KFH", "2026-08-04", open_price=2.2)}, frozen_events=[daily_state])
    runner.ingest_session(session="2026-08-05", market_sessions={"KFH": _market("KFH", "2026-08-05", open_price=2.1)}, frozen_events=[])

    with ledger.connect() as conn:
        sell = conn.execute("SELECT portfolio, transaction_type, reason FROM transactions WHERE transaction_type = 'SELL'").fetchone()
    assert sell["portfolio"] == "WATCHLIST"
    assert sell["reason"] == "EXIT_STRUCTURAL_EMA30_2C"


def test_day_zero_inventory_reads_sealed_v53_replay():
    inventory = build_day_zero_inventory()

    assert inventory["frozen_variant"] == "A"
    assert len(inventory["symbols"]) == 139
    assert "KFH" in inventory["symbols"]
    assert {"lifecycle", "tier"}.issubset(inventory["symbols"]["KFH"])


def test_live_market_source_reads_single_session_from_ohlcv_cache(tmp_path: Path):
    db_path = tmp_path / "dev_portfolio.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE ee_ohlcv_cache (ticker TEXT, bar_date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, turnover_kwd REAL, fetched_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE ee_symbol_segment_map (original_symbol TEXT, segment_symbol TEXT, segment_id INTEGER, bars_count INTEGER, start_trade_date INTEGER, end_trade_date INTEGER)"
    )
    conn.execute(
        "INSERT INTO ee_ohlcv_cache (ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("KFH", "2026-08-20", 1.0, 1.1, 0.9, 1.05, 1000.0, 1050.0, "2026-08-20T12:00:00Z"),
    )
    conn.execute(
        "INSERT INTO ee_ohlcv_cache (ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("KFH", "2026-08-19", 0.95, 1.0, 0.9, 0.98, 1000.0, 980.0, "2026-08-19T12:00:00Z"),
    )
    conn.execute(
        "INSERT INTO ee_ohlcv_cache (ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("AUB", "2026-08-20", 2.0, 2.2, 1.9, 2.1, 2000.0, 4200.0, "2026-08-20T12:00:00Z"),
    )
    conn.execute(
        "INSERT INTO ee_symbol_segment_map (original_symbol, segment_symbol, segment_id, bars_count, start_trade_date, end_trade_date) VALUES (?, ?, ?, ?, ?, ?)",
        ("KFH", "KFH__SEG0001", 1, 2, 1758259200, 1800000000),
    )
    conn.execute(
        "INSERT INTO ee_symbol_segment_map (original_symbol, segment_symbol, segment_id, bars_count, start_trade_date, end_trade_date) VALUES (?, ?, ?, ?, ?, ?)",
        ("AUB", "AUB__SEG0001", 1, 1, 1758259200, 1800000000),
    )
    conn.commit()
    conn.close()

    from app.services.eagle_eye_v2.simulator.market_data_source import LiveMarketDataSource

    surface_db = tmp_path / "forward_surface.db"
    with sqlite3.connect(str(surface_db)) as surface_conn:
        surface_conn.execute(
            "CREATE TABLE forward_surface_rows (run_key TEXT, symbol TEXT, trade_date TEXT, row_json TEXT, calendar_version_id TEXT, mask_manifest_version_id TEXT, status TEXT)"
        )
        surface_conn.execute(
            "INSERT INTO forward_surface_rows (run_key, symbol, trade_date, row_json, calendar_version_id, mask_manifest_version_id, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("FORWARD_SURFACE", "KFH__SEG0001", "2026-08-20", '{"symbol":"KFH__SEG0001","open":1.0,"high":1.1,"low":0.9,"close":1.05,"volume":1000,"turnover_kwd":1050}', "BK_CAL_V4_1783783330", "R12_MASKED_INTERVALS_MANIFEST_V4_3_FINAL", "READY"),
        )
        surface_conn.execute(
            "INSERT INTO forward_surface_rows (run_key, symbol, trade_date, row_json, calendar_version_id, mask_manifest_version_id, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("FORWARD_SURFACE", "AUB__SEG0001", "2026-08-20", '{"symbol":"AUB__SEG0001","open":2.0,"high":2.2,"low":1.9,"close":2.1,"volume":2000,"turnover_kwd":4200}', "BK_CAL_V4_1783783330", "R12_MASKED_INTERVALS_MANIFEST_V4_3_FINAL", "READY"),
        )
        surface_conn.commit()

    source = LiveMarketDataSource(db_path=db_path, session_date="2026-08-20", expected_symbol_count=2, surface_db_path=surface_db)
    rows = source.load_session_rows()

    assert len(rows) == 2
    assert set(rows.keys()) == {"KFH__SEG0001", "AUB__SEG0001"}
    assert rows["KFH__SEG0001"].session == "2026-08-20"
    assert rows["AUB__SEG0001"].close_price == pytest.approx(2.1)


def test_live_market_source_assertions_abort_on_mismatch(tmp_path: Path):
    db_path = tmp_path / "dev_portfolio.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE ee_ohlcv_cache (ticker TEXT, bar_date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, turnover_kwd REAL, fetched_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE ee_symbol_segment_map (original_symbol TEXT, segment_symbol TEXT, segment_id INTEGER, bars_count INTEGER, start_trade_date INTEGER, end_trade_date INTEGER)"
    )
    conn.execute(
        "INSERT INTO ee_ohlcv_cache (ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("KFH", "2026-08-19", 1.0, 1.1, 0.9, 1.05, 1000.0, 1050.0, "2026-08-19T12:00:00Z"),
    )
    conn.execute(
        "INSERT INTO ee_symbol_segment_map (original_symbol, segment_symbol, segment_id, bars_count, start_trade_date, end_trade_date) VALUES (?, ?, ?, ?, ?, ?)",
        ("KFH", "KFH__SEG0001", 1, 1, 1758259200, 1800000000),
    )
    conn.commit()
    conn.close()

    from app.services.eagle_eye_v2.simulator.market_data_source import LiveMarketDataSource

    surface_db = tmp_path / "forward_surface.db"
    with sqlite3.connect(str(surface_db)) as surface_conn:
        surface_conn.execute(
            "CREATE TABLE forward_surface_rows (run_key TEXT, symbol TEXT, trade_date TEXT, row_json TEXT, calendar_version_id TEXT, mask_manifest_version_id TEXT, status TEXT)"
        )
        surface_conn.execute(
            "INSERT INTO forward_surface_rows (run_key, symbol, trade_date, row_json, calendar_version_id, mask_manifest_version_id, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("FORWARD_SURFACE", "KFH__SEG0001", "2026-08-20", '{"symbol":"KFH__SEG0001","open":1.0,"high":1.1,"low":0.9,"close":1.05,"volume":1000,"turnover_kwd":1050}', "BK_CAL_V4_1783783330", "R12_MASKED_INTERVALS_MANIFEST_V4_3_FINAL", "READY"),
        )
        surface_conn.commit()

    source = LiveMarketDataSource(db_path=db_path, session_date="2026-08-20", expected_symbol_count=139, surface_db_path=surface_db)
    with pytest.raises(RuntimeError, match="expected symbol count"):
        source.load_session_rows()


def test_genesis_mode_uses_sealed_replay_source():
    from app.services.eagle_eye_v2.simulator.market_data_source import SealedReplayMarketDataSource

    source = SealedReplayMarketDataSource()
    rows = source.load_session_rows(expected_symbol_count=139)

    assert len(rows) == 139
    assert "KFH" in rows
    assert rows["KFH"].symbol == "KFH"
    assert rows["KFH"].session.startswith("2026-07-")


def test_forward_surface_builder_requires_sealed_authority_map(tmp_path: Path):
    from app.services.eagle_eye_v2.simulator.forward_surface import ForwardSurfaceBuilder

    live_db = tmp_path / "live.db"
    conn = sqlite3.connect(str(live_db))
    conn.execute(
        "CREATE TABLE ee_ohlcv_cache (ticker TEXT, bar_date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, turnover_kwd REAL, fetched_at TEXT)"
    )
    conn.execute(
        "INSERT INTO ee_ohlcv_cache (ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("MABANEE", "2026-07-12", 12.0, 12.5, 11.8, 12.2, 1500.0, 18300.0, "2026-07-12T12:00:00Z"),
    )
    conn.commit()
    conn.close()

    sealed_db = tmp_path / "missing_sealed.db"
    surface_db = tmp_path / "forward_surface.db"
    builder = ForwardSurfaceBuilder(live_db_path=live_db, sealed_db_path=sealed_db, surface_db_path=surface_db)

    with pytest.raises((FileNotFoundError, RuntimeError), match="sealed authority DB|ee_symbol_segment_map|not found"):
        builder.append_session_rows("2026-07-12")


def test_forward_surface_builder_stamps_authorities_and_appends_rows(tmp_path: Path):
    from app.services.eagle_eye_v2.simulator.forward_surface import ForwardSurfaceBuilder

    live_db = tmp_path / "live.db"
    conn = sqlite3.connect(str(live_db))
    conn.execute(
        "CREATE TABLE ee_ohlcv_cache (ticker TEXT, bar_date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, turnover_kwd REAL, fetched_at TEXT)"
    )
    conn.execute(
        "INSERT INTO ee_ohlcv_cache (ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("SANAM", "2026-07-10", 10.0, 10.5, 9.8, 10.2, 1000.0, 10200.0, "2026-07-10T12:00:00Z"),
    )
    conn.execute(
        "INSERT INTO ee_ohlcv_cache (ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("TIJARA", "2026-07-10", 20.0, 20.6, 19.8, 20.2, 2000.0, 40400.0, "2026-07-10T12:00:00Z"),
    )
    conn.commit()
    conn.close()

    sealed_db = tmp_path / "sealed.db"
    _write_sealed_segment_map(sealed_db, [
        ("SANAM", "SANAM__SEG0001", 1, 3, 1758259200, 1800000000),
        ("TIJARA", "TIJARA__SEG0001", 2, 3, 1758259200, 1800000000),
    ])

    surface_db = tmp_path / "forward_surface.db"
    builder = ForwardSurfaceBuilder(live_db_path=live_db, sealed_db_path=sealed_db, surface_db_path=surface_db)
    written = builder.append_session_rows("2026-07-10")

    assert written["rows_written"] == 2
    assert builder.surface_authority_ids["calendar_version_id"] == "BK_CAL_V4_1783783330"
    assert builder.surface_authority_ids["mask_manifest_version_id"] == "R12_MASKED_INTERVALS_MANIFEST_V4_3_FINAL"

    with sqlite3.connect(str(surface_db)) as surface_conn:
        rows = surface_conn.execute(
            "SELECT symbol, trade_date, calendar_version_id, mask_manifest_version_id FROM forward_surface_rows ORDER BY symbol"
        ).fetchall()

    assert {row[0] for row in rows} == {"SANAM__SEG0001", "TIJARA__SEG0001"}
    assert all(row[2] == "BK_CAL_V4_1783783330" for row in rows)
    assert all(row[3] == "R12_MASKED_INTERVALS_MANIFEST_V4_3_FINAL" for row in rows)

def test_forward_surface_builder_resolves_segment_qualified_keys(tmp_path: Path):
    from app.services.eagle_eye_v2.simulator.forward_surface import ForwardSurfaceBuilder

    live_db = tmp_path / "live.db"
    conn = sqlite3.connect(str(live_db))
    conn.execute(
        "CREATE TABLE ee_ohlcv_cache (ticker TEXT, bar_date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, turnover_kwd REAL, fetched_at TEXT)"
    )
    conn.execute(
        "INSERT INTO ee_ohlcv_cache (ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("SANAM", "2026-07-10", 10.0, 10.5, 9.8, 10.2, 1000.0, 10200.0, "2026-07-10T12:00:00Z"),
    )
    conn.commit(); conn.close()

    sealed_db = tmp_path / "sealed.db"
    _write_sealed_segment_map(sealed_db, [
        ("SANAM", "SANAM__SEG0001", 1, 13, 1767398400, 1786924800),
    ])

    surface_db = tmp_path / "forward_surface.db"
    builder = ForwardSurfaceBuilder(live_db_path=live_db, sealed_db_path=sealed_db, surface_db_path=surface_db)
    written = builder.append_session_rows("2026-07-10")

    assert written["rows_written"] == 1
    with sqlite3.connect(str(surface_db)) as surface_conn:
        row = surface_conn.execute("SELECT symbol FROM forward_surface_rows").fetchone()
    assert row[0] == "SANAM__SEG0001"


def test_forward_surface_builder_resolves_open_segment_after_sealed_window(tmp_path: Path):
    from app.services.eagle_eye_v2.simulator.forward_surface import ForwardSurfaceBuilder

    live_db = tmp_path / "live.db"
    conn = sqlite3.connect(str(live_db))
    conn.execute(
        "CREATE TABLE ee_ohlcv_cache (ticker TEXT, bar_date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, turnover_kwd REAL, fetched_at TEXT)"
    )
    conn.execute(
        "INSERT INTO ee_ohlcv_cache (ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("SANAM", "2026-07-12", 11.0, 11.5, 10.8, 11.2, 1500.0, 16800.0, "2026-07-12T12:00:00Z"),
    )
    conn.commit(); conn.close()

    sealed_db = tmp_path / "sealed.db"
    _write_sealed_segment_map(sealed_db, [
        ("SANAM", "SANAM__SEG0001", 1, 13, 1767398400, 1786924800),
    ])

    surface_db = tmp_path / "forward_surface.db"
    builder = ForwardSurfaceBuilder(live_db_path=live_db, sealed_db_path=sealed_db, surface_db_path=surface_db)
    written = builder.append_session_rows("2026-07-12")

    assert written["rows_written"] == 1
    with sqlite3.connect(str(surface_db)) as surface_conn:
        row = surface_conn.execute("SELECT symbol FROM forward_surface_rows").fetchone()
    assert row[0] == "SANAM__SEG0001"


def test_forward_surface_builder_quarantines_symbol_closed_before_window_end(tmp_path: Path):
    from app.services.eagle_eye_v2.simulator.forward_surface import ForwardSurfaceBuilder

    live_db = tmp_path / "live.db"
    conn = sqlite3.connect(str(live_db))
    conn.execute(
        "CREATE TABLE ee_ohlcv_cache (ticker TEXT, bar_date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, turnover_kwd REAL, fetched_at TEXT)"
    )
    conn.execute(
        "INSERT INTO ee_ohlcv_cache (ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("KFH", "2026-08-25", 9.8, 10.1, 9.6, 9.9, 1200.0, 11880.0, "2026-08-25T12:00:00Z"),
    )
    conn.commit(); conn.close()

    sealed_db = tmp_path / "sealed.db"
    _write_sealed_segment_map(sealed_db, [
        ("KFH", "KFH__SEG0001", 1, 5, 1767398400, 1784217600),
        ("SANAM", "SANAM__SEG0001", 2, 10, 1767398400, 1786924800),
    ])

    builder = ForwardSurfaceBuilder(live_db_path=live_db, sealed_db_path=sealed_db, surface_db_path=tmp_path / "forward_surface.db")
    result = builder.append_session_rows("2026-08-25")

    assert result["rows_written"] == 0
    assert result["quarantined"] == 1
    with sqlite3.connect(str(tmp_path / "forward_surface.db")) as surface_conn:
        quarantine = surface_conn.execute("SELECT symbol, session, reason FROM forward_surface_quarantine").fetchall()
    assert quarantine[0][0] == "KFH"
    assert quarantine[0][1] == "2026-08-25"


def test_forward_surface_builder_quarantines_unmapped_canonical_symbol(tmp_path: Path):
    from app.services.eagle_eye_v2.simulator.forward_surface import ForwardSurfaceBuilder

    live_db = tmp_path / "live.db"
    conn = sqlite3.connect(str(live_db))
    conn.execute(
        "CREATE TABLE ee_ohlcv_cache (ticker TEXT, bar_date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, turnover_kwd REAL, fetched_at TEXT)"
    )
    conn.execute(
        "INSERT INTO ee_ohlcv_cache (ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("KFH", "2026-07-10", 10.0, 10.5, 9.8, 10.2, 1000.0, 10200.0, "2026-07-10T12:00:00Z"),
    )
    conn.commit(); conn.close()

    sealed_db = tmp_path / "sealed.db"
    _write_sealed_segment_map(sealed_db, [
        ("SANAM", "SANAM__SEG0001", 1, 3, 1767398400, 1786924800),
    ])

    builder = ForwardSurfaceBuilder(live_db_path=live_db, sealed_db_path=sealed_db, surface_db_path=tmp_path / "forward_surface.db")
    result = builder.append_session_rows("2026-07-10")

    assert result["rows_written"] == 0
    assert result["quarantined"] == 1
    with sqlite3.connect(str(tmp_path / "forward_surface.db")) as surface_conn:
        quarantine = surface_conn.execute("SELECT symbol, reason FROM forward_surface_quarantine").fetchall()
    assert quarantine[0][0] == "KFH"
    assert "unmapped" in quarantine[0][1].lower()


def test_forward_surface_builder_quarantines_on_segment_change(tmp_path: Path):
    from app.services.eagle_eye_v2.simulator.forward_surface import ForwardSurfaceBuilder

    live_db = tmp_path / "live.db"
    conn = sqlite3.connect(str(live_db))
    conn.execute(
        "CREATE TABLE ee_ohlcv_cache (ticker TEXT, bar_date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, turnover_kwd REAL, fetched_at TEXT)"
    )
    conn.execute(
        "INSERT INTO ee_ohlcv_cache (ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("SANAM", "2026-07-10", 10.0, 10.5, 9.8, 10.2, 1000.0, 10200.0, "2026-07-10T12:00:00Z"),
    )
    conn.commit(); conn.close()

    sealed_db = tmp_path / "sealed.db"
    _write_sealed_segment_map(sealed_db, [
        ("SANAM", "SANAM__SEG0001", 1, 13, 1758259200, 1784206800),
        ("SANAM", "SANAM__SEG0002", 2, 17, 1783630800, 1800000000),
    ])

    builder = ForwardSurfaceBuilder(live_db_path=live_db, sealed_db_path=sealed_db, surface_db_path=tmp_path / "forward_surface.db")
    result = builder.append_session_rows("2026-07-10")

    assert result["rows_written"] == 0
    assert result["quarantined"] == 1
    with sqlite3.connect(str(tmp_path / "forward_surface.db")) as surface_conn:
        quarantine = surface_conn.execute("SELECT symbol, reason FROM forward_surface_quarantine").fetchall()
    assert quarantine[0][0] == "SANAM"
    assert "segment change" in quarantine[0][1].lower() or "owner ruling" in quarantine[0][1].lower()


def _synthetic_replay_target() -> dict:
    """A target row shaped like forward_replay.replay_symbol()'s output, with a
    fully populated `daily` dict (as the frozen harness produces every session)."""
    return {
        "state": "MARKUP_ACTIVE",
        "tier": "NONE",
        "confirmation_state": "CONFIRMED",
        "candidate_intent_state": "EXECUTE_DIRECT",
        "canonical_symbol": "SANAM",
        "position": {"sessions_held": 12, "mfe": 0.04},
        "daily": {
            "close": 105.2,
            "ema10": 104.1,
            "ema30": 101.8,
            "atr14": 2.3,
            "obv": 15234.0,
            "base_state": "BASE_FROZEN",
            "base_reference": {"id": "SANAM::2026-05-01::BASE01", "top": 98.5, "low": 92.0, "width_pct": 7.1, "validity_state": "VALID"},
            "usable_pivots": {"last_markup_swing_low": {"price": 96.0}},
            "confirmation_gates": [{"name": "FLOW_OBV_SLOPE_OK", "value": 0.12, "threshold": 0.10, "pass": True}],
            "disposition_state": "HOLD",
            "execution": [],
        },
    }


def test_decision_log_snapshot_field_coverage():
    """SNAPSHOT_FIELD_DROP guard: every field the mobile UI reads from a symbol's
    projected state must be a real key in the persisted decision-log snapshot.
    If a future ctx/daily field the UI needs is added without wiring it into the
    snapshot builders, this test fails instead of the screen silently showing
    n/a (see SIM-APP-3d X3 / SIM-APP-3e Y1-Y2)."""
    target = _synthetic_replay_target()
    events = SimulatorRunner._events_from_target("SANAM", "2026-08-20", target)
    assert events, "expected at least one DAILY_STATE event"
    snapshot = events[-1].state_snapshot

    missing = sorted(field for field in UI_REQUIRED_SNAPSHOT_FIELDS if field not in snapshot)
    assert not missing, f"decision-log snapshot is missing UI-required fields: {missing}"

    # Spot-check the values actually flowed through, not just key presence.
    assert snapshot["ema10"] == 104.1
    assert snapshot["ema30"] == 101.8
    assert snapshot["atr14"] == 2.3
    assert snapshot["obv"] == 15234.0
    assert snapshot["base_reference"]["top"] == 98.5
    assert snapshot["usable_pivots"]["last_markup_swing_low"]["price"] == 96.0
    assert snapshot["entry_paths"]["gates_passing"] == 1
    assert snapshot["exit_watch"]["sessions_held"] == 12

def test_live_source_prefers_forward_surface_when_available(tmp_path: Path):
    from app.services.eagle_eye_v2.simulator.market_data_source import LiveMarketDataSource

    live_db = tmp_path / "live.db"
    surface_db = tmp_path / "forward_surface.db"
    conn = sqlite3.connect(str(live_db))
    conn.execute(
        "CREATE TABLE ee_ohlcv_cache (ticker TEXT, bar_date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, turnover_kwd REAL, fetched_at TEXT)"
    )
    conn.execute(
        "INSERT INTO ee_ohlcv_cache (ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("SANAM", "2026-07-10", 10.0, 10.5, 9.8, 10.2, 1000.0, 10200.0, "2026-07-10T12:00:00Z"),
    )
    conn.commit(); conn.close()

    with sqlite3.connect(str(surface_db)) as surface_conn:
        surface_conn.execute(
            "CREATE TABLE forward_surface_rows (run_key TEXT, symbol TEXT, trade_date TEXT, row_json TEXT, calendar_version_id TEXT, mask_manifest_version_id TEXT, status TEXT)"
        )
        surface_conn.execute(
            "INSERT INTO forward_surface_rows (run_key, symbol, trade_date, row_json, calendar_version_id, mask_manifest_version_id, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("FORWARD_SURFACE", "SANAM", "2026-07-10", '{"symbol":"SANAM","open":11.0,"high":11.5,"low":10.8,"close":11.1,"volume":1200,"turnover_kwd":13200}', "BK_CAL_V4_1783783330", "R12_MASKED_INTERVALS_MANIFEST_V4_3_FINAL", "READY"),
        )
        surface_conn.commit()

    source = LiveMarketDataSource(db_path=live_db, session_date="2026-07-10", expected_symbol_count=1, surface_db_path=surface_db)
    rows = source.load_session_rows()

    assert rows["SANAM"].close_price == pytest.approx(11.1)
    assert rows["SANAM"].ingestion_ts == "2026-07-10T12:00:00+00:00"


def test_live_source_aborts_when_forward_surface_is_missing(tmp_path: Path):
    from app.services.eagle_eye_v2.simulator.market_data_source import LiveMarketDataSource

    live_db = tmp_path / "live.db"
    conn = sqlite3.connect(str(live_db))
    conn.execute(
        "CREATE TABLE ee_ohlcv_cache (ticker TEXT, bar_date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, turnover_kwd REAL, fetched_at TEXT)"
    )
    conn.execute(
        "INSERT INTO ee_ohlcv_cache (ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("SANAM", "2026-07-10", 10.0, 10.5, 9.8, 10.2, 1000.0, 10200.0, "2026-07-10T12:00:00Z"),
    )
    conn.commit(); conn.close()

    source = LiveMarketDataSource(db_path=live_db, session_date="2026-07-10", expected_symbol_count=1, surface_db_path=tmp_path / "missing_forward_surface.db")
    with pytest.raises(RuntimeError, match="forward surface is required|forward surface is missing"):
        source.load_session_rows()

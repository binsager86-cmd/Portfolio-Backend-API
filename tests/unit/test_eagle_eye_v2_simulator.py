from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.services.eagle_eye_v2.simulator.accounting import PaperPortfolioEngine
from app.services.eagle_eye_v2.simulator.ledger import BackfillGuardError, SimulatorLedger
from app.services.eagle_eye_v2.simulator.models import DecisionKind, FrozenEvent, MarketSession, TransactionType
from app.services.eagle_eye_v2.simulator.runner import SimulatorRunner, build_day_zero_inventory
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

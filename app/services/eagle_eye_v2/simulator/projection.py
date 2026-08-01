from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.services.eagle_eye_v2.simulator.constants import INITIAL_CAPITAL_KWD, LEDGER_PATH

BOOKS = ("BUY", "WATCHLIST")
SCHEMA = "eagle_eye_sim"
INTEGRITY_ID = 1

POSTGRES_GRANTS = """
CREATE ROLE eagle_eye_sim_reader LOGIN PASSWORD '<managed-secret>';
GRANT USAGE ON SCHEMA eagle_eye_sim TO eagle_eye_sim_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA eagle_eye_sim TO eagle_eye_sim_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA eagle_eye_sim GRANT SELECT ON TABLES TO eagle_eye_sim_reader;
""".strip()

PROJECTION_COLUMNS: dict[str, tuple[str, ...]] = {
    "sim_portfolios": ("book", "nav_kwd", "cash_kwd", "invested_kwd", "open_position_count", "total_pnl_kwd", "change_since_inception_pct", "inception_date", "projected_at"),
    "sim_positions": ("book", "symbol", "entry_date", "entry_price", "entry_reason", "sessions_held", "last_close", "unrealized_pnl_pct", "unrealized_pnl_kwd", "current_lifecycle", "avoid_tier", "projected_at"),
    "sim_transactions": ("id", "created_at", "portfolio", "transaction_type", "symbol", "quantity", "price", "gross_value_kwd", "commission_kwd", "net_cash_delta_kwd", "decision_session", "fill_session", "source_event_id", "reason", "status", "voids_transaction_id", "suspension_gap_sessions", "data_ingested_at", "decision_close_ts", "state_snapshot_json", "projected_at"),
    "sim_decisions": ("id", "created_at", "symbol", "decision_session", "kind", "reason", "portfolio", "frozen_action_json", "state_snapshot_json", "veto_tier", "would_have_entry_reason", "data_ingested_at", "decision_close_ts", "disposition", "tier", "projected_at"),
    "sim_nav_daily": ("book", "session", "nav_kwd", "cash_kwd", "invested_kwd", "projected_at"),
    "sim_symbol_state": ("symbol", "book", "lifecycle", "tier", "session", "source", "last_kind", "last_disposition", "confidence", "gates_passing", "gates_json", "soft_conditions_json", "hard_refs_json", "base_json", "entry_paths_json", "exit_watch_json", "projected_at"),
    "sim_symbol_events": ("id", "symbol", "decision_session", "created_at", "kind", "disposition", "payload_json", "projected_at"),
    "sim_cycles": ("id", "book", "symbol", "base_start", "base_end", "entry_date", "entry_path", "entry_price", "peak_mfe", "shakeout_dates_json", "exit_date", "exit_reason", "exit_price", "pnl_pct", "projected_at"),
}

INTEGRITY_COLUMNS = (
    "id", "status", "last_projected_session", "projection_started_at", "projection_completed_at",
    "sqlite_row_counts_json", "postgres_row_counts_json", "row_count_match", "checksum_match",
    "sqlite_checksum", "postgres_checksum", "stale_reason", "guard_trips_count", "ledger_sha256",
)

COLUMN_DDL: dict[str, dict[str, str]] = {
    "sim_transactions": {"data_ingested_at": "TEXT", "decision_close_ts": "TEXT"},
    "sim_decisions": {"data_ingested_at": "TEXT", "decision_close_ts": "TEXT"},
}

SQL_MAP_POSTGRES: dict[str, str] = {
    "GET /api/v2/simulator/portfolios": "SELECT book, nav_kwd, cash_kwd, invested_kwd, open_position_count, total_pnl_kwd, change_since_inception_pct, inception_date FROM eagle_eye_sim.sim_portfolios ORDER BY book",
    "GET /api/v2/simulator/portfolios/{book}/positions": "SELECT symbol, entry_date, entry_price, entry_reason, sessions_held, last_close, unrealized_pnl_pct, unrealized_pnl_kwd, current_lifecycle, avoid_tier FROM eagle_eye_sim.sim_positions WHERE book = :book ORDER BY symbol",
    "GET /api/v2/simulator/portfolios/{book}/nav": "SELECT session, nav_kwd, cash_kwd, invested_kwd FROM eagle_eye_sim.sim_nav_daily WHERE book = :book ORDER BY session DESC LIMIT :days",
    "GET /api/v2/simulator/transactions": "SELECT * FROM eagle_eye_sim.sim_transactions WHERE (:book IS NULL OR portfolio = :book) AND (:symbol IS NULL OR symbol = :symbol) ORDER BY id DESC LIMIT :limit",
    "GET /api/v2/simulator/decisions": "SELECT * FROM eagle_eye_sim.sim_decisions WHERE (:symbol IS NULL OR symbol = :symbol) ORDER BY id DESC LIMIT :limit",
    "GET /api/v2/simulator/symbols/state": "SELECT symbol, book, lifecycle, tier, session, source, last_kind, last_disposition, confidence, gates_passing, gates_json, soft_conditions_json, hard_refs_json, base_json, entry_paths_json, exit_watch_json FROM eagle_eye_sim.sim_symbol_state ORDER BY symbol",
    "GET /api/v2/simulator/symbols/{symbol}/trace": "SELECT * FROM eagle_eye_sim.sim_symbol_state WHERE symbol = :symbol ORDER BY projected_at DESC LIMIT 1",
    "GET /api/v2/simulator/symbols/{symbol}/events": "SELECT * FROM eagle_eye_sim.sim_symbol_events WHERE symbol = :symbol ORDER BY decision_session DESC, id DESC LIMIT :limit",
    "GET /api/v2/simulator/symbols/{symbol}/cycles": "SELECT * FROM eagle_eye_sim.sim_cycles WHERE symbol = :symbol ORDER BY COALESCE(exit_date, base_start) DESC, id DESC",
    "GET /api/v2/simulator/scanner/v2-columns": "SELECT symbol, book, lifecycle, tier, gates_passing, confidence, last_kind, last_disposition, base_json FROM eagle_eye_sim.sim_symbol_state ORDER BY symbol",
    "GET /api/v2/simulator/system/integrity": "SELECT * FROM eagle_eye_sim.sim_integrity WHERE id = 1",
}


@dataclass(frozen=True)
class ProjectionResult:
    status: str
    last_projected_session: str | None
    row_count_match: bool
    checksum_match: bool
    sqlite_row_counts: dict[str, int]
    postgres_row_counts: dict[str, int]
    sqlite_checksum: str
    postgres_checksum: str
    stale_reason: str | None
    ledger_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "last_projected_session": self.last_projected_session,
            "row_count_match": self.row_count_match,
            "checksum_match": self.checksum_match,
            "sqlite_row_counts": self.sqlite_row_counts,
            "postgres_row_counts": self.postgres_row_counts,
            "sqlite_checksum": self.sqlite_checksum,
            "postgres_checksum": self.postgres_checksum,
            "stale_reason": self.stale_reason,
            "ledger_sha256": self.ledger_sha256,
        }


def ledger_path_from_env() -> Path:
    return Path(os.environ.get("SIMULATOR_LEDGER_PATH", str(LEDGER_PATH)))


def project_simulator_ledger(db: Session | None = None, ledger_path: Path | None = None) -> ProjectionResult:
    owns_session = db is None
    session = db or SessionLocal()
    try:
        result = _project_with_session(session, ledger_path or ledger_path_from_env())
        session.commit()
        return result
    except Exception:
        session.rollback()
        raise
    finally:
        if owns_session:
            session.close()


def ensure_projection_schema(db: Session) -> None:
    if _use_postgres():
        db.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
    _drop_incompatible_projection_tables(db)
    for statement in _projection_ddl():
        db.execute(text(statement))
    _ensure_projection_columns(db)


def table_name(name: str) -> str:
    return f"{SCHEMA}.{name}" if _use_postgres() else name


def _ensure_projection_columns(db: Session) -> None:
    for table, columns in COLUMN_DDL.items():
        existing = _existing_columns(db, table)
        for column, column_type in columns.items():
            if column not in existing:
                db.execute(text(f"ALTER TABLE {table_name(table)} ADD COLUMN {column} {column_type}"))


def _drop_incompatible_projection_tables(db: Session) -> None:
    expected = {**PROJECTION_COLUMNS, "sim_integrity": INTEGRITY_COLUMNS}
    for table, columns in expected.items():
        existing = _existing_columns(db, table)
        if not existing:
            continue
        if not set(columns).issubset(existing):
            db.execute(text(f"DROP TABLE IF EXISTS {table_name(table)}"))


def _existing_columns(db: Session, table: str) -> set[str]:
    if _use_postgres():
        rows = db.execute(
            text("SELECT column_name FROM information_schema.columns WHERE table_schema = :schema AND table_name = :table"),
            {"schema": SCHEMA, "table": table},
        ).all()
        return {str(row[0]) for row in rows}
    rows = db.execute(text(f"PRAGMA table_info({table})")).all()
    return {str(row[1]) for row in rows}


def _project_with_session(db: Session, ledger_path: Path) -> ProjectionResult:
    ensure_projection_schema(db)
    started = _utc_now()
    with _connect_sqlite_ro(ledger_path) as sqlite_conn:
        payload = _read_sqlite_projection(sqlite_conn)
        sqlite_row_counts = _sqlite_source_counts(sqlite_conn)
        sqlite_checksum = _sqlite_transaction_checksum(sqlite_conn)
        ledger_sha = _sha256(ledger_path)

    _replace_read_model(db, payload, started)
    postgres_row_counts = _projection_row_counts(db)
    postgres_checksum = _projection_transaction_checksum(db)
    row_count_match = all(sqlite_row_counts.get(key) == postgres_row_counts.get(key) for key in sqlite_row_counts)
    checksum_match = sqlite_checksum == postgres_checksum
    stale_reason = None if row_count_match and checksum_match else "projection verification mismatch"
    status = "FRESH" if stale_reason is None else "STALE"
    last_session = payload["integrity"]["last_projected_session"]

    _replace_integrity(
        db,
        {
            "id": INTEGRITY_ID,
            "status": status,
            "last_projected_session": last_session,
            "projection_started_at": started,
            "projection_completed_at": _utc_now(),
            "sqlite_row_counts_json": json.dumps(sqlite_row_counts, sort_keys=True),
            "postgres_row_counts_json": json.dumps(postgres_row_counts, sort_keys=True),
            "row_count_match": row_count_match,
            "checksum_match": checksum_match,
            "sqlite_checksum": sqlite_checksum,
            "postgres_checksum": postgres_checksum,
            "stale_reason": stale_reason,
            "guard_trips_count": payload["integrity"]["guard_trips_count"],
            "ledger_sha256": ledger_sha,
        },
    )
    return ProjectionResult(status, last_session, row_count_match, checksum_match, sqlite_row_counts, postgres_row_counts, sqlite_checksum, postgres_checksum, stale_reason, ledger_sha)


def _projection_ddl() -> list[str]:
    bool_type = "BOOLEAN" if _use_postgres() else "INTEGER"
    return [
        f"""
        CREATE TABLE IF NOT EXISTS {table_name('sim_portfolios')} (
            book TEXT PRIMARY KEY,
            nav_kwd DOUBLE PRECISION NOT NULL,
            cash_kwd DOUBLE PRECISION NOT NULL,
            invested_kwd DOUBLE PRECISION NOT NULL,
            open_position_count INTEGER NOT NULL,
            total_pnl_kwd DOUBLE PRECISION NOT NULL,
            change_since_inception_pct DOUBLE PRECISION NOT NULL,
            inception_date TEXT,
            projected_at TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {table_name('sim_positions')} (
            book TEXT NOT NULL,
            symbol TEXT NOT NULL,
            entry_date TEXT,
            entry_price DOUBLE PRECISION NOT NULL,
            entry_reason TEXT,
            sessions_held INTEGER,
            last_close DOUBLE PRECISION NOT NULL,
            unrealized_pnl_pct DOUBLE PRECISION NOT NULL,
            unrealized_pnl_kwd DOUBLE PRECISION NOT NULL,
            current_lifecycle TEXT,
            avoid_tier TEXT NOT NULL,
            projected_at TEXT NOT NULL,
            PRIMARY KEY (book, symbol)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {table_name('sim_transactions')} (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            portfolio TEXT NOT NULL,
            transaction_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            quantity DOUBLE PRECISION NOT NULL,
            price DOUBLE PRECISION NOT NULL,
            gross_value_kwd DOUBLE PRECISION NOT NULL,
            commission_kwd DOUBLE PRECISION NOT NULL,
            net_cash_delta_kwd DOUBLE PRECISION NOT NULL,
            decision_session TEXT NOT NULL,
            fill_session TEXT NOT NULL,
            source_event_id TEXT,
            reason TEXT NOT NULL,
            status TEXT NOT NULL,
            voids_transaction_id INTEGER,
            suspension_gap_sessions INTEGER NOT NULL,
            data_ingested_at TEXT,
            decision_close_ts TEXT,
            state_snapshot_json TEXT NOT NULL,
            projected_at TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {table_name('sim_decisions')} (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            decision_session TEXT NOT NULL,
            kind TEXT NOT NULL,
            reason TEXT NOT NULL,
            portfolio TEXT,
            frozen_action_json TEXT NOT NULL,
            state_snapshot_json TEXT NOT NULL,
            veto_tier TEXT,
            would_have_entry_reason TEXT,
            data_ingested_at TEXT,
            decision_close_ts TEXT,
            disposition TEXT NOT NULL,
            tier TEXT,
            projected_at TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {table_name('sim_nav_daily')} (
            book TEXT NOT NULL,
            session TEXT NOT NULL,
            nav_kwd DOUBLE PRECISION NOT NULL,
            cash_kwd DOUBLE PRECISION NOT NULL,
            invested_kwd DOUBLE PRECISION NOT NULL,
            projected_at TEXT NOT NULL,
            PRIMARY KEY (book, session)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {table_name('sim_symbol_state')} (
            symbol TEXT PRIMARY KEY,
            book TEXT,
            lifecycle TEXT NOT NULL,
            tier TEXT NOT NULL,
            session TEXT,
            source TEXT NOT NULL,
            last_kind TEXT,
            last_disposition TEXT,
            confidence DOUBLE PRECISION,
            gates_passing INTEGER,
            gates_json TEXT,
            soft_conditions_json TEXT,
            hard_refs_json TEXT,
            base_json TEXT,
            entry_paths_json TEXT,
            exit_watch_json TEXT,
            projected_at TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {table_name('sim_symbol_events')} (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            decision_session TEXT NOT NULL,
            created_at TEXT NOT NULL,
            kind TEXT NOT NULL,
            disposition TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            projected_at TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {table_name('sim_cycles')} (
            id INTEGER PRIMARY KEY,
            book TEXT NOT NULL,
            symbol TEXT NOT NULL,
            base_start TEXT,
            base_end TEXT,
            entry_date TEXT,
            entry_path TEXT,
            entry_price DOUBLE PRECISION NOT NULL,
            peak_mfe DOUBLE PRECISION NOT NULL,
            shakeout_dates_json TEXT NOT NULL,
            exit_date TEXT,
            exit_reason TEXT,
            exit_price DOUBLE PRECISION,
            pnl_pct DOUBLE PRECISION NOT NULL,
            projected_at TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {table_name('sim_integrity')} (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            last_projected_session TEXT,
            projection_started_at TEXT NOT NULL,
            projection_completed_at TEXT NOT NULL,
            sqlite_row_counts_json TEXT NOT NULL,
            postgres_row_counts_json TEXT NOT NULL,
            row_count_match {bool_type} NOT NULL,
            checksum_match {bool_type} NOT NULL,
            sqlite_checksum TEXT NOT NULL,
            postgres_checksum TEXT NOT NULL,
            stale_reason TEXT,
            guard_trips_count INTEGER NOT NULL,
            ledger_sha256 TEXT NOT NULL
        )
        """,
    ]


def _read_sqlite_projection(conn: sqlite3.Connection) -> dict[str, Any]:
    return {
        "portfolios": _portfolio_rows(conn),
        "positions": _position_rows(conn),
        "transactions": [dict(row) for row in conn.execute("SELECT * FROM transactions ORDER BY id")],
        "decisions": _decision_rows(conn),
        "nav_daily": _nav_rows(conn),
        "symbol_state": _symbol_state_rows(conn),
        "symbol_events": _symbol_event_rows(conn),
        "cycles": _cycle_rows(conn),
        "integrity": {
            "last_projected_session": conn.execute("SELECT MAX(session) FROM daily_valuations").fetchone()[0],
            "guard_trips_count": int(conn.execute("SELECT COUNT(*) FROM guard_trips").fetchone()[0] or 0),
        },
    }


def _portfolio_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = []
    for book in BOOKS:
        latest = conn.execute(
            """
            SELECT session, MAX(nav_kwd) AS nav_kwd, MAX(cash_kwd) AS cash_kwd, SUM(market_value_kwd) AS invested_kwd
            FROM daily_valuations WHERE portfolio = ? GROUP BY session ORDER BY session DESC LIMIT 1
            """,
            (book,),
        ).fetchone()
        net_cash = conn.execute("SELECT SUM(net_cash_delta_kwd) FROM transactions WHERE portfolio = ? AND status = 'POSTED'", (book,)).fetchone()[0] or 0.0
        inception = conn.execute("SELECT MIN(fill_session) FROM transactions WHERE portfolio = ? AND status = 'POSTED'", (book,)).fetchone()[0]
        nav = float(latest["nav_kwd"] if latest else INITIAL_CAPITAL_KWD)
        cash = float(latest["cash_kwd"] if latest else INITIAL_CAPITAL_KWD + net_cash)
        invested = float(latest["invested_kwd"] if latest else 0.0)
        rows.append({
            "book": book,
            "nav_kwd": nav,
            "cash_kwd": cash,
            "invested_kwd": invested,
            "open_position_count": len(_open_quantities(conn, book)),
            "total_pnl_kwd": nav - INITIAL_CAPITAL_KWD,
            "change_since_inception_pct": _pct(nav - INITIAL_CAPITAL_KWD, INITIAL_CAPITAL_KWD),
            "inception_date": inception,
        })
    return rows


def _position_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = []
    for book in BOOKS:
        for symbol, quantity in _open_quantities(conn, book).items():
            buy = conn.execute(
                "SELECT fill_session, price, reason FROM transactions WHERE portfolio = ? AND symbol = ? AND transaction_type = 'BUY' AND status = 'POSTED' ORDER BY id LIMIT 1",
                (book, symbol),
            ).fetchone()
            valuation = conn.execute(
                "SELECT session, close_price, state_snapshot_json FROM daily_valuations WHERE portfolio = ? AND symbol = ? ORDER BY session DESC, id DESC LIMIT 1",
                (book, symbol),
            ).fetchone()
            entry_price = float(buy["price"] if buy else 0.0)
            last_close = float(valuation["close_price"] if valuation else entry_price)
            state = _json_object(valuation["state_snapshot_json"] if valuation else None)
            rows.append({
                "book": book,
                "symbol": symbol,
                "entry_date": buy["fill_session"] if buy else None,
                "entry_price": entry_price,
                "entry_reason": buy["reason"] if buy else None,
                "sessions_held": _sessions_between(buy["fill_session"] if buy else None, valuation["session"] if valuation else None),
                "last_close": last_close,
                "unrealized_pnl_pct": _pct(last_close - entry_price, entry_price),
                "unrealized_pnl_kwd": (last_close - entry_price) * float(quantity),
                "current_lifecycle": state.get("lifecycle_state") or state.get("lifecycle") or state.get("lifecycle_status"),
                "avoid_tier": state.get("avoid_tier") or state.get("tier") or "NONE",
            })
    return rows


def _decision_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = []
    for row in conn.execute("SELECT * FROM decision_log ORDER BY id"):
        item = dict(row)
        state = _json_object(item.get("state_snapshot_json"))
        item["disposition"] = item.get("kind")
        item["tier"] = item.get("veto_tier") or state.get("avoid_tier") or state.get("tier")
        rows.append(item)
    return rows


def _nav_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(
            """
            SELECT portfolio AS book, session, MAX(nav_kwd) AS nav_kwd, MAX(cash_kwd) AS cash_kwd, SUM(market_value_kwd) AS invested_kwd
            FROM daily_valuations GROUP BY portfolio, session ORDER BY portfolio, session
            """
        )
    ]


def _symbol_state_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = []
    for row in conn.execute(
        """
        SELECT d.symbol, d.decision_session, d.portfolio, d.kind, d.veto_tier, d.state_snapshot_json
        FROM decision_log d JOIN (SELECT symbol, MAX(id) AS id FROM decision_log GROUP BY symbol) latest ON latest.id = d.id
        ORDER BY d.symbol
        """
    ):
        state = _json_object(row["state_snapshot_json"])
        rows.append({
            "symbol": row["symbol"],
            "book": row["portfolio"] or state.get("book") or state.get("portfolio"),
            "lifecycle": state.get("lifecycle_state") or state.get("lifecycle") or "NEUTRAL",
            "tier": state.get("avoid_tier") or state.get("tier") or row["veto_tier"] or "NONE",
            "session": row["decision_session"],
            "source": "decision_log",
            "last_kind": row["kind"],
            "last_disposition": row["kind"],
            "confidence": _maybe_float(state.get("confidence") or state.get("confidence_pct") or state.get("score")),
            "gates_passing": _maybe_int(state.get("gates_passing") or state.get("gate_count") or state.get("gates_passed")),
            "gates_json": _json_text(state.get("gates_json") or state.get("gates")),
            "soft_conditions_json": _json_text(state.get("soft_conditions_json") or state.get("soft_conditions")),
            "hard_refs_json": _json_text(state.get("hard_refs_json") or state.get("hard_refs")),
            "base_json": _json_text(state.get("base_json") or state.get("base")),
            "entry_paths_json": _json_text(state.get("entry_paths_json") or state.get("entry_paths")),
            "exit_watch_json": _json_text(state.get("exit_watch_json") or state.get("exit_watch")),
        })
    return rows


def _symbol_event_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = []
    for row in conn.execute("SELECT * FROM decision_log ORDER BY created_at, id"):
        item = dict(row)
        state = _json_object(item.get("state_snapshot_json"))
        payload = {
            "created_at": item.get("created_at"),
            "decision_session": item.get("decision_session"),
            "portfolio": item.get("portfolio"),
            "reason": item.get("reason"),
            "veto_tier": item.get("veto_tier"),
            "would_have_entry_reason": item.get("would_have_entry_reason"),
            "state_snapshot": state,
            "frozen_action": _json_object(item.get("frozen_action_json")),
        }
        rows.append({
            "id": int(item["id"]),
            "symbol": item["symbol"],
            "decision_session": item["decision_session"],
            "created_at": item["created_at"],
            "kind": item["kind"],
            "disposition": item["kind"],
            "payload_json": json.dumps(payload, sort_keys=True),
        })
    return rows


def _cycle_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tx_rows = [dict(row) for row in conn.execute("SELECT * FROM transactions WHERE status = 'POSTED' ORDER BY id")]
    for book in BOOKS:
        by_symbol: dict[str, list[dict[str, Any]]] = {}
        for tx in tx_rows:
            if tx["portfolio"] != book:
                continue
            by_symbol.setdefault(str(tx["symbol"]), []).append(tx)
        for symbol, symbol_rows in by_symbol.items():
            open_tx: dict[str, Any] | None = None
            for tx in symbol_rows:
                tx_type = str(tx["transaction_type"])
                if tx_type == "BUY" and open_tx is None:
                    open_tx = tx
                    continue
                if tx_type != "SELL" or open_tx is None:
                    continue
                entry_state = _json_object(open_tx.get("state_snapshot_json"))
                exit_state = _json_object(tx.get("state_snapshot_json"))
                base_start = entry_state.get("base_start") or entry_state.get("base_start_date") or open_tx["decision_session"]
                base_end = entry_state.get("base_end") or entry_state.get("base_end_date") or open_tx["fill_session"]
                entry_price = float(open_tx.get("price") or 0.0)
                exit_price = float(tx.get("price") or 0.0)
                valuations = [
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT session, close_price FROM daily_valuations
                        WHERE portfolio = ? AND symbol = ? AND session BETWEEN ? AND ?
                        ORDER BY session, id
                        """,
                        (book, symbol, open_tx["fill_session"], tx["fill_session"]),
                    )
                ]
                peak_close = max([entry_price] + [float(row["close_price"] or 0.0) for row in valuations])
                peak_mfe = _pct(peak_close - entry_price, entry_price)
                shakeouts = [row["session"] for row in valuations if float(row["close_price"] or 0.0) < entry_price]
                rows.append({
                    "id": len(rows) + 1,
                    "book": book,
                    "symbol": symbol,
                    "base_start": base_start,
                    "base_end": base_end,
                    "entry_date": open_tx["fill_session"],
                    "entry_path": entry_state.get("entry_path") or entry_state.get("path") or open_tx["reason"],
                    "entry_price": entry_price,
                    "peak_mfe": peak_mfe,
                    "shakeout_dates_json": json.dumps(shakeouts, sort_keys=True),
                    "exit_date": tx["fill_session"],
                    "exit_reason": tx["reason"],
                    "exit_price": exit_price,
                    "pnl_pct": _pct(exit_price - entry_price, entry_price),
                })
                open_tx = None
    return rows


def _replace_read_model(db: Session, payload: dict[str, Any], projected_at: str) -> None:
    for table in ("sim_portfolios", "sim_positions", "sim_transactions", "sim_decisions", "sim_nav_daily", "sim_symbol_state", "sim_symbol_events", "sim_cycles"):
        db.execute(text(f"DELETE FROM {table_name(table)}"))
    _bulk_insert(db, "sim_portfolios", payload["portfolios"], projected_at)
    _bulk_insert(db, "sim_positions", payload["positions"], projected_at)
    _bulk_insert(db, "sim_transactions", payload["transactions"], projected_at)
    _bulk_insert(db, "sim_decisions", payload["decisions"], projected_at)
    _bulk_insert(db, "sim_nav_daily", payload["nav_daily"], projected_at)
    _bulk_insert(db, "sim_symbol_state", payload["symbol_state"], projected_at)
    _bulk_insert(db, "sim_symbol_events", payload["symbol_events"], projected_at)
    _bulk_insert(db, "sim_cycles", payload["cycles"], projected_at)


def _bulk_insert(db: Session, table: str, rows: list[dict[str, Any]], projected_at: str) -> None:
    if not rows:
        return
    columns = list(PROJECTION_COLUMNS[table])
    normalized = [
        {column: {**row, "projected_at": projected_at}.get(column) for column in columns}
        for row in rows
    ]
    placeholders = ", ".join(f":{column}" for column in columns)
    db.execute(text(f"INSERT INTO {table_name(table)} ({', '.join(columns)}) VALUES ({placeholders})"), normalized)


def _replace_integrity(db: Session, row: dict[str, Any]) -> None:
    db.execute(text(f"DELETE FROM {table_name('sim_integrity')} WHERE id = :id"), {"id": row["id"]})
    columns = list(row.keys())
    db.execute(text(f"INSERT INTO {table_name('sim_integrity')} ({', '.join(columns)}) VALUES ({', '.join(':' + c for c in columns)})"), row)


def _connect_sqlite_ro(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _open_quantities(conn: sqlite3.Connection, book: str) -> dict[str, float]:
    rows = conn.execute(
        """
        SELECT symbol, SUM(CASE transaction_type WHEN 'BUY' THEN quantity WHEN 'SELL' THEN -quantity ELSE 0 END) AS quantity
        FROM transactions WHERE portfolio = ? AND status = 'POSTED' GROUP BY symbol
        """,
        (book,),
    ).fetchall()
    return {row["symbol"]: float(row["quantity"] or 0.0) for row in rows if float(row["quantity"] or 0.0) > 0.000001}


def _sqlite_source_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        "sim_transactions": int(conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]),
        "sim_decisions": int(conn.execute("SELECT COUNT(*) FROM decision_log").fetchone()[0]),
        "sim_nav_daily": int(conn.execute("SELECT COUNT(*) FROM (SELECT portfolio, session FROM daily_valuations GROUP BY portfolio, session)").fetchone()[0]),
        "sim_symbol_state": int(conn.execute("SELECT COUNT(*) FROM (SELECT symbol FROM decision_log GROUP BY symbol)").fetchone()[0]),
        "sim_symbol_events": int(conn.execute("SELECT COUNT(*) FROM decision_log").fetchone()[0]),
        "sim_cycles": int(_sqlite_cycle_count(conn)),
    }


def _projection_row_counts(db: Session) -> dict[str, int]:
    return {table: int(db.execute(text(f"SELECT COUNT(*) FROM {table_name(table)}")).scalar() or 0) for table in ("sim_transactions", "sim_decisions", "sim_nav_daily", "sim_symbol_state", "sim_symbol_events", "sim_cycles")}


def _sqlite_transaction_checksum(conn: sqlite3.Connection) -> str:
    rows = conn.execute("SELECT id, portfolio, transaction_type, symbol, net_cash_delta_kwd FROM transactions ORDER BY id").fetchall()
    return _md5_rows(f"{row['id']}|{row['portfolio']}|{row['transaction_type']}|{row['symbol']}|{float(row['net_cash_delta_kwd']):.12f}" for row in rows)


def _projection_transaction_checksum(db: Session) -> str:
    rows = db.execute(text(f"SELECT id, portfolio, transaction_type, symbol, net_cash_delta_kwd FROM {table_name('sim_transactions')} ORDER BY id")).mappings().all()
    return _md5_rows(f"{row['id']}|{row['portfolio']}|{row['transaction_type']}|{row['symbol']}|{float(row['net_cash_delta_kwd']):.12f}" for row in rows)


def _md5_rows(rows: Iterable[str]) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    for row in rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            json.loads(value)
        except json.JSONDecodeError:
            return json.dumps(value, sort_keys=True)
        return value
    return json.dumps(value, sort_keys=True)


def _maybe_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _sqlite_cycle_count(conn: sqlite3.Connection) -> int:
    count = 0
    tx_rows = [dict(row) for row in conn.execute("SELECT * FROM transactions WHERE status = 'POSTED' ORDER BY id")]
    for book in BOOKS:
        by_symbol: dict[str, list[dict[str, Any]]] = {}
        for tx in tx_rows:
            if tx["portfolio"] != book:
                continue
            by_symbol.setdefault(str(tx["symbol"]), []).append(tx)
        for symbol_rows in by_symbol.values():
            open_tx: dict[str, Any] | None = None
            for tx in symbol_rows:
                if tx["transaction_type"] == "BUY" and open_tx is None:
                    open_tx = tx
                    continue
                if tx["transaction_type"] == "SELL" and open_tx is not None:
                    count += 1
                    open_tx = None
    return count


def _pct(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return (numerator / denominator) * 100.0


def _sessions_between(start: str | None, end: str | None) -> int | None:
    if not start or not end:
        return None
    from datetime import date

    try:
        return max(0, (date.fromisoformat(end) - date.fromisoformat(start)).days)
    except ValueError:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _use_postgres() -> bool:
    return get_settings().use_postgres

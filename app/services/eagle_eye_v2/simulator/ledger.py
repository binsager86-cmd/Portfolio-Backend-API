from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.eagle_eye_v2.simulator.constants import LEDGER_PATH, MANIFEST_PATH
from app.services.eagle_eye_v2.simulator.models import DecisionKind, FrozenEvent, MarketSession, TransactionType, parse_timestamp


class BackfillGuardError(RuntimeError):
    pass


class SimulatorLedger:
    def __init__(self, path: Path = LEDGER_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=DELETE")
        return conn

    def ensure_schema(self) -> None:
        with sqlite3.connect(str(self.path)) as conn:
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    portfolio TEXT NOT NULL CHECK (portfolio IN ('BUY', 'WATCHLIST')),
                    transaction_type TEXT NOT NULL CHECK (transaction_type IN ('BUY', 'SELL', 'VOID')),
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL DEFAULT 0,
                    price REAL NOT NULL DEFAULT 0,
                    gross_value_kwd REAL NOT NULL DEFAULT 0,
                    commission_kwd REAL NOT NULL DEFAULT 0,
                    net_cash_delta_kwd REAL NOT NULL DEFAULT 0,
                    decision_session TEXT NOT NULL,
                    fill_session TEXT NOT NULL,
                    source_event_id TEXT,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'POSTED' CHECK (status IN ('POSTED', 'VOID')),
                    voids_transaction_id INTEGER,
                    suspension_gap_sessions INTEGER NOT NULL DEFAULT 0,
                    data_ingested_at TEXT NOT NULL,
                    decision_close_ts TEXT NOT NULL,
                    state_snapshot_json TEXT NOT NULL,
                    CHECK (fill_session > decision_session),
                    CHECK (data_ingested_at <= decision_close_ts)
                );

                CREATE TRIGGER IF NOT EXISTS transactions_no_update
                BEFORE UPDATE ON transactions
                BEGIN
                    SELECT RAISE(ABORT, 'transactions are append-only');
                END;

                CREATE TRIGGER IF NOT EXISTS transactions_no_delete
                BEFORE DELETE ON transactions
                BEGIN
                    SELECT RAISE(ABORT, 'transactions are append-only');
                END;

                CREATE TABLE IF NOT EXISTS daily_valuations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    portfolio TEXT NOT NULL CHECK (portfolio IN ('BUY', 'WATCHLIST')),
                    symbol TEXT NOT NULL,
                    session TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    close_price REAL NOT NULL,
                    market_value_kwd REAL NOT NULL,
                    cash_kwd REAL NOT NULL,
                    nav_kwd REAL NOT NULL,
                    state_snapshot_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS decision_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                    data_ingested_at TEXT NOT NULL,
                    decision_close_ts TEXT NOT NULL,
                    CHECK (data_ingested_at <= decision_close_ts)
                );

                CREATE TABLE IF NOT EXISTS guard_trips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    decision_session TEXT NOT NULL,
                    affected_transaction_id INTEGER,
                    reason TEXT NOT NULL,
                    detail_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS monthly_hashes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    ledger_month TEXT NOT NULL,
                    ledger_sha256 TEXT NOT NULL,
                    manifest_path TEXT,
                    UNIQUE (ledger_month, ledger_sha256)
                );

                CREATE TABLE IF NOT EXISTS machine_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    session TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    pivot_json TEXT NOT NULL,
                    windows_json TEXT NOT NULL,
                    base_recovery_stamps_json TEXT NOT NULL,
                    prev_ready TEXT NOT NULL,
                    prev_segment_json TEXT,
                    prev_masked INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_machine_state_symbol_session
                    ON machine_state(symbol, session, id);
                """
            )
            conn.commit()

    def append_machine_state(self, *, symbol: str, session: str, state: dict[str, Any]) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO machine_state (
                    symbol, session, state_json, pivot_json, windows_json,
                    base_recovery_stamps_json, prev_ready, prev_segment_json,
                    prev_masked, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol.upper(), session,
                    json.dumps(state.get("machine") or {}, sort_keys=True),
                    json.dumps(state.get("pivot") or {}, sort_keys=True),
                    json.dumps({"history_window": state.get("history_window") or [], "flow_window": state.get("flow_window") or [], "coverage_dates": state.get("coverage_dates") or [], "segment_dates": state.get("segment_dates") or [], "flag_rows": state.get("flag_rows") or [], "prior_base": state.get("prior_base")}, sort_keys=True),
                    json.dumps(state.get("base_recovery_stamps") or {}, sort_keys=True),
                    str(state.get("prev_ready") or "READINESS_PENDING"),
                    json.dumps(state.get("prev_segment"), sort_keys=True) if state.get("prev_segment") is not None else None,
                    1 if state.get("prev_masked") else 0,
                    self.utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def latest_machine_state(self, symbol: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM machine_state WHERE symbol = ? ORDER BY session DESC, id DESC LIMIT 1",
                (symbol.upper(),),
            ).fetchone()
        if row is None:
            return None
        windows = json.loads(row["windows_json"])
        return {
            "machine": json.loads(row["state_json"]),
            "pivot": json.loads(row["pivot_json"]),
            "history_window": windows.get("history_window", []),
            "flow_window": windows.get("flow_window", []),
            "coverage_dates": windows.get("coverage_dates", []),
            "segment_dates": windows.get("segment_dates", []),
            "flag_rows": windows.get("flag_rows", []),
            "prior_base": windows.get("prior_base"),
            "base_recovery_stamps": json.loads(row["base_recovery_stamps_json"]),
            "prev_ready": row["prev_ready"],
            "prev_segment": json.loads(row["prev_segment_json"]) if row["prev_segment_json"] else None,
            "prev_masked": bool(row["prev_masked"]),
            "session": row["session"],
        }

    @staticmethod
    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def append_decision(self, event: FrozenEvent, market_session: MarketSession, portfolio: str | None = None) -> int:
        self.assert_no_backfill(event.decision_session, market_session)
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO decision_log (
                    created_at, symbol, decision_session, kind, reason, portfolio,
                    frozen_action_json, state_snapshot_json, veto_tier,
                    would_have_entry_reason, data_ingested_at, decision_close_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.utc_now(),
                    event.symbol.upper(),
                    event.decision_session,
                    event.kind.value,
                    event.reason,
                    portfolio,
                    json.dumps(event.action, sort_keys=True),
                    json.dumps(event.state_snapshot, sort_keys=True),
                    event.veto_tier,
                    event.would_have_entry_reason,
                    market_session.ingestion_ts,
                    market_session.decision_close_ts,
                ),
            )
            return int(cursor.lastrowid)

    def append_transaction(
        self,
        *,
        portfolio: str,
        transaction_type: TransactionType,
        symbol: str,
        quantity: float,
        price: float,
        gross_value_kwd: float,
        commission_kwd: float,
        net_cash_delta_kwd: float,
        decision_session: str,
        fill_session: str,
        reason: str,
        market_session: MarketSession,
        state_snapshot: dict[str, Any],
        source_event_id: str | None = None,
        status: str = "POSTED",
        voids_transaction_id: int | None = None,
        suspension_gap_sessions: int = 0,
    ) -> int:
        self.assert_fill_after_decision(decision_session, fill_session)
        self.assert_no_backfill(decision_session, market_session)
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO transactions (
                    created_at, portfolio, transaction_type, symbol, quantity, price,
                    gross_value_kwd, commission_kwd, net_cash_delta_kwd,
                    decision_session, fill_session, source_event_id, reason, status,
                    voids_transaction_id, suspension_gap_sessions, data_ingested_at,
                    decision_close_ts, state_snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.utc_now(), portfolio, transaction_type.value, symbol.upper(), quantity, price,
                    gross_value_kwd, commission_kwd, net_cash_delta_kwd, decision_session,
                    fill_session, source_event_id, reason, status, voids_transaction_id,
                    suspension_gap_sessions, market_session.ingestion_ts,
                    market_session.decision_close_ts, json.dumps(state_snapshot, sort_keys=True),
                ),
            )
            return int(cursor.lastrowid)

    def append_void_transaction(self, transaction_id: int, reason: str, market_session: MarketSession, detail: dict[str, Any]) -> int:
        original = self.get_transaction(transaction_id)
        if original is None:
            raise BackfillGuardError(f"cannot void missing transaction {transaction_id}")
        event_session = str(original["decision_session"])
        row_id = self.append_transaction(
            portfolio=str(original["portfolio"]),
            transaction_type=TransactionType.VOID,
            symbol=str(original["symbol"]),
            quantity=0.0,
            price=0.0,
            gross_value_kwd=0.0,
            commission_kwd=0.0,
            net_cash_delta_kwd=0.0,
            decision_session=event_session,
            fill_session=market_session.session,
            reason=reason,
            market_session=market_session,
            state_snapshot=detail,
            status="VOID",
            voids_transaction_id=transaction_id,
        )
        self.append_guard_trip(str(original["symbol"]), event_session, transaction_id, reason, detail)
        return row_id

    def append_daily_valuation(
        self,
        *,
        portfolio: str,
        symbol: str,
        session: str,
        quantity: float,
        close_price: float,
        market_value_kwd: float,
        cash_kwd: float,
        nav_kwd: float,
        state_snapshot: dict[str, Any],
    ) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO daily_valuations (
                    created_at, portfolio, symbol, session, quantity, close_price,
                    market_value_kwd, cash_kwd, nav_kwd, state_snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.utc_now(), portfolio, symbol.upper(), session, quantity, close_price,
                    market_value_kwd, cash_kwd, nav_kwd, json.dumps(state_snapshot, sort_keys=True),
                ),
            )
            return int(cursor.lastrowid)

    def append_guard_trip(self, symbol: str, decision_session: str, affected_transaction_id: int | None, reason: str, detail: dict[str, Any]) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO guard_trips (created_at, symbol, decision_session, affected_transaction_id, reason, detail_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (self.utc_now(), symbol.upper(), decision_session, affected_transaction_id, reason, json.dumps(detail, sort_keys=True)),
            )
            return int(cursor.lastrowid)

    def get_transaction(self, transaction_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM transactions WHERE id = ?", (transaction_id,)).fetchone()

    def compute_ledger_sha256(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def append_monthly_hash(self, ledger_month: str, manifest_path: Path = MANIFEST_PATH) -> str:
        ledger_sha = self.compute_ledger_sha256()
        with self.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO monthly_hashes (created_at, ledger_month, ledger_sha256, manifest_path) VALUES (?, ?, ?, ?)",
                (self.utc_now(), ledger_month, ledger_sha, str(manifest_path)),
            )
        append_monthly_ledger_hash_to_manifest(manifest_path, ledger_month, ledger_sha, self.path)
        return ledger_sha

    @staticmethod
    def assert_fill_after_decision(decision_session: str, fill_session: str) -> None:
        if fill_session <= decision_session:
            raise BackfillGuardError(f"fill_session must be after decision_session: {fill_session} <= {decision_session}")

    @staticmethod
    def assert_no_backfill(decision_session: str, market_session: MarketSession) -> None:
        if market_session.session < decision_session:
            raise BackfillGuardError(f"market session {market_session.session} predates decision {decision_session}")
        if parse_timestamp(market_session.ingestion_ts) > parse_timestamp(market_session.decision_close_ts):
            raise BackfillGuardError("decision consumed data ingested after decision-session close")


def append_monthly_ledger_hash_to_manifest(manifest_path: Path, ledger_month: str, ledger_sha: str, ledger_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema": "eagle_eye_archive_manifest_v1", "files": []}
    monthly = manifest.setdefault("monthly_ledger_hashes", [])
    entry = {
        "ledger_month": ledger_month,
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger_sha,
        "appended_at": SimulatorLedger.utc_now(),
        "campaign": "SIM-1",
        "role": "monthly simulator ledger seal",
    }
    if not any(item.get("ledger_month") == ledger_month and item.get("ledger_sha256") == ledger_sha for item in monthly):
        monthly.append(entry)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        Path(str(manifest_path) + ".sha256").write_text(f"{manifest_sha}  {manifest_path.name}\n", encoding="ascii")

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.services.eagle_eye_v2.telemetry_schema import TABLE_TO_FIELDS, TABLES


def _open_connection() -> sqlite3.Connection:
    bound_db = (os.environ.get("EE_V2_RUNTIME_DB_PATH") or "").strip()
    settings = get_settings()
    if settings.use_postgres:
        raise RuntimeError("R14-B module (a) harness is SQLite-only in this phase; PostgreSQL migration path requires a dedicated directive.")
    db_path = bound_db if bound_db else settings.database_abs_path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _ddl_sqlite() -> list[str]:
    return [
        """
        CREATE TABLE IF NOT EXISTS daily_term_row (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            segment_id TEXT,
            segment_day_index INTEGER,
            phase_before TEXT,
            phase_after TEXT,
            readiness_state TEXT,
            readiness_transition_event TEXT,
            readiness_transition_from_state TEXT,
            readiness_transition_to_state TEXT,
            segment_restart_flag INTEGER,
            masked_context_flag INTEGER,
            lookback_long_sessions INTEGER,
            lookback_segment_sessions INTEGER,
            lookback_fallback_sessions INTEGER,
            base_reference_id TEXT,
            intent_id TEXT,
            predicate_namespace TEXT,
            predicate_name TEXT,
            predicate_value REAL,
            predicate_threshold_parameter TEXT,
            predicate_pass INTEGER,
            recoverability_state TEXT,
            recoverability_reason TEXT,
            source_payload_fields TEXT,
            base_reference_version TEXT,
            base_reference_origin TEXT,
            base_reference_current_flag INTEGER,
            extension_pct_vs_current_valid_reference REAL,
            chase_advisory_flag INTEGER,
            current_day_value_kwd REAL,
            trailing_liquidity_context_value REAL,
            early_tier_flag INTEGER,
            dead_money_sessions INTEGER,
            flow_obv_slope_40 REAL,
            flow_anv_slope_40 REAL,
            flow_accumulation_divergence REAL,
            accumulation_context_ok INTEGER,
            participation_cap_pct REAL,
            pilot_size_fraction REAL,
            time_stop_sessions INTEGER,
            entry_tier TEXT,
            flow_evidence_snapshot TEXT,
            current_valid_reference_value REAL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_daily_term_row_symbol_date ON daily_term_row(symbol, trade_date)",
        """
        CREATE TABLE IF NOT EXISTS daily_state_snapshot (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            readiness_state TEXT,
            phase_state TEXT,
            base_reference_snapshot TEXT,
            intent_snapshot TEXT,
            avoid_state TEXT,
            risk_budget_state TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_daily_state_snapshot_symbol_date ON daily_state_snapshot(symbol, trade_date)",
        """
        CREATE TABLE IF NOT EXISTS execution_outcome_row (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            candidate_intent_state TEXT,
            execution_state TEXT,
            veto_plane TEXT,
            veto_reason TEXT,
            opened_trade_flag INTEGER,
            trade_id TEXT,
            chase_advisory_emitted INTEGER,
            chase_advisory_extension_pct REAL,
            entry_tier TEXT,
            dead_money_sessions INTEGER
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_execution_outcome_row_symbol_date ON execution_outcome_row(symbol, trade_date)",
        """
        CREATE TABLE IF NOT EXISTS ledger_daily_hash_chain (
            chain_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            previous_hash TEXT,
            chain_hash TEXT NOT NULL,
            emitted_at_utc TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_ledger_daily_hash_chain_date ON ledger_daily_hash_chain(trade_date)",
        """
        CREATE TRIGGER IF NOT EXISTS trg_daily_term_row_block_update
        BEFORE UPDATE ON daily_term_row
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: daily_term_row update blocked');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_daily_term_row_block_delete
        BEFORE DELETE ON daily_term_row
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: daily_term_row delete blocked');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_daily_state_snapshot_block_update
        BEFORE UPDATE ON daily_state_snapshot
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: daily_state_snapshot update blocked');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_daily_state_snapshot_block_delete
        BEFORE DELETE ON daily_state_snapshot
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: daily_state_snapshot delete blocked');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_execution_outcome_row_block_update
        BEFORE UPDATE ON execution_outcome_row
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: execution_outcome_row update blocked');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_execution_outcome_row_block_delete
        BEFORE DELETE ON execution_outcome_row
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: execution_outcome_row delete blocked');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_ledger_daily_hash_chain_block_update
        BEFORE UPDATE ON ledger_daily_hash_chain
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: ledger_daily_hash_chain update blocked');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_ledger_daily_hash_chain_block_delete
        BEFORE DELETE ON ledger_daily_hash_chain
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: ledger_daily_hash_chain delete blocked');
        END
        """,
    ]


def apply_schema_migration() -> dict[str, Any]:
    ddls = _ddl_sqlite()
    conn = _open_connection()
    try:
        for ddl in ddls:
            conn.execute(ddl)
        _ensure_daily_term_row_columns(conn)
        conn.commit()
        return {"dialect": "sqlite", "ddl_emitted": ddls}
    finally:
        conn.close()


def _ensure_daily_term_row_columns(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(daily_term_row)")
    existing = {str(r["name"]).lower() for r in cur.fetchall()}
    additions = [
        ("segment_day_index", "INTEGER"),
        ("readiness_transition_event", "TEXT"),
        ("readiness_transition_from_state", "TEXT"),
        ("readiness_transition_to_state", "TEXT"),
        ("segment_restart_flag", "INTEGER"),
        ("masked_context_flag", "INTEGER"),
        ("lookback_long_sessions", "INTEGER"),
        ("lookback_segment_sessions", "INTEGER"),
        ("lookback_fallback_sessions", "INTEGER"),
    ]
    for col, typ in additions:
        if col.lower() not in existing:
            conn.execute(f"ALTER TABLE daily_term_row ADD COLUMN {col} {typ}")


def append_row(table: str, row: dict[str, Any]) -> None:
    if table not in TABLE_TO_FIELDS:
        raise ValueError(f"Unsupported table: {table}")
    cols = TABLE_TO_FIELDS[table]
    missing = [c for c in cols if c not in row]
    if missing:
        raise ValueError(f"Missing required columns for {table}: {missing}")

    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    params = tuple(row[c] for c in cols)

    conn = _open_connection()
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def fetch_rows(table: str, trade_date: str) -> list[dict[str, Any]]:
    conn = _open_connection()
    try:
        cur = conn.execute(f"SELECT * FROM {table} WHERE trade_date = ? ORDER BY row_id", (trade_date,))
        rows = [dict(r) for r in cur.fetchall()]
        return rows
    finally:
        conn.close()


def get_table_columns(table: str) -> list[str]:
    conn = _open_connection()
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return [str(r["name"]) for r in cur.fetchall()]
    finally:
        conn.close()


def verify_update_delete_blocked(table: str, trade_date: str) -> dict[str, str]:
    conn = _open_connection()
    out: dict[str, str] = {}
    try:
        try:
            conn.execute(f"UPDATE {table} SET symbol = symbol WHERE trade_date = ?", (trade_date,))
            conn.commit()
            out["update"] = "FAILED_NOT_BLOCKED"
        except sqlite3.DatabaseError as e:
            out["update"] = f"BLOCKED:{e}"
            conn.rollback()

        try:
            conn.execute(f"DELETE FROM {table} WHERE trade_date = ?", (trade_date,))
            conn.commit()
            out["delete"] = "FAILED_NOT_BLOCKED"
        except sqlite3.DatabaseError as e:
            out["delete"] = f"BLOCKED:{e}"
            conn.rollback()
        return out
    finally:
        conn.close()


def _canonical_daily_content(trade_date: str) -> str:
    payload: dict[str, Any] = {"trade_date": trade_date, "tables": {}}
    for table in TABLES:
        payload["tables"][table] = fetch_rows(table, trade_date)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def emit_daily_hash_chain(trade_date: str, sidecar_path: Path) -> dict[str, Any]:
    canonical = _canonical_daily_content(trade_date)
    content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    conn = _open_connection()
    try:
        prev_row = conn.execute(
            "SELECT chain_hash FROM ledger_daily_hash_chain ORDER BY chain_id DESC LIMIT 1"
        ).fetchone()
        previous_hash = str(prev_row["chain_hash"]) if prev_row else "GENESIS"
        chain_hash = hashlib.sha256(f"{previous_hash}:{trade_date}:{content_hash}".encode("utf-8")).hexdigest()

        conn.execute(
            """
            INSERT INTO ledger_daily_hash_chain(trade_date, content_hash, previous_hash, chain_hash, emitted_at_utc)
            VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (trade_date, content_hash, previous_hash, chain_hash),
        )
        conn.commit()

        rows = conn.execute(
            "SELECT trade_date, content_hash, previous_hash, chain_hash, emitted_at_utc FROM ledger_daily_hash_chain ORDER BY chain_id"
        ).fetchall()
        sidecar_lines = [
            f"{r['chain_hash']}  trade_date={r['trade_date']} content_hash={r['content_hash']} prev={r['previous_hash']} emitted_at={r['emitted_at_utc']}"
            for r in rows
        ]
        sidecar_path.write_text("\n".join(sidecar_lines) + "\n", encoding="utf-8")
        return {
            "trade_date": trade_date,
            "content_hash": content_hash,
            "previous_hash": previous_hash,
            "chain_hash": chain_hash,
            "sidecar_path": str(sidecar_path),
        }
    finally:
        conn.close()

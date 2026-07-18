from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any


PREDICTION_FIELDS = [
    "prediction_id",
    "symbol",
    "prediction_date",
    "engine_baseline_id",
    "freeze_version_hash",
    "intent_state",
    "execution_state",
    "entry_tier",
    "reference_price",
    "base_reference",
    "avoid_state",
    "predicate_snapshot_json",
    "event_type",
    "source_run_key",
    "created_utc",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_prediction_id(
    *,
    source_run_key: str,
    symbol: str,
    prediction_date: str,
    event_type: str,
    intent_state: str,
    execution_state: str,
    entry_tier: str,
) -> str:
    payload = "|".join(
        [
            source_run_key,
            symbol.upper().strip(),
            prediction_date,
            event_type,
            intent_state,
            execution_state,
            entry_tier,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def ddl_sqlite() -> list[str]:
    return [
        """
        CREATE TABLE IF NOT EXISTS ee_v2_forward_predictions (
            prediction_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            engine_baseline_id TEXT NOT NULL,
            freeze_version_hash TEXT NOT NULL,
            intent_state TEXT NOT NULL,
            execution_state TEXT NOT NULL,
            entry_tier TEXT NOT NULL,
            reference_price REAL NOT NULL,
            base_reference TEXT NOT NULL,
            avoid_state TEXT NOT NULL,
            predicate_snapshot_json TEXT NOT NULL,
            event_type TEXT NOT NULL,
            source_run_key TEXT NOT NULL,
            created_utc TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_ee_v2_forward_predictions_symbol_date ON ee_v2_forward_predictions(symbol, prediction_date)",
        "CREATE INDEX IF NOT EXISTS idx_ee_v2_forward_predictions_event_type ON ee_v2_forward_predictions(event_type)",
        """
        CREATE TRIGGER IF NOT EXISTS trg_ee_v2_forward_predictions_block_update
        BEFORE UPDATE ON ee_v2_forward_predictions
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: ee_v2_forward_predictions update blocked');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_ee_v2_forward_predictions_block_delete
        BEFORE DELETE ON ee_v2_forward_predictions
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: ee_v2_forward_predictions delete blocked');
        END
        """,
    ]


def apply_schema_migration(conn: sqlite3.Connection) -> None:
    for ddl in ddl_sqlite():
        conn.execute(ddl)
    conn.commit()


class ForwardPredictionLedger:
    """Append-only writer for execution-relevant Eagle Eye v2 predictions."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        apply_schema_migration(conn)

    def append_prediction(
        self,
        *,
        symbol: str,
        prediction_date: str,
        engine_baseline_id: str,
        freeze_version_hash: str,
        intent_state: str,
        execution_state: str,
        entry_tier: str,
        reference_price: float,
        base_reference: dict[str, Any] | str,
        avoid_state: str,
        predicate_snapshot: dict[str, Any],
        event_type: str,
        source_run_key: str,
        created_utc: str | None = None,
    ) -> str:
        prediction_id = make_prediction_id(
            source_run_key=source_run_key,
            symbol=symbol,
            prediction_date=prediction_date,
            event_type=event_type,
            intent_state=intent_state,
            execution_state=execution_state,
            entry_tier=entry_tier,
        )
        base_reference_text = (
            base_reference
            if isinstance(base_reference, str)
            else json.dumps(base_reference, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        )
        row = {
            "prediction_id": prediction_id,
            "symbol": symbol.upper().strip(),
            "prediction_date": prediction_date,
            "engine_baseline_id": engine_baseline_id,
            "freeze_version_hash": freeze_version_hash,
            "intent_state": intent_state,
            "execution_state": execution_state,
            "entry_tier": entry_tier,
            "reference_price": float(reference_price),
            "base_reference": base_reference_text,
            "avoid_state": avoid_state,
            "predicate_snapshot_json": json.dumps(
                predicate_snapshot,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "event_type": event_type,
            "source_run_key": source_run_key,
            "created_utc": created_utc or _utc_now(),
        }
        placeholders = ", ".join(["?"] * len(PREDICTION_FIELDS))
        self.conn.execute(
            f"INSERT INTO ee_v2_forward_predictions ({', '.join(PREDICTION_FIELDS)}) VALUES ({placeholders})",
            tuple(row[field] for field in PREDICTION_FIELDS),
        )
        self.conn.commit()
        return prediction_id


def fetch_predictions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT *
        FROM ee_v2_forward_predictions
        ORDER BY symbol, prediction_date, event_type, prediction_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def verify_update_delete_blocked(conn: sqlite3.Connection, prediction_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        conn.execute(
            "UPDATE ee_v2_forward_predictions SET symbol = symbol WHERE prediction_id = ?",
            (prediction_id,),
        )
        conn.commit()
        out["update"] = "FAILED_NOT_BLOCKED"
    except sqlite3.DatabaseError as exc:
        out["update"] = str(exc)
        conn.rollback()
    try:
        conn.execute("DELETE FROM ee_v2_forward_predictions WHERE prediction_id = ?", (prediction_id,))
        conn.commit()
        out["delete"] = "FAILED_NOT_BLOCKED"
    except sqlite3.DatabaseError as exc:
        out["delete"] = str(exc)
        conn.rollback()
    return out
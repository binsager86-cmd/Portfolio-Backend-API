from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


GRADE_FIELDS = [
    "prediction_id",
    "symbol",
    "prediction_date",
    "return_20",
    "return_60",
    "return_120",
    "mfe_120",
    "materialization_verdict",
    "grade_status",
    "grade_date",
    "grader_version",
    "sealed_data_last_date",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ddl_sqlite() -> list[str]:
    return [
        """
        CREATE TABLE IF NOT EXISTS ee_v2_prediction_grades (
            grade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            return_20 REAL,
            return_60 REAL,
            return_120 REAL,
            mfe_120 REAL,
            materialization_verdict TEXT NOT NULL,
            grade_status TEXT NOT NULL,
            grade_date TEXT NOT NULL,
            grader_version TEXT NOT NULL,
            sealed_data_last_date TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_ee_v2_prediction_grades_prediction_id ON ee_v2_prediction_grades(prediction_id)",
        "CREATE INDEX IF NOT EXISTS idx_ee_v2_prediction_grades_symbol_date ON ee_v2_prediction_grades(symbol, prediction_date)",
        """
        CREATE TRIGGER IF NOT EXISTS trg_ee_v2_prediction_grades_block_update
        BEFORE UPDATE ON ee_v2_prediction_grades
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: ee_v2_prediction_grades update blocked');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_ee_v2_prediction_grades_block_delete
        BEFORE DELETE ON ee_v2_prediction_grades
        BEGIN
            SELECT RAISE(ABORT, 'append-only table: ee_v2_prediction_grades delete blocked');
        END
        """,
    ]


def apply_schema_migration(conn: sqlite3.Connection) -> None:
    for ddl in ddl_sqlite():
        conn.execute(ddl)
    conn.commit()


def open_prediction_reader(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def read_predictions_read_only(db_path: Path) -> list[dict[str, Any]]:
    conn = open_prediction_reader(db_path)
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM ee_v2_forward_predictions
            ORDER BY symbol, prediction_date, event_type, prediction_id
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def apply_grades(
    *,
    predictions_db_path: Path,
    grades_conn: sqlite3.Connection,
    sealed_ohlcv_by_symbol: dict[str, list[dict[str, Any]]],
    horizons: Iterable[int] = (20, 60, 120),
    grade_date: str | None = None,
    grader_version: str = "R14G_PREDICTION_GRADER_V1",
) -> list[dict[str, Any]]:
    """Grade predictions while opening the prediction table through SQLite mode=ro."""

    apply_schema_migration(grades_conn)
    horizons = tuple(horizons)
    predictions = read_predictions_read_only(predictions_db_path)
    emitted: list[dict[str, Any]] = []
    grade_date = grade_date or _utc_now()

    for prediction in predictions:
        symbol = str(prediction["symbol"]).upper().strip()
        rows = sealed_ohlcv_by_symbol.get(symbol, [])
        date_to_index = {str(row["trade_date"]): idx for idx, row in enumerate(rows)}
        prediction_date = str(prediction["prediction_date"])
        index = date_to_index.get(prediction_date)
        reference_price = float(prediction.get("reference_price") or 0.0)
        sealed_last = str(rows[-1]["trade_date"]) if rows else "NONE"

        returns: dict[int, float | None] = {h: None for h in horizons}
        mfe_120: float | None = None
        verdict = "PENDING_HORIZON"
        grade_status = "PENDING_HORIZON"

        if index is not None and reference_price > 0:
            for horizon in horizons:
                horizon_index = index + horizon
                if horizon_index < len(rows):
                    returns[horizon] = (float(rows[horizon_index]["close"]) / reference_price) - 1.0
            if index + 120 < len(rows):
                forward_highs = [float(row["high"] or row["close"] or 0.0) for row in rows[index + 1 : index + 121]]
                if forward_highs:
                    mfe_120 = (max(forward_highs) / reference_price) - 1.0
                    verdict = "MATERIALIZED" if mfe_120 >= 0.20 else "NOT_MATERIALIZED"
                    grade_status = "GRADED"

        row = {
            "prediction_id": prediction["prediction_id"],
            "symbol": symbol,
            "prediction_date": prediction_date,
            "return_20": returns.get(20),
            "return_60": returns.get(60),
            "return_120": returns.get(120),
            "mfe_120": mfe_120,
            "materialization_verdict": verdict,
            "grade_status": grade_status,
            "grade_date": grade_date,
            "grader_version": grader_version,
            "sealed_data_last_date": sealed_last,
        }
        placeholders = ", ".join(["?"] * len(GRADE_FIELDS))
        grades_conn.execute(
            f"INSERT INTO ee_v2_prediction_grades ({', '.join(GRADE_FIELDS)}) VALUES ({placeholders})",
            tuple(row[field] for field in GRADE_FIELDS),
        )
        emitted.append(row)

    grades_conn.commit()
    return emitted


def verify_prediction_reader_cannot_write(db_path: Path) -> dict[str, str]:
    conn = open_prediction_reader(db_path)
    try:
        try:
            conn.execute("INSERT INTO ee_v2_forward_predictions (prediction_id) VALUES ('SHOULD_FAIL')")
            return {"prediction_reader_write_attempt": "FAILED_NOT_BLOCKED"}
        except sqlite3.DatabaseError as exc:
            return {"prediction_reader_write_attempt": str(exc)}
    finally:
        conn.close()
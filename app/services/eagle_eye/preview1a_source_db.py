from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.database import query_all


@dataclass(frozen=True)
class SourceStreamSpec:
    source_table: str
    primary_key: tuple[str, str]
    stream_type: str
    adjustment_version: str
    corporate_action_ledger_version: str
    dataset_id: str


def market_date_to_utc_epoch(date_text: str) -> int:
    day = datetime.strptime(date_text.strip()[:10], "%Y-%m-%d").date()
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def _open_readonly_sqlite(path: str) -> sqlite3.Connection:
    p = Path(path).resolve()
    uri = f"file:{p.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def extract_bars_from_source(
    source_db_path: str,
    stream: SourceStreamSpec,
    symbols: list[str],
    warmup_data_start: int,
    milestone_t: int,
) -> list[dict[str, Any]]:
    if not symbols:
        return []
    placeholders = ",".join(["?"] * len(symbols))
    sql = (
        f"SELECT symbol, trade_date, open, high, low, close, volume, value_kwd, "
        f"COALESCE(adjustment_status, 'raw_unadjusted') AS adjustment_status, "
        f"COALESCE(corporate_action_version, 'none') AS corporate_action_version "
        f"FROM {stream.source_table} "
        f"WHERE symbol IN ({placeholders}) AND trade_date BETWEEN ? AND ? "
        f"ORDER BY symbol, trade_date"
    )
    params = [s.upper() for s in symbols] + [int(warmup_data_start), int(milestone_t)]
    out: list[dict[str, Any]] = []

    settings = get_settings()
    if settings.use_postgres and not str(source_db_path or "").strip():
        rows = query_all(sql, tuple(params))
        for row in rows:
            out.append(dict(row))
        return out

    with _open_readonly_sqlite(source_db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        for row in rows:
            out.append(dict(row))
    return out


def initialize_output_ohlcv(output_db_path: str) -> None:
    with sqlite3.connect(output_db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ee_ohlcv (
                symbol TEXT NOT NULL,
                trade_date INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                raw_close REAL,
                adjusted_close REAL,
                volume REAL,
                value_kwd REAL,
                value_unit TEXT,
                source TEXT NOT NULL DEFAULT 'preview_source_copy',
                source_type TEXT,
                source_ref TEXT,
                data_environment TEXT,
                ingestion_run_id TEXT,
                request_parameters_hash TEXT,
                payload_hash TEXT,
                code_commit TEXT,
                parser_version TEXT,
                synthetic_flag INTEGER NOT NULL DEFAULT 0,
                adjustment_status TEXT NOT NULL DEFAULT 'raw_unadjusted',
                corporate_action_version TEXT NOT NULL DEFAULT 'none',
                ingested_at INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(symbol, trade_date)
            )
            """
        )
        conn.execute("DELETE FROM ee_ohlcv")
        conn.commit()


def load_bars_into_output(output_db_path: str, bars: list[dict[str, Any]]) -> int:
    if not bars:
        return 0
    with sqlite3.connect(output_db_path) as conn:
        conn.executemany(
            """
            INSERT INTO ee_ohlcv (
                symbol, trade_date, open, high, low, close, raw_close, adjusted_close,
                volume, value_kwd, value_unit, source, source_type, source_ref,
                data_environment, ingestion_run_id, request_parameters_hash, payload_hash,
                code_commit, parser_version, synthetic_flag, adjustment_status,
                corporate_action_version, ingested_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, trade_date) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                raw_close=excluded.raw_close,
                adjusted_close=excluded.adjusted_close,
                volume=excluded.volume,
                value_kwd=excluded.value_kwd,
                value_unit=excluded.value_unit,
                source=excluded.source,
                source_type=excluded.source_type,
                source_ref=excluded.source_ref,
                data_environment=excluded.data_environment,
                adjustment_status=excluded.adjustment_status,
                corporate_action_version=excluded.corporate_action_version
            """,
            [
                (
                    str(b["symbol"]).upper(),
                    int(b["trade_date"]),
                    float(b.get("open") or 0.0),
                    float(b.get("high") or 0.0),
                    float(b.get("low") or 0.0),
                    float(b.get("close") or 0.0),
                    float(b.get("close") or 0.0),
                    float(b.get("close") or 0.0),
                    float(b.get("volume") or 0.0),
                    float(b.get("value_kwd") or 0.0),
                    "kwd",
                    "preview_source_copy",
                    "preview_copy",
                    "preview:source",
                    "preview",
                    "preview",
                    "preview",
                    "preview",
                    "preview",
                    "preview",
                    0,
                    str(b.get("adjustment_status") or "raw_unadjusted"),
                    str(b.get("corporate_action_version") or "none"),
                    0,
                )
                for b in bars
            ],
        )
        conn.commit()
        return len(bars)

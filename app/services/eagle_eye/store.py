"""
Eagle Eye — Persistent DB store.

Creates and manages 4 tables in the shared SQLite/PostgreSQL database:
  - ee_ohlcv_cache      : daily OHLCV bars per ticker
  - ee_dna_profiles     : behavioral DNA JSON blobs
  - ee_ratings_cache    : current scanner ratings (one row per ticker)
  - ee_compute_log      : audit trail for pipeline runs

All DDL uses CREATE TABLE/INDEX IF NOT EXISTS — fully idempotent.
Single-row writes use the backend's exec_sql helper (?-style params).
Bulk OHLCV writes bypass the proxy layer for performance:
  - SQLite: raw sqlite3.executemany with INSERT OR REPLACE
  - PostgreSQL: delete-then-insert via pandas to_sql
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Table creation — idempotent, additive only
# ---------------------------------------------------------------------------

def ensure_tables() -> None:
    """Create all Eagle Eye tables if they do not already exist."""
    from app.core.database import exec_sql

    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS ee_ohlcv_cache (
            ticker       TEXT    NOT NULL,
            bar_date     TEXT    NOT NULL,
            open         REAL,
            high         REAL,
            low          REAL,
            close        REAL,
            volume       REAL,
            turnover_kwd REAL,
            fetched_at   INTEGER,
            PRIMARY KEY (ticker, bar_date)
        )
        """,
        (),
    )

    exec_sql(
        "CREATE INDEX IF NOT EXISTS idx_ee_ohlcv_td ON ee_ohlcv_cache(ticker, bar_date)",
        (),
    )

    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS ee_dna_profiles (
            ticker           TEXT PRIMARY KEY,
            dna_json         TEXT    NOT NULL,
            total_events     INTEGER DEFAULT 0,
            dominant_pattern TEXT,
            computed_at      TEXT,
            updated_at       INTEGER
        )
        """,
        (),
    )

    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS ee_ratings_cache (
            ticker               TEXT PRIMARY KEY,
            name_en              TEXT,
            sector               TEXT,
            stage                TEXT,
            rating               TEXT,
            confidence           REAL,
            thesis               TEXT,
            entry_primary        REAL,
            entry_aggressive     REAL,
            entry_conservative   REAL,
            stop_loss            REAL,
            tp1                  REAL,
            tp1_probability      REAL,
            tp2                  REAL,
            tp2_probability      REAL,
            tp3                  REAL,
            tp3_probability      REAL,
            last_price           REAL,
            supports_json        TEXT,
            resistances_json     TEXT,
            signals_json         TEXT,
            indicators_json      TEXT,
            days_of_history      INTEGER,
            computed_at          TEXT,
            updated_at           INTEGER
        )
        """,
        (),
    )

    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS ee_compute_log (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            run_type TEXT,
            ticker   TEXT,
            status   TEXT,
            message  TEXT,
            run_at   INTEGER
        )
        """,
        (),
    )


# ---------------------------------------------------------------------------
# OHLCV helpers
# ---------------------------------------------------------------------------

def save_ohlcv(ticker: str, df: pd.DataFrame) -> int:
    """
    Bulk-upsert OHLCV rows for *ticker* into ee_ohlcv_cache.

    *df* must be indexed by datetime (DatetimeIndex) with columns:
    open, high, low, close, volume, turnover_kwd.

    Returns the number of rows written.
    """
    if df is None or df.empty:
        return 0

    from app.core.config import get_settings

    settings = get_settings()
    ts = int(time.time())
    upper = ticker.upper()

    rows = []
    for dt_idx, row in df.iterrows():
        bar_d = str(dt_idx.date()) if hasattr(dt_idx, "date") else str(dt_idx)[:10]
        rows.append((
            upper, bar_d,
            _f(row.get("open")), _f(row.get("high")), _f(row.get("low")),
            _f(row.get("close")), _f(row.get("volume")), _f(row.get("turnover_kwd")),
            ts,
        ))

    if not rows:
        return 0

    if not settings.use_postgres:
        import sqlite3
        conn = sqlite3.connect(settings.database_abs_path, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executemany(
                """
                INSERT OR REPLACE INTO ee_ohlcv_cache
                    (ticker, bar_date, open, high, low, close, volume, turnover_kwd, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()
    else:
        # PostgreSQL: remove affected dates then bulk-insert via pandas
        from app.core.database import exec_sql, engine as db_engine

        dates = [r[1] for r in rows]
        placeholders = ", ".join(["?"] * len(dates))
        exec_sql(
            f"DELETE FROM ee_ohlcv_cache WHERE ticker = ? AND bar_date IN ({placeholders})",
            tuple([upper] + dates),
        )
        frame = pd.DataFrame(
            rows,
            columns=[
                "ticker", "bar_date", "open", "high", "low",
                "close", "volume", "turnover_kwd", "fetched_at",
            ],
        )
        frame.to_sql(
            "ee_ohlcv_cache", db_engine,
            if_exists="append", index=False,
            method="multi", chunksize=500,
        )

    return len(rows)


def load_ohlcv(
    ticker: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """
    Load cached OHLCV rows for *ticker* from the DB.

    Returns a DataFrame indexed by datetime with columns:
    open, high, low, close, volume, turnover_kwd.
    Empty DataFrame (same columns) if no data found.
    """
    sql = (
        "SELECT bar_date, open, high, low, close, volume, turnover_kwd "
        "FROM ee_ohlcv_cache WHERE ticker = ?"
    )
    params: list = [ticker.upper()]

    if start:
        sql += " AND bar_date >= ?"
        params.append(start.isoformat())
    if end:
        sql += " AND bar_date <= ?"
        params.append(end.isoformat())
    sql += " ORDER BY bar_date"

    from app.core.database import query_all

    rows = query_all(sql, tuple(params))
    if not rows:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "turnover_kwd"]
        )

    data = {
        "date": [r["bar_date"] for r in rows],
        "open": [r["open"] for r in rows],
        "high": [r["high"] for r in rows],
        "low": [r["low"] for r in rows],
        "close": [r["close"] for r in rows],
        "volume": [r["volume"] for r in rows],
        "turnover_kwd": [r["turnover_kwd"] for r in rows],
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def get_latest_ohlcv_date(ticker: str) -> Optional[date]:
    """Return the most recent bar_date stored for *ticker*, or None."""
    from app.core.database import query_one

    row = query_one(
        "SELECT MAX(bar_date) AS max_date FROM ee_ohlcv_cache WHERE ticker = ?",
        (ticker.upper(),),
    )
    if row is None:
        return None
    val = row["max_date"]
    if not val:
        return None
    try:
        return date.fromisoformat(str(val))
    except Exception:
        return None


def list_tickers_with_ohlcv() -> List[str]:
    """Return all distinct tickers that have data in ee_ohlcv_cache."""
    from app.core.database import query_all

    rows = query_all(
        "SELECT DISTINCT ticker FROM ee_ohlcv_cache ORDER BY ticker", ()
    )
    return [r["ticker"] for r in rows] if rows else []


# ---------------------------------------------------------------------------
# DNA helpers
# ---------------------------------------------------------------------------

def save_dna(
    ticker: str,
    dna_dict: dict,
    total_events: int = 0,
    dominant_pattern: Optional[str] = None,
) -> None:
    """Upsert a DNA profile for *ticker*."""
    from app.core.database import exec_sql

    exec_sql(
        """
        INSERT OR REPLACE INTO ee_dna_profiles
            (ticker, dna_json, total_events, dominant_pattern, computed_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            ticker.upper(),
            json.dumps(dna_dict),
            total_events,
            dominant_pattern,
            date.today().isoformat(),
            int(time.time()),
        ),
    )


def load_dna(ticker: str) -> Optional[dict]:
    """Load and deserialize the DNA JSON blob for *ticker*, or None."""
    from app.core.database import query_one

    row = query_one(
        "SELECT dna_json FROM ee_dna_profiles WHERE ticker = ?",
        (ticker.upper(),),
    )
    if row is None:
        return None
    try:
        return json.loads(row["dna_json"])
    except Exception:
        return None


def list_tickers_with_dna() -> List[str]:
    """Return all tickers that have a stored DNA profile."""
    from app.core.database import query_all

    rows = query_all("SELECT ticker FROM ee_dna_profiles ORDER BY ticker", ())
    return [r["ticker"] for r in rows] if rows else []


# ---------------------------------------------------------------------------
# Ratings helpers
# ---------------------------------------------------------------------------

def save_rating(
    ticker: str,
    name_en: str,
    sector: str,
    result: dict,
) -> None:
    """
    Upsert one computed rating row into ee_ratings_cache.

    *result* is the dict produced by the rating engine (same shape as
    ``_run_analysis`` returns in the eagle_eye router).
    """
    from app.core.database import exec_sql

    et = result.get("entry") or {}
    ind = result.get("indicators") or {}

    exec_sql(
        """
        INSERT OR REPLACE INTO ee_ratings_cache (
            ticker, name_en, sector, stage, rating, confidence, thesis,
            entry_primary, entry_aggressive, entry_conservative,
            stop_loss, tp1, tp1_probability, tp2, tp2_probability, tp3, tp3_probability,
            last_price, supports_json, resistances_json, signals_json, indicators_json,
            days_of_history, computed_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            ticker.upper(),
            name_en,
            sector,
            result.get("stage"),
            result.get("rating"),
            result.get("confidence"),
            result.get("thesis"),
            _f(et.get("entry_primary")),
            _f(et.get("entry_aggressive")),
            _f(et.get("entry_conservative")),
            _f(et.get("stop_loss")),
            _f(et.get("tp1")),
            _f(et.get("tp1_probability")),
            _f(et.get("tp2")),
            _f(et.get("tp2_probability")),
            _f(et.get("tp3")),
            _f(et.get("tp3_probability")),
            _f(ind.get("close")),
            json.dumps(result.get("supports") or []),
            json.dumps(result.get("resistances") or []),
            json.dumps([]),
            json.dumps({k: _j(v) for k, v in ind.items()}),
            result.get("days_of_history"),
            result.get("computed_at", date.today().isoformat()),
            int(time.time()),
        ),
    )


def load_all_ratings() -> List[dict]:
    """
    Load all rows from ee_ratings_cache, ordered by confidence descending.
    Fast path for the scanner endpoint.
    """
    from app.core.database import query_all

    rows = query_all(
        """
        SELECT ticker, name_en, sector, stage, rating, confidence, thesis,
               entry_primary, stop_loss, tp1, last_price, computed_at
        FROM   ee_ratings_cache
        ORDER  BY confidence DESC
        """,
        (),
    )
    return [dict(r.items()) for r in rows] if rows else []


def load_rating(ticker: str) -> Optional[dict]:
    """Load the full rating row for a single ticker, or None."""
    from app.core.database import query_one

    row = query_one(
        """
        SELECT ticker, name_en, sector, stage, rating, confidence, thesis,
               entry_primary, entry_aggressive, entry_conservative,
               stop_loss, tp1, tp1_probability, tp2, tp2_probability,
               tp3, tp3_probability, last_price,
               supports_json, resistances_json, indicators_json,
               days_of_history, computed_at
        FROM   ee_ratings_cache
        WHERE  ticker = ?
        """,
        (ticker.upper(),),
    )
    if row is None:
        return None
    d = dict(row.items())
    for key in ("supports_json", "resistances_json", "indicators_json"):
        if d.get(key):
            try:
                d[key] = json.loads(d[key])
            except Exception:
                d[key] = []
    return d


# ---------------------------------------------------------------------------
# Compute log
# ---------------------------------------------------------------------------

def log_compute(
    run_type: str,
    ticker: Optional[str],
    status: str,
    message: str = "",
) -> None:
    """Append a row to ee_compute_log. Never raises."""
    try:
        from app.core.database import exec_sql

        exec_sql(
            "INSERT INTO ee_compute_log (run_type, ticker, status, message, run_at) VALUES (?,?,?,?,?)",
            (run_type, ticker, status, message[:500], int(time.time())),
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Private numeric helpers
# ---------------------------------------------------------------------------

def _f(v: Any) -> Optional[float]:
    """Safely coerce *v* to float; return None for NaN / Inf / non-numeric."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _j(v: Any) -> Any:
    """Make *v* JSON-serializable (replaces NaN/Inf with None)."""
    try:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if isinstance(v, (bool, int, str, type(None))):
            return v
        return float(v)
    except Exception:
        return None

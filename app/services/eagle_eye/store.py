"""
Eagle Eye — Persistent DB store.

Creates and manages 5 tables in the shared SQLite/PostgreSQL database:
  - ee_ohlcv_cache      : daily OHLCV bars per ticker
  - ee_dna_profiles     : behavioral DNA JSON blobs
  - ee_ratings_cache    : current scanner ratings (one row per ticker)
    - ratings_history     : daily point-in-time ratings snapshots
  - ee_compute_log      : audit trail for pipeline runs

All DDL uses CREATE TABLE/INDEX IF NOT EXISTS — fully idempotent.
Single-row writes use portable ``ON CONFLICT`` upserts via the backend's
``exec_sql`` helper (?-style params) so both SQLite and PostgreSQL work.
Bulk OHLCV writes bypass the proxy layer for performance:
  - SQLite: raw sqlite3.executemany with INSERT OR REPLACE
  - PostgreSQL: delete-then-insert via pandas to_sql
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import date, datetime
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
            market_tier          TEXT,
            stage                TEXT,
            rating               TEXT,
            confidence           REAL,
            ml_score             REAL,
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
            computed_date        TEXT,
            run_id               TEXT,
            run_started_at       TEXT,
            code_fingerprint     TEXT,
            updated_at           INTEGER
        )
        """,
        (),
    )

    # Additive migration: volume_context_json (added Phase 2)
    from app.core.database import add_column_if_missing as _acim
    _acim("ee_ratings_cache", "market_tier", "TEXT")
    _acim("ee_ratings_cache", "volume_context_json", "TEXT")
    _acim("ee_ratings_cache", "ml_score", "REAL")
    _acim("ee_ratings_cache", "computed_date", "TEXT")
    _acim("ee_ratings_cache", "run_id", "TEXT")
    _acim("ee_ratings_cache", "run_started_at", "TEXT")
    _acim("ee_ratings_cache", "code_fingerprint", "TEXT")
    _acim("ee_ratings_cache", "risky_near_resistance", "INTEGER")
    _acim("ee_ratings_cache", "risk_reward_ratio", "REAL")

    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS ratings_history (
            ticker               TEXT NOT NULL,
            computed_date        TEXT NOT NULL,
            name_en              TEXT,
            sector               TEXT,
            market_tier          TEXT,
            stage                TEXT,
            rating               TEXT,
            confidence           REAL,
            thesis               TEXT,
            entry_primary        REAL,
            stop_loss            REAL,
            tp1                  REAL,
            tp2                  REAL,
            tp3                  REAL,
            last_price           REAL,
            signals_json         TEXT,
            indicators_json      TEXT,
            volume_context_json  TEXT,
            computed_at          TEXT,
            updated_at           INTEGER,
            PRIMARY KEY (ticker, computed_date)
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


def get_trailing_ohlcv_start_date(ticker: str, trailing_sessions: int = 10) -> Optional[date]:
    """
    Return the earliest date within the trailing *trailing_sessions* cached bars
    for *ticker*.

    This is used by ingestion to re-fetch and overwrite recent bars on each run,
    capturing late exchange corrections without re-downloading full history.
    """
    from app.core.database import query_all

    n = max(1, int(trailing_sessions))
    rows = query_all(
        f"""
        SELECT bar_date
        FROM ee_ohlcv_cache
        WHERE ticker = ?
        ORDER BY bar_date DESC
        LIMIT {n}
        """,
        (ticker.upper(),),
    )
    if not rows:
        return None

    dates: List[date] = []
    for row in rows:
        raw = row.get("bar_date") if hasattr(row, "get") else row["bar_date"]
        if not raw:
            continue
        try:
            dates.append(date.fromisoformat(str(raw)))
        except Exception:
            continue

    if not dates:
        return None
    return min(dates)


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

    computed_at = date.today().isoformat()
    updated_at = int(time.time())

    exec_sql(
        """
        INSERT INTO ee_dna_profiles
            (ticker, dna_json, total_events, dominant_pattern, computed_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (ticker) DO UPDATE SET
            dna_json = excluded.dna_json,
            total_events = excluded.total_events,
            dominant_pattern = excluded.dominant_pattern,
            computed_at = excluded.computed_at,
            updated_at = excluded.updated_at
        """,
        (
            ticker.upper(),
            json.dumps(dna_dict),
            total_events,
            dominant_pattern,
            computed_at,
            updated_at,
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
    rr_cached = _f(ind.get("risk_reward_ratio"))
    if rr_cached is None:
        rr_cached = 0.0
    computed_at = result.get("computed_at")
    if not computed_at:
        computed_at = datetime.now().isoformat(timespec="seconds")
    computed_date = result.get("computed_date")
    if not computed_date:
        computed_date = str(computed_at)[:10]

    exec_sql(
        """
        INSERT INTO ee_ratings_cache (
            ticker, name_en, sector, market_tier, stage, rating, confidence, ml_score, thesis,
            entry_primary, entry_aggressive, entry_conservative,
            stop_loss, tp1, tp1_probability, tp2, tp2_probability, tp3, tp3_probability,
            last_price, supports_json, resistances_json, signals_json, indicators_json,
            days_of_history, computed_at, computed_date, run_id, run_started_at, code_fingerprint,
            updated_at, volume_context_json, risky_near_resistance, risk_reward_ratio
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT (ticker) DO UPDATE SET
            name_en = excluded.name_en,
            sector = excluded.sector,
            market_tier = excluded.market_tier,
            stage = excluded.stage,
            rating = excluded.rating,
            confidence = excluded.confidence,
            ml_score = excluded.ml_score,
            thesis = excluded.thesis,
            entry_primary = excluded.entry_primary,
            entry_aggressive = excluded.entry_aggressive,
            entry_conservative = excluded.entry_conservative,
            stop_loss = excluded.stop_loss,
            tp1 = excluded.tp1,
            tp1_probability = excluded.tp1_probability,
            tp2 = excluded.tp2,
            tp2_probability = excluded.tp2_probability,
            tp3 = excluded.tp3,
            tp3_probability = excluded.tp3_probability,
            last_price = excluded.last_price,
            supports_json = excluded.supports_json,
            resistances_json = excluded.resistances_json,
            signals_json = excluded.signals_json,
            indicators_json = excluded.indicators_json,
            days_of_history = excluded.days_of_history,
            computed_at = excluded.computed_at,
            computed_date = excluded.computed_date,
            run_id = excluded.run_id,
            run_started_at = excluded.run_started_at,
            code_fingerprint = excluded.code_fingerprint,
            updated_at = excluded.updated_at,
            volume_context_json = excluded.volume_context_json,
            risky_near_resistance = excluded.risky_near_resistance,
            risk_reward_ratio = excluded.risk_reward_ratio
        """,
        (
            ticker.upper(),
            name_en,
            sector,
            result.get("market_tier"),
            result.get("stage"),
            result.get("rating"),
            result.get("confidence"),
            _f(result.get("ml_score")),
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
            computed_at,
            computed_date,
            result.get("run_id"),
            result.get("run_started_at"),
            result.get("code_fingerprint"),
            int(time.time()),
            json.dumps(result.get("volume_context") or {}),
            int(result.get("risky_near_resistance", False)),
            rr_cached,
        ),
    )


def load_all_ratings(
    min_confidence: float = 0.0,
    limit: int = 500,
    computed_at: Optional[str] = None,
) -> List[dict]:
    """
    Load rows from ee_ratings_cache ordered by confidence descending.

    min_confidence and limit are pushed to SQL so the DB does the work
    instead of loading every row and filtering in Python.

    By default, this returns the latest row per ticker regardless of date.
    This keeps the scanner populated even if an in-progress recompute is
    interrupted before all tickers are refreshed. When *computed_at* is
    provided, rows are pinned to that date for explicit historical/day views.
    """
    from app.core.database import query_all

    sql = """
           SELECT ticker, name_en, sector, market_tier, stage, rating, confidence, ml_score, thesis,
               entry_primary, stop_loss, tp1, last_price, computed_at, computed_date,
               volume_context_json, indicators_json, risky_near_resistance, risk_reward_ratio
        FROM   ee_ratings_cache
        WHERE  confidence >= ?
    """
    params: List[object] = [float(min_confidence)]

    if computed_at:
        target_date = computed_at
        if "T" in target_date:
            target_date = target_date[:10]
        sql += """
          AND  (computed_date = ? OR computed_at = ? OR computed_at LIKE ?)
        """
        params.extend([target_date, target_date, f"{target_date}%"])

    sql += """
        ORDER  BY confidence DESC
        LIMIT  ?
    """
    params.append(int(limit))

    rows = query_all(sql, tuple(params))
    if not rows:
        return []

    def _safe_float(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return f

    result = []
    for r in rows:
        d = dict(r.items())
        vc_raw = d.pop("volume_context_json", None)
        indicators_raw = d.pop("indicators_json", None)
        try:
            d["volume_context"] = json.loads(vc_raw) if vc_raw else {}
        except Exception:
            d["volume_context"] = {}
        indicators = {}
        if indicators_raw:
            try:
                indicators = json.loads(indicators_raw)
            except Exception:
                indicators = {}
        if isinstance(indicators, dict):
            from app.services.eagle_eye.scoring.recommendation_engine import compute_continue_rising

            d.update(compute_continue_rising(indicators, str(d.get("stage") or "")))

            rr = _safe_float(d.get("risk_reward_ratio"))
            if rr is None:
                rr = _safe_float(indicators.get("risk_reward_ratio"))
            d["risk_reward_ratio"] = rr
            d["risky_near_resistance"] = bool(d.get("risky_near_resistance", False))
        else:
            d["risk_reward_ratio"] = None
            d["risky_near_resistance"] = False
        result.append(d)
    return result


def load_rating(ticker: str) -> Optional[dict]:
    """Load the full rating row for a single ticker, or None."""
    from app.core.database import query_one

    row = query_one(
        """
            SELECT ticker, name_en, sector, market_tier, stage, rating, confidence, ml_score, thesis,
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
    indicators = d.get("indicators_json")
    if isinstance(indicators, dict):
        from app.services.eagle_eye.scoring.recommendation_engine import compute_continue_rising

        d.update(compute_continue_rising(indicators, str(d.get("stage") or "")))
    return d


def snapshot_ratings_history(computed_date: Optional[str] = None) -> int:
    """Upsert today's current ratings cache into the append-only ratings history table."""
    from app.core.database import exec_sql, query_all

    ensure_tables()
    snapshot_date = computed_date or date.today().isoformat()
    rows = query_all(
        """
        SELECT ticker, name_en, sector, market_tier, stage, rating, confidence, thesis,
               entry_primary, stop_loss, tp1, tp2, tp3, last_price,
               signals_json, indicators_json, volume_context_json, computed_at
        FROM ee_ratings_cache
        """,
        (),
    )
    if not rows:
        return 0

    written = 0
    updated_at = int(time.time())
    for row in rows:
        exec_sql(
            """
            INSERT INTO ratings_history (
                ticker, computed_date, name_en, sector, market_tier,
                stage, rating, confidence, thesis,
                entry_primary, stop_loss, tp1, tp2, tp3, last_price,
                signals_json, indicators_json, volume_context_json,
                computed_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (ticker, computed_date) DO UPDATE SET
                name_en = excluded.name_en,
                sector = excluded.sector,
                market_tier = excluded.market_tier,
                stage = excluded.stage,
                rating = excluded.rating,
                confidence = excluded.confidence,
                thesis = excluded.thesis,
                entry_primary = excluded.entry_primary,
                stop_loss = excluded.stop_loss,
                tp1 = excluded.tp1,
                tp2 = excluded.tp2,
                tp3 = excluded.tp3,
                last_price = excluded.last_price,
                signals_json = excluded.signals_json,
                indicators_json = excluded.indicators_json,
                volume_context_json = excluded.volume_context_json,
                computed_at = excluded.computed_at,
                updated_at = excluded.updated_at
            """,
            (
                row["ticker"],
                snapshot_date,
                row.get("name_en"),
                row.get("sector"),
                row.get("market_tier"),
                row.get("stage"),
                row.get("rating"),
                row.get("confidence"),
                row.get("thesis"),
                row.get("entry_primary"),
                row.get("stop_loss"),
                row.get("tp1"),
                row.get("tp2"),
                row.get("tp3"),
                row.get("last_price"),
                row.get("signals_json"),
                row.get("indicators_json"),
                row.get("volume_context_json"),
                row.get("computed_at") or snapshot_date,
                updated_at,
            ),
        )
        written += 1
    return written


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

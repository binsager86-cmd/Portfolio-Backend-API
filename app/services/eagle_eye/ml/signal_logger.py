"""
ml/signal_logger.py — Addendum A.4: Considered-signal logger.

Logs every signal evaluated by the rule engine, regardless of whether it led to
an entry.  Records full feature snapshot + skip reason.  Realized outcomes are
filled by a forward-only daily job — NEVER at signal time (no look-ahead).

Table layout:
    considered_signals(
        signal_id               TEXT PRIMARY KEY,
        stock_ticker            TEXT NOT NULL,
        signal_date             TEXT NOT NULL,
        rule_score              REAL NOT NULL,
        would_have_entered      INTEGER NOT NULL,   -- 1/0
        skip_reason             TEXT,               -- NULL if would_have_entered=1
        full_feature_snapshot_json TEXT,
        realized_outcome_20d    REAL,               -- filled later
        outcome_filled          INTEGER NOT NULL DEFAULT 0,
        created_at              INTEGER NOT NULL
    )

Valid skip_reason values (SIGNAL_SKIP_REASONS):
    BELOW_CONFIDENCE_THRESHOLD
    STAGE_NOT_ALLOWED
    LIQUIDITY_GATE
    SECTOR_CAP_REACHED
    CIRCUIT_BREAKER
    OTHER

Usage
-----
From the rule engine (rating_engine.py):

    from app.services.eagle_eye.ml.signal_logger import log_considered_signal

    log_considered_signal(
        ticker="2222",
        signal_date="2025-01-15",
        rule_score=0.72,
        would_have_entered=False,
        skip_reason="BELOW_CONFIDENCE_THRESHOLD",
        features={"stage": 2, "vol_ratio": 1.4, ...},
    )
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import date, timedelta
from typing import Any, Dict, Literal, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGNAL_SKIP_REASONS = frozenset({
    "BELOW_CONFIDENCE_THRESHOLD",
    "STAGE_NOT_ALLOWED",
    "LIQUIDITY_GATE",
    "SECTOR_CAP_REACHED",
    "CIRCUIT_BREAKER",
    "OTHER",
})

SkipReason = Literal[
    "BELOW_CONFIDENCE_THRESHOLD",
    "STAGE_NOT_ALLOWED",
    "LIQUIDITY_GATE",
    "SECTOR_CAP_REACHED",
    "CIRCUIT_BREAKER",
    "OTHER",
]


# ---------------------------------------------------------------------------
# DB helper (ensure table exists)
# ---------------------------------------------------------------------------

def _ensure_considered_signals_table() -> None:
    """Idempotent table creation for `considered_signals`."""
    from app.core.database import exec_sql
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS considered_signals (
            signal_id                   TEXT PRIMARY KEY,
            stock_ticker                TEXT NOT NULL,
            signal_date                 TEXT NOT NULL,
            rule_score                  REAL NOT NULL,
            would_have_entered          INTEGER NOT NULL CHECK (would_have_entered IN (0,1)),
            skip_reason                 TEXT,
            full_feature_snapshot_json  TEXT,
            realized_outcome_20d        REAL,
            outcome_filled              INTEGER NOT NULL DEFAULT 0,
            created_at                  INTEGER NOT NULL
        )
        """,
        (),
    )
    exec_sql(
        "CREATE INDEX IF NOT EXISTS idx_cs_ticker_date ON considered_signals(stock_ticker, signal_date)",
        (),
    )
    exec_sql(
        "CREATE INDEX IF NOT EXISTS idx_cs_outcome_filled ON considered_signals(outcome_filled)",
        (),
    )


# ---------------------------------------------------------------------------
# Write a considered signal
# ---------------------------------------------------------------------------

def log_considered_signal(
    ticker: str,
    signal_date: Union[str, date],
    rule_score: float,
    would_have_entered: bool,
    skip_reason: Optional[SkipReason] = None,
    features: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Persist one evaluated signal to ``considered_signals``.

    Parameters
    ----------
    ticker             : stock ticker symbol
    signal_date        : ISO date string or date object (YYYY-MM-DD)
    rule_score         : raw rule-engine confidence / score
    would_have_entered : True if the rules would have triggered a trade
    skip_reason        : why entry was skipped (None if would_have_entered=True)
    features           : dict of feature values at signal time (snapshot)

    Returns
    -------
    signal_id : UUID string for the new row
    """
    if isinstance(signal_date, date):
        signal_date = signal_date.isoformat()

    if not would_have_entered and skip_reason is None:
        skip_reason = "OTHER"

    if skip_reason is not None and skip_reason not in SIGNAL_SKIP_REASONS:
        logger.warning(
            "Unknown skip_reason %r — defaulting to OTHER", skip_reason
        )
        skip_reason = "OTHER"

    signal_id = str(uuid.uuid4())
    features_json = json.dumps(features or {}, default=str)

    try:
        _ensure_considered_signals_table()
        from app.core.database import exec_sql
        exec_sql(
            """
            INSERT INTO considered_signals
                (signal_id, stock_ticker, signal_date, rule_score,
                 would_have_entered, skip_reason, full_feature_snapshot_json,
                 realized_outcome_20d, outcome_filled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 0, ?)
            ON CONFLICT(signal_id) DO NOTHING
            """,
            (
                signal_id, ticker, signal_date, float(rule_score),
                1 if would_have_entered else 0,
                skip_reason, features_json, int(time.time()),
            ),
        )
        logger.debug("Logged considered signal %s for %s on %s", signal_id, ticker, signal_date)
    except Exception as exc:  # noqa: BLE001
        # Signal logging must NEVER crash the rule engine
        logger.warning("Failed to log considered signal: %s", exc)

    return signal_id


# ---------------------------------------------------------------------------
# Forward-only outcome fill job (DAILY — never at signal time)
# ---------------------------------------------------------------------------

def fill_realized_outcomes(
    lookback_days: int = 30,
) -> int:
    """
    Fill ``realized_outcome_20d`` for signals that:
      - are NOT yet filled (outcome_filled = 0)
      - have a signal_date at least 20 trading days ago (proxy: 30 calendar days)

    Forward-only: we look 20 days AHEAD from the signal_date using OHLCV close
    prices.  This must run as a scheduled daily job, not at signal time.

    Parameters
    ----------
    lookback_days : how far back to scan for unfilled signals (default 30)

    Returns
    -------
    n_filled : number of rows updated
    """
    from app.core.database import exec_sql, exec_sql_fetch
    _ensure_considered_signals_table()

    cutoff_date = (date.today() - timedelta(days=20)).isoformat()
    scan_start  = (date.today() - timedelta(days=lookback_days + 20)).isoformat()

    try:
        rows = exec_sql_fetch(
            """
            SELECT signal_id, stock_ticker, signal_date
            FROM considered_signals
            WHERE outcome_filled = 0
              AND signal_date <= ?
              AND signal_date >= ?
            ORDER BY signal_date
            """,
            (cutoff_date, scan_start),
        )
    except Exception as exc:
        logger.warning("fill_realized_outcomes: fetch failed: %s", exc)
        return 0

    if not rows:
        return 0

    n_filled = 0
    for signal_id, ticker, signal_date_str in rows:
        outcome = _compute_20d_outcome(ticker, signal_date_str)
        if outcome is None:
            continue  # price data not yet available — retry next day
        try:
            exec_sql(
                """
                UPDATE considered_signals
                   SET realized_outcome_20d = ?, outcome_filled = 1
                 WHERE signal_id = ?
                """,
                (outcome, signal_id),
            )
            n_filled += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("fill_realized_outcomes: update failed for %s: %s", signal_id, exc)

    logger.info("fill_realized_outcomes: filled %d outcomes", n_filled)
    return n_filled


def _compute_20d_outcome(ticker: str, signal_date_str: str) -> Optional[float]:
    """
    Calculate the 20-calendar-day forward log-return using OHLCV close prices.

    Returns None if prices are not available for both dates.
    NEVER called at signal time — only by fill_realized_outcomes().
    """
    try:
        from app.services.eagle_eye.store import load_ohlcv
        df = load_ohlcv(ticker)
        if df is None or df.empty or "close" not in df.columns:
            return None

        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()

        import pandas as pd
        signal_date = pd.Timestamp(signal_date_str)
        future_date = signal_date + pd.DateOffset(days=20)

        # Closest trading day on or after signal_date
        avail_start = df.index[df.index >= signal_date]
        avail_end   = df.index[df.index >= future_date]

        if avail_start.empty or avail_end.empty:
            return None

        price_start = float(df.loc[avail_start[0], "close"])
        price_end   = float(df.loc[avail_end[0], "close"])

        if price_start <= 0:
            return None

        import numpy as np
        return float(np.log(price_end / price_start))
    except Exception as exc:  # noqa: BLE001
        logger.debug("_compute_20d_outcome failed for %s @ %s: %s", ticker, signal_date_str, exc)
        return None


# ---------------------------------------------------------------------------
# Leakage audit assertion (used in CI)
# ---------------------------------------------------------------------------

def audit_signal_logger() -> None:
    """
    Assert that this module contains no look-ahead patterns.
    Called by leakage_audit.audit_callable on the module's source text.
    """
    import inspect
    from app.services.eagle_eye.ml.leakage_audit import scan_source_for_leakage

    source = inspect.getsource(__import__(__name__))
    issues = scan_source_for_leakage(source)
    if issues:
        raise RuntimeError(
            f"signal_logger.py contains potential look-ahead patterns: {issues}"
        )

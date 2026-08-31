"""
Trend-Hold Book — persistent paper-trading ledger.

Mechanically "executes" (virtual money only) the decisions already written
by trend_hold_batch.py into ee_trend_hold_state -- BUY / SCALE_OUT /
SELL_SIGNAL -- into a separate, independent 3-table ledger:

  - ee_trend_hold_book_state       : singleton row, cash + starting capital
  - ee_trend_hold_book_positions   : open paper positions, one row per ticker
  - ee_trend_hold_book_trades      : append-only fill ledger (BUY/SCALE_OUT/EXIT)
  - ee_trend_hold_book_nav_history : daily equity/cash snapshot (equity curve)
  - ee_trend_hold_book_lessons     : post-trade "autopsy" for each closed leg
    (SCALE_OUT/EXIT), written by trend_hold_lessons.py -- see that module

This is NOT the real portfolio (app/models/portfolio.py -- real user money)
and NOT the unrelated eagle_eye_v2/simulator/ system (3-symbol backtest
simulator, its own external ledger DB). Fully independent tables, fully
independent of both.

All DDL uses CREATE TABLE IF NOT EXISTS -- idempotent. Writes use the same
portable ``exec_sql``/``exec_sql_batch`` (?-style params) helpers as
app/services/eagle_eye/store.py so both SQLite and PostgreSQL work.
"""
from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

STARTING_CAPITAL_KWD = 100_000.0


def ensure_trend_hold_book_tables() -> None:
    """Create the Trend-Hold Book tables if they do not already exist."""
    from app.core.database import exec_sql

    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS ee_trend_hold_book_state (
            id                  INTEGER PRIMARY KEY,
            cash_kwd            REAL,
            starting_capital_kwd REAL,
            updated_at          INTEGER
        )
        """,
        (),
    )

    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS ee_trend_hold_book_positions (
            ticker              TEXT PRIMARY KEY,
            quantity            REAL,
            avg_cost            REAL,
            entry_commission_kwd REAL,
            opened_date         TEXT,
            updated_at          INTEGER
        )
        """,
        (),
    )

    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS ee_trend_hold_book_nav_history (
            nav_date            TEXT PRIMARY KEY,
            cash_kwd            REAL,
            equity_kwd          REAL,
            open_position_count INTEGER,
            updated_at          INTEGER
        )
        """,
        (),
    )

    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS ee_trend_hold_book_trades (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker           TEXT NOT NULL,
            side             TEXT NOT NULL,
            trade_date       TEXT NOT NULL,
            quantity         REAL,
            price            REAL,
            gross_kwd        REAL,
            commission_kwd   REAL,
            realized_pnl_kwd REAL,
            reason           TEXT,
            executed_at      INTEGER,
            UNIQUE (ticker, trade_date)
        )
        """,
        (),
    )

    # Post-trade "autopsy" for each closed leg -- shares the trades table's
    # natural (ticker, trade_date) key, since exactly one trade fires per
    # ticker per session (see UNIQUE above), so no surrogate FK is needed.
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS ee_trend_hold_book_lessons (
            ticker          TEXT NOT NULL,
            trade_date      TEXT NOT NULL,
            side            TEXT NOT NULL,
            classification  TEXT NOT NULL,
            outcome         TEXT NOT NULL,
            mae_pct         REAL,
            mfe_pct         REAL,
            giveback_pct    REAL,
            holding_days    INTEGER,
            reason          TEXT,
            enhancement     TEXT,
            computed_at     INTEGER,
            PRIMARY KEY (ticker, trade_date)
        )
        """,
        (),
    )


# ---------------------------------------------------------------------------
# Numeric helpers (same convention as app/services/eagle_eye/store.py's _f)
# ---------------------------------------------------------------------------

def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Book state (cash)
# ---------------------------------------------------------------------------

def load_book_state() -> Dict[str, Any]:
    """Return the singleton book state, initializing it on first use."""
    from app.core.database import exec_sql, query_one

    row = query_one(
        "SELECT id, cash_kwd, starting_capital_kwd, updated_at FROM ee_trend_hold_book_state WHERE id = 1",
        (),
    )
    if row is not None:
        d = dict(row.items())
        return {
            "cash_kwd": _f(d.get("cash_kwd")) or 0.0,
            "starting_capital_kwd": _f(d.get("starting_capital_kwd")) or STARTING_CAPITAL_KWD,
            "updated_at": d.get("updated_at"),
        }

    exec_sql(
        """
        INSERT INTO ee_trend_hold_book_state (id, cash_kwd, starting_capital_kwd, updated_at)
        VALUES (1, ?, ?, ?)
        """,
        (STARTING_CAPITAL_KWD, STARTING_CAPITAL_KWD, int(time.time())),
    )
    return {
        "cash_kwd": STARTING_CAPITAL_KWD,
        "starting_capital_kwd": STARTING_CAPITAL_KWD,
        "updated_at": int(time.time()),
    }


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def load_all_positions() -> Dict[str, dict]:
    """Return {ticker: row} for every currently open paper position."""
    from app.core.database import query_all

    rows = query_all(
        """
        SELECT ticker, quantity, avg_cost, entry_commission_kwd, opened_date, updated_at
        FROM   ee_trend_hold_book_positions
        """,
        (),
    )
    return {str(r["ticker"]).upper(): dict(r.items()) for r in rows or []}


def delete_position(ticker: str) -> None:
    from app.core.database import exec_sql

    exec_sql("DELETE FROM ee_trend_hold_book_positions WHERE ticker = ?", (ticker.upper(),))


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

def trade_exists(ticker: str, trade_date: str) -> bool:
    """Idempotency check: has this ticker already been actioned for this session?"""
    from app.core.database import query_one

    row = query_one(
        "SELECT id FROM ee_trend_hold_book_trades WHERE ticker = ? AND trade_date = ?",
        (ticker.upper(), trade_date),
    )
    return row is not None


# ---------------------------------------------------------------------------
# Atomic fills — each bundles the position change + trade row + cash update
# into a single transaction (exec_sql_batch) so a crash mid-write can never
# leave a trade recorded without its matching position/cash change, or vice
# versa -- which would otherwise silently break the (ticker, trade_date)
# idempotency guard on the next scheduler run.
# ---------------------------------------------------------------------------

def record_buy_fill(
    ticker: str,
    trade_date: str,
    quantity: float,
    price: float,
    gross_kwd: float,
    commission_kwd: float,
    reason: Optional[str],
    cash_kwd: float,
) -> None:
    from app.core.database import exec_sql_batch

    now = int(time.time())
    exec_sql_batch(
        [
            (
                """
                INSERT INTO ee_trend_hold_book_positions (
                    ticker, quantity, avg_cost, entry_commission_kwd, opened_date, updated_at
                ) VALUES (?,?,?,?,?,?)
                ON CONFLICT (ticker) DO UPDATE SET
                    quantity = excluded.quantity,
                    avg_cost = excluded.avg_cost,
                    entry_commission_kwd = excluded.entry_commission_kwd,
                    opened_date = excluded.opened_date,
                    updated_at = excluded.updated_at
                """,
                (ticker.upper(), _f(quantity), _f(price), _f(commission_kwd), trade_date, now),
            ),
            (
                """
                INSERT INTO ee_trend_hold_book_trades (
                    ticker, side, trade_date, quantity, price, gross_kwd,
                    commission_kwd, realized_pnl_kwd, reason, executed_at
                ) VALUES (?,'BUY',?,?,?,?,?,NULL,?,?)
                """,
                (ticker.upper(), trade_date, _f(quantity), _f(price), _f(gross_kwd), _f(commission_kwd), reason, now),
            ),
            (
                "UPDATE ee_trend_hold_book_state SET cash_kwd = ?, updated_at = ? WHERE id = 1",
                (_f(cash_kwd), now),
            ),
        ]
    )


def _lesson_insert_statement(ticker: str, trade_date: str, side: str, lesson: Optional[dict], now: int):
    """Build the (sql, params) tuple for one lesson row, or None if no lesson was computed."""
    if lesson is None:
        return None
    return (
        """
        INSERT INTO ee_trend_hold_book_lessons (
            ticker, trade_date, side, classification, outcome, mae_pct,
            mfe_pct, giveback_pct, holding_days, reason, enhancement, computed_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT (ticker, trade_date) DO UPDATE SET
            side = excluded.side,
            classification = excluded.classification,
            outcome = excluded.outcome,
            mae_pct = excluded.mae_pct,
            mfe_pct = excluded.mfe_pct,
            giveback_pct = excluded.giveback_pct,
            holding_days = excluded.holding_days,
            reason = excluded.reason,
            enhancement = excluded.enhancement,
            computed_at = excluded.computed_at
        """,
        (
            ticker.upper(),
            trade_date,
            side,
            lesson["classification"],
            lesson["outcome"],
            _f(lesson.get("mae_pct")),
            _f(lesson.get("mfe_pct")),
            _f(lesson.get("giveback_pct")),
            lesson.get("holding_days"),
            lesson.get("reason"),
            lesson.get("enhancement"),
            now,
        ),
    )


def record_scale_out_fill(
    ticker: str,
    trade_date: str,
    sell_quantity: float,
    price: float,
    gross_kwd: float,
    commission_kwd: float,
    realized_pnl_kwd: float,
    reason: Optional[str],
    cash_kwd: float,
    remaining_quantity: float,
    remaining_entry_commission_kwd: float,
    avg_cost: float,
    opened_date: str,
    lesson: Optional[dict] = None,
) -> None:
    from app.core.database import exec_sql_batch

    now = int(time.time())
    statements = [
        (
            """
            UPDATE ee_trend_hold_book_positions
            SET quantity = ?, entry_commission_kwd = ?, avg_cost = ?, opened_date = ?, updated_at = ?
            WHERE ticker = ?
            """,
            (_f(remaining_quantity), _f(remaining_entry_commission_kwd), _f(avg_cost), opened_date, now, ticker.upper()),
        ),
        (
            """
            INSERT INTO ee_trend_hold_book_trades (
                ticker, side, trade_date, quantity, price, gross_kwd,
                commission_kwd, realized_pnl_kwd, reason, executed_at
            ) VALUES (?,'SCALE_OUT',?,?,?,?,?,?,?,?)
            """,
            (
                ticker.upper(), trade_date, _f(sell_quantity), _f(price), _f(gross_kwd),
                _f(commission_kwd), _f(realized_pnl_kwd), reason, now,
            ),
        ),
        (
            "UPDATE ee_trend_hold_book_state SET cash_kwd = ?, updated_at = ? WHERE id = 1",
            (_f(cash_kwd), now),
        ),
    ]
    lesson_stmt = _lesson_insert_statement(ticker, trade_date, "SCALE_OUT", lesson, now)
    if lesson_stmt is not None:
        statements.append(lesson_stmt)
    exec_sql_batch(statements)


def record_exit_fill(
    ticker: str,
    trade_date: str,
    sell_quantity: float,
    price: float,
    gross_kwd: float,
    commission_kwd: float,
    realized_pnl_kwd: float,
    reason: Optional[str],
    cash_kwd: float,
    lesson: Optional[dict] = None,
) -> None:
    from app.core.database import exec_sql_batch

    now = int(time.time())
    statements = [
        ("DELETE FROM ee_trend_hold_book_positions WHERE ticker = ?", (ticker.upper(),)),
        (
            """
            INSERT INTO ee_trend_hold_book_trades (
                ticker, side, trade_date, quantity, price, gross_kwd,
                commission_kwd, realized_pnl_kwd, reason, executed_at
            ) VALUES (?,'EXIT',?,?,?,?,?,?,?,?)
            """,
            (
                ticker.upper(), trade_date, _f(sell_quantity), _f(price), _f(gross_kwd),
                _f(commission_kwd), _f(realized_pnl_kwd), reason, now,
            ),
        ),
        (
            "UPDATE ee_trend_hold_book_state SET cash_kwd = ?, updated_at = ? WHERE id = 1",
            (_f(cash_kwd), now),
        ),
    ]
    lesson_stmt = _lesson_insert_statement(ticker, trade_date, "EXIT", lesson, now)
    if lesson_stmt is not None:
        statements.append(lesson_stmt)
    exec_sql_batch(statements)


def load_recent_trades(limit: int = 300) -> List[dict]:
    from app.core.database import query_all

    rows = query_all(
        """
        SELECT id, ticker, side, trade_date, quantity, price, gross_kwd,
               commission_kwd, realized_pnl_kwd, reason, executed_at
        FROM   ee_trend_hold_book_trades
        ORDER BY id DESC
        LIMIT  ?
        """,
        (limit,),
    )
    return [dict(r.items()) for r in rows or []]


# ---------------------------------------------------------------------------
# NAV history (daily equity snapshot, powers the equity curve)
# ---------------------------------------------------------------------------

def save_nav_snapshot(nav_date: str, cash_kwd: float, equity_kwd: float, open_position_count: int) -> None:
    """Upsert today's book equity snapshot -- safe to call more than once per day."""
    from app.core.database import exec_sql

    exec_sql(
        """
        INSERT INTO ee_trend_hold_book_nav_history (
            nav_date, cash_kwd, equity_kwd, open_position_count, updated_at
        ) VALUES (?,?,?,?,?)
        ON CONFLICT (nav_date) DO UPDATE SET
            cash_kwd = excluded.cash_kwd,
            equity_kwd = excluded.equity_kwd,
            open_position_count = excluded.open_position_count,
            updated_at = excluded.updated_at
        """,
        (nav_date, _f(cash_kwd), _f(equity_kwd), open_position_count, int(time.time())),
    )


def load_nav_history(days: int = 180) -> List[dict]:
    """Return the last *days* daily equity snapshots, oldest first (chart order)."""
    from app.core.database import query_all

    rows = query_all(
        """
        SELECT nav_date, cash_kwd, equity_kwd, open_position_count
        FROM   ee_trend_hold_book_nav_history
        ORDER BY nav_date DESC
        LIMIT  ?
        """,
        (days,),
    )
    return list(reversed([dict(r.items()) for r in rows or []]))


# ---------------------------------------------------------------------------
# Lessons (post-trade autopsy, see trend_hold_lessons.py)
# ---------------------------------------------------------------------------

def load_lessons(limit: int = 200) -> List[dict]:
    """Return recent trade lessons, newest first."""
    from app.core.database import query_all

    rows = query_all(
        """
        SELECT ticker, trade_date, side, classification, outcome, mae_pct,
               mfe_pct, giveback_pct, holding_days, reason, enhancement, computed_at
        FROM   ee_trend_hold_book_lessons
        ORDER BY trade_date DESC, ticker ASC
        LIMIT  ?
        """,
        (limit,),
    )
    return [dict(r.items()) for r in rows or []]


def load_lessons_summary() -> Dict[str, Any]:
    """
    Aggregate the lessons log into a "what's actually going wrong" rollup:
    counts per classification/outcome, plus average excursion metrics for
    losing trades -- the evidence a human would want before touching any
    trend_hold_engine.py parameter.
    """
    from app.core.database import query_all

    rows = query_all(
        """
        SELECT classification, outcome, mae_pct, mfe_pct, giveback_pct, holding_days
        FROM   ee_trend_hold_book_lessons
        """,
        (),
    )
    rows = [dict(r.items()) for r in rows or []]

    by_classification: Dict[str, int] = {}
    by_outcome: Dict[str, int] = {}
    loss_mae: List[float] = []
    win_giveback: List[float] = []

    for r in rows:
        cls = r.get("classification") or "UNKNOWN"
        outcome = r.get("outcome") or "UNKNOWN"
        by_classification[cls] = by_classification.get(cls, 0) + 1
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
        if outcome == "LOSS" and r.get("mae_pct") is not None:
            loss_mae.append(float(r["mae_pct"]))
        if outcome == "WIN" and r.get("giveback_pct") is not None:
            win_giveback.append(float(r["giveback_pct"]))

    return {
        "total_closed": len(rows),
        "by_classification": by_classification,
        "by_outcome": by_outcome,
        "avg_loss_mae_pct": round(sum(loss_mae) / len(loss_mae), 2) if loss_mae else None,
        "avg_win_giveback_pct": round(sum(win_giveback) / len(win_giveback), 2) if win_giveback else None,
    }

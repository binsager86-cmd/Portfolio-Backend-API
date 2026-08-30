"""
Trend-Hold Book — persistent paper-trading ledger.

Mechanically "executes" (virtual money only) the decisions already written
by trend_hold_batch.py into ee_trend_hold_state -- BUY / SCALE_OUT /
SELL_SIGNAL -- into a separate, independent 3-table ledger:

  - ee_trend_hold_book_state     : singleton row, cash + starting capital
  - ee_trend_hold_book_positions : open paper positions, one row per ticker
  - ee_trend_hold_book_trades    : append-only fill ledger (BUY/SCALE_OUT/EXIT)

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
) -> None:
    from app.core.database import exec_sql_batch

    now = int(time.time())
    exec_sql_batch(
        [
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
    )


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
) -> None:
    from app.core.database import exec_sql_batch

    now = int(time.time())
    exec_sql_batch(
        [
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
    )


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

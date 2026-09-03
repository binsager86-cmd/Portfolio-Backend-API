"""
Paper Book — persistent, multi-book paper-trading ledger.

Mechanically "executes" (virtual money only) the decisions from an
independent signal source into a ``book_id``-scoped 5-table ledger:

  - ee_trend_hold_book_state       : one row per book, cash + starting capital
  - ee_trend_hold_book_positions   : open paper positions, one row per (book, ticker)
  - ee_trend_hold_book_trades      : append-only fill ledger (BUY/SCALE_OUT/EXIT)
  - ee_trend_hold_book_nav_history : daily equity/cash snapshot (equity curve)
  - ee_trend_hold_book_lessons     : post-trade "autopsy" for each closed leg,
    written by trend_hold_lessons.py -- see that module

Two books exist today, each fully independent -- own starting capital, own
cash pool, own positions, own trades, never sharing state:
  - "trend_hold" : driven by trend_hold_engine.py (trend_hold_book.py)
  - "v1_rating"  : driven by the V1 rating engine (v1_rating_book.py)

This is NOT the real portfolio (app/models/portfolio.py -- real user money)
and NOT the unrelated eagle_eye_v2/simulator/ system (3-symbol backtest
simulator, its own external ledger DB). Fully independent tables, fully
independent of both.

Table names are unchanged from the original single-book ("trend_hold"-only)
version of this module -- only the keys widened to include ``book_id``, via
a one-time, idempotent migration (see ``_migrate_add_book_id``) that
preserves every row already written by the live Trend-Hold Book, backfilled
as book_id='trend_hold'.

All DDL uses CREATE TABLE IF NOT EXISTS -- idempotent. Writes use the same
portable ``exec_sql``/``exec_sql_batch`` (?-style params) helpers as
app/services/eagle_eye/store.py so both SQLite and PostgreSQL work.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

# Shared economics -- both books use identical starting capital, sizing,
# and commission so their performance scorecards are directly comparable.
STARTING_CAPITAL_KWD = 100_000.0
POSITION_SIZE_FRACTION = 0.10
MAX_CONCURRENT_POSITIONS = 10
COMMISSION_RATE = 0.00325


# ---------------------------------------------------------------------------
# Schema — creation + the one-time "add book_id" migration
# ---------------------------------------------------------------------------

def _migrate_add_book_id(table: str, create_sql: str, copy_columns: str, order_by: str) -> None:
    """
    If *table* already exists but predates the multi-book schema (no
    book_id column -- true only for tables written by the original,
    "trend_hold"-only version of this module, which may already hold live
    production data), rebuild it with book_id folded into the key,
    backfilling every existing row as book_id='trend_hold'.

    No-op if the table doesn't exist yet (the caller's CREATE TABLE IF NOT
    EXISTS handles a fresh install directly) or already has book_id
    (migration already ran on a previous startup).

    Runs the whole rename/create/copy/drop sequence on a single raw
    connection so it's one atomic unit (all-or-nothing on failure), rather
    than four separate auto-committed exec_sql() calls that could leave
    the schema half-migrated if the process died partway through.

    On SQLite specifically: ``ALTER TABLE ... RENAME TO`` triggers a
    schema-wide re-validation of every view in the database (unrelated to
    the table being renamed) unless ``PRAGMA legacy_alter_table`` is
    enabled first -- without it, a RENAME here can fail with an unrelated
    "no such table" error if *any* other, unrelated view in the DB
    happens to be broken (observed against this app's dev DB, which has a
    pre-existing broken `holdings` view from an older schema generation).
    """
    from app.core.config import get_settings
    from app.core.database import _normalize_ddl_for_pg, column_exists, get_connection, table_exists

    if not table_exists(table) or column_exists(table, "book_id"):
        return

    old = f"{table}__pre_book_id"
    is_pg = get_settings().use_postgres

    with get_connection() as conn:
        cur = conn.cursor()
        try:
            if not is_pg:
                cur.execute("PRAGMA legacy_alter_table = ON")
            cur.execute(f"ALTER TABLE {table} RENAME TO {old}")
            cur.execute(_normalize_ddl_for_pg(create_sql) if is_pg else create_sql)
            cur.execute(
                f"INSERT INTO {table} (book_id, {copy_columns}) "
                f"SELECT 'trend_hold', {copy_columns} FROM {old} ORDER BY {order_by}"
            )
            cur.execute(f"DROP TABLE {old}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def ensure_paper_book_tables() -> None:
    """Create the Paper Book tables (or migrate them to the multi-book schema) if needed."""
    from app.core.database import exec_sql

    state_sql = """
        CREATE TABLE IF NOT EXISTS ee_trend_hold_book_state (
            book_id              TEXT PRIMARY KEY,
            cash_kwd             REAL,
            starting_capital_kwd REAL,
            updated_at           INTEGER
        )
        """
    _migrate_add_book_id(
        "ee_trend_hold_book_state", state_sql,
        "cash_kwd, starting_capital_kwd, updated_at", "updated_at",
    )
    exec_sql(state_sql, ())

    positions_sql = """
        CREATE TABLE IF NOT EXISTS ee_trend_hold_book_positions (
            book_id               TEXT NOT NULL,
            ticker                TEXT NOT NULL,
            quantity              REAL,
            avg_cost              REAL,
            entry_commission_kwd  REAL,
            opened_date           TEXT,
            updated_at            INTEGER,
            PRIMARY KEY (book_id, ticker)
        )
        """
    _migrate_add_book_id(
        "ee_trend_hold_book_positions", positions_sql,
        "ticker, quantity, avg_cost, entry_commission_kwd, opened_date, updated_at", "ticker",
    )
    exec_sql(positions_sql, ())

    nav_sql = """
        CREATE TABLE IF NOT EXISTS ee_trend_hold_book_nav_history (
            book_id              TEXT NOT NULL,
            nav_date             TEXT NOT NULL,
            cash_kwd             REAL,
            equity_kwd           REAL,
            open_position_count  INTEGER,
            updated_at           INTEGER,
            PRIMARY KEY (book_id, nav_date)
        )
        """
    _migrate_add_book_id(
        "ee_trend_hold_book_nav_history", nav_sql,
        "nav_date, cash_kwd, equity_kwd, open_position_count, updated_at", "nav_date",
    )
    exec_sql(nav_sql, ())

    trades_sql = """
        CREATE TABLE IF NOT EXISTS ee_trend_hold_book_trades (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id          TEXT NOT NULL,
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
            confidence       REAL,
            UNIQUE (book_id, ticker, trade_date)
        )
        """
    _migrate_add_book_id(
        "ee_trend_hold_book_trades", trades_sql,
        "ticker, side, trade_date, quantity, price, gross_kwd, commission_kwd, "
        "realized_pnl_kwd, reason, executed_at, confidence", "id",
    )
    exec_sql(trades_sql, ())

    # Post-trade "autopsy" for each closed leg -- shares the trades table's
    # natural (book_id, ticker, trade_date) key, since exactly one trade
    # fires per ticker per session per book (see UNIQUE above), so no
    # surrogate FK is needed.
    lessons_sql = """
        CREATE TABLE IF NOT EXISTS ee_trend_hold_book_lessons (
            book_id         TEXT NOT NULL,
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
            PRIMARY KEY (book_id, ticker, trade_date)
        )
        """
    _migrate_add_book_id(
        "ee_trend_hold_book_lessons", lessons_sql,
        "ticker, trade_date, side, classification, outcome, mae_pct, mfe_pct, "
        "giveback_pct, holding_days, reason, enhancement, computed_at", "trade_date",
    )
    exec_sql(lessons_sql, ())

    # Intraday mark-to-market cache for currently open positions only (see
    # trend_hold_book.py::run_open_position_price_refresh). Brand-new table,
    # multi-book (book_id-scoped) from day one -- no _migrate_add_book_id
    # needed. Deliberately separate from ee_trend_hold_state: this is a
    # display-only price overlay, never read by the decision engine.
    live_prices_sql = """
        CREATE TABLE IF NOT EXISTS ee_trend_hold_book_live_prices (
            book_id     TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            price       REAL,
            updated_at  INTEGER,
            PRIMARY KEY (book_id, ticker)
        )
        """
    exec_sql(live_prices_sql, ())


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

def load_book_state(book_id: str) -> Dict[str, Any]:
    """Return *book_id*'s state, initializing it (with a fresh starting balance) on first use."""
    from app.core.database import exec_sql, query_one

    row = query_one(
        "SELECT book_id, cash_kwd, starting_capital_kwd, updated_at FROM ee_trend_hold_book_state WHERE book_id = ?",
        (book_id,),
    )
    if row is not None:
        d = dict(row.items())
        return {
            "cash_kwd": _f(d.get("cash_kwd")) or 0.0,
            "starting_capital_kwd": _f(d.get("starting_capital_kwd")) or STARTING_CAPITAL_KWD,
            "updated_at": d.get("updated_at"),
        }

    exec_sql(
        "INSERT INTO ee_trend_hold_book_state (book_id, cash_kwd, starting_capital_kwd, updated_at) VALUES (?,?,?,?)",
        (book_id, STARTING_CAPITAL_KWD, STARTING_CAPITAL_KWD, int(time.time())),
    )
    return {
        "cash_kwd": STARTING_CAPITAL_KWD,
        "starting_capital_kwd": STARTING_CAPITAL_KWD,
        "updated_at": int(time.time()),
    }


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def load_all_positions(book_id: str) -> Dict[str, dict]:
    """Return {ticker: row} for every currently open position in *book_id*."""
    from app.core.database import query_all

    rows = query_all(
        """
        SELECT ticker, quantity, avg_cost, entry_commission_kwd, opened_date, updated_at
        FROM   ee_trend_hold_book_positions
        WHERE  book_id = ?
        """,
        (book_id,),
    )
    return {str(r["ticker"]).upper(): dict(r.items()) for r in rows or []}


# ---------------------------------------------------------------------------
# Live prices — intraday mark-to-market cache for open positions only
# ---------------------------------------------------------------------------

def save_live_prices(book_id: str, prices: Dict[str, float]) -> None:
    """
    Replace *book_id*'s intraday price cache with *prices* -- a full
    replace-all-for-book write (delete then insert), not a merge, so a
    ticker whose position has since closed has its stale price dropped
    automatically instead of lingering until some other cleanup runs.
    """
    from app.core.database import exec_sql_batch

    now = int(time.time())
    statements: list = [("DELETE FROM ee_trend_hold_book_live_prices WHERE book_id = ?", (book_id,))]
    for ticker, price in prices.items():
        p = _f(price)
        if p is None:
            continue
        statements.append(
            (
                "INSERT INTO ee_trend_hold_book_live_prices (book_id, ticker, price, updated_at) VALUES (?,?,?,?)",
                (book_id, ticker.upper(), p, now),
            )
        )
    exec_sql_batch(statements)


def load_live_prices(book_id: str) -> Dict[str, float]:
    """Return {ticker: price} for book_id's cached intraday prices (open positions only)."""
    from app.core.database import query_all

    rows = query_all(
        "SELECT ticker, price FROM ee_trend_hold_book_live_prices WHERE book_id = ?",
        (book_id,),
    )
    out: Dict[str, float] = {}
    for r in rows or []:
        d = dict(r.items())
        p = _f(d.get("price"))
        if p is not None:
            out[str(d["ticker"]).upper()] = p
    return out


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

def trade_exists(book_id: str, ticker: str, trade_date: str) -> bool:
    """Idempotency check: has this ticker already been actioned for this book/session?"""
    from app.core.database import query_one

    row = query_one(
        "SELECT id FROM ee_trend_hold_book_trades WHERE book_id = ? AND ticker = ? AND trade_date = ?",
        (book_id, ticker.upper(), trade_date),
    )
    return row is not None


# ---------------------------------------------------------------------------
# Atomic fills — each bundles the position change + trade row + cash update
# into a single transaction (exec_sql_batch) so a crash mid-write can never
# leave a trade recorded without its matching position/cash change, or vice
# versa -- which would otherwise silently break the (book_id, ticker,
# trade_date) idempotency guard on the next scheduler run.
# ---------------------------------------------------------------------------

def record_buy_fill(
    book_id: str,
    ticker: str,
    trade_date: str,
    quantity: float,
    price: float,
    gross_kwd: float,
    commission_kwd: float,
    reason: Optional[str],
    cash_kwd: float,
    confidence: Optional[float] = None,
) -> None:
    from app.core.database import exec_sql_batch

    now = int(time.time())
    exec_sql_batch(
        [
            (
                """
                INSERT INTO ee_trend_hold_book_positions (
                    book_id, ticker, quantity, avg_cost, entry_commission_kwd, opened_date, updated_at
                ) VALUES (?,?,?,?,?,?,?)
                ON CONFLICT (book_id, ticker) DO UPDATE SET
                    quantity = excluded.quantity,
                    avg_cost = excluded.avg_cost,
                    entry_commission_kwd = excluded.entry_commission_kwd,
                    opened_date = excluded.opened_date,
                    updated_at = excluded.updated_at
                """,
                (book_id, ticker.upper(), _f(quantity), _f(price), _f(commission_kwd), trade_date, now),
            ),
            (
                """
                INSERT INTO ee_trend_hold_book_trades (
                    book_id, ticker, side, trade_date, quantity, price, gross_kwd,
                    commission_kwd, realized_pnl_kwd, reason, executed_at, confidence
                ) VALUES (?,?,'BUY',?,?,?,?,?,NULL,?,?,?)
                """,
                (
                    book_id, ticker.upper(), trade_date, _f(quantity), _f(price), _f(gross_kwd),
                    _f(commission_kwd), reason, now, _f(confidence),
                ),
            ),
            (
                "UPDATE ee_trend_hold_book_state SET cash_kwd = ?, updated_at = ? WHERE book_id = ?",
                (_f(cash_kwd), now, book_id),
            ),
        ]
    )


def _lesson_insert_statement(book_id: str, ticker: str, trade_date: str, side: str, lesson: Optional[dict], now: int):
    """Build the (sql, params) tuple for one lesson row, or None if no lesson was computed."""
    if lesson is None:
        return None
    return (
        """
        INSERT INTO ee_trend_hold_book_lessons (
            book_id, ticker, trade_date, side, classification, outcome, mae_pct,
            mfe_pct, giveback_pct, holding_days, reason, enhancement, computed_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT (book_id, ticker, trade_date) DO UPDATE SET
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
            book_id,
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
    book_id: str,
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
            WHERE book_id = ? AND ticker = ?
            """,
            (
                _f(remaining_quantity), _f(remaining_entry_commission_kwd), _f(avg_cost),
                opened_date, now, book_id, ticker.upper(),
            ),
        ),
        (
            """
            INSERT INTO ee_trend_hold_book_trades (
                book_id, ticker, side, trade_date, quantity, price, gross_kwd,
                commission_kwd, realized_pnl_kwd, reason, executed_at
            ) VALUES (?,?,'SCALE_OUT',?,?,?,?,?,?,?,?)
            """,
            (
                book_id, ticker.upper(), trade_date, _f(sell_quantity), _f(price), _f(gross_kwd),
                _f(commission_kwd), _f(realized_pnl_kwd), reason, now,
            ),
        ),
        (
            "UPDATE ee_trend_hold_book_state SET cash_kwd = ?, updated_at = ? WHERE book_id = ?",
            (_f(cash_kwd), now, book_id),
        ),
    ]
    lesson_stmt = _lesson_insert_statement(book_id, ticker, trade_date, "SCALE_OUT", lesson, now)
    if lesson_stmt is not None:
        statements.append(lesson_stmt)
    exec_sql_batch(statements)


def record_exit_fill(
    book_id: str,
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
    confidence: Optional[float] = None,
) -> None:
    from app.core.database import exec_sql_batch

    now = int(time.time())
    statements = [
        ("DELETE FROM ee_trend_hold_book_positions WHERE book_id = ? AND ticker = ?", (book_id, ticker.upper())),
        (
            """
            INSERT INTO ee_trend_hold_book_trades (
                book_id, ticker, side, trade_date, quantity, price, gross_kwd,
                commission_kwd, realized_pnl_kwd, reason, executed_at, confidence
            ) VALUES (?,?,'EXIT',?,?,?,?,?,?,?,?,?)
            """,
            (
                book_id, ticker.upper(), trade_date, _f(sell_quantity), _f(price), _f(gross_kwd),
                _f(commission_kwd), _f(realized_pnl_kwd), reason, now, _f(confidence),
            ),
        ),
        (
            "UPDATE ee_trend_hold_book_state SET cash_kwd = ?, updated_at = ? WHERE book_id = ?",
            (_f(cash_kwd), now, book_id),
        ),
    ]
    lesson_stmt = _lesson_insert_statement(book_id, ticker, trade_date, "EXIT", lesson, now)
    if lesson_stmt is not None:
        statements.append(lesson_stmt)
    exec_sql_batch(statements)


def load_recent_trades(book_id: str, limit: int = 300) -> List[dict]:
    from app.core.database import query_all

    rows = query_all(
        """
        SELECT id, ticker, side, trade_date, quantity, price, gross_kwd,
               commission_kwd, realized_pnl_kwd, reason, executed_at, confidence
        FROM   ee_trend_hold_book_trades
        WHERE  book_id = ?
        ORDER BY id DESC
        LIMIT  ?
        """,
        (book_id, limit),
    )
    return [dict(r.items()) for r in rows or []]


# ---------------------------------------------------------------------------
# NAV history (daily equity snapshot, powers the equity curve)
# ---------------------------------------------------------------------------

def save_nav_snapshot(book_id: str, nav_date: str, cash_kwd: float, equity_kwd: float, open_position_count: int) -> None:
    """Upsert *book_id*'s equity snapshot for today -- safe to call more than once per day."""
    from app.core.database import exec_sql

    exec_sql(
        """
        INSERT INTO ee_trend_hold_book_nav_history (
            book_id, nav_date, cash_kwd, equity_kwd, open_position_count, updated_at
        ) VALUES (?,?,?,?,?,?)
        ON CONFLICT (book_id, nav_date) DO UPDATE SET
            cash_kwd = excluded.cash_kwd,
            equity_kwd = excluded.equity_kwd,
            open_position_count = excluded.open_position_count,
            updated_at = excluded.updated_at
        """,
        (book_id, nav_date, _f(cash_kwd), _f(equity_kwd), open_position_count, int(time.time())),
    )


def load_nav_history(book_id: str, days: int = 180) -> List[dict]:
    """Return *book_id*'s last *days* daily equity snapshots, oldest first (chart order)."""
    from app.core.database import query_all

    rows = query_all(
        """
        SELECT nav_date, cash_kwd, equity_kwd, open_position_count
        FROM   ee_trend_hold_book_nav_history
        WHERE  book_id = ?
        ORDER BY nav_date DESC
        LIMIT  ?
        """,
        (book_id, days),
    )
    return list(reversed([dict(r.items()) for r in rows or []]))


# ---------------------------------------------------------------------------
# Lessons (post-trade autopsy, see trend_hold_lessons.py)
# ---------------------------------------------------------------------------

def load_lessons(book_id: str, limit: int = 200) -> List[dict]:
    """Return *book_id*'s recent trade lessons, newest first."""
    from app.core.database import query_all

    rows = query_all(
        """
        SELECT ticker, trade_date, side, classification, outcome, mae_pct,
               mfe_pct, giveback_pct, holding_days, reason, enhancement, computed_at
        FROM   ee_trend_hold_book_lessons
        WHERE  book_id = ?
        ORDER BY trade_date DESC, ticker ASC
        LIMIT  ?
        """,
        (book_id, limit),
    )
    return [dict(r.items()) for r in rows or []]


def load_lessons_summary(book_id: str) -> Dict[str, Any]:
    """
    Aggregate *book_id*'s lessons log into a "what's actually going wrong"
    rollup: counts per classification/outcome, plus average excursion
    metrics for losing trades -- the evidence a human would want before
    touching any decision-engine parameter.
    """
    from app.core.database import query_all

    rows = query_all(
        """
        SELECT classification, outcome, mae_pct, mfe_pct, giveback_pct, holding_days
        FROM   ee_trend_hold_book_lessons
        WHERE  book_id = ?
        """,
        (book_id,),
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


# ---------------------------------------------------------------------------
# Performance statistics (standard trading scorecard, from realized P&L)
# ---------------------------------------------------------------------------

def load_performance_stats(book_id: str) -> Dict[str, Any]:
    """
    Standard trading performance scorecard for *book_id*, computed directly
    from ee_trend_hold_book_trades.realized_pnl_kwd -- win/loss counts,
    best/worst trade, profit factor, expectancy. Independent of the lessons
    classifier (works even before any lesson has been computed), and the
    KWD-denominated companion to load_lessons_summary()'s percentage-based
    excursion metrics.
    """
    from app.core.database import query_all

    rows = query_all(
        "SELECT realized_pnl_kwd, commission_kwd FROM ee_trend_hold_book_trades WHERE book_id = ?",
        (book_id,),
    )
    rows = [dict(r.items()) for r in rows or []]

    total_commission = sum(_f(r.get("commission_kwd")) or 0.0 for r in rows)
    pnls = [float(r["realized_pnl_kwd"]) for r in rows if r.get("realized_pnl_kwd") is not None]
    wins = [p for p in pnls if p >= 0]
    losses = [p for p in pnls if p < 0]

    total_closed = len(pnls)
    win_count = len(wins)
    loss_count = len(losses)
    total_pnl = sum(pnls)
    gross_profit = sum(wins)
    gross_loss = sum(losses)  # <= 0

    # profit_factor is undefined (not 0, not infinite) until there's at
    # least one loss to divide by -- returned as None rather than a
    # non-JSON-safe float('inf') when every closed trade so far has won.
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None

    return {
        "total_closed": total_closed,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate_pct": round(win_count / total_closed * 100.0, 2) if total_closed else None,
        "total_realized_pnl_kwd": round(total_pnl, 3),
        "max_profit_kwd": round(max(wins), 3) if wins else None,
        "max_loss_kwd": round(min(losses), 3) if losses else None,
        "avg_win_kwd": round(gross_profit / win_count, 3) if win_count else None,
        "avg_loss_kwd": round(gross_loss / loss_count, 3) if loss_count else None,
        "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
        "expectancy_kwd": round(total_pnl / total_closed, 3) if total_closed else None,
        "total_commission_paid_kwd": round(total_commission, 3),
    }

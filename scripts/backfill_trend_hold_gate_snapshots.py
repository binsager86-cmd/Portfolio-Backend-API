"""
One-time backfill: reconstruct entry/exit gate snapshots (and the lesson
table's denormalized price/quantity/P&L columns) for Trend-Hold Book rows
that were written before those columns existed.

Why this is safe and exact, not an approximation: trend_hold_engine's
replay_symbol() is a pure function of a ticker's OHLCV history (only
rolling/backward-looking windows, no lookahead), so replaying it today over
the same (now slightly longer) history reproduces the exact same decision --
and the exact same gate inputs -- on every historical date as the engine
actually produced live. Re-running the full replay per ticker is cheap
(trend_hold_batch.py's own docstring: "low seconds for the whole universe").

Idempotent: every write is COALESCE(existing, new), so it never overwrites
a value some later, already-migrated code path already filled in, and it's
safe to re-run (e.g. after a partial failure, or once more tickers exist).

Usage (from backend-api root):
    python scripts/backfill_trend_hold_gate_snapshots.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())

from app.core.database import exec_sql, query_all, query_one  # noqa: E402
from app.services.eagle_eye.store import load_ohlcv  # noqa: E402
from app.services.eagle_eye_v2 import paper_book_store as book  # noqa: E402
from app.services.eagle_eye_v2.trend_hold_batch import _adapt_ohlcv  # noqa: E402
from app.services.eagle_eye_v2.trend_hold_book import BOOK_ID  # noqa: E402
from app.services.eagle_eye_v2.trend_hold_engine import compute_daily_features, replay_symbol  # noqa: E402
from app.services.eagle_eye_v2.trend_hold_lessons import analyze_trade  # noqa: E402


def _replay_by_date(ticker: str, raw) -> dict:
    if raw is None or raw.empty:
        return {}
    features = compute_daily_features(_adapt_ohlcv(raw))
    rows = replay_symbol(features)
    return {str(r["trade_date"]): r for r in rows}


def main() -> None:
    book.ensure_paper_book_tables()

    tickers = [
        str(r["ticker"]).upper()
        for r in query_all("SELECT DISTINCT ticker FROM ee_trend_hold_book_trades WHERE book_id = ?", (BOOK_ID,)) or []
    ]
    print(f"backfill: {len(tickers)} ticker(s) with Trend-Hold Book trade history")

    trades_updated = 0
    lessons_updated = 0
    positions_updated = 0
    skipped_no_replay_row = 0

    for ticker in tickers:
        raw = load_ohlcv(ticker)
        by_date = _replay_by_date(ticker, raw)
        if not by_date:
            print(f"  {ticker}: no OHLCV/replay data, skipping")
            continue

        trade_rows = query_all(
            """
            SELECT id, side, trade_date, price
            FROM   ee_trend_hold_book_trades
            WHERE  book_id = ? AND ticker = ?
            ORDER BY trade_date ASC, id ASC
            """,
            (BOOK_ID, ticker),
        ) or []

        current_entry_price = None
        current_entry_date = None
        for tr in trade_rows:
            trade_date = str(tr["trade_date"])
            side = tr["side"]
            day = by_date.get(trade_date)
            if day is None:
                skipped_no_replay_row += 1
                continue

            if side == "BUY":
                current_entry_price = float(tr["price"]) if tr["price"] is not None else None
                current_entry_date = trade_date

            entry_gate = day.get("entry_gate")
            entry_gate_json = json.dumps(entry_gate) if entry_gate else None
            exit_gate_json = None
            if side in ("SCALE_OUT", "EXIT"):
                exit_gate = day.get("gate_snapshot")
                if exit_gate:
                    exit_gate_json = json.dumps(exit_gate)

            if entry_gate_json or exit_gate_json:
                exec_sql(
                    """
                    UPDATE ee_trend_hold_book_trades
                    SET entry_gate_json = COALESCE(entry_gate_json, ?),
                        exit_gate_json  = COALESCE(exit_gate_json, ?)
                    WHERE id = ?
                    """,
                    (entry_gate_json, exit_gate_json, tr["id"]),
                )
                trades_updated += 1

            if side in ("SCALE_OUT", "EXIT"):
                lesson = query_one(
                    "SELECT entry_price FROM ee_trend_hold_book_lessons "
                    "WHERE book_id = ? AND ticker = ? AND trade_date = ?",
                    (BOOK_ID, ticker, trade_date),
                )
                if lesson is not None:
                    trade_full = query_one(
                        "SELECT quantity, price, realized_pnl_kwd, commission_kwd FROM ee_trend_hold_book_trades "
                        "WHERE book_id = ? AND ticker = ? AND trade_date = ?",
                        (BOOK_ID, ticker, trade_date),
                    )
                    # Always re-run the classifier here (not just when the
                    # lesson previously lacked entry_gate): analyze_trade()
                    # is a pure function of OHLCV + dates/prices + entry_gate,
                    # so re-running with the same inputs reproduces the same
                    # classification every time -- this is what upgrades an
                    # existing QUICK_STOP's generic template enhancement into
                    # one that cites this trade's actual rel-volume/CMF
                    # numbers, and keeps re-runs fully idempotent.
                    reclassified = None
                    if entry_gate and current_entry_price:
                        try:
                            reclassified = analyze_trade(
                                side=side,
                                entry_date=current_entry_date,
                                entry_price=current_entry_price,
                                exit_date=trade_date,
                                exit_price=float(tr["price"]),
                                ohlcv=raw,
                                entry_gate=entry_gate,
                            )
                        except Exception as exc:
                            print(f"    {ticker} {trade_date}: reclassify failed ({exc}), keeping stored lesson")

                    if reclassified is not None:
                        exec_sql(
                            """
                            UPDATE ee_trend_hold_book_lessons
                            SET classification   = ?,
                                outcome          = ?,
                                mae_pct          = ?,
                                mfe_pct          = ?,
                                giveback_pct     = ?,
                                holding_days     = ?,
                                reason           = ?,
                                enhancement      = ?,
                                entry_price      = COALESCE(entry_price, ?),
                                exit_price       = COALESCE(exit_price, ?),
                                quantity         = COALESCE(quantity, ?),
                                realized_pnl_kwd = COALESCE(realized_pnl_kwd, ?),
                                commission_kwd   = COALESCE(commission_kwd, ?),
                                entry_gate_json  = COALESCE(entry_gate_json, ?),
                                exit_gate_json   = COALESCE(exit_gate_json, ?)
                            WHERE book_id = ? AND ticker = ? AND trade_date = ?
                            """,
                            (
                                reclassified.classification,
                                reclassified.outcome,
                                reclassified.mae_pct,
                                reclassified.mfe_pct,
                                reclassified.giveback_pct,
                                reclassified.holding_days,
                                reclassified.reason,
                                reclassified.enhancement,
                                current_entry_price,
                                float(trade_full["price"]) if trade_full and trade_full["price"] is not None else None,
                                float(trade_full["quantity"]) if trade_full and trade_full["quantity"] is not None else None,
                                float(trade_full["realized_pnl_kwd"]) if trade_full and trade_full["realized_pnl_kwd"] is not None else None,
                                float(trade_full["commission_kwd"]) if trade_full and trade_full["commission_kwd"] is not None else None,
                                entry_gate_json,
                                exit_gate_json,
                                BOOK_ID, ticker, trade_date,
                            ),
                        )
                    else:
                        exec_sql(
                            """
                            UPDATE ee_trend_hold_book_lessons
                            SET entry_price      = COALESCE(entry_price, ?),
                                exit_price       = COALESCE(exit_price, ?),
                                quantity         = COALESCE(quantity, ?),
                                realized_pnl_kwd = COALESCE(realized_pnl_kwd, ?),
                                commission_kwd   = COALESCE(commission_kwd, ?),
                                entry_gate_json  = COALESCE(entry_gate_json, ?),
                                exit_gate_json   = COALESCE(exit_gate_json, ?)
                            WHERE book_id = ? AND ticker = ? AND trade_date = ?
                            """,
                            (
                                current_entry_price,
                                float(trade_full["price"]) if trade_full and trade_full["price"] is not None else None,
                                float(trade_full["quantity"]) if trade_full and trade_full["quantity"] is not None else None,
                                float(trade_full["realized_pnl_kwd"]) if trade_full and trade_full["realized_pnl_kwd"] is not None else None,
                                float(trade_full["commission_kwd"]) if trade_full and trade_full["commission_kwd"] is not None else None,
                                entry_gate_json,
                                exit_gate_json,
                                BOOK_ID, ticker, trade_date,
                            ),
                        )
                    lessons_updated += 1

        # Currently open position (if any): carry the latest replay row's
        # entry_gate onto ee_trend_hold_book_positions.
        pos = query_one(
            "SELECT ticker FROM ee_trend_hold_book_positions WHERE book_id = ? AND ticker = ?", (BOOK_ID, ticker)
        )
        if pos is not None:
            latest_dates = sorted(by_date.keys())
            latest_row = by_date[latest_dates[-1]] if latest_dates else None
            entry_gate = latest_row.get("entry_gate") if latest_row else None
            if entry_gate:
                exec_sql(
                    """
                    UPDATE ee_trend_hold_book_positions
                    SET entry_gate_json = COALESCE(entry_gate_json, ?)
                    WHERE book_id = ? AND ticker = ?
                    """,
                    (json.dumps(entry_gate), BOOK_ID, ticker),
                )
                positions_updated += 1

        print(f"  {ticker}: {len(trade_rows)} trade row(s) processed")

    print(
        json.dumps(
            {
                "tickers": len(tickers),
                "trades_updated": trades_updated,
                "lessons_updated": lessons_updated,
                "positions_updated": positions_updated,
                "skipped_no_replay_row": skipped_no_replay_row,
            }
        )
    )


if __name__ == "__main__":
    main()

"""
Eagle Eye - Inspect Recommendation Tracker Database (READ-ONLY)

Shows what the tracker has accumulated:
  - ee_rating_snapshots : daily rating history
  - ee_signal_tracker   : BUY/SELL signals + forward P&L
  - ee_weekly_reviews   : saved weekly reports

Usage (from backend-api root):
    python scripts/inspect_tracker_db.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.getcwd())

from app.core.database import query_all, query_val


def _hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def inspect_snapshots() -> None:
    _hr("RATING SNAPSHOTS (daily history)")

    total = query_val("SELECT COUNT(*) FROM ee_rating_snapshots", ()) or 0
    print(f"Total snapshot rows: {total}")
    if total == 0:
        print("No snapshots yet. Run a recompute (compute_all_ratings) first.")
        return

    days = query_all(
        """
        SELECT snapshot_date, COUNT(*) AS rows
        FROM ee_rating_snapshots
        GROUP BY snapshot_date
        ORDER BY snapshot_date DESC
        LIMIT 15
        """,
        (),
    )
    print("\nSnapshot days captured (most recent 15):")
    print(f"{'Date':<14}{'Stocks':>8}")
    print("-" * 22)
    for day in days:
        print(f"{day['snapshot_date']:<14}{day['rows']:>8}")

    latest = query_val("SELECT MAX(snapshot_date) FROM ee_rating_snapshots", ())
    print(f"\nLatest snapshot date: {latest}")
    dist = query_all(
        """
        SELECT rating, COUNT(*) AS cnt
        FROM ee_rating_snapshots
        WHERE snapshot_date = ?
        GROUP BY rating
        ORDER BY cnt DESC
        """,
        (latest,),
    )
    print(f"\nRating distribution on {latest}:")
    print(f"{'Rating':<16}{'Count':>6}")
    print("-" * 22)
    for row in dist:
        print(f"{row['rating']:<16}{row['cnt']:>6}")


def inspect_signals() -> None:
    _hr("SIGNAL TRACKER (BUY/SELL signals + forward P&L)")

    total = query_val("SELECT COUNT(*) FROM ee_signal_tracker", ()) or 0
    open_cnt = query_val("SELECT COUNT(*) FROM ee_signal_tracker WHERE status='OPEN'", ()) or 0
    closed_cnt = query_val("SELECT COUNT(*) FROM ee_signal_tracker WHERE status='CLOSED'", ()) or 0
    buy_cnt = query_val("SELECT COUNT(*) FROM ee_signal_tracker WHERE signal_type='BUY'", ()) or 0
    sell_cnt = query_val("SELECT COUNT(*) FROM ee_signal_tracker WHERE signal_type='SELL'", ()) or 0

    print(f"Total signals tracked: {total}")
    print(f"  BUY: {buy_cnt}   SELL: {sell_cnt}")
    print(f"  OPEN (awaiting maturity): {open_cnt}   CLOSED (20d matured): {closed_cnt}")

    if total == 0:
        print("\nNo signals tracked yet.")
        return

    recent = query_all(
        """
        SELECT ticker, signal_type, signal_date, signal_price, confidence,
               stage, status, pnl_5d_pct, pnl_20d_pct, outcome_label
        FROM ee_signal_tracker
        ORDER BY signal_date DESC, ticker
        LIMIT 20
        """,
        (),
    )
    print("\nMost recent 20 signals:")
    print(
        f"{'Ticker':<10}{'Type':<6}{'Date':<12}{'Price':>8}{'Conf':>6}"
        f"{'Status':>8}{'5dPnL':>8}{'20dPnL':>8}{'Outcome':>12}"
    )
    print("-" * 88)
    for signal in recent:
        price = f"{signal['signal_price']:.2f}" if signal.get("signal_price") is not None else "-"
        conf = f"{signal['confidence']:.0f}" if signal.get("confidence") is not None else "-"
        pnl_5d = f"{signal['pnl_5d_pct']:+.1f}" if signal.get("pnl_5d_pct") is not None else "-"
        pnl_20d = f"{signal['pnl_20d_pct']:+.1f}" if signal.get("pnl_20d_pct") is not None else "-"
        outcome = signal.get("outcome_label") or "-"
        print(
            f"{signal['ticker']:<10}{signal['signal_type']:<6}{signal['signal_date']:<12}"
            f"{price:>8}{conf:>6}{signal['status']:>8}{pnl_5d:>8}{pnl_20d:>8}{outcome:>12}"
        )

    if closed_cnt > 0:
        _hr("CLOSED SIGNAL PERFORMANCE")
        perf = query_all(
            """
            SELECT signal_type,
                   COUNT(*) AS n,
                   ROUND(AVG(pnl_20d_pct), 2) AS avg_pnl,
                   ROUND(AVG(max_gain_20d), 2) AS avg_max_gain,
                   ROUND(AVG(max_drawdown_20d), 2) AS avg_max_dd,
                   SUM(hit_tp1) AS tp1_hits,
                   SUM(hit_stop) AS stop_hits
            FROM ee_signal_tracker
            WHERE status='CLOSED'
            GROUP BY signal_type
            """,
            (),
        )
        for row in perf:
            print(f"\n{row['signal_type']} signals ({row['n']} closed):")
            print(f"  Avg 20d P&L:      {row['avg_pnl']}%")
            print(f"  Avg max gain:     {row['avg_max_gain']}%")
            print(f"  Avg max drawdown: {row['avg_max_dd']}%")
            print(f"  TP1 hits:         {row['tp1_hits']}/{row['n']}")
            print(f"  Stop hits:        {row['stop_hits']}/{row['n']}")

        outcomes = query_all(
            """
            SELECT outcome_label, COUNT(*) AS cnt
            FROM ee_signal_tracker
            WHERE status='CLOSED' AND outcome_label IS NOT NULL
            GROUP BY outcome_label
            ORDER BY cnt DESC
            """,
            (),
        )
        if outcomes:
            print("\nOutcome breakdown:")
            for outcome in outcomes:
                print(f"  {outcome['outcome_label']:<14}{outcome['cnt']:>4}")
    else:
        print("\n(No signals have matured to 20 days yet - outcomes will populate")
        print(" as forward price data becomes available.)")


def inspect_reviews() -> None:
    _hr("WEEKLY REVIEWS (saved reports)")

    total = query_val("SELECT COUNT(*) FROM ee_weekly_reviews", ()) or 0
    print(f"Total saved reviews: {total}")
    if total == 0:
        print("No weekly reviews generated yet. Run: python run_weekly_review.py")
        return

    reviews = query_all(
        """
        SELECT week_start, week_end, created_at
        FROM ee_weekly_reviews
        ORDER BY week_start DESC
        LIMIT 10
        """,
        (),
    )
    print(f"\n{'Week Start':<14}{'Week End':<14}")
    print("-" * 28)
    for review in reviews:
        print(f"{review['week_start']:<14}{review['week_end']:<14}")


def main() -> None:
    print("\nEAGLE EYE - RECOMMENDATION TRACKER DATABASE INSPECTION")
    print("(read-only - no changes made)")

    try:
        inspect_snapshots()
    except Exception as exc:
        print(f"\n[ee_rating_snapshots] not available: {exc}")

    try:
        inspect_signals()
    except Exception as exc:
        print(f"\n[ee_signal_tracker] not available: {exc}")

    try:
        inspect_reviews()
    except Exception as exc:
        print(f"\n[ee_weekly_reviews] not available: {exc}")

    print("\n" + "=" * 70)
    print("  Inspection complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
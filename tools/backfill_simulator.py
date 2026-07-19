#!/usr/bin/env python3
"""
backfill_simulator.py
─────────────────────
Backfills the Eagle Eye Paper Trading Simulator for historical dates.

Run from the backend-api directory:

    PYTHONPATH=. python tools/backfill_simulator.py [--from 2025-01-01] [--to today] [--dry-run]

The script iterates each Kuwait trading day (Sun–Thu) in the range and calls
SimulatorEngine.run_daily(date).  It relies on historical data that is already
present in ee_ratings_cache and ee_ohlcv_cache.

If the simulator has already processed a date (a daily snapshot exists for that
date) it is skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta

# ── Path bootstrap ────────────────────────────────────────────────────────────
# When PYTHONPATH=. the app package is importable directly.
# This check gives a clearer error if the user forgot to set PYTHONPATH.
try:
    from app.services.eagle_eye.simulator import get_engine
    from app.core.database import query_val
except ModuleNotFoundError as exc:
    print(
        f"\n[ERROR] Could not import backend modules: {exc}\n"
        "       Make sure you run this from the backend-api/ directory with:\n"
        "           PYTHONPATH=. python tools/backfill_simulator.py\n",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Helpers ───────────────────────────────────────────────────────────────────

KUWAIT_WEEKEND = {5, 6}  # Friday=5, Saturday=6  (Mon=0 … Sun=6)


def is_trading_day(d: date) -> bool:
    """Kuwait Stock Exchange is open Sun–Thu."""
    return d.weekday() not in KUWAIT_WEEKEND


def date_range(start: date, end: date):
    """Yield every calendar date in [start, end]."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def already_processed(d: date) -> bool:
    """Return True if all configured simulator cards already have a snapshot for this date."""
    try:
        count = query_val(
            "SELECT COUNT(*) FROM simulator_daily_snapshots WHERE date = ?",
            (d.isoformat(),),
        )
        portfolio_count = query_val("SELECT COUNT(*) FROM simulator_portfolios", ()) or 0
        return (count or 0) >= max(1, int(portfolio_count))
    except Exception:
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Eagle Eye simulator")
    parser.add_argument(
        "--from",
        dest="from_date",
        default="2025-01-01",
        help="Start date inclusive (YYYY-MM-DD). Default: 2025-01-01",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        default=date.today().isoformat(),
        help="End date inclusive (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which dates would be processed without actually running them",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process dates even if snapshots already exist",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Seconds to sleep between dates (default 0.1) — avoids hammering DB",
    )
    args = parser.parse_args()

    try:
        start = date.fromisoformat(args.from_date)
        end = date.fromisoformat(args.to_date)
    except ValueError as e:
        print(f"[ERROR] Invalid date: {e}", file=sys.stderr)
        sys.exit(1)

    if start > end:
        print("[ERROR] --from date must be ≤ --to date", file=sys.stderr)
        sys.exit(1)

    trading_days = [d for d in date_range(start, end) if is_trading_day(d)]
    print(
        f"\nEagle Eye Simulator Backfill"
        f"\n  Range  : {start} to {end}"
        f"\n  Trading days: {len(trading_days)}"
        f"\n  Dry run: {args.dry_run}"
        f"\n  Force  : {args.force}"
        f"\n{'-' * 60}"
    )

    if not trading_days:
        print("Nothing to process.")
        return

    engine = get_engine()
    processed = skipped = errors = 0

    for d in trading_days:
        date_str = d.isoformat()

        if not args.force and already_processed(d):
            skipped += 1
            continue

        try:
            engine._assert_rating_snapshot_available(date_str)
        except Exception as exc:
            print(f"  ✗ {date_str}  ERROR: {exc}")
            errors += 1
            if not args.dry_run:
                continue
            continue

        if args.dry_run:
            print(f"  [DRY RUN] Would process {date_str}")
            processed += 1
            continue

        try:
            engine.run_daily(d)
            print(f"  ✓ {date_str}")
            processed += 1
        except Exception as exc:
            print(f"  ✗ {date_str}  ERROR: {exc}")
            errors += 1

        if args.delay > 0:
            time.sleep(args.delay)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Done.  Processed: {processed}  Skipped: {skipped}  Errors: {errors}")

    if not args.dry_run:
        print("\nPortfolio summary:")
        try:
            from app.core.database import query_all
            rows = query_all(
                """
                SELECT p.strategy_name,
                       p.total_value_kwd,
                       COALESCE(t.trade_count, 0) AS total_trades,
                       ROUND((p.total_value_kwd - p.starting_capital_kwd) / p.starting_capital_kwd * 100, 2) AS ret_pct
                FROM simulator_portfolios p
                LEFT JOIN (
                    SELECT portfolio_id, COUNT(*) AS trade_count
                    FROM sim_transactions
                    GROUP BY portfolio_id
                ) t ON t.portfolio_id = p.id
                ORDER BY p.id
                """,
            )
            for r in rows:
                ret = r.get("ret_pct", 0) or 0
                sign = "+" if ret >= 0 else ""
                print(
                    f"  {r['strategy_name']:20s}  "
                    f"Value: {r['total_value_kwd']:9.2f} KWD  "
                    f"Trades: {r['total_trades']:4d}  "
                    f"Return: {sign}{ret:.2f}%"
                )
        except Exception as e:
            print(f"  (Could not fetch summary: {e})")


if __name__ == "__main__":
    main()

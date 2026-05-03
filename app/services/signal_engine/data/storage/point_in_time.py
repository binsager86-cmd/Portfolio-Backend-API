"""Point-in-time (PIT) universe management for survivorship-bias-free backtesting.

Problem
-------
If we test against the *current* list of Premier Market stocks we introduce
survivorship bias — stocks that were delisted, suspended, or merged between
2020 and 2025 are invisible to our backtest.  This inflates the apparent win
rate because we inadvertently only test against "survivors."

Solution
--------
Maintain a time-indexed snapshot of which stocks were in the Premier universe
on each date.  A walk-forward backtest then uses get_universe_at(date) to
obtain the correct set of stocks for that point in time.

Snapshot format (in-memory or JSON file)
-----------------------------------------
A list of snapshot dicts ordered ascending by date:

    [
        {"date": "2020-01-01", "stocks": ["NBK", "KFH", "ZAIN", ...]},
        {"date": "2021-01-01", "stocks": ["NBK", "KFH", "ZAIN", "BURG", ...]},
        ...
    ]

The last snapshot whose "date" ≤ query_date is used (i.e., changes take
effect the day they are recorded).  If no snapshot predates the query the
current PREMIER_STOCKS list from kuwait_constants is used as fallback.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from app.services.signal_engine.config.kuwait_constants import PREMIER_STOCKS

logger = logging.getLogger(__name__)

_DEFAULT_SNAPSHOTS_PATH = Path(os.environ.get(
    "SIGNAL_PIT_PATH",
    Path(__file__).parent.parent.parent.parent.parent / "data" / "pit_snapshots.json",
))


# ── Load / persist snapshots ──────────────────────────────────────────────────

def load_snapshots(path: Path | None = None) -> list[dict[str, Any]]:
    """Load universe snapshots from a JSON file.

    Returns an empty list (triggering fallback to PREMIER_STOCKS) if the
    file does not exist or cannot be parsed.
    """
    p = path or _DEFAULT_SNAPSHOTS_PATH
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("PIT snapshots file has unexpected format at %s", p)
            return []
        return sorted(data, key=lambda s: s.get("date", ""))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load PIT snapshots from %s: %s", p, exc)
        return []


def save_snapshots(snapshots: list[dict[str, Any]], path: Path | None = None) -> bool:
    """Persist universe snapshots to a JSON file."""
    p = path or _DEFAULT_SNAPSHOTS_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(sorted(snapshots, key=lambda s: s.get("date", "")), f, indent=2)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save PIT snapshots to %s: %s", p, exc)
        return False


def add_snapshot(
    date: str,
    stocks: list[str],
    snapshots: list[dict[str, Any]] | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Add or update a universe snapshot for a given date.

    If a snapshot already exists for the exact date it is replaced.

    Args:
        date:      Snapshot date in YYYY-MM-DD format.
        stocks:    List of stock codes active on this date.
        snapshots: Existing snapshots list (if already loaded).
        path:      File path for persistence.

    Returns:
        Updated snapshots list.
    """
    current = snapshots if snapshots is not None else load_snapshots(path)
    # Replace existing entry for same date
    updated = [s for s in current if s.get("date") != date]
    updated.append({"date": date, "stocks": [c.upper() for c in stocks]})
    updated.sort(key=lambda s: s.get("date", ""))
    if path or _DEFAULT_SNAPSHOTS_PATH.parent.exists():
        save_snapshots(updated, path)
    return updated


# ── Point-in-time universe lookup ─────────────────────────────────────────────

def get_universe_at(
    query_date: str,
    snapshots: list[dict[str, Any]] | None = None,
    path: Path | None = None,
) -> list[str]:
    """Return the Premier Market universe as it existed on query_date.

    Implements the "as-of" lookup:
      - Find the most recent snapshot whose date ≤ query_date.
      - If none exists, fall back to the current PREMIER_STOCKS constant.

    Args:
        query_date: Date string in YYYY-MM-DD format.
        snapshots:  Pre-loaded snapshots (avoids disk read if already in memory).
        path:       Override file path for snapshots.

    Returns:
        List of active stock codes on query_date.
    """
    loaded = snapshots if snapshots is not None else load_snapshots(path)
    if not loaded:
        return list(PREMIER_STOCKS)

    # Find the latest snapshot whose date ≤ query_date
    best: dict[str, Any] | None = None
    for snapshot in loaded:
        snap_date = snapshot.get("date", "")
        if snap_date <= query_date:
            if best is None or snap_date > best.get("date", ""):
                best = snapshot

    if best is None:
        logger.debug("No PIT snapshot found before %s — using PREMIER_STOCKS fallback", query_date)
        return list(PREMIER_STOCKS)

    stocks = best.get("stocks", [])
    if not stocks:
        return list(PREMIER_STOCKS)

    logger.debug("PIT universe at %s (snapshot %s): %d stocks", query_date, best["date"], len(stocks))
    return list(stocks)


# ── Survivorship bias helpers ─────────────────────────────────────────────────

def get_delisted_stocks(
    from_date: str,
    to_date: str,
    snapshots: list[dict[str, Any]] | None = None,
    path: Path | None = None,
) -> list[str]:
    """Return stocks that were in the universe at from_date but not at to_date.

    These are the "survivorship bias" candidates — stocks that would be
    invisible if we only used the current universe for backtesting.

    Args:
        from_date: Start date (inclusive) in YYYY-MM-DD format.
        to_date:   End date (inclusive) in YYYY-MM-DD format.
        snapshots: Pre-loaded snapshots.
        path:      Override file path.

    Returns:
        List of stock codes that were delisted between from_date and to_date.
    """
    loaded = snapshots if snapshots is not None else load_snapshots(path)
    start_universe = set(get_universe_at(from_date, snapshots=loaded))
    end_universe = set(get_universe_at(to_date, snapshots=loaded))
    return sorted(start_universe - end_universe)


def get_newly_listed_stocks(
    from_date: str,
    to_date: str,
    snapshots: list[dict[str, Any]] | None = None,
    path: Path | None = None,
) -> list[str]:
    """Return stocks added to the universe between from_date and to_date."""
    loaded = snapshots if snapshots is not None else load_snapshots(path)
    start_universe = set(get_universe_at(from_date, snapshots=loaded))
    end_universe = set(get_universe_at(to_date, snapshots=loaded))
    return sorted(end_universe - start_universe)


def build_backtest_universe(
    test_from: str,
    test_to: str,
    snapshots: list[dict[str, Any]] | None = None,
    path: Path | None = None,
) -> list[str]:
    """Return the union of all stocks active at any point during the test window.

    This is the correct bias-free universe for a walk-forward backtest window:
    we must include stocks that were listed at the START of the test window
    (even if later delisted) to avoid survivorship bias.

    Args:
        test_from: Test window start date.
        test_to:   Test window end date.

    Returns:
        Deduplicated sorted list of stock codes.
    """
    loaded = snapshots if snapshots is not None else load_snapshots(path)
    start = set(get_universe_at(test_from, snapshots=loaded))
    end = set(get_universe_at(test_to, snapshots=loaded))
    # Union: test any stock that was ever active during the window
    return sorted(start | end)

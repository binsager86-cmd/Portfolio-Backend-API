"""Unit tests for the Kuwait Signal Engine — Point-in-Time Universe.

Snapshot format: list[dict] where each dict has {"date": str, "stocks": list[str]}.

Validates:
  - get_universe_at with no snapshots falls back to PREMIER_STOCKS
  - add_snapshot + get_universe_at returns saved universe
  - build_backtest_universe returns combined list of all stocks
  - get_delisted_stocks detects removals from universe
  - get_newly_listed_stocks detects additions to universe
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services.signal_engine.config.kuwait_constants import PREMIER_STOCKS
from app.services.signal_engine.data.storage.point_in_time import (
    add_snapshot,
    build_backtest_universe,
    get_delisted_stocks,
    get_newly_listed_stocks,
    get_universe_at,
    load_snapshots,
    save_snapshots,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def snap_path(tmp_path):
    return Path(tmp_path) / "snapshots.json"


def _snaps(*items: tuple[str, list[str]]) -> list[dict]:
    """Create snapshots list from (date, stocks) tuples."""
    return [{"date": d, "stocks": s} for d, s in items]


# ── Fallback to PREMIER_STOCKS ────────────────────────────────────────────────

class TestFallback:
    def test_no_snapshots_returns_premier_stocks(self):
        universe = get_universe_at("2024-01-15", snapshots=[])
        assert isinstance(universe, list)
        premier = set(PREMIER_STOCKS)
        assert len(set(universe) & premier) > 0

    def test_before_all_snapshot_dates_returns_fallback(self):
        # Only snapshot is in 2030 — before that should fall back to PREMIER_STOCKS
        snaps = _snaps(("2030-01-01", ["NBK", "ZAIN"]))
        universe = get_universe_at("2020-01-01", snapshots=snaps)
        assert len(universe) > 0


# ── Round-trip snapshot ───────────────────────────────────────────────────────

class TestSnapshotRoundTrip:
    def test_add_and_retrieve_snapshot(self, snap_path):
        snaps = add_snapshot("2024-06-01", ["NBK", "BOUBYAN", "BURG"],
                             snapshots=[], path=snap_path)
        universe = get_universe_at("2024-06-01", snapshots=snaps)
        assert set(universe) == {"NBK", "BOUBYAN", "BURG"}

    def test_retrieves_most_recent_snapshot_on_or_before(self, snap_path):
        snaps = _snaps(("2024-01-01", ["NBK"]), ("2024-06-01", ["NBK", "ZAIN"]))
        universe = get_universe_at("2024-03-15", snapshots=snaps)
        # Closest snapshot on-or-before 2024-03-15 is 2024-01-01
        assert "ZAIN" not in universe

    def test_exact_date_match(self):
        snaps = _snaps(("2024-01-01", ["NBK"]), ("2024-06-01", ["NBK", "ZAIN"]))
        universe = get_universe_at("2024-06-01", snapshots=snaps)
        assert "ZAIN" in universe


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load_snapshots(self, snap_path):
        snaps = _snaps(("2024-01-01", ["NBK", "ZAIN"]))
        save_snapshots(snaps, path=snap_path)
        loaded = load_snapshots(path=snap_path)
        assert isinstance(loaded, list)
        assert len(loaded) == 1
        assert loaded[0]["date"] == "2024-01-01"
        assert "ZAIN" in loaded[0]["stocks"]

    def test_load_missing_file_returns_empty(self, snap_path):
        missing = snap_path.parent / "nonexistent.json"
        loaded = load_snapshots(path=missing)
        assert loaded == []


# ── build_backtest_universe ───────────────────────────────────────────────────

class TestBuildBacktestUniverse:
    def test_returns_list(self):
        snaps = _snaps(("2022-01-01", ["NBK"]), ("2023-01-01", ["NBK", "ZAIN"]))
        result = build_backtest_universe("2022-01-01", "2023-12-31", snapshots=snaps)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_includes_stocks_from_both_ends(self):
        snaps = _snaps(("2022-01-01", ["NBK", "BURG"]), ("2023-01-01", ["NBK", "ZAIN"]))
        result = build_backtest_universe("2022-01-01", "2023-01-01", snapshots=snaps)
        # Union: should include all stocks active at either end
        assert "NBK" in result
        assert "BURG" in result
        assert "ZAIN" in result


# ── Delisted / newly listed ───────────────────────────────────────────────────

class TestDelistedAndNewlyListed:
    def test_delisted_stocks_detected(self):
        snaps = _snaps(("2022-01-01", ["NBK", "BOUBYAN", "BURG"]),
                        ("2023-01-01", ["NBK", "BOUBYAN"]))
        delisted = get_delisted_stocks("2022-01-01", "2023-01-01", snapshots=snaps)
        assert "BURG" in delisted

    def test_newly_listed_detected(self):
        snaps = _snaps(("2022-01-01", ["NBK"]), ("2023-01-01", ["NBK", "ZAIN"]))
        new_stocks = get_newly_listed_stocks("2022-01-01", "2023-01-01", snapshots=snaps)
        assert "ZAIN" in new_stocks

    def test_no_changes_returns_empty(self):
        snaps = _snaps(("2022-01-01", ["NBK", "ZAIN"]), ("2023-01-01", ["NBK", "ZAIN"]))
        assert get_delisted_stocks("2022-01-01", "2023-01-01", snapshots=snaps) == []
        assert get_newly_listed_stocks("2022-01-01", "2023-01-01", snapshots=snaps) == []

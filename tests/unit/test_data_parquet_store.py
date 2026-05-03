"""Unit tests for the Kuwait Signal Engine — Parquet Store.

Skipped entirely when pyarrow is not installed.

Validates:
  - save_rows / load_rows round-trip preserves all rows
  - load_rows with date filter returns only matching dates
  - list_stored_stocks returns saved symbol
  - get_date_range returns (min_date, max_date) strings
  - delete_stock removes the file
  - Saving additional rows merges/upserts (no duplicates by date)
"""
from __future__ import annotations

import os
import tempfile

import pytest

pyarrow = pytest.importorskip("pyarrow")   # skip entire module when pyarrow absent

from app.services.signal_engine.data.storage.parquet_store import (
    delete_stock,
    get_date_range,
    list_stored_stocks,
    load_rows,
    save_rows,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_rows(symbol: str = "NBK", n: int = 5) -> list[dict]:
    base = "2024-01-0"
    return [
        {"symbol": symbol, "date": f"2024-01-0{i + 1}", "open": 500.0, "high": 505.0,
         "low": 495.0, "close": 501.0 + i, "volume": 1_000_000.0, "value": 500_000.0}
        for i in range(n)
    ]


@pytest.fixture()
def store_dir(tmp_path):
    """Provide a unique temporary directory for each test."""
    return str(tmp_path)


# ── Round-trip ────────────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_save_and_load(self, store_dir):
        rows = _sample_rows("NBK", n=5)
        save_rows("NBK", rows, store_dir=store_dir)
        loaded = load_rows("NBK", store_dir=store_dir)
        assert len(loaded) == 5

    def test_all_dates_preserved(self, store_dir):
        rows = _sample_rows("BOUBYAN", n=3)
        save_rows("BOUBYAN", rows, store_dir=store_dir)
        loaded = load_rows("BOUBYAN", store_dir=store_dir)
        dates = {r["date"] for r in loaded}
        assert "2024-01-01" in dates
        assert "2024-01-03" in dates

    def test_close_values_preserved(self, store_dir):
        rows = _sample_rows("NBK", n=3)
        save_rows("NBK", rows, store_dir=store_dir)
        loaded = load_rows("NBK", store_dir=store_dir)
        loaded_sorted = sorted(loaded, key=lambda r: r["date"])
        assert loaded_sorted[0]["close"] == pytest.approx(501.0)


# ── Date filtering ────────────────────────────────────────────────────────────

class TestDateFilter:
    def test_load_with_start_date(self, store_dir):
        rows = _sample_rows("NBK", n=5)
        save_rows("NBK", rows, store_dir=store_dir)
        loaded = load_rows("NBK", start_date="2024-01-03", store_dir=store_dir)
        for r in loaded:
            assert r["date"] >= "2024-01-03"

    def test_load_with_end_date(self, store_dir):
        rows = _sample_rows("NBK", n=5)
        save_rows("NBK", rows, store_dir=store_dir)
        loaded = load_rows("NBK", end_date="2024-01-03", store_dir=store_dir)
        for r in loaded:
            assert r["date"] <= "2024-01-03"


# ── Metadata ─────────────────────────────────────────────────────────────────

class TestMetadata:
    def test_list_stored_stocks_includes_saved(self, store_dir):
        save_rows("ZAIN", _sample_rows("ZAIN", 2), store_dir=store_dir)
        symbols = list_stored_stocks(store_dir=store_dir)
        assert "ZAIN" in symbols

    def test_get_date_range(self, store_dir):
        rows = _sample_rows("NBK", n=5)
        save_rows("NBK", rows, store_dir=store_dir)
        min_d, max_d = get_date_range("NBK", store_dir=store_dir)
        assert min_d == "2024-01-01"
        assert max_d == "2024-01-05"


# ── Delete ────────────────────────────────────────────────────────────────────

class TestDelete:
    def test_delete_removes_stock(self, store_dir):
        save_rows("BURG", _sample_rows("BURG", 2), store_dir=store_dir)
        delete_stock("BURG", store_dir=store_dir)
        assert "BURG" not in list_stored_stocks(store_dir=store_dir)


# ── Upsert/merge ──────────────────────────────────────────────────────────────

class TestUpsert:
    def test_no_duplicates_when_resaving_same_dates(self, store_dir):
        rows = _sample_rows("NBK", n=3)
        save_rows("NBK", rows, store_dir=store_dir)
        save_rows("NBK", rows, store_dir=store_dir)   # same data again
        loaded = load_rows("NBK", store_dir=store_dir)
        assert len(loaded) == 3   # no duplicates

    def test_merge_adds_new_rows(self, store_dir):
        rows1 = _sample_rows("NBK", n=3)
        rows2 = [{"symbol": "NBK", "date": "2024-01-10", "open": 510.0, "high": 515.0,
                   "low": 505.0, "close": 512.0, "volume": 1_500_000.0, "value": 600_000.0}]
        save_rows("NBK", rows1, store_dir=store_dir)
        save_rows("NBK", rows2, store_dir=store_dir)
        loaded = load_rows("NBK", store_dir=store_dir)
        dates = {r["date"] for r in loaded}
        assert "2024-01-10" in dates
        assert len(dates) == 4

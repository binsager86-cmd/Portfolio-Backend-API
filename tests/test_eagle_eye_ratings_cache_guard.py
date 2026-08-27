# ruff: noqa: E402

from __future__ import annotations

import os
import sys

import pytest

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

from app.services.eagle_eye import ingest


def test_check_ratings_cache_drop_warns_on_large_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[tuple[str, str]] = []

    monkeypatch.setattr("app.services.eagle_eye.store.get_ratings_cache_row_count", lambda: 70)
    monkeypatch.setattr("app.services.eagle_eye.store.get_last_ratings_cache_row_count", lambda: 100)
    monkeypatch.setattr(
        "app.services.eagle_eye.store.log_compute",
        lambda run_type, ticker, status, message="": logged.append((status, message)),
    )
    recorded: list[tuple[str, int]] = []
    monkeypatch.setattr(
        "app.services.eagle_eye.store.record_ratings_cache_row_count",
        lambda run_id, row_count: recorded.append((run_id, row_count)),
    )

    ingest.check_ratings_cache_drop("run_test_1")

    warnings = [msg for status, msg in logged if status == "warning"]
    assert warnings, "expected a warning log row for a >20% cache drop"
    assert "RATINGS_CACHE_UNEXPLAINED_LOSS" in warnings[0]
    assert "previous_rows=100" in warnings[0]
    assert "current_rows=70" in warnings[0]
    assert recorded == [("run_test_1", 70)]


def test_check_ratings_cache_drop_no_warning_on_small_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[tuple[str, str]] = []

    monkeypatch.setattr("app.services.eagle_eye.store.get_ratings_cache_row_count", lambda: 95)
    monkeypatch.setattr("app.services.eagle_eye.store.get_last_ratings_cache_row_count", lambda: 100)
    monkeypatch.setattr(
        "app.services.eagle_eye.store.log_compute",
        lambda run_type, ticker, status, message="": logged.append((status, message)),
    )
    monkeypatch.setattr("app.services.eagle_eye.store.record_ratings_cache_row_count", lambda run_id, row_count: None)

    ingest.check_ratings_cache_drop("run_test_2")

    assert not [msg for status, msg in logged if status == "warning"]


def test_check_ratings_cache_drop_no_previous_record(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[tuple[str, str]] = []
    recorded: list[tuple[str, int]] = []

    monkeypatch.setattr("app.services.eagle_eye.store.get_ratings_cache_row_count", lambda: 42)
    monkeypatch.setattr("app.services.eagle_eye.store.get_last_ratings_cache_row_count", lambda: None)
    monkeypatch.setattr(
        "app.services.eagle_eye.store.log_compute",
        lambda run_type, ticker, status, message="": logged.append((status, message)),
    )
    monkeypatch.setattr(
        "app.services.eagle_eye.store.record_ratings_cache_row_count",
        lambda run_id, row_count: recorded.append((run_id, row_count)),
    )

    ingest.check_ratings_cache_drop("run_test_3")

    assert not [msg for status, msg in logged if status == "warning"]
    assert recorded == [("run_test_3", 42)]

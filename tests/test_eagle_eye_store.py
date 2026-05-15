from __future__ import annotations

import logging
import sys
from types import ModuleType

import pytest

from app.services.eagle_eye import store


def _install_fake_database_module(monkeypatch: pytest.MonkeyPatch, exec_sql) -> None:
    fake_module = ModuleType("app.core.database")
    fake_module.exec_sql = exec_sql
    monkeypatch.setitem(sys.modules, "app.core.database", fake_module)


def _sample_rating_result() -> dict:
    return {
        "stage": "markup",
        "rating": "buy",
        "confidence": 87.5,
        "thesis": "Momentum and volume confirm the breakout.",
        "entry": {
            "entry_primary": 710.0,
            "entry_aggressive": 705.0,
            "entry_conservative": 715.0,
            "stop_loss": 690.0,
            "tp1": 740.0,
            "tp1_probability": 0.6,
            "tp2": 760.0,
            "tp2_probability": 0.3,
            "tp3": 780.0,
            "tp3_probability": 0.1,
        },
        "indicators": {"close": 720.0, "rsi": 61.2},
        "supports": [700.0],
        "resistances": [735.0],
        "days_of_history": 120,
        "computed_at": "2026-05-15",
        "volume_context": {
            "relative_volume": 1.4,
            "liquidity_tier": "TRADEABLE",
            "is_volume_confirmed": True,
        },
    }


def test_save_rating_uses_postgres_upsert(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, tuple]] = []

    def exec_sql(sql: str, params: tuple = ()) -> None:
        captured.append((sql, params))

    _install_fake_database_module(monkeypatch, exec_sql)
    monkeypatch.setattr(store, "_use_postgres_backend", lambda: True)

    store.save_rating("kfh", "KFH", "Banking", _sample_rating_result())

    assert len(captured) == 1
    sql, params = captured[0]
    assert "INSERT INTO ee_ratings_cache" in sql
    assert "ON CONFLICT (ticker) DO UPDATE SET" in sql
    assert "INSERT OR REPLACE" not in sql
    assert "volume_context_json = EXCLUDED.volume_context_json" in sql
    assert params[0] == "KFH"


def test_save_dna_uses_postgres_upsert(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, tuple]] = []

    def exec_sql(sql: str, params: tuple = ()) -> None:
        captured.append((sql, params))

    _install_fake_database_module(monkeypatch, exec_sql)
    monkeypatch.setattr(store, "_use_postgres_backend", lambda: True)

    store.save_dna("kfh", {"pattern": "markup"}, total_events=4, dominant_pattern="markup")

    assert len(captured) == 1
    sql, params = captured[0]
    assert "INSERT INTO ee_dna_profiles" in sql
    assert "ON CONFLICT (ticker) DO UPDATE SET" in sql
    assert "INSERT OR REPLACE" not in sql
    assert "dominant_pattern = EXCLUDED.dominant_pattern" in sql
    assert params[0] == "KFH"


def test_save_rating_logs_persistence_failures(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def exec_sql(sql: str, params: tuple = ()) -> None:  # noqa: ARG001
        raise RuntimeError("syntax error at or near OR")

    _install_fake_database_module(monkeypatch, exec_sql)
    monkeypatch.setattr(store, "_use_postgres_backend", lambda: True)

    with caplog.at_level(logging.ERROR), pytest.raises(
        RuntimeError,
        match="syntax error at or near OR",
    ):
        store.save_rating("kfh", "KFH", "Banking", _sample_rating_result())

    assert "Failed to persist Eagle Eye rating cache row for KFH" in caplog.text

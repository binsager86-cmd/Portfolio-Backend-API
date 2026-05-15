from __future__ import annotations

import sys
from types import ModuleType

from app.services.eagle_eye import store


def _install_fake_database_module(monkeypatch, captured):
    fake_module = ModuleType("app.core.database")

    def exec_sql(sql, params=()):  # noqa: ARG001
        captured["sql"] = sql
        captured["params"] = params

    fake_module.exec_sql = exec_sql
    monkeypatch.setitem(sys.modules, "app.core.database", fake_module)


def test_save_rating_uses_portable_on_conflict_upsert(monkeypatch):
    captured = {}
    _install_fake_database_module(monkeypatch, captured)

    store.save_rating(
        "aginv",
        "Agility",
        "Industrials",
        {
            "stage": "markup",
            "rating": "A",
            "confidence": 88.5,
            "thesis": "Momentum confirmed",
            "entry": {"entry_primary": 1.2},
            "indicators": {"close": 1.3},
            "supports": [1.1],
            "resistances": [1.4],
            "days_of_history": 250,
            "computed_at": "2026-05-15",
            "volume_context": {"liquidity_tier": "TRADEABLE"},
        },
    )

    normalized_sql = " ".join(captured["sql"].split())

    assert "INSERT OR REPLACE" not in normalized_sql
    assert "ON CONFLICT (ticker) DO UPDATE SET" in normalized_sql
    assert captured["params"][0] == "AGINV"


def test_save_dna_uses_portable_on_conflict_upsert(monkeypatch):
    captured = {}
    _install_fake_database_module(monkeypatch, captured)

    store.save_dna("aginv", {"pattern": "breakout"}, total_events=12, dominant_pattern="breakout")

    normalized_sql = " ".join(captured["sql"].split())

    assert "INSERT OR REPLACE" not in normalized_sql
    assert "ON CONFLICT (ticker) DO UPDATE SET" in normalized_sql
    assert captured["params"][0] == "AGINV"

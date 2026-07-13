from __future__ import annotations

from pathlib import Path

import pytest

from scripts.migrate_data import validate_migration_runtime_isolation


def test_migration_script_blocks_test_env_on_production_target(tmp_path: Path) -> None:
    base = tmp_path / "workspace"
    base.mkdir(parents=True, exist_ok=True)
    (base / ".env.production").write_text("ENVIRONMENT=production\nDATABASE_PATH=/data/portfolio.db\n", encoding="utf-8")

    # Run with sqlite path matching production and a postgres URL set.
    with pytest.raises(RuntimeError, match="TEST_OR_DEBUG_MUST_NOT_USE_DATABASE_URL"):
        validate_migration_runtime_isolation(
            sqlite_path="/data/portfolio.db",
            postgres_url="postgresql://prod-host/db",
            environment="test",
        )


def test_migration_script_blocks_debug_env_on_production_sqlite_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRODUCTION_DATABASE_PATH", "/data/portfolio.db")
    with pytest.raises(RuntimeError, match="TEST_OR_DEBUG_DATABASE_MUST_NOT_EQUAL_PRODUCTION"):
        validate_migration_runtime_isolation(
            sqlite_path="/data/portfolio.db",
            postgres_url="",
            environment="debug",
        )

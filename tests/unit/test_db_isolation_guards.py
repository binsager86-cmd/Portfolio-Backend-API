from __future__ import annotations

from pathlib import Path

import pytest

from app.core.db_isolation import (
    enforce_environment_database_isolation,
    ensure_debug_fixture_write_allowed,
)


def _base_dir(tmp_path: Path) -> Path:
    base = tmp_path / "workspace"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _write_env_production(base: Path, db_path: str) -> None:
    (base / ".env.production").write_text(
        f"ENVIRONMENT=production\nDATABASE_PATH={db_path}\n",
        encoding="utf-8",
    )


def _call(
    *,
    env: str,
    db_path: str,
    base_dir: Path,
    use_postgres: bool = False,
    database_url: str = "",
) -> None:
    enforce_environment_database_isolation(
        environment=env,
        database_abs_path=db_path,
        use_postgres=use_postgres,
        database_url=database_url,
        base_dir=base_dir,
    )


def test_test_env_blocked_on_production_sqlite_path(tmp_path: Path) -> None:
    base = _base_dir(tmp_path)
    prod = "/data/portfolio.db"
    _write_env_production(base, prod)
    with pytest.raises(RuntimeError, match="TEST_OR_DEBUG_DATABASE_MUST_NOT_EQUAL_PRODUCTION"):
        _call(env="test", db_path=prod, base_dir=base)


def test_debug_env_blocked_on_production_sqlite_path(tmp_path: Path) -> None:
    base = _base_dir(tmp_path)
    prod = "/data/portfolio.db"
    _write_env_production(base, prod)
    with pytest.raises(RuntimeError, match="TEST_OR_DEBUG_DATABASE_MUST_NOT_EQUAL_PRODUCTION"):
        _call(env="debug", db_path=prod, base_dir=base)


def test_development_blocked_on_production_without_explicit_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _base_dir(tmp_path)
    prod = "/data/portfolio.db"
    _write_env_production(base, prod)
    monkeypatch.delenv("ALLOW_DEVELOPMENT_ON_PRODUCTION_DB", raising=False)
    with pytest.raises(RuntimeError, match="PYTEST_DATABASE_MUST_NOT_EQUAL_PRODUCTION"):
        _call(env="development", db_path=prod, base_dir=base)


def test_development_allowed_on_production_with_explicit_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _base_dir(tmp_path)
    prod = "/data/portfolio.db"
    _write_env_production(base, prod)
    monkeypatch.setenv("ALLOW_DEVELOPMENT_ON_PRODUCTION_DB", "1")
    with pytest.raises(RuntimeError, match="PYTEST_DATABASE_MUST_NOT_EQUAL_PRODUCTION"):
        _call(env="development", db_path=prod, base_dir=base)


def test_test_or_debug_blocked_for_production_postgres_url(tmp_path: Path) -> None:
    base = _base_dir(tmp_path)
    _write_env_production(base, "/data/portfolio.db")
    with pytest.raises(RuntimeError, match="TEST_OR_DEBUG_MUST_NOT_USE_DATABASE_URL"):
        _call(
            env="test",
            db_path=str((tmp_path / "test_portfolio_one.db").resolve()),
            base_dir=base,
            use_postgres=True,
            database_url="postgresql://prod-host/portfolio",
        )


def test_relative_path_resolving_to_production_is_blocked(tmp_path: Path) -> None:
    base = _base_dir(tmp_path)
    prod_rel = "../shared/prod.db"
    _write_env_production(base, prod_rel)
    rel = "../shared/prod.db"
    with pytest.raises(RuntimeError, match="TEST_OR_DEBUG_DATABASE_MUST_NOT_EQUAL_PRODUCTION"):
        _call(env="test", db_path=str((base / rel).resolve()), base_dir=base)


def test_symlink_resolving_to_production_is_blocked(tmp_path: Path) -> None:
    base = _base_dir(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    prod = shared / "prod.db"
    prod.write_text("x", encoding="utf-8")
    _write_env_production(base, str(prod))

    link = base / "prod_link.db"
    try:
        link.symlink_to(prod)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Symlink creation requires elevated privileges on this Windows host")
        raise

    with pytest.raises(RuntimeError, match="TEST_OR_DEBUG_DATABASE_MUST_NOT_EQUAL_PRODUCTION"):
        _call(env="debug", db_path=str(link.resolve()), base_dir=base)


def test_missing_environment_is_blocked(tmp_path: Path) -> None:
    base = _base_dir(tmp_path)
    _write_env_production(base, "/data/portfolio.db")
    with pytest.raises(RuntimeError, match="ENVIRONMENT_MUST_BE_SET"):
        _call(env="", db_path=str((tmp_path / "x.db").resolve()), base_dir=base)


def test_unknown_environment_is_blocked(tmp_path: Path) -> None:
    base = _base_dir(tmp_path)
    _write_env_production(base, "/data/portfolio.db")
    with pytest.raises(RuntimeError, match="ENVIRONMENT_UNKNOWN"):
        _call(env="staging", db_path=str((tmp_path / "x.db").resolve()), base_dir=base)


def test_pytest_without_temp_database_is_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _base_dir(tmp_path)
    _write_env_production(base, "/data/portfolio.db")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests::x")
    monkeypatch.delenv("PYTEST_ALLOW_NON_TEMP_DB", raising=False)
    with pytest.raises(RuntimeError, match="PYTEST_DATABASE_MUST_BE_TEMP_FILE"):
        _call(env="test", db_path=str((tmp_path / "dev_portfolio.db").resolve()), base_dir=base)


def test_pytest_temp_database_is_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _base_dir(tmp_path)
    _write_env_production(base, "/data/portfolio.db")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests::x")
    _call(env="test", db_path=str((tmp_path / "test_portfolio_abc.db").resolve()), base_dir=base)


def test_debug_fixture_loader_without_opt_in_is_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _base_dir(tmp_path)
    _write_env_production(base, "/data/portfolio.db")
    monkeypatch.delenv("ALLOW_DEBUG_FIXTURE_WRITE", raising=False)
    with pytest.raises(RuntimeError, match="DEBUG_FIXTURE_WRITE_REQUIRES_ALLOW_DEBUG_FIXTURE_WRITE=1"):
        ensure_debug_fixture_write_allowed(
            environment="debug",
            database_abs_path=str((tmp_path / "debug.db").resolve()),
            base_dir=base,
        )


def test_debug_fixture_loader_is_blocked_on_production_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _base_dir(tmp_path)
    prod = "/data/portfolio.db"
    _write_env_production(base, prod)
    monkeypatch.setenv("ALLOW_DEBUG_FIXTURE_WRITE", "1")
    with pytest.raises(RuntimeError, match="DEBUG_FIXTURE_WRITE_DATABASE_MUST_NOT_EQUAL_PRODUCTION"):
        ensure_debug_fixture_write_allowed(
            environment="debug",
            database_abs_path=prod,
            base_dir=base,
        )

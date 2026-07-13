from __future__ import annotations

import os
from pathlib import Path


def _parse_env_file_var(env_file: Path, key: str) -> str | None:
    if not env_file.exists():
        return None
    try:
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() != key:
                continue
            value = v.strip().strip('"').strip("'")
            return value or None
    except Exception:
        return None
    return None


def resolve_production_db_path(base_dir: Path) -> Path | None:
    explicit = (os.getenv("PRODUCTION_DATABASE_PATH") or "").strip()
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else (base_dir / p).resolve()

    env_prod = base_dir / ".env.production"
    raw = _parse_env_file_var(env_prod, "DATABASE_PATH")
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else (base_dir / p).resolve()


def running_under_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def enforce_environment_database_isolation(
    *,
    environment: str,
    database_abs_path: str,
    use_postgres: bool,
    database_url: str,
    base_dir: Path,
) -> None:
    env = (environment or "").strip().lower()
    if not env:
        raise RuntimeError("ENVIRONMENT_MUST_BE_SET")

    allowed_envs = {"production", "development", "test", "debug"}
    if env not in allowed_envs:
        raise RuntimeError(f"ENVIRONMENT_UNKNOWN: {env}")

    db_path = Path(database_abs_path).resolve()
    prod_db_path = resolve_production_db_path(base_dir)

    if not database_abs_path:
        raise RuntimeError("DATABASE_PATH_MUST_BE_SET")

    if env in {"test", "debug"} and (database_url or "").strip():
        raise RuntimeError(
            "TEST_OR_DEBUG_MUST_NOT_USE_DATABASE_URL: clear DATABASE_URL for test/debug sessions"
        )

    if env in {"test", "debug"} and use_postgres:
        raise RuntimeError("TEST_OR_DEBUG_MUST_NOT_USE_POSTGRES")

    if prod_db_path is not None and db_path == prod_db_path and env in {"test", "debug"}:
        raise RuntimeError("TEST_OR_DEBUG_DATABASE_MUST_NOT_EQUAL_PRODUCTION")

    if running_under_pytest() and prod_db_path is not None and db_path == prod_db_path:
        raise RuntimeError("PYTEST_DATABASE_MUST_NOT_EQUAL_PRODUCTION")

    if env == "development" and prod_db_path is not None and db_path == prod_db_path:
        allow = (os.getenv("ALLOW_DEVELOPMENT_ON_PRODUCTION_DB") or "").strip()
        if allow != "1":
            raise RuntimeError("DEVELOPMENT_DATABASE_MUST_NOT_EQUAL_PRODUCTION")

    if running_under_pytest() and env == "test":
        allow_non_temp = (os.getenv("PYTEST_ALLOW_NON_TEMP_DB") or "").strip()
        db_name = db_path.name.lower()
        if allow_non_temp != "1" and not db_name.startswith("test_portfolio_"):
            raise RuntimeError("PYTEST_DATABASE_MUST_BE_TEMP_FILE")


def ensure_debug_fixture_write_allowed(environment: str, database_abs_path: str, base_dir: Path) -> None:
    env = (environment or "").strip().lower()
    if env == "production":
        raise RuntimeError("DEBUG_FIXTURE_WRITE_FORBIDDEN_IN_PRODUCTION")

    allow = (os.getenv("ALLOW_DEBUG_FIXTURE_WRITE") or "").strip()
    if allow != "1":
        raise RuntimeError(
            "DEBUG_FIXTURE_WRITE_REQUIRES_ALLOW_DEBUG_FIXTURE_WRITE=1"
        )

    prod_db_path = resolve_production_db_path(base_dir)
    if prod_db_path is not None and Path(database_abs_path).resolve() == prod_db_path:
        raise RuntimeError("DEBUG_FIXTURE_WRITE_DATABASE_MUST_NOT_EQUAL_PRODUCTION")

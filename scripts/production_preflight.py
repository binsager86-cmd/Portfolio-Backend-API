"""
Production preflight checks for backend-api.

Run from backend-api/:
    python scripts/production_preflight.py

Optional:
    python scripts/production_preflight.py --allow-sqlite

Exit codes:
    0 = all required checks passed
    1 = one or more required checks failed
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from sqlalchemy import text

from app.core.config import get_settings
from app.core.database import engine


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _is_placeholder_secret(value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return True
    lower = v.lower()
    placeholder_tokens = [
        "change_me",
        "change_this",
        "before_production",
        "placeholder",
        "example",
    ]
    if any(token in lower for token in placeholder_tokens):
        return True
    return len(v) < 32


def _record(results: list[CheckResult], name: str, ok: bool, detail: str, required: bool = True) -> None:
    results.append(CheckResult(name=name, ok=ok, detail=detail, required=required))


def run_preflight(allow_sqlite: bool) -> int:
    settings = get_settings()
    results: list[CheckResult] = []

    _record(
        results,
        "ENVIRONMENT is production",
        settings.ENVIRONMENT == "production",
        f"ENVIRONMENT={settings.ENVIRONMENT}",
    )

    if allow_sqlite:
        _record(
            results,
            "Database mode",
            True,
            "SQLite allowed for this run (--allow-sqlite)",
        )
    else:
        _record(
            results,
            "Database mode is PostgreSQL",
            settings.use_postgres,
            "DATABASE_URL must be postgresql://... and non-empty",
        )

    _record(
        results,
        "SECRET_KEY is production-safe",
        not _is_placeholder_secret(settings.SECRET_KEY),
        "SECRET_KEY must be non-placeholder and >= 32 chars",
    )

    _record(
        results,
        "CRON_SECRET_KEY configured",
        not _is_placeholder_secret(settings.CRON_SECRET_KEY),
        "CRON_SECRET_KEY must be non-placeholder and >= 32 chars",
    )

    _record(
        results,
        "CORS_ORIGINS configured",
        "your-app" not in (settings.CORS_ORIGINS or "") and bool(settings.cors_origins_list),
        f"CORS_ORIGINS={settings.CORS_ORIGINS}",
    )

    # Database connectivity and schema checks
    db_ok = False
    db_detail = ""
    version = ""
    with engine.connect() as conn:
        db_ok = conn.execute(text("SELECT 1")).scalar() == 1
        if settings.use_postgres:
            version = str(conn.execute(text("SELECT version()")).scalar() or "")
            db_detail = "PostgreSQL connection OK"
        else:
            version = "SQLite"
            db_detail = "SQLite connection OK"

        required_tables = [
            "users",
            "stocks",
            "portfolios",
            "portfolio_holdings",
            "financial_statements",
            "financial_line_items",
        ]

        if settings.use_postgres:
            found = set(
                row[0]
                for row in conn.execute(
                    text(
                        """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                        """
                    )
                ).fetchall()
            )
        else:
            found = set(
                row[0]
                for row in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                ).fetchall()
            )

        missing = [t for t in required_tables if t not in found]
        _record(
            results,
            "Core tables present",
            len(missing) == 0,
            "missing=" + (", ".join(missing) if missing else "none"),
        )

        if settings.use_postgres:
            has_alembic = conn.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema='public' AND table_name='alembic_version'
                    )
                    """
                )
            ).scalar()
            if has_alembic:
                version_num = conn.execute(text("SELECT version_num FROM alembic_version LIMIT 1")).scalar()
                _record(
                    results,
                    "Alembic version table",
                    bool(version_num),
                    f"version={version_num}",
                )
            else:
                _record(
                    results,
                    "Alembic version table",
                    False,
                    "alembic_version table not found",
                )

    _record(results, "Database connectivity", db_ok, db_detail)
    _record(results, "Database engine version", bool(version), version, required=False)

    print("\n=== Production Preflight Report ===")
    failed_required = 0
    for r in results:
        status = "PASS" if r.ok else ("FAIL" if r.required else "WARN")
        req = "required" if r.required else "advisory"
        print(f"[{status}] {r.name} ({req})")
        print(f"       {r.detail}")
        if not r.ok and r.required:
            failed_required += 1

    if failed_required:
        print(f"\nPreflight failed: {failed_required} required check(s) failed.")
        return 1

    print("\nPreflight passed: required checks are green.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backend production preflight checks.")
    parser.add_argument(
        "--allow-sqlite",
        action="store_true",
        help="Allow SQLite mode for local dry-runs (not for production pushes).",
    )
    args = parser.parse_args()

    try:
        rc = run_preflight(allow_sqlite=args.allow_sqlite)
    except Exception as exc:
        print(f"\nPreflight crashed: {exc}")
        rc = 1
    raise SystemExit(rc)


if __name__ == "__main__":
    main()

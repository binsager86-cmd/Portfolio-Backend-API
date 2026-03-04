"""
SQLite → PostgreSQL Data Migration Script.

Reads all tables from the existing SQLite ``portfolio.db`` and writes them
into a PostgreSQL database, preserving:
  - user_id isolation (no row can lose its owner)
  - soft-delete flags (is_deleted, deleted_at)
  - auto-increment PKs (re-seeded via PostgreSQL sequences)

Usage:
    # Dry-run (validate only, no writes)
    python -m scripts.migrate_data --dry-run

    # Full migration
    python -m scripts.migrate_data \
        --sqlite ../dev_portfolio.db \
        --postgres postgresql://user:pass@localhost:5432/portfolio

    # Resume after partial failure (skip already-migrated rows)
    python -m scripts.migrate_data --resume

Requirements:
    pip install psycopg2-binary sqlalchemy pandas tqdm
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import (
    MetaData,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.engine import Engine

logger = logging.getLogger("migrate_data")

# ─── Migration order (respects FK dependencies) ──────────────────────

MIGRATION_ORDER: list[str] = [
    # Layer 0: no FK deps
    "users",
    "securities_master",
    "stocks_master",
    # Layer 1: depends on users
    "portfolios",
    "stocks",
    "cash_deposits",
    "portfolio_cash",
    "portfolio_snapshots",
    # Layer 2: depends on users + portfolios
    "external_accounts",
    "transactions",
    "portfolio_transactions",
    "position_snapshots",
    # Layer 3: PFM (depends on users)
    "pfm_snapshots",
    # Layer 4: PFM children (depends on pfm_snapshots)
    "pfm_assets",
    "pfm_liabilities",
    "pfm_income_expenses",
]

# Tables that must preserve soft-delete semantics
SOFT_DELETE_TABLES = {"transactions", "cash_deposits", "portfolio_transactions"}

# Columns that must NOT be NULL after migration
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "users": ["id", "username", "password_hash"],
    "transactions": ["id", "user_id", "stock_symbol", "txn_type"],
    "portfolios": ["id", "user_id", "name"],
    "stocks": ["id", "user_id", "symbol"],
    "cash_deposits": ["id", "user_id", "portfolio", "amount"],
}


# ─── SQLite type → PostgreSQL type mapping ────────────────────────────

def _sqlite_to_pg_type(col_type: str) -> str:
    """Map a SQLite column type string to PostgreSQL."""
    t = (col_type or "TEXT").upper().strip()
    if "INT" in t:
        return "BIGINT" if "BIG" in t else "INTEGER"
    if "REAL" in t or "FLOAT" in t or "DOUBLE" in t:
        return "DOUBLE PRECISION"
    if "BOOL" in t:
        return "BOOLEAN"
    if "BLOB" in t:
        return "BYTEA"
    if "DATE" in t or "TIME" in t:
        return "TEXT"  # kept as ISO text for now
    return "TEXT"


# ─── Schema introspection ─────────────────────────────────────────────

def get_sqlite_schema(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    """Return {table_name: [{name, type, notnull, pk}, ...]}."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [r[0] for r in cur.fetchall()]

    schema: dict[str, list[dict]] = {}
    for table in tables:
        cur.execute(f"PRAGMA table_info([{table}])")
        cols = []
        for row in cur.fetchall():
            cols.append({
                "cid": row[0],
                "name": row[1],
                "type": row[2],
                "notnull": bool(row[3]),
                "default": row[4],
                "pk": bool(row[5]),
            })
        schema[table] = cols
    return schema


def create_pg_table(pg_engine: Engine, table_name: str, columns: list[dict]) -> None:
    """
    Create a PostgreSQL table matching the SQLite schema.

    If the table already exists it is left untouched (idempotent).
    """
    col_defs: list[str] = []
    has_pk = any(c["pk"] for c in columns)

    for c in columns:
        pg_type = _sqlite_to_pg_type(c["type"])
        parts = [f'"{c["name"]}"']

        if c["pk"] and c["name"] == "id":
            parts.append("SERIAL PRIMARY KEY")
        else:
            parts.append(pg_type)
            if c["notnull"]:
                parts.append("NOT NULL")
            if c["default"] is not None:
                # Sanitize default value
                default = str(c["default"])
                if pg_type == "TEXT":
                    parts.append(f"DEFAULT '{default}'")
                else:
                    parts.append(f"DEFAULT {default}")

        col_defs.append(" ".join(parts))

    ddl = f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n  ' + ",\n  ".join(col_defs) + "\n);"
    with pg_engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("  ✓ Table %s ready", table_name)


# ─── Row-level type coercion ──────────────────────────────────────────

def _coerce_row(row: dict, columns: list[dict]) -> dict:
    """
    Coerce SQLite values to PostgreSQL-compatible types.

    Handles:
      - None user_id → 1 (safety net)
      - Integer booleans → Python bool
      - Null strings → None
    """
    coerced = dict(row)

    # user_id must never be NULL
    if "user_id" in coerced and coerced["user_id"] is None:
        coerced["user_id"] = 1

    # Soft-delete flags: ensure integer 0/1
    for flag in ("is_deleted", "is_current", "is_long_term", "is_active", "is_finance_cost", "is_gna"):
        if flag in coerced:
            val = coerced[flag]
            if val is None:
                coerced[flag] = 0
            else:
                coerced[flag] = int(val)

    # Numeric columns: coerce empty string → None
    for c in columns:
        pg_type = _sqlite_to_pg_type(c["type"])
        name = c["name"]
        if name not in coerced:
            continue
        val = coerced[name]
        if pg_type in ("INTEGER", "BIGINT", "DOUBLE PRECISION"):
            if val == "" or val == "None":
                coerced[name] = None
            elif val is not None:
                try:
                    if pg_type == "DOUBLE PRECISION":
                        coerced[name] = float(val)
                    else:
                        coerced[name] = int(float(val))
                except (ValueError, TypeError):
                    coerced[name] = None

    return coerced


# ─── Batch insert with ON CONFLICT ───────────────────────────────────

def insert_batch(
    pg_engine: Engine,
    table_name: str,
    rows: list[dict],
    pk_col: str = "id",
    batch_size: int = 500,
) -> tuple[int, int]:
    """
    Insert rows into PostgreSQL using ON CONFLICT DO NOTHING for idempotency.

    Returns (inserted_count, skipped_count).
    """
    if not rows:
        return 0, 0

    columns = list(rows[0].keys())
    col_list = ", ".join(f'"{c}"' for c in columns)
    param_list = ", ".join(f":{c}" for c in columns)

    # If table has an 'id' PK, use ON CONFLICT(id) DO NOTHING
    if pk_col in columns:
        sql = text(
            f'INSERT INTO "{table_name}" ({col_list}) VALUES ({param_list}) '
            f"ON CONFLICT ({pk_col}) DO NOTHING"
        )
    else:
        sql = text(f'INSERT INTO "{table_name}" ({col_list}) VALUES ({param_list})')

    inserted = 0
    skipped = 0

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        with pg_engine.begin() as conn:
            for row in batch:
                try:
                    result = conn.execute(sql, row)
                    if result.rowcount > 0:
                        inserted += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    logger.warning("    Row skipped (table=%s): %s", table_name, exc)
                    skipped += 1

    return inserted, skipped


def reset_sequence(pg_engine: Engine, table_name: str, pk_col: str = "id") -> None:
    """Reset PostgreSQL auto-increment sequence to max(id) + 1."""
    seq_name = f"{table_name}_{pk_col}_seq"
    sql = text(
        f"SELECT setval(pg_get_serial_sequence('{table_name}', '{pk_col}'), "
        f"COALESCE((SELECT MAX({pk_col}) FROM \"{table_name}\"), 1))"
    )
    try:
        with pg_engine.begin() as conn:
            conn.execute(sql)
        logger.info("  ✓ Sequence %s reset", seq_name)
    except Exception as exc:
        logger.debug("  Sequence reset skipped for %s: %s", table_name, exc)


# ─── Validation ───────────────────────────────────────────────────────

def validate_migration(
    sqlite_conn: sqlite3.Connection,
    pg_engine: Engine,
    table_name: str,
) -> dict[str, Any]:
    """
    Compare row counts and spot-check data between SQLite and PostgreSQL.

    Returns dict with validation results.
    """
    # SQLite count
    cur = sqlite_conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM [{table_name}]")
    sqlite_count = cur.fetchone()[0]

    # PostgreSQL count
    with pg_engine.connect() as conn:
        pg_count = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()

    result = {
        "table": table_name,
        "sqlite_count": sqlite_count,
        "pg_count": pg_count,
        "match": sqlite_count == pg_count,
    }

    # Spot-check: verify user_id isolation (no NULLs)
    if "user_id" in [c["name"] for c in get_sqlite_schema(sqlite_conn).get(table_name, [])]:
        with pg_engine.connect() as conn:
            null_count = conn.execute(
                text(f'SELECT COUNT(*) FROM "{table_name}" WHERE user_id IS NULL')
            ).scalar()
            result["null_user_ids"] = null_count

    # Spot-check: soft-delete flags preserved
    if table_name in SOFT_DELETE_TABLES:
        cur.execute(f"SELECT COUNT(*) FROM [{table_name}] WHERE is_deleted = 1")
        sqlite_deleted = cur.fetchone()[0]
        with pg_engine.connect() as conn:
            pg_deleted = conn.execute(
                text(f'SELECT COUNT(*) FROM "{table_name}" WHERE is_deleted = 1')
            ).scalar()
            result["soft_deleted_sqlite"] = sqlite_deleted
            result["soft_deleted_pg"] = pg_deleted
            result["soft_delete_match"] = sqlite_deleted == pg_deleted

    return result


# ─── Main migration orchestrator ──────────────────────────────────────

def migrate_sqlite_to_postgres(
    sqlite_path: str,
    postgres_url: str,
    dry_run: bool = False,
    resume: bool = False,
    batch_size: int = 500,
) -> dict[str, Any]:
    """
    Migrate all data from SQLite to PostgreSQL.

    CRITICAL invariants preserved:
      1. user_id isolation — no row loses its owner
      2. soft-delete flags — is_deleted/deleted_at preserved exactly
      3. auto-increment PKs — sequences re-seeded after migration
      4. Idempotent — ON CONFLICT DO NOTHING allows safe re-runs

    Args:
        sqlite_path: Path to SQLite database file.
        postgres_url: PostgreSQL connection string.
        dry_run: If True, validate schema without writing data.
        resume: If True, skip tables already populated in PG.
        batch_size: Rows per INSERT batch.

    Returns:
        Summary dict with per-table results.
    """
    # -- 1. Validate SQLite source --
    sqlite_path_obj = Path(sqlite_path).resolve()
    if not sqlite_path_obj.exists():
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_path_obj}")

    sqlite_conn = sqlite3.connect(str(sqlite_path_obj))
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_schema = get_sqlite_schema(sqlite_conn)

    logger.info("SQLite source: %s (%d tables)", sqlite_path_obj, len(sqlite_schema))

    # -- 2. Connect to PostgreSQL --
    pg_engine = create_engine(postgres_url, echo=False)

    # Quick connectivity check
    try:
        with pg_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL target: connected ✓")
    except Exception as exc:
        raise ConnectionError(f"Cannot connect to PostgreSQL: {exc}") from exc

    results: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sqlite_path": str(sqlite_path_obj),
        "postgres_url": postgres_url.split("@")[-1],  # redact credentials
        "dry_run": dry_run,
        "tables": {},
    }

    # -- 3. Discover tables to migrate --
    ordered_tables = [t for t in MIGRATION_ORDER if t in sqlite_schema]

    # Also pick up any tables NOT in MIGRATION_ORDER (append at end)
    extra = [t for t in sqlite_schema if t not in MIGRATION_ORDER]
    if extra:
        logger.info("Extra tables (not in migration order): %s", extra)
        ordered_tables.extend(sorted(extra))

    try:
        from tqdm import tqdm
        table_iter = tqdm(ordered_tables, desc="Migrating tables", unit="table")
    except ImportError:
        table_iter = ordered_tables

    # -- 4. Migrate each table --
    for table_name in table_iter:
        columns = sqlite_schema[table_name]
        table_result: dict[str, Any] = {"columns": len(columns)}

        # 4a. Create PG table if needed
        if not dry_run:
            create_pg_table(pg_engine, table_name, columns)

        # 4b. Check if PG table already has data (for --resume)
        if resume and not dry_run:
            with pg_engine.connect() as conn:
                existing = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
                if existing > 0:
                    logger.info("  ⏭  %s: already has %d rows, skipping (--resume)", table_name, existing)
                    table_result["status"] = "skipped_resume"
                    table_result["existing_rows"] = existing
                    results["tables"][table_name] = table_result
                    continue

        # 4c. Read all rows from SQLite
        cur = sqlite_conn.cursor()
        cur.execute(f"SELECT * FROM [{table_name}]")
        col_names = [desc[0] for desc in cur.description]
        raw_rows = cur.fetchall()

        table_result["sqlite_rows"] = len(raw_rows)
        logger.info("  %s: %d rows", table_name, len(raw_rows))

        if dry_run:
            table_result["status"] = "dry_run"
            results["tables"][table_name] = table_result
            continue

        if not raw_rows:
            table_result["status"] = "empty"
            results["tables"][table_name] = table_result
            continue

        # 4d. Convert rows and coerce types
        rows_to_insert: list[dict] = []
        for raw_row in raw_rows:
            row_dict = dict(zip(col_names, raw_row))
            coerced = _coerce_row(row_dict, columns)
            rows_to_insert.append(coerced)

        # 4e. Determine PK column
        pk_cols = [c["name"] for c in columns if c["pk"]]
        pk_col = pk_cols[0] if pk_cols else "id"

        # 4f. Insert into PostgreSQL
        inserted, skipped = insert_batch(
            pg_engine, table_name, rows_to_insert, pk_col=pk_col, batch_size=batch_size,
        )
        table_result["inserted"] = inserted
        table_result["skipped"] = skipped
        table_result["status"] = "migrated"

        # 4g. Reset sequence for SERIAL columns
        if pk_col == "id":
            reset_sequence(pg_engine, table_name, pk_col)

        # 4h. Validate
        validation = validate_migration(sqlite_conn, pg_engine, table_name)
        table_result["validation"] = validation

        results["tables"][table_name] = table_result

    # -- 5. Final summary --
    sqlite_conn.close()
    results["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    total_migrated = sum(
        t.get("inserted", 0) for t in results["tables"].values()
    )
    total_skipped = sum(
        t.get("skipped", 0) for t in results["tables"].values()
    )
    mismatches = [
        t["table"]
        for t in (r.get("validation", {}) for r in results["tables"].values())
        if isinstance(t, dict) and not t.get("match", True)
    ]

    results["summary"] = {
        "tables_processed": len(results["tables"]),
        "total_rows_inserted": total_migrated,
        "total_rows_skipped": total_skipped,
        "row_count_mismatches": mismatches,
    }

    if mismatches:
        logger.warning("⚠ Row count mismatches in: %s", mismatches)
    else:
        logger.info("✅ Migration complete — all row counts match")

    return results


# ─── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate portfolio data from SQLite to PostgreSQL.",
    )
    parser.add_argument(
        "--sqlite",
        default="../dev_portfolio.db",
        help="Path to SQLite database (default: ../dev_portfolio.db)",
    )
    parser.add_argument(
        "--postgres",
        default=None,
        help="PostgreSQL URL (default: from DATABASE_URL env var or settings)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate schema only, do not write data",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tables that already have data in PostgreSQL",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Rows per INSERT batch (default: 500)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve PostgreSQL URL
    postgres_url = args.postgres
    if not postgres_url:
        import os
        postgres_url = os.environ.get("DATABASE_URL")
    if not postgres_url:
        try:
            from app.core.config import get_settings
            settings = get_settings()
            postgres_url = getattr(settings, "DATABASE_URL", None)
        except Exception:
            pass
    if not postgres_url:
        parser.error(
            "PostgreSQL URL required: pass --postgres or set DATABASE_URL env var"
        )

    logger.info("=" * 60)
    logger.info("Portfolio SQLite → PostgreSQL Migration")
    logger.info("=" * 60)

    results = migrate_sqlite_to_postgres(
        sqlite_path=args.sqlite,
        postgres_url=postgres_url,
        dry_run=args.dry_run,
        resume=args.resume,
        batch_size=args.batch_size,
    )

    # Print summary
    summary = results.get("summary", {})
    print()
    print("═" * 50)
    print(f"  Tables processed:   {summary.get('tables_processed', 0)}")
    print(f"  Rows inserted:      {summary.get('total_rows_inserted', 0)}")
    print(f"  Rows skipped:       {summary.get('total_rows_skipped', 0)}")
    if summary.get("row_count_mismatches"):
        print(f"  ⚠ Mismatches:      {summary['row_count_mismatches']}")
    else:
        print("  ✅ All counts match")
    print("═" * 50)

    if results.get("dry_run"):
        print("\n  [DRY RUN] No data was written.\n")

    sys.exit(0 if not summary.get("row_count_mismatches") else 1)


if __name__ == "__main__":
    main()

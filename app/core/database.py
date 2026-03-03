"""
Database connection layer for the FastAPI backend.

Provides both:
  - **Raw SQLite** helpers (`get_connection`, `query_df`, etc.)
    for existing service functions that use hand-written SQL.
  - **SQLAlchemy** engine + session factory for new code using ORM models.

Supports two modes:
  - **SQLite** (default, development): uses DATABASE_PATH
  - **PostgreSQL** (production): uses DATABASE_URL

Both layers share the same underlying connection.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

import pandas as pd
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker, DeclarativeBase

from app.core.config import get_settings

_settings = get_settings()
_DB_PATH = _settings.database_abs_path
_USE_PG = _settings.use_postgres


# ── SQLAlchemy Base ──────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Declarative base for all SQLAlchemy models."""
    pass


# ── SQLAlchemy Engine & Session ──────────────────────────────────────

_SQLALCHEMY_URL = _settings.sqlalchemy_url

if _USE_PG:
    engine = create_engine(
        _SQLALCHEMY_URL,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
    )
else:
    engine = create_engine(
        _SQLALCHEMY_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_wal(dbapi_conn, connection_record):
        """Enable WAL + FK enforcement on every new SQLAlchemy connection."""
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency — yields a SQLAlchemy session and ensures cleanup.

    Usage in routes:
        @router.get("/items")
        def list_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Raw SQLite helpers (backward-compatible) ─────────────────────────
#    These functions use raw sqlite3 when running against SQLite,
#    or route through SQLAlchemy when running against PostgreSQL.


def _positional_to_named(params: tuple) -> dict:
    """
    Convert positional ``?``-style params to named ``:p0, :p1, …`` dict
    for SQLAlchemy ``text()`` execution against PostgreSQL.

    Also rewrites the SQL ``%s`` placeholders to ``:p0, :p1, …``.
    Called *after* ``sql.replace('?', '%s')``.
    """
    if not params:
        return {}
    return {f"p{i}": v for i, v in enumerate(params)}


def _pg_sql_named(sql: str, params: tuple) -> tuple[str, dict]:
    """
    Rewrite ``?``-style SQL for PostgreSQL by:
      1. Replacing each ``?`` with :p0, :p1, … in order
      2. Building the matching parameter dict.
    """
    named: dict[str, Any] = {}
    parts: list[str] = []
    idx = 0
    for ch in sql:
        if ch == "?":
            key = f"p{idx}"
            parts.append(f":{key}")
            if idx < len(params):
                named[key] = params[idx]
            idx += 1
        else:
            parts.append(ch)
    return "".join(parts), named


def _ensure_wal_mode(conn: sqlite3.Connection) -> None:
    """Enable WAL journal mode for safe concurrent reads."""
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.fetchone()
    cur.close()


@contextmanager
def get_connection():
    """
    Context-managed database connection.

    For SQLite: yields a sqlite3.Connection with WAL mode.
    For PostgreSQL: yields a SQLAlchemy connection (raw DBAPI).
    """
    if _USE_PG:
        with engine.connect() as conn:
            yield conn
    else:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _ensure_wal_mode(conn)
        conn.row_factory = sqlite3.Row  # dict-like rows
        try:
            yield conn
        finally:
            conn.close()


def get_conn() -> sqlite3.Connection:
    """Non-context-manager connection (for legacy-compatible code paths)."""
    if _USE_PG:
        # Return a raw DBAPI connection from the pool
        return engine.raw_connection()
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    _ensure_wal_mode(conn)
    return conn


def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute a SELECT and return a DataFrame."""
    if _USE_PG:
        pg_sql, named = _pg_sql_named(sql, params)
        with engine.connect() as conn:
            return pd.read_sql_query(text(pg_sql), conn, params=named)
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def query_val(sql: str, params: tuple = ()) -> Any:
    """Execute a query and return a single scalar value."""
    if _USE_PG:
        pg_sql, named = _pg_sql_named(sql, params)
        with engine.connect() as conn:
            result = conn.execute(text(pg_sql), named)
            row = result.fetchone()
            return row[0] if row else None
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else None


def query_one(sql: str, params: tuple = ()):
    """Execute a query and return a single row."""
    if _USE_PG:
        pg_sql, named = _pg_sql_named(sql, params)
        with engine.connect() as conn:
            result = conn.execute(text(pg_sql), named)
            row = result.mappings().fetchone()
            return dict(row) if row else None
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchone()


def query_all(sql: str, params: tuple = ()) -> list:
    """Execute a query and return all rows."""
    if _USE_PG:
        pg_sql, named = _pg_sql_named(sql, params)
        with engine.connect() as conn:
            result = conn.execute(text(pg_sql), named)
            return [dict(r) for r in result.mappings().fetchall()]
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()


def exec_sql(sql: str, params: tuple = ()) -> None:
    """Execute a write statement (INSERT / UPDATE / DELETE)."""
    if _USE_PG:
        pg_sql, named = _pg_sql_named(sql, params)
        with engine.begin() as conn:
            conn.execute(text(pg_sql), named)
        return
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()


def exec_sql_fetchone(sql: str, params: tuple = ()):
    """Execute SQL and return first row (works for both SQLite and PostgreSQL)."""
    if _USE_PG:
        pg_sql, named = _pg_sql_named(sql, params)
        with engine.connect() as conn:
            result = conn.execute(text(pg_sql), named)
            return result.fetchone()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchone()


def exec_sql_fetch(sql: str, params: tuple = ()):
    """Execute SQL and return all result rows (works for both SQLite and PostgreSQL)."""
    if _USE_PG:
        pg_sql, named = _pg_sql_named(sql, params)
        with engine.connect() as conn:
            result = conn.execute(text(pg_sql), named)
            return result.fetchall()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()


def exec_sql_returning_id(sql: str, params: tuple = ()) -> int:
    """Execute an INSERT and return the new row's id."""
    if _USE_PG:
        clean = sql.rstrip().rstrip(";")
        pg_sql, named = _pg_sql_named(clean + " RETURNING id", params)
        with engine.begin() as conn:
            result = conn.execute(text(pg_sql), named)
            row = result.fetchone()
            return row[0] if row else 0
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        return cur.lastrowid or 0


def exec_sql_batch(statements: list[tuple[str, tuple]]) -> None:
    """Execute multiple write statements in a single transaction."""
    if _USE_PG:
        with engine.begin() as conn:
            for sql, params in statements:
                pg_sql, named = _pg_sql_named(sql, params)
                conn.execute(text(pg_sql), named)
    else:
        with get_connection() as conn:
            cur = conn.cursor()
            for sql, params in statements:
                cur.execute(sql, params)
            conn.commit()


def column_exists(table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    if _USE_PG:
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = :table AND column_name = :column"
                ),
                {"table": table, "column": column},
            ).fetchone()
            return result is not None
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cur.fetchall()]
        return column in cols


def add_column_if_missing(table: str, column: str, col_type: str = "REAL") -> None:
    """Additive migration — add a column to a table if it does not already exist.

    Mirrors the Streamlit ``add_column_if_missing()`` helper so that the
    backend can safely reference optional columns added by later migrations.
    """
    if column_exists(table, column):
        return
    if _USE_PG:
        pg_type = col_type.replace("REAL", "DOUBLE PRECISION").replace("INTEGER", "INT")
        with engine.begin() as conn:
            conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {pg_type}'))
    else:
        with get_connection() as conn:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            conn.commit()


def check_db_exists() -> bool:
    """Verify that the database is accessible."""
    if _USE_PG:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
    return Path(_DB_PATH).exists()

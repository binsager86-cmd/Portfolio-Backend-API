"""
Migration: Security hardening tables & columns (Phase 3.1)

Adds:
  - audit_log table
  - token_blacklist table
  - users.failed_login_attempts, users.locked_until, users.last_failed_login columns

Idempotent — safe to run multiple times.
"""

import sqlite3
import os
import sys

DB_PATH = os.environ.get(
    "DATABASE_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "dev_portfolio.db"),
)
DB_PATH = os.path.abspath(DB_PATH)


def _col_exists(cur: sqlite3.Cursor, table: str, column: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def _table_exists(cur: sqlite3.Cursor, table: str) -> bool:
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cur.fetchone() is not None


def migrate(db_path: str = DB_PATH):
    print(f"Migrating: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # ── audit_log ────────────────────────────────────────────────────
    if not _table_exists(cur, "audit_log"):
        cur.execute("""
            CREATE TABLE audit_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER,
                action      TEXT NOT NULL,
                resource_type TEXT,
                resource_id INTEGER,
                details     TEXT,
                ip_address  TEXT,
                user_agent  TEXT,
                created_at  INTEGER NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS ix_audit_log_user_id ON audit_log (user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_audit_log_action ON audit_log (action)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_audit_log_created_at ON audit_log (created_at)")
        print("  ✅ Created audit_log table")
    else:
        print("  ⏭️  audit_log already exists")

    # ── token_blacklist ──────────────────────────────────────────────
    if not _table_exists(cur, "token_blacklist"):
        cur.execute("""
            CREATE TABLE token_blacklist (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                jti             TEXT NOT NULL UNIQUE,
                user_id         INTEGER,
                blacklisted_at  INTEGER NOT NULL,
                expires_at      INTEGER NOT NULL
            )
        """)
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_token_blacklist_jti ON token_blacklist (jti)")
        print("  ✅ Created token_blacklist table")
    else:
        print("  ⏭️  token_blacklist already exists")

    # ── users lockout columns ────────────────────────────────────────
    for col, col_type, default in [
        ("failed_login_attempts", "INTEGER", "0"),
        ("locked_until", "INTEGER", "NULL"),
        ("last_failed_login", "INTEGER", "NULL"),
    ]:
        if not _col_exists(cur, "users", col):
            cur.execute(
                f"ALTER TABLE users ADD COLUMN {col} {col_type} DEFAULT {default}"
            )
            print(f"  ✅ Added users.{col}")
        else:
            print(f"  ⏭️  users.{col} already exists")

    conn.commit()
    conn.close()
    print("Migration complete ✅")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DB_PATH
    migrate(path)

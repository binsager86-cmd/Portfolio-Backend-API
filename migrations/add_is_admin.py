"""
Migration: Add is_admin column to users table.

Idempotent — safe to run multiple times.
Uses add_column_if_missing from core database module.
"""

from app.core.database import add_column_if_missing


def run():
    add_column_if_missing("users", "is_admin", "INTEGER DEFAULT 0")
    print("✓ is_admin column ensured on users table")


if __name__ == "__main__":
    run()

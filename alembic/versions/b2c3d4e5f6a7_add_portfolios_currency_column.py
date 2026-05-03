"""add portfolios.currency column if missing

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-05-03 00:00:00.000000

Production fix: the `currency` column was missing from the `portfolios`
table on the live PostgreSQL database.  The initial schema migration
included it, but the DB was provisioned before that migration contained
the column, leaving it absent.

This migration is **idempotent** — it checks for the column before adding
it so it is safe to run against both affected and already-correct databases.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    cols = [c["name"] for c in inspector.get_columns(table)]
    return column in cols


def upgrade() -> None:
    if not _column_exists("portfolios", "currency"):
        op.add_column(
            "portfolios",
            sa.Column(
                "currency",
                sa.String(length=10),
                nullable=False,
                server_default="KWD",
            ),
        )


def downgrade() -> None:
    # Only drop if present (mirrors the idempotent upgrade)
    if _column_exists("portfolios", "currency"):
        op.drop_column("portfolios", "currency")

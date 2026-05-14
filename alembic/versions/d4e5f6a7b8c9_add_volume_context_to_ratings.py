"""add_volume_context_to_ratings

Revision ID: d4e5f6a7b8c9
Revises: c1d2e3f4a5b6
Create Date: 2026-05-14 00:00:00.000000

Adds volume_context_json column to ee_ratings_cache so volume awareness
data is persisted alongside each rating and can be served by the scanner API.

Also adds entry_relative_volume to simulator_positions for post-trade analysis.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "d4e5f6a7b8c9"
down_revision: str | None = "c1d2e3f4a5b6"
branch_labels = None
depends_on = None


def _col_exists(table: str, col: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return any(c["name"] == col for c in inspector.get_columns(table))


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table in inspector.get_table_names()


def upgrade() -> None:
    # ee_ratings_cache.volume_context_json
    if _table_exists("ee_ratings_cache") and not _col_exists("ee_ratings_cache", "volume_context_json"):
        with op.batch_alter_table("ee_ratings_cache") as batch_op:
            batch_op.add_column(sa.Column("volume_context_json", sa.Text(), nullable=True))

    # simulator_positions.entry_relative_volume
    if _table_exists("simulator_positions") and not _col_exists("simulator_positions", "entry_relative_volume"):
        with op.batch_alter_table("simulator_positions") as batch_op:
            batch_op.add_column(
                sa.Column("entry_relative_volume", sa.Numeric(precision=8, scale=2), nullable=True)
            )


def downgrade() -> None:
    if _table_exists("ee_ratings_cache") and _col_exists("ee_ratings_cache", "volume_context_json"):
        with op.batch_alter_table("ee_ratings_cache") as batch_op:
            batch_op.drop_column("volume_context_json")

    if _table_exists("simulator_positions") and _col_exists("simulator_positions", "entry_relative_volume"):
        with op.batch_alter_table("simulator_positions") as batch_op:
            batch_op.drop_column("entry_relative_volume")

"""eagle_eye_simulator_tables

Revision ID: c1d2e3f4a5b6
Revises: b2c3d4e5f6a7
Create Date: 2026-05-14 00:00:00.000000

Creates the 4 simulator tables and seeds the 3 initial portfolios.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect, text

# revision identifiers
revision: str = "c1d2e3f4a5b6"
down_revision: str | None = "b2c3d4e5f6a7"
branch_labels = None
depends_on = None


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table in inspector.get_table_names()


def upgrade() -> None:
    # ── simulator_portfolios ─────────────────────────────────────────────
    if not _table_exists("simulator_portfolios"):
        op.create_table(
            "simulator_portfolios",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("strategy_name", sa.String(length=20), nullable=False),
            sa.Column("starting_capital_kwd", sa.Numeric(precision=16, scale=4), nullable=False, server_default="10000"),
            sa.Column("cash_balance_kwd", sa.Numeric(precision=16, scale=4), nullable=False, server_default="10000"),
            sa.Column("total_value_kwd", sa.Numeric(precision=16, scale=4), nullable=False, server_default="10000"),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        with op.batch_alter_table("simulator_portfolios") as batch_op:
            batch_op.create_index("ix_sim_portfolios_strategy", ["strategy_name"], unique=True)

    # ── simulator_positions ──────────────────────────────────────────────
    if not _table_exists("simulator_positions"):
        op.create_table(
            "simulator_positions",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("portfolio_id", sa.Integer(), nullable=False),
            sa.Column("ticker", sa.String(length=20), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="OPEN"),
            sa.Column("entry_date", sa.String(length=10), nullable=True),
            sa.Column("entry_price", sa.Numeric(precision=16, scale=6), nullable=True),
            sa.Column("shares", sa.Numeric(precision=16, scale=4), nullable=True),
            sa.Column("size_kwd", sa.Numeric(precision=16, scale=4), nullable=True),
            sa.Column("size_pct_of_portfolio", sa.Numeric(precision=8, scale=4), nullable=True),
            # entry context
            sa.Column("entry_confidence", sa.Numeric(precision=8, scale=4), nullable=True),
            sa.Column("entry_stage", sa.String(length=40), nullable=True),
            sa.Column("entry_rating", sa.String(length=20), nullable=True),
            sa.Column("entry_thesis", sa.Text(), nullable=True),
            sa.Column("entry_signal_breakdown", sa.Text(), nullable=True),  # JSON
            sa.Column("entry_accumulation_score", sa.Numeric(precision=8, scale=4), nullable=True),
            sa.Column("entry_indicators_snapshot", sa.Text(), nullable=True),  # JSON
            # trade plan
            sa.Column("planned_stop_loss", sa.Numeric(precision=16, scale=6), nullable=True),
            sa.Column("planned_tp1", sa.Numeric(precision=16, scale=6), nullable=True),
            sa.Column("planned_tp2", sa.Numeric(precision=16, scale=6), nullable=True),
            sa.Column("planned_tp3", sa.Numeric(precision=16, scale=6), nullable=True),
            # partial TP tracking flags (stored as int 0/1 for SQLite compat)
            sa.Column("tp1_hit", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("tp2_hit", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("shares_remaining", sa.Numeric(precision=16, scale=4), nullable=True),
            # exit data
            sa.Column("exit_date", sa.String(length=10), nullable=True),
            sa.Column("exit_price", sa.Numeric(precision=16, scale=6), nullable=True),
            sa.Column("exit_reason", sa.String(length=30), nullable=True),
            sa.Column("pnl_kwd", sa.Numeric(precision=16, scale=4), nullable=True),
            sa.Column("pnl_pct", sa.Numeric(precision=10, scale=4), nullable=True),
            sa.Column("days_held", sa.Integer(), nullable=True),
            sa.Column("max_unrealized_gain_pct", sa.Numeric(precision=10, scale=4), nullable=True),
            sa.Column("max_unrealized_loss_pct", sa.Numeric(precision=10, scale=4), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["portfolio_id"], ["simulator_portfolios.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        with op.batch_alter_table("simulator_positions") as batch_op:
            batch_op.create_index("ix_sim_positions_portfolio_status", ["portfolio_id", "status"])
            batch_op.create_index("ix_sim_positions_ticker", ["ticker"])

    # ── simulator_daily_snapshots ────────────────────────────────────────
    if not _table_exists("simulator_daily_snapshots"):
        op.create_table(
            "simulator_daily_snapshots",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("portfolio_id", sa.Integer(), nullable=False),
            sa.Column("date", sa.String(length=10), nullable=False),
            sa.Column("cash_balance_kwd", sa.Numeric(precision=16, scale=4), nullable=True),
            sa.Column("open_positions_value_kwd", sa.Numeric(precision=16, scale=4), nullable=True),
            sa.Column("total_value_kwd", sa.Numeric(precision=16, scale=4), nullable=True),
            sa.Column("daily_pnl_kwd", sa.Numeric(precision=16, scale=4), nullable=True),
            sa.Column("cumulative_return_pct", sa.Numeric(precision=10, scale=4), nullable=True),
            sa.Column("drawdown_from_peak_pct", sa.Numeric(precision=10, scale=4), nullable=True),
            sa.Column("open_position_count", sa.Integer(), nullable=True),
            sa.ForeignKeyConstraint(["portfolio_id"], ["simulator_portfolios.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        with op.batch_alter_table("simulator_daily_snapshots") as batch_op:
            batch_op.create_index("ix_sim_snapshots_portfolio_date", ["portfolio_id", "date"], unique=True)

    # ── simulator_considered_trades ──────────────────────────────────────
    if not _table_exists("simulator_considered_trades"):
        op.create_table(
            "simulator_considered_trades",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("portfolio_id", sa.Integer(), nullable=False),
            sa.Column("date", sa.String(length=10), nullable=True),
            sa.Column("ticker", sa.String(length=20), nullable=True),
            sa.Column("confidence", sa.Numeric(precision=8, scale=4), nullable=True),
            sa.Column("stage", sa.String(length=40), nullable=True),
            sa.Column("reason_skipped", sa.String(length=50), nullable=True),
            sa.ForeignKeyConstraint(["portfolio_id"], ["simulator_portfolios.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        with op.batch_alter_table("simulator_considered_trades") as batch_op:
            batch_op.create_index("ix_sim_considered_portfolio_date", ["portfolio_id", "date"])

    # ── Seed initial portfolios ──────────────────────────────────────────
    bind = op.get_bind()
    count = bind.execute(text("SELECT COUNT(*) FROM simulator_portfolios")).scalar()
    if count == 0:
        now_ts = "2026-05-14 00:00:00"
        bind.execute(
            text(
                """
                INSERT INTO simulator_portfolios
                    (strategy_name, starting_capital_kwd, cash_balance_kwd, total_value_kwd, created_at, updated_at)
                VALUES
                    ('CONSERVATIVE', 10000, 10000, 10000, :ts, :ts),
                    ('MODERATE',     10000, 10000, 10000, :ts, :ts),
                    ('AGGRESSIVE',   10000, 10000, 10000, :ts, :ts)
                """
            ),
            {"ts": now_ts},
        )


def downgrade() -> None:
    for table in [
        "simulator_considered_trades",
        "simulator_daily_snapshots",
        "simulator_positions",
        "simulator_portfolios",
    ]:
        if _table_exists(table):
            op.drop_table(table)

"""KFH broker import staging and durable transaction identity.

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-09-01 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: str | None = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def _table_exists(name: str) -> bool:
    return name in inspect(op.get_bind()).get_table_names()


def _column_exists(table: str, column: str) -> bool:
    return column in {item["name"] for item in inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    if not _table_exists("broker_connections"):
        op.create_table(
            "broker_connections",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("broker", sa.String(length=50), nullable=False),
            sa.Column("broker_account_key", sa.String(length=64), nullable=False),
            sa.Column("account_label", sa.String(length=100), nullable=True),
            sa.Column(
                "auth_mode",
                sa.String(length=40),
                nullable=False,
                server_default="LOCAL_BROWSER_SESSION",
            ),
            sa.Column("status", sa.String(length=40), nullable=False, server_default="DISCONNECTED"),
            sa.Column("last_connected_at", sa.Integer(), nullable=True),
            sa.Column("last_successful_sync", sa.String(length=30), nullable=True),
            sa.Column("last_sync_status", sa.String(length=40), nullable=True),
            sa.Column("created_at", sa.Integer(), nullable=False),
            sa.Column("updated_at", sa.Integer(), nullable=True),
            sa.UniqueConstraint(
                "user_id", "broker", "broker_account_key", name="uq_broker_connection_account"
            ),
        )

    if not _table_exists("broker_raw_transactions"):
        op.create_table(
            "broker_raw_transactions",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column(
                "broker_connection_id",
                sa.Integer(),
                sa.ForeignKey("broker_connections.id"),
                nullable=False,
            ),
            sa.Column("sync_batch_id", sa.Integer(), nullable=True),
            sa.Column("broker_transaction_ref", sa.String(length=200), nullable=False),
            sa.Column("secondary_fingerprint", sa.String(length=64), nullable=False),
            sa.Column("record_kind", sa.String(length=20), nullable=False),
            sa.Column("transaction_date", sa.String(length=10), nullable=False),
            sa.Column("transaction_timestamp", sa.String(length=30), nullable=True),
            sa.Column("settlement_date", sa.String(length=10), nullable=True),
            sa.Column("transaction_type", sa.String(length=30), nullable=False),
            sa.Column("symbol", sa.String(length=50), nullable=True),
            sa.Column("quantity", sa.Numeric(precision=38, scale=12), nullable=True),
            sa.Column("price", sa.Numeric(precision=38, scale=12), nullable=True),
            sa.Column("amount", sa.Numeric(precision=38, scale=12), nullable=True),
            sa.Column("fees", sa.Numeric(precision=38, scale=12), nullable=True),
            sa.Column("canonical_payload", sa.Text(), nullable=False),
            sa.Column("raw_payload", sa.Text(), nullable=True),
            sa.Column("raw_transaction_type", sa.String(length=100), nullable=True),
            sa.Column("raw_description", sa.Text(), nullable=True),
            sa.Column("raw_date", sa.String(length=100), nullable=True),
            sa.Column("raw_hash", sa.String(length=64), nullable=False),
            sa.Column("parser_version", sa.String(length=100), nullable=False),
            sa.Column("adapter_version", sa.String(length=100), nullable=False),
            sa.Column("committed_at", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.Integer(), nullable=False),
            sa.Column("updated_at", sa.Integer(), nullable=True),
            sa.UniqueConstraint(
                "user_id",
                "broker_connection_id",
                "broker_transaction_ref",
                name="uq_broker_record_identity",
            ),
        )

    if not _table_exists("broker_import_batches"):
        op.create_table(
            "broker_import_batches",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column(
                "broker_connection_id",
                sa.Integer(),
                sa.ForeignKey("broker_connections.id"),
                nullable=False,
            ),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("mode", sa.String(length=30), nullable=True),
            sa.Column("requested_from", sa.String(length=10), nullable=True),
            sa.Column("requested_to", sa.String(length=10), nullable=True),
            sa.Column("started_at", sa.Integer(), nullable=False),
            sa.Column("fetched_at", sa.Integer(), nullable=True),
            sa.Column("committed_at", sa.Integer(), nullable=True),
            sa.Column("scope_json", sa.Text(), nullable=True),
            sa.Column("fetched_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("unsettled_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("new_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("duplicate_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("matched_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("conflict_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("unsupported_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("kfh_open_balance", sa.Numeric(precision=38, scale=3), nullable=True),
            sa.Column("kfh_close_balance", sa.Numeric(precision=38, scale=3), nullable=True),
            sa.Column("kfh_total_buy", sa.Numeric(precision=38, scale=3), nullable=True),
            sa.Column("kfh_total_sell", sa.Numeric(precision=38, scale=3), nullable=True),
            sa.Column("kfh_total_deposit", sa.Numeric(precision=38, scale=3), nullable=True),
            sa.Column("kfh_total_withdrawal", sa.Numeric(precision=38, scale=3), nullable=True),
            sa.Column("counts_json", sa.Text(), nullable=False),
            sa.Column("reconciliation_json", sa.Text(), nullable=True),
            sa.Column("cash_summary_json", sa.Text(), nullable=True),
            sa.Column("commit_result_json", sa.Text(), nullable=True),
            sa.Column("created_at", sa.Integer(), nullable=False),
            sa.Column("confirmed_at", sa.Integer(), nullable=True),
        )
    else:
        batch_columns = {
            "reconciliation_json": sa.Text(),
            "cash_summary_json": sa.Text(),
            "mode": sa.String(length=30),
            "requested_from": sa.String(length=10),
            "requested_to": sa.String(length=10),
            "started_at": sa.Integer(),
            "fetched_at": sa.Integer(),
            "committed_at": sa.Integer(),
            "unsettled_count": sa.Integer(),
            "new_count": sa.Integer(),
            "duplicate_count": sa.Integer(),
            "matched_count": sa.Integer(),
            "conflict_count": sa.Integer(),
            "unsupported_count": sa.Integer(),
            "kfh_open_balance": sa.Numeric(precision=38, scale=3),
            "kfh_close_balance": sa.Numeric(precision=38, scale=3),
            "kfh_total_buy": sa.Numeric(precision=38, scale=3),
            "kfh_total_sell": sa.Numeric(precision=38, scale=3),
            "kfh_total_deposit": sa.Numeric(precision=38, scale=3),
            "kfh_total_withdrawal": sa.Numeric(precision=38, scale=3),
        }
        for name, column_type in batch_columns.items():
            if not _column_exists("broker_import_batches", name):
                op.add_column(
                    "broker_import_batches",
                    sa.Column(name, column_type, nullable=True),
                )

    connection_columns = {
        "auth_mode": sa.String(length=40),
        "status": sa.String(length=40),
        "last_connected_at": sa.Integer(),
        "last_sync_status": sa.String(length=40),
    }
    for name, column_type in connection_columns.items():
        if not _column_exists("broker_connections", name):
            op.add_column("broker_connections", sa.Column(name, column_type, nullable=True))

    raw_columns = {
        "sync_batch_id": sa.Integer(),
        "transaction_timestamp": sa.String(length=30),
        "raw_transaction_type": sa.String(length=100),
        "raw_description": sa.Text(),
        "raw_date": sa.String(length=100),
        "raw_hash": sa.String(length=64),
        "parser_version": sa.String(length=100),
        "adapter_version": sa.String(length=100),
    }
    for name, column_type in raw_columns.items():
        if not _column_exists("broker_raw_transactions", name):
            op.add_column(
                "broker_raw_transactions",
                sa.Column(name, column_type, nullable=True),
            )

    if not _table_exists("broker_import_items"):
        op.create_table(
            "broker_import_items",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "batch_id", sa.Integer(), sa.ForeignKey("broker_import_batches.id"), nullable=False
            ),
            sa.Column(
                "broker_raw_transaction_id",
                sa.Integer(),
                sa.ForeignKey("broker_raw_transactions.id"),
                nullable=False,
            ),
            sa.Column("classification", sa.String(length=30), nullable=False),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("matched_record_kind", sa.String(length=20), nullable=True),
            sa.Column("matched_record_id", sa.Integer(), nullable=True),
            sa.Column("match_type", sa.String(length=30), nullable=True),
            sa.Column("confidence", sa.Float(), nullable=True),
            sa.Column("saham_value_json", sa.Text(), nullable=True),
            sa.Column("kfh_value_json", sa.Text(), nullable=True),
            sa.Column("selected_default", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.Integer(), nullable=False),
            sa.UniqueConstraint("batch_id", "broker_raw_transaction_id", name="uq_broker_batch_record"),
        )

    if not _table_exists("broker_transaction_links"):
        op.create_table(
            "broker_transaction_links",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column(
                "broker_raw_transaction_id",
                sa.Integer(),
                sa.ForeignKey("broker_raw_transactions.id"),
                nullable=False,
                unique=True,
            ),
            sa.Column("saham_record_kind", sa.String(length=20), nullable=False),
            sa.Column("saham_record_id", sa.Integer(), nullable=False),
            sa.Column("match_type", sa.String(length=30), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("linked_at", sa.Integer(), nullable=False),
        )

    for name in ("saham_value_json", "kfh_value_json"):
        if not _column_exists("broker_import_items", name):
            op.add_column(
                "broker_import_items",
                sa.Column(name, sa.Text(), nullable=True),
            )

    if not _table_exists("broker_unsettled_transactions"):
        op.create_table(
            "broker_unsettled_transactions",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column(
                "sync_batch_id",
                sa.Integer(),
                sa.ForeignKey("broker_import_batches.id"),
                nullable=False,
            ),
            sa.Column(
                "broker_connection_id",
                sa.Integer(),
                sa.ForeignKey("broker_connections.id"),
                nullable=False,
            ),
            sa.Column("broker_transaction_ref", sa.String(length=200), nullable=True),
            sa.Column("raw_transaction_type", sa.String(length=100), nullable=True),
            sa.Column("raw_description", sa.Text(), nullable=True),
            sa.Column("raw_date", sa.String(length=100), nullable=True),
            sa.Column("raw_payload", sa.Text(), nullable=False),
            sa.Column("raw_hash", sa.String(length=64), nullable=False),
            sa.Column("parser_version", sa.String(length=100), nullable=False),
            sa.Column("adapter_version", sa.String(length=100), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="UNSETTLED"),
            sa.Column("created_at", sa.Integer(), nullable=False),
        )

    op.create_index("idx_broker_conn_user", "broker_connections", ["user_id"], if_not_exists=True)
    op.create_index(
        "idx_broker_raw_identity",
        "broker_raw_transactions",
        ["user_id", "broker_connection_id", "broker_transaction_ref"],
        if_not_exists=True,
    )
    op.create_index(
        "idx_broker_raw_fingerprint",
        "broker_raw_transactions",
        ["user_id", "secondary_fingerprint"],
        if_not_exists=True,
    )
    op.create_index(
        "idx_broker_batch_user", "broker_import_batches", ["user_id", "created_at"], if_not_exists=True
    )
    op.create_index(
        "idx_broker_item_batch",
        "broker_import_items",
        ["batch_id", "classification"],
        if_not_exists=True,
    )
    op.create_index(
        "idx_broker_link_record",
        "broker_transaction_links",
        ["user_id", "saham_record_kind", "saham_record_id"],
        if_not_exists=True,
    )
    op.create_index(
        "idx_broker_unsettled_batch",
        "broker_unsettled_transactions",
        ["sync_batch_id", "status"],
        if_not_exists=True,
    )


def downgrade() -> None:
    # Intentionally preserve additive broker audit/staging data. The previous
    # application release does not reference these tables, so code rollback is
    # safe without destructive schema changes. A verified backup restore is the
    # only approved path when physical removal is required.
    pass

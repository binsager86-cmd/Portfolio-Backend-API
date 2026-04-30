"""add_price_news_indexes

Revision ID: a3f9c1d2e8b4
Revises: 15d6c76a75e2
Create Date: 2026-04-30 00:00:00.000000

Adds targeted indexes to eliminate full-table-scans on the most
frequently queried tables.

The original patch request referenced a few column names that do not exist in
this repository (`portfolio_snapshots.symbol`, `portfolio_snapshots.trade_date`,
`portfolio_transactions.trade_date`). This migration applies the same intent to
the actual schema (`position_snapshots.stock_symbol`, `snapshot_date`,
`portfolio_transactions.txn_date`).

  news_articles
    - idx_news_content_hash    (content_hash)          — unique dedup key
    - idx_news_published       (published_at)          — feed/history sort
    - idx_news_source_hash     (source, content_hash) — dedup + source filters
    - idx_news_category_lang   (category, language)   — category filter on /feed

  portfolio_snapshots
    - idx_snapshots_user_date  (user_id, snapshot_date) — daily portfolio lookups

  position_snapshots
    - idx_snapshots_symbol_date (stock_symbol, snapshot_date) — per-ticker history

  portfolio_transactions
    - idx_transactions_user_date (user_id, txn_date)  — holdings aggregation

  transactions  (legacy)
    - idx_txn_user_symbol      (user_id, stock_symbol) — portfolio build queries
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision = "a3f9c1d2e8b4"
down_revision = "15d6c76a75e2"
branch_labels = None
depends_on = None


def upgrade() -> None:
  bind = op.get_bind()
  inspector = sa.inspect(bind)
  existing_tables = set(inspector.get_table_names())

    # ── news_articles ─────────────────────────────────────────────
    # content_hash: unique index for fast dedup on bulk-import.
    # main.py lifespan also creates this index via CREATE INDEX IF NOT EXISTS,
    # so we use if_not_exists=True to stay idempotent.
  if "news_articles" in existing_tables:
    op.create_index(
      "idx_news_content_hash",
      "news_articles",
      ["content_hash"],
      unique=True,
      if_not_exists=True,
    )
    op.create_index(
      "idx_news_published",
      "news_articles",
      ["published_at"],
      if_not_exists=True,
    )
    op.create_index(
      "idx_news_source_hash",
      "news_articles",
      ["source", "content_hash"],
      if_not_exists=True,
    )
    op.create_index(
      "idx_news_category_lang",
      "news_articles",
      ["category", "language"],
      if_not_exists=True,
    )

    # ── portfolio_snapshots ───────────────────────────────────────
    if "portfolio_snapshots" in existing_tables:
      op.create_index(
        "idx_snapshots_user_date",
        "portfolio_snapshots",
        ["user_id", "snapshot_date"],
        if_not_exists=True,
      )

    # ── position_snapshots ────────────────────────────────────────
    if "position_snapshots" in existing_tables:
      op.create_index(
        "idx_snapshots_symbol_date",
        "position_snapshots",
        ["stock_symbol", "snapshot_date"],
        if_not_exists=True,
      )

    # ── portfolio_transactions ────────────────────────────────────
    if "portfolio_transactions" in existing_tables:
      op.create_index(
        "idx_transactions_user_date",
        "portfolio_transactions",
        ["user_id", "txn_date"],
        if_not_exists=True,
      )

    # ── transactions (legacy) ─────────────────────────────────────
    if "transactions" in existing_tables:
      op.create_index(
        "idx_txn_user_symbol",
        "transactions",
        ["user_id", "stock_symbol"],
        if_not_exists=True,
      )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "transactions" in existing_tables:
      op.drop_index("idx_txn_user_symbol", table_name="transactions", if_exists=True)
    if "portfolio_transactions" in existing_tables:
      op.drop_index("idx_transactions_user_date", table_name="portfolio_transactions", if_exists=True)
    if "position_snapshots" in existing_tables:
      op.drop_index("idx_snapshots_symbol_date", table_name="position_snapshots", if_exists=True)
    if "portfolio_snapshots" in existing_tables:
      op.drop_index("idx_snapshots_user_date", table_name="portfolio_snapshots", if_exists=True)
    if "news_articles" in existing_tables:
      op.drop_index("idx_news_source_hash", table_name="news_articles", if_exists=True)
      op.drop_index("idx_news_category_lang", table_name="news_articles", if_exists=True)
      op.drop_index("idx_news_published", table_name="news_articles", if_exists=True)
      op.drop_index("idx_news_content_hash", table_name="news_articles", if_exists=True)

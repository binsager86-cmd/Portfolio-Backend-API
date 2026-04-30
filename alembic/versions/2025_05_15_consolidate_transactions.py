"""consolidate_transactions_to_fk_guard

Revision ID: a1b2c3d4e5f6
Revises: a3f9c1d2e8b4
Create Date: 2025-05-15 10:00:00.000000

[B-1] Guard revision only.

The originally requested `transactions -> portfolio_transactions` swap is NOT
safe for this repository state:

- live API routes still read/write `transactions` directly
- the schemas are not column-compatible (`shares/purchase_cost/sell_value/...`
  vs `amount/price_per_share/...`)
- dropping or renaming either table would be a breaking change, not a
  zero-downtime migration

This revision intentionally performs no schema mutation. It exists to record
that the consolidation was audited and remains blocked until the application is
migrated off the legacy table end-to-end.
"""

from alembic import op

revision = "a1b2c3d4e5f6"
down_revision = "a3f9c1d2e8b4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Intentionally no-op. See module docstring for the blocking constraints.
    pass


def downgrade() -> None:
    # Intentionally no-op.
    pass

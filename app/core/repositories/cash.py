"""
[B-2] Cash deposit repository — ORM-backed CRUD for cash_deposits.

Complex analytics queries (UNION ALL cash recalculations, export SELECTs
with COALESCE, paginated list with dynamic WHERE) intentionally stay as
raw SQL in the route layer because the ORM overhead adds no value there.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.repositories.base import BaseRepository
from app.models.cash import CashDeposit


class CashDepositRepository(BaseRepository[CashDeposit]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, CashDeposit)

    def get_active(self, deposit_id: int, user_id: int) -> CashDeposit | None:
        """
        Fetch a non-deleted deposit owned by *user_id*.
        Returns ``None`` when not found or soft-deleted.
        """
        return (
            self._session.query(CashDeposit)
            .filter(
                CashDeposit.id == deposit_id,
                CashDeposit.user_id == user_id,
                (CashDeposit.is_deleted.is_(None)) | (CashDeposit.is_deleted == 0),
            )
            .first()
        )

    def get_deleted(self, deposit_id: int, user_id: int) -> CashDeposit | None:
        """Fetch a soft-deleted deposit. Used by the restore endpoint."""
        return (
            self._session.query(CashDeposit)
            .filter(
                CashDeposit.id == deposit_id,
                CashDeposit.user_id == user_id,
                CashDeposit.is_deleted == 1,
            )
            .first()
        )

    def get_any(self, deposit_id: int, user_id: int) -> CashDeposit | None:
        """Fetch a deposit regardless of its deleted state. Used for snapshot syncs."""
        return (
            self._session.query(CashDeposit)
            .filter(
                CashDeposit.id == deposit_id,
                CashDeposit.user_id == user_id,
            )
            .first()
        )

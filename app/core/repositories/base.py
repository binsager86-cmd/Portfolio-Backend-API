"""
[B-2] Typed generic repository base.

Provides a consistent SQLAlchemy session interface — replaces scattered
raw ``query_df()`` / ``exec_sql()`` calls with ORM-backed operations.

Raw SQL is intentionally preserved only for complex analytics queries
(UNION ALL, dynamic GROUP BY, window functions) where the ORM overhead
is unjustified and the readability gain is nil.
"""

from __future__ import annotations

from typing import Any, Generic, Sequence, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """
    Typed CRUD repository for a single SQLAlchemy model.

    Example::

        repo = BaseRepository(session, CashDeposit)
        deposit = repo.get(42)
        deposits = repo.filter_by(user_id=7, is_deleted=0)
        repo.bulk_insert([{"user_id": 7, "amount": 100.0, ...}])
    """

    def __init__(self, session: Session, model: type[ModelT]) -> None:
        self._session = session
        self._model = model

    # ── Single-record access ──────────────────────────────────────────

    def get(self, pk: int) -> ModelT | None:
        """Fetch by primary key. Returns ``None`` when not found."""
        return self._session.get(self._model, pk)

    def get_one_by(self, **kwargs: Any) -> ModelT | None:
        """Return the first record matching all keyword filters, or ``None``."""
        return (
            self._session.query(self._model)
            .filter_by(**kwargs)
            .first()
        )

    # ── Multi-record access ───────────────────────────────────────────

    def filter_by(self, **kwargs: Any) -> list[ModelT]:
        """
        Return all records matching keyword equality filters.

        Equivalent to: ``SELECT * FROM table WHERE col=val AND ...``
        """
        return (
            self._session.query(self._model)
            .filter_by(**kwargs)
            .all()
        )

    def select(self, *criteria: Any) -> list[ModelT]:
        """
        Return all records matching arbitrary SQLAlchemy filter expressions.

        Example::

            repo.select(
                CashDeposit.user_id == user_id,
                CashDeposit.is_deleted.is_(None) | (CashDeposit.is_deleted == 0),
            )
        """
        return (
            self._session.query(self._model)
            .filter(*criteria)
            .all()
        )

    def all(self) -> list[ModelT]:
        """Return every row (use sparingly on large tables)."""
        return self._session.query(self._model).all()

    # ── Write operations ──────────────────────────────────────────────

    def add(self, instance: ModelT) -> ModelT:
        """
        Persist a new instance and flush so ``instance.id`` is populated.
        Does **not** commit — the caller controls the transaction boundary.
        """
        self._session.add(instance)
        self._session.flush()
        return instance

    def bulk_insert(self, rows: list[dict[str, Any]]) -> None:
        """
        Insert multiple rows as plain mappings (no ORM overhead per row).
        Flushes after insert; does **not** commit.

        Prefer for large datasets where individual ``add()`` calls are slow.
        """
        if rows:
            self._session.bulk_insert_mappings(self._model, rows)
            self._session.flush()

    def delete(self, instance: ModelT) -> None:
        """Hard-delete an ORM instance from the session (and DB on commit)."""
        self._session.delete(instance)
        self._session.flush()

    # ── Session control (delegate) ─────────────────────────────────────

    def commit(self) -> None:
        self._session.commit()

    def rollback(self) -> None:
        self._session.rollback()

    def refresh(self, instance: ModelT) -> ModelT:
        """Reload the instance from the DB (updates in-memory attributes)."""
        self._session.refresh(instance)
        return instance

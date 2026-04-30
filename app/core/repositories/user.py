"""
[B-2] User repository — ORM-backed auth queries.

Replaces the raw ``query_one()`` call in ``auth_service.authenticate_user()``.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.repositories.base import BaseRepository
from app.models.user import User


class UserRepository(BaseRepository[User]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, User)

    def get_by_username(self, username: str) -> User | None:
        """Case-sensitive username lookup. Returns ``None`` when not found."""
        return (
            self._session.query(User)
            .filter(User.username == username)
            .first()
        )

"""
Repository layer — typed SQLAlchemy data-access abstractions.

Usage:
    from app.core.repositories.user import UserRepository
    from app.core.repositories.cash import CashDepositRepository
"""

from app.core.repositories.base import BaseRepository
from app.core.repositories.user import UserRepository
from app.core.repositories.cash import CashDepositRepository

__all__ = ["BaseRepository", "UserRepository", "CashDepositRepository"]

"""
User model — maps to the existing ``users`` table.
"""

from typing import Optional

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Account lockout fields
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    locked_until: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # unix timestamp
    last_failed_login: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    portfolios = relationship("Portfolio", back_populates="user", lazy="selectin")
    stocks = relationship("Stock", back_populates="user", lazy="selectin")
    transactions = relationship("PortfolioTransaction", back_populates="user", lazy="select")

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r}>"

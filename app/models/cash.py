"""
Cash-related models — deposits, external accounts.

Maps to existing tables:
  - cash_deposits
  - external_accounts
"""

from typing import Optional

from sqlalchemy import Integer, String, Text, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class CashDeposit(Base):
    """Cash deposit / withdrawal record."""
    __tablename__ = "cash_deposits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio: Mapped[str] = mapped_column(String(50), nullable=False)
    deposit_date: Mapped[str] = mapped_column(String(10), nullable=False)  # YYYY-MM-DD
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default="KWD")
    bank_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    deposit_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, default="deposit")
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_deleted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=0)
    deleted_at: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return f"<CashDeposit id={self.id} amount={self.amount} {self.currency}>"


class ExternalAccount(Base):
    """External brokerage / bank account with cash balance."""
    __tablename__ = "external_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("portfolios.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    account_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default="KWD")
    current_balance: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_reconciled_date: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    portfolio = relationship("Portfolio", back_populates="external_accounts")

    def __repr__(self) -> str:
        return f"<ExternalAccount id={self.id} name={self.name!r} balance={self.current_balance}>"

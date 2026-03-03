"""
PFM (Personal Finance Management) models.

Maps to existing tables:
  - pfm_snapshots
  - pfm_assets
  - pfm_liabilities
  - pfm_income_expenses
"""

from typing import Optional

from sqlalchemy import Integer, String, Float, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class PfmSnapshot(Base):
    """Monthly/periodic net-worth snapshot."""
    __tablename__ = "pfm_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    snapshot_date: Mapped[str] = mapped_column(String(10), nullable=False)  # YYYY-MM-DD
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    total_assets: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_liabilities: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    net_worth: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    assets = relationship("PfmAsset", back_populates="snapshot", cascade="all, delete-orphan")
    liabilities = relationship("PfmLiability", back_populates="snapshot", cascade="all, delete-orphan")
    income_expenses = relationship("PfmIncomeExpense", back_populates="snapshot", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<PfmSnapshot id={self.id} date={self.snapshot_date} net_worth={self.net_worth}>"


class PfmAsset(Base):
    """Individual asset line item within a PFM snapshot."""
    __tablename__ = "pfm_assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(Integer, ForeignKey("pfm_snapshots.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(50), nullable=False)  # real_estate|shares|gold|cash|crypto|other
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    quantity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default="KWD")
    value_kwd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    snapshot = relationship("PfmSnapshot", back_populates="assets")

    def __repr__(self) -> str:
        return f"<PfmAsset id={self.id} name={self.name!r} value={self.value_kwd}>"


class PfmLiability(Base):
    """Liability line item within a PFM snapshot."""
    __tablename__ = "pfm_liabilities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(Integer, ForeignKey("pfm_snapshots.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    amount_kwd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_current: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_long_term: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    snapshot = relationship("PfmSnapshot", back_populates="liabilities")

    def __repr__(self) -> str:
        return f"<PfmLiability id={self.id} category={self.category!r} amount={self.amount_kwd}>"


class PfmIncomeExpense(Base):
    """Monthly income or expense line item within a PFM snapshot."""
    __tablename__ = "pfm_income_expenses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(Integer, ForeignKey("pfm_snapshots.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)  # income|expense
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    monthly_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_finance_cost: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_gna: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    snapshot = relationship("PfmSnapshot", back_populates="income_expenses")

    def __repr__(self) -> str:
        return f"<PfmIncomeExpense id={self.id} kind={self.kind} category={self.category!r}>"

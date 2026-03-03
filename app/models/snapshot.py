"""
Snapshot-related models — portfolio snapshots, position snapshots.

Maps to existing tables:
  - portfolio_snapshots  (daily portfolio-level tracking)
  - position_snapshots   (per-stock position tracking)
  - portfolio_cash       (cash balance per portfolio)
  - stocks_master        (global stock catalogue)
"""

from typing import Optional

from sqlalchemy import Integer, String, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class PortfolioSnapshot(Base):
    """Daily portfolio-level value snapshot."""
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    snapshot_date: Mapped[str] = mapped_column(String(10), nullable=False)  # YYYY-MM-DD
    portfolio_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    daily_movement: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    beginning_difference: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    deposit_cash: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    accumulated_cash: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    net_gain: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    change_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roi_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    twr_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mwrr_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return f"<PortfolioSnapshot id={self.id} date={self.snapshot_date} value={self.portfolio_value}>"


class PositionSnapshot(Base):
    """Per-stock position snapshot."""
    __tablename__ = "position_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    stock_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    stock_symbol: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    portfolio_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    snapshot_date: Mapped[str] = mapped_column(String(10), nullable=False)
    total_shares: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    realized_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cash_dividends_received: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, default="OPEN")

    def __repr__(self) -> str:
        return f"<PositionSnapshot id={self.id} symbol={self.stock_symbol} date={self.snapshot_date}>"


class PortfolioCash(Base):
    """Reconciled cash balance per portfolio (5-source UNION ALL result)."""
    __tablename__ = "portfolio_cash"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio: Mapped[str] = mapped_column(String(50), nullable=False)
    balance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default="KWD")
    last_updated: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    manual_override: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:
        return f"<PortfolioCash portfolio={self.portfolio} balance={self.balance}>"


class StocksMaster(Base):
    """Global stock catalogue (exchange-independent)."""
    __tablename__ = "stocks_master"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    exchange: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default="KWD")

    def __repr__(self) -> str:
        return f"<StocksMaster id={self.id} symbol={self.symbol!r}>"

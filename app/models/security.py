"""
Securities Master model — 3-layer securities data architecture.

Maps to existing table:
  - securities_master  (exchange-listed securities with Yahoo/TradingView metadata)
"""

from typing import Optional

from sqlalchemy import Integer, String, Float, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SecuritiesMaster(Base):
    """
    Exchange-listed securities with metadata.
    Layer 1: Securities Master (this table)
    Layer 2: stocks table (user's positions + current prices)
    Layer 3: prices cache (historical)
    """
    __tablename__ = "securities_master"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    exchange: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default="KWD")
    asset_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, default="EQUITY")
    sector: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    yahoo_symbol: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    tradingview_symbol: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    tradingview_exchange: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    isin: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    market_cap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    outstanding_shares: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return f"<SecuritiesMaster id={self.id} symbol={self.symbol!r} exchange={self.exchange}>"

"""
Portfolio-related SQLAlchemy models.

Maps the Streamlit-era tables: stocks, transactions.
Aligned with the actual sqlite schema used by ui.py and conftest.py.
"""

import time
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import relationship

from app.core.database import Base


class Portfolio(Base):
    """
    Portfolio grouping — KFH, BBYN, USA.

    Maps the ``portfolios`` table (has user_id per conftest schema).
    """

    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(50), nullable=False)  # KFH, BBYN, USA
    currency = Column(String(10), nullable=False, default="KWD")
    description = Column(Text, nullable=True)
    created_at = Column(Integer, default=lambda: int(time.time()))

    user = relationship("User", back_populates="portfolios")
    external_accounts = relationship("ExternalAccount", back_populates="portfolio", lazy="selectin")
    portfolio_transactions = relationship("PortfolioTransaction2", back_populates="portfolio", lazy="selectin")

    def __repr__(self) -> str:
        return f"<Portfolio {self.name}>"


class Stock(Base):
    """
    Mirrors the ``stocks`` table created by ui.py ``init_db()``.

    Aligned with actual schema: name (not company), current_price,
    tradingview_symbol, etc.
    """

    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    symbol = Column(String(50), nullable=False)
    name = Column(String(200), nullable=True)
    portfolio = Column(String(50), nullable=True)
    currency = Column(String(10), default="KWD")
    current_price = Column(Float, nullable=True)
    last_updated = Column(Integer, nullable=True)
    price_source = Column(String(100), nullable=True)
    tradingview_symbol = Column(String(50), nullable=True)
    tradingview_exchange = Column(String(50), nullable=True)
    market_cap = Column(Float, nullable=True)
    sector = Column(String(100), nullable=True)
    industry = Column(String(100), nullable=True)
    pe_ratio = Column(Float, nullable=True)

    user = relationship("User", back_populates="stocks")

    def __repr__(self) -> str:
        return f"<Stock {self.symbol} ({self.portfolio})>"


class PortfolioTransaction(Base):
    """
    Mirrors the ``transactions`` table created by ui.py ``init_db()``.

    Uses text-based portfolio/stock_symbol (not FK integers) to match
    the actual production schema.
    """

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio = Column(String(50), nullable=False)       # text: "KFH", "BBYN", "USA"
    stock_symbol = Column(String(50), nullable=False)     # text: "HUMANSOFT", "AAPL"
    txn_date = Column(String(20), nullable=True)          # ISO date
    txn_type = Column(String(20), nullable=False)         # Buy, Sell, Bonus, Dividend
    shares = Column(Float, default=0.0)
    purchase_cost = Column(Float, default=0.0)
    sell_value = Column(Float, default=0.0)
    bonus_shares = Column(Float, default=0.0)
    cash_dividend = Column(Float, default=0.0)
    reinvested_dividend = Column(Float, default=0.0)
    fees = Column(Float, default=0.0)
    price_override = Column(Float, nullable=True)
    planned_cum_shares = Column(Float, nullable=True)
    broker = Column(String(100), nullable=True)
    reference = Column(String(200), nullable=True)
    notes = Column(Text, nullable=True)
    category = Column(String(50), default="portfolio")
    is_deleted = Column(Integer, default=0)
    deleted_at = Column(Integer, nullable=True)
    created_at = Column(Integer, default=lambda: int(time.time()))

    user = relationship("User", back_populates="transactions")

    def __repr__(self) -> str:
        return f"<Transaction {self.id} {self.txn_type} {self.shares}sh>"


class Transaction(Base):
    """
    Legacy ``ledger_entries`` table from the accounting backend (setup_db.py).

    Kept for backward compat with report scripts.
    """

    __tablename__ = "ledger_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer)
    entry_date = Column(String(20))
    entry_type = Column(String(30))
    quantity = Column(Float, default=0.0)
    price_per_unit = Column(Float, default=0.0)
    total_value = Column(Float, default=0.0)
    fees = Column(Float, default=0.0)
    notes = Column(Text)
    created_at = Column(Integer, default=lambda: int(time.time()))

    def __repr__(self) -> str:
        return f"<LedgerEntry {self.id} {self.entry_type}>"


class PortfolioTransaction2(Base):
    """
    The ``portfolio_transactions`` table — a newer unified ledger
    linking portfolios, accounts, and stocks via FKs.

    Distinct from PortfolioTransaction which maps the legacy ``transactions`` table.
    """

    __tablename__ = "portfolio_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    account_id = Column(Integer, nullable=True)
    stock_id = Column(Integer, nullable=True)
    txn_type = Column(String(20), nullable=False)
    txn_date = Column(String(20), nullable=False)
    amount = Column(Float, default=0.0)
    shares = Column(Float, nullable=True)
    price_per_share = Column(Float, nullable=True)
    fees = Column(Float, default=0.0)
    currency = Column(String(10), default="KWD")
    fx_rate = Column(Float, nullable=True)
    symbol = Column(String(50), nullable=True)
    description = Column(Text, nullable=True)
    reference = Column(String(200), nullable=True)
    notes = Column(Text, nullable=True)
    is_deleted = Column(Integer, default=0)
    deleted_at = Column(Integer, nullable=True)
    created_at = Column(Integer, default=lambda: int(time.time()))
    updated_at = Column(Integer, nullable=True)

    portfolio = relationship("Portfolio", back_populates="portfolio_transactions")

    def __repr__(self) -> str:
        return f"<PortfolioTransaction2 {self.id} {self.txn_type}>"

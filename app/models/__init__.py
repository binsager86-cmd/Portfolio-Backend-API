"""
SQLAlchemy Models — package-level exports.

All models inherit from `app.core.database.Base`.
Import models here so Alembic's `target_metadata` can discover them.
"""

from app.models.user import User
from app.models.portfolio import (
    Portfolio,
    PortfolioTransaction,
    PortfolioTransaction2,
    Transaction,
    Stock,
)
from app.models.cash import CashDeposit, ExternalAccount
from app.models.security import SecuritiesMaster
from app.models.pfm import PfmSnapshot, PfmAsset, PfmLiability, PfmIncomeExpense
from app.models.snapshot import (
    PortfolioSnapshot,
    PositionSnapshot,
    PortfolioCash,
    StocksMaster,
)
from app.models.audit import AuditLog, TokenBlacklist

__all__ = [
    "User",
    "Portfolio",
    "PortfolioTransaction",
    "PortfolioTransaction2",
    "Transaction",
    "Stock",
    "CashDeposit",
    "ExternalAccount",
    "SecuritiesMaster",
    "PfmSnapshot",
    "PfmAsset",
    "PfmLiability",
    "PfmIncomeExpense",
    "PortfolioSnapshot",
    "PositionSnapshot",
    "PortfolioCash",
    "StocksMaster",
    "AuditLog",
    "TokenBlacklist",
]

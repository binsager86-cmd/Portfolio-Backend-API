"""
Validators — input validation helpers for stock symbols, portfolios, etc.
"""

import re
from typing import Optional

from app.services.fx_service import PORTFOLIO_CCY


# Valid stock symbol: 1-20 alphanumeric chars, may include . or -
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9\.\-]{1,20}$")


def validate_stock_symbol(symbol: str) -> Optional[str]:
    """
    Validate and normalize a stock symbol.

    Returns the cleaned symbol or None if invalid.
    """
    if not symbol:
        return None
    cleaned = symbol.strip().upper()
    if not _SYMBOL_RE.match(cleaned):
        return None
    return cleaned


def validate_portfolio_name(name: str) -> bool:
    """Check if a portfolio name is valid (KFH, BBYN, USA)."""
    return name in PORTFOLIO_CCY


def validate_txn_type(txn_type: str) -> bool:
    """Check if a transaction type is valid."""
    return txn_type in ("Buy", "Sell")


def validate_deposit_type(deposit_type: str) -> bool:
    """Check if a deposit type is valid."""
    return deposit_type in ("deposit", "withdrawal")


def validate_currency(currency: str) -> bool:
    """Check if a currency code is valid."""
    return currency in ("KWD", "USD")


def validate_date_format(date_str: str) -> bool:
    """Check if a string is in YYYY-MM-DD format."""
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", date_str))

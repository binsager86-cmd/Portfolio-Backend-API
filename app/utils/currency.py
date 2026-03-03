"""
Currency utilities — convert_to_kwd(), safe_float().

These are the canonical implementations — fx_service.py delegates here.
"""

from typing import Optional


DEFAULT_USD_TO_KWD: float = 0.307190

PORTFOLIO_CCY: dict[str, str] = {
    "KFH": "KWD",
    "BBYN": "KWD",
    "USA": "USD",
}


def safe_float(v, default: float = 0.0) -> float:
    """Safely convert a value to float, returning *default* on failure."""
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def convert_to_kwd(
    amount: float,
    ccy: str,
    usd_kwd_rate: Optional[float] = None,
) -> float:
    """
    Convert *amount* from *ccy* to KWD.

    Parameters:
        amount: The monetary value.
        ccy: Source currency code ('KWD', 'USD', etc.).
        usd_kwd_rate: Optional override for the USD→KWD rate.
                      If None, uses the live/cached rate from fx_service.

    Returns:
        The KWD-equivalent amount.
    """
    if amount is None:
        return 0.0
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return 0.0

    if ccy is None:
        ccy = "KWD"

    if ccy == "KWD":
        return amount
    if ccy == "USD":
        rate = usd_kwd_rate or _get_rate()
        return amount * rate
    return amount  # other currencies pass-through


def _get_rate() -> float:
    """Lazy import to avoid circular dependency with fx_service."""
    try:
        from app.services.fx_service import get_usd_kwd_rate
        return get_usd_kwd_rate()
    except Exception:
        return DEFAULT_USD_TO_KWD


def format_kwd(amount: float, decimals: int = 3) -> str:
    """Format a KWD amount with thousands separator."""
    return f"{amount:,.{decimals}f}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Format a percentage value."""
    return f"{value:.{decimals}f}%"

"""
FX (Foreign Exchange) Service

Handles USD→KWD conversion with 1-hour caching.
Extracts and de-Streamlit-ifies the logic from legacy ui.py.
"""

import time
import random
import logging
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

# ── Constants ────────────────────────────────────────────────────────
DEFAULT_USD_TO_KWD: float = 0.307190  # Approximate fallback rate
BASE_CCY = "KWD"
USD_CCY = "USD"

PORTFOLIO_CCY: dict[str, str] = {
    "KFH": "KWD",
    "BBYN": "KWD",
    "USA": "USD",
}

# ── In-memory FX cache ──────────────────────────────────────────────
_fx_cache: dict[str, dict] = {}


def _cache_key() -> str:
    return "usd_kwd"


def _get_cached_rate() -> Optional[float]:
    """Return cached rate if still fresh, else None."""
    entry = _fx_cache.get(_cache_key())
    if entry is None:
        return None
    if time.time() - entry["ts"] > _settings.FX_CACHE_TTL:
        return None  # expired
    return entry["rate"]


def _set_cached_rate(rate: float) -> None:
    _fx_cache[_cache_key()] = {"rate": rate, "ts": time.time()}


# ── Public API ──────────────────────────────────────────────────────

def fetch_usd_kwd_rate(max_retries: int = 3) -> float:
    """
    Fetch live USD→KWD rate via yfinance with retry + exponential back-off.
    Returns cached rate if available and fresh (< FX_CACHE_TTL seconds old).
    Falls back to DEFAULT_USD_TO_KWD on failure.
    """
    # 1) Check cache first
    cached = _get_cached_rate()
    if cached is not None:
        return cached

    # 2) Try yfinance
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed – using fallback USD/KWD rate")
        return DEFAULT_USD_TO_KWD

    for attempt in range(1, max_retries + 1):
        try:
            ticker = yf.Ticker("KWD=X")
            hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                rate = float(hist["Close"].dropna().iloc[-1])
                if rate > 0:
                    _set_cached_rate(rate)
                    logger.info(f"Fetched USD/KWD rate: {rate}")
                    return rate
        except Exception as exc:
            logger.debug(f"USD/KWD fetch attempt {attempt} failed: {exc}")
            if attempt < max_retries:
                wait = (2 ** attempt) + random.uniform(0.3, 1.0)
                time.sleep(wait)

    logger.warning("All USD/KWD fetch attempts failed – using fallback rate")
    return DEFAULT_USD_TO_KWD


def get_usd_kwd_rate() -> float:
    """Convenience wrapper – always returns a rate (cached → live → fallback)."""
    return fetch_usd_kwd_rate()


def convert_to_kwd(amount: float, ccy: str) -> float:
    """
    Convert *amount* from *ccy* to KWD.
    No Streamlit dependency – uses cached or live FX rate.
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
        rate = get_usd_kwd_rate()
        return amount * rate
    return amount  # other currencies pass-through


def safe_float(v, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default

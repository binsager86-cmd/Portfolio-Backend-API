"""
Price Service — stock price update logic.

Migrated from the legacy cron handler in ui.py (lines 1-125).
Handles:
  - Kuwait stocks via Yahoo Finance ({SYMBOL}.KW suffix)
  - US stocks via Yahoo Finance (raw symbol)
  - KWD price normalisation (÷1000 when value >50)
  - Reference list lookup (matches Streamlit's resolve_yf_ticker)
  - Tracks update results for caller logging / API response
"""

import asyncio
import time
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, Dict

from app.core.cache import cache_key, price_cache, get_cached, set_cached
from app.core.database import get_conn, add_column_if_missing

logger = logging.getLogger(__name__)


# ── Reference list lookup (mirrors Streamlit resolve_yf_ticker) ──────

# Build {symbol → yf_ticker} maps from hardcoded stock lists
_KW_MAP: Dict[str, str] = {}
_US_MAP: Dict[str, str] = {}

def _ensure_maps():
    """Lazy-load symbol→yf_ticker maps from stock_lists.py."""
    if _KW_MAP:
        return
    try:
        from app.data.stock_lists import KUWAIT_STOCKS, US_STOCKS
        for entry in KUWAIT_STOCKS:
            _KW_MAP[entry["symbol"].upper()] = entry["yf_ticker"]
        for entry in US_STOCKS:
            _US_MAP[entry["symbol"].upper()] = entry["yf_ticker"]
    except ImportError:
        logger.warning("stock_lists.py not found — falling back to suffix rules")

# Variation aliases matching Streamlit's KUWAIT_VARIATIONS
_VARIATIONS: Dict[str, str] = {
    "AGILITY": "AGLTY",
    "AGILITY PLC": "AGLTY",
    "MABNEE": "MABANEE",
    "H-SOFT": "HUMANSOFT",
    "INCYTE": "INCY",
}


# ── Yahoo symbol mapping ─────────────────────────────────────────────
# Kuwait stocks on Yahoo use a .KW suffix and are quoted in fils (×1000).

def _yahoo_symbol(symbol: str, currency: str) -> str:
    """
    Convert an internal symbol to a Yahoo Finance ticker.

    Resolution order (matches Streamlit resolve_yf_ticker):
      1. If symbol already has a market suffix (.KW, .BH, etc.) → use as-is
      2. Look up in KUWAIT_STOCKS / US_STOCKS reference lists
      3. Apply variation mapping (AGILITY→AGLTY, etc.)
      4. Currency fallback: KWD→.KW, else raw symbol
    """
    sym_upper = symbol.strip().upper()

    # 1. Already has a suffix
    if "." in sym_upper:
        return sym_upper

    # 2. Reference list lookup
    _ensure_maps()
    if currency == "KWD" and sym_upper in _KW_MAP:
        return _KW_MAP[sym_upper]
    if currency == "USD" and sym_upper in _US_MAP:
        return _US_MAP[sym_upper]
    # Also check the other list as fallback
    if sym_upper in _KW_MAP and currency == "KWD":
        return _KW_MAP[sym_upper]
    if sym_upper in _US_MAP:
        return _US_MAP[sym_upper]

    # 3. Variation mapping
    canonical = _VARIATIONS.get(sym_upper)
    if canonical:
        if currency == "KWD" and canonical in _KW_MAP:
            return _KW_MAP[canonical]
        if canonical in _US_MAP:
            return _US_MAP[canonical]
        # Apply currency suffix to canonical
        if currency == "KWD":
            return f"{canonical}.KW"
        return canonical

    # 4. Currency suffix fallback
    if currency == "KWD":
        return f"{sym_upper}.KW"
    return sym_upper          # USD / other


def _normalise_kwd_price(raw: float, currency: str) -> float:
    """
    Kuwait Exchange quotes are in fils → always divide by 1000 to get KWD.
    yfinance returns .KW prices in fils; dividing converts to KWD.
    """
    if currency == "KWD":
        return raw / 1000.0
    return raw


def _fetch_price_snapshot_sync(symbol: str, currency: str) -> dict:
    """Fetch one live quote from Yahoo Finance in a worker thread."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed") from exc

    yahoo_sym = _yahoo_symbol(symbol, currency)
    ticker = yf.Ticker(yahoo_sym)
    hist = ticker.history(period="5d", interval="1d")

    if hist is not None and getattr(hist.columns, "nlevels", 1) > 1:
        hist.columns = hist.columns.get_level_values(0)

    if hist is None or hist.empty or "Close" not in hist.columns:
        raise ValueError("No market data")

    closes = hist["Close"].dropna()
    if closes.empty:
        raise ValueError("No market data")

    price = _normalise_kwd_price(float(closes.iloc[-1]), currency)
    previous_close = None
    if len(closes) >= 2:
        previous_close = _normalise_kwd_price(float(closes.iloc[-2]), currency)

    pe_ratio = None
    try:
        info = ticker.info
        pe_val = info.get("trailingPE") or info.get("forwardPE")
        if pe_val is not None:
            pe_ratio = round(float(pe_val), 2)
    except Exception as exc:
        logger.debug("P/E fetch failed for %s: %s", yahoo_sym, exc)

    return {
        "symbol": symbol,
        "yahoo_symbol": yahoo_sym,
        "price": round(price, 6),
        "previous_close": round(previous_close, 6) if previous_close is not None else None,
        "pe_ratio": pe_ratio,
        "currency": currency,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


async def get_price_snapshot(symbol: str, currency: str = "KWD") -> dict:
    """Return a cached live quote, degrading gracefully on upstream failures."""
    key = cache_key("price", symbol.strip().upper(), currency.upper())
    cached = price_cache.get(key)
    if cached is not None:
        return cached

    try:
        result = await asyncio.to_thread(_fetch_price_snapshot_sync, symbol, currency)
        price_cache[key] = result
        return result
    except Exception as exc:
        fallback = cached if cached is not None else {
            "symbol": symbol,
            "price": None,
            "previous_close": None,
            "pe_ratio": None,
            "currency": currency,
            "error": str(exc),
        }
        return fallback


# ── Result container ─────────────────────────────────────────────────

@dataclass
class PriceUpdateResult:
    """Summary of a single run of the price updater."""
    stocks_found: int = 0
    updated: int = 0
    failed: int = 0
    skipped: int = 0
    details: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "stocks_found": self.stocks_found,
            "updated": self.updated,
            "failed": self.failed,
            "skipped": self.skipped,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "details": self.details,
            "errors": self.errors,
        }


# ── Core updater ─────────────────────────────────────────────────────

def update_all_prices(
    user_id: int = 1,
    only_with_holdings: bool = True,
) -> PriceUpdateResult:
    """
    Fetch the latest closing price for every stock in the ``stocks`` table
    and write it back.  Mirrors the legacy cron handler in ui.py.

    Parameters
    ----------
    user_id : int
        Which user's stocks to update (default 1).
    only_with_holdings : bool
        If True, only update stocks that have a positive share balance
        (i.e. net buys − sells > 0.001).  Saves API calls on dead positions.
    """
    # Lazy-import so the module loads even if yfinance is missing in test envs
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance is not installed – cannot update prices.")
        res = PriceUpdateResult()
        res.errors.append("yfinance not installed")
        return res

    t0 = time.time()
    result = PriceUpdateResult()

    conn = get_conn()
    cur = conn.cursor()

    try:
        # ── Fetch eligible stocks ────────────────────────────────────
        if only_with_holdings:
            cur.execute(
                """
                SELECT s.id, s.symbol, s.currency, s.yf_ticker,
                    COALESCE(
                        SUM(CASE WHEN t.txn_type = 'Buy'  THEN t.shares ELSE 0 END) -
                        SUM(CASE WHEN t.txn_type = 'Sell' THEN t.shares ELSE 0 END),
                    0) AS net_shares
                FROM stocks s
                LEFT JOIN transactions t
                    ON s.symbol = t.stock_symbol AND s.user_id = t.user_id
                WHERE s.user_id = ?
                  AND s.symbol IS NOT NULL AND s.symbol != ''
                GROUP BY s.id, s.symbol, s.currency, s.yf_ticker
                HAVING COALESCE(
                        SUM(CASE WHEN t.txn_type = 'Buy'  THEN t.shares ELSE 0 END) -
                        SUM(CASE WHEN t.txn_type = 'Sell' THEN t.shares ELSE 0 END),
                       0) > 0.001
                """,
                (user_id,),
            )
        else:
            cur.execute(
                """
                SELECT s.id, s.symbol, s.currency, s.yf_ticker, 0 AS net_shares
                FROM stocks s
                WHERE s.user_id = ?
                  AND s.symbol IS NOT NULL AND s.symbol != ''
                """,
                (user_id,),
            )

        stocks = cur.fetchall()
        result.stocks_found = len(stocks)
        logger.info("Price updater: found %d stocks to update", len(stocks))

        # Ensure additive columns exist across SQLite/PostgreSQL
        add_column_if_missing("stocks", "pe_ratio", "REAL")
        add_column_if_missing("stocks", "previous_close", "REAL")

        # ── Fetch & write prices ─────────────────────────────────────
        for stock_id, symbol, currency, stored_yf_ticker, _ in stocks:
            try:
                # Prefer stored yf_ticker if available, else derive from symbol+currency
                yahoo_sym = stored_yf_ticker if stored_yf_ticker else _yahoo_symbol(symbol, currency)

                # Check price cache before hitting yfinance network
                cached_data = get_cached(price_cache, yahoo_sym)
                if cached_data is not None:
                    price = cached_data["price"]
                    previous_close = cached_data["previous_close"]
                    pe_ratio = cached_data["pe_ratio"]
                else:
                    ticker = yf.Ticker(yahoo_sym)

                    # Use 5d window so weekends / holidays still return data
                    hist = ticker.history(period="5d", interval="1d")

                    # yfinance ≥ 1.0 may return MultiIndex columns
                    if hist is not None and hist.columns.nlevels > 1:
                        hist.columns = hist.columns.get_level_values(0)

                    if hist is None or hist.empty or "Close" not in hist.columns:
                        logger.warning("No data for %s (yahoo: %s)", symbol, yahoo_sym)
                        result.skipped += 1
                        result.details.append({"symbol": symbol, "status": "no_data"})
                        continue

                    closes = hist["Close"].dropna()
                    raw_price = float(closes.iloc[-1])
                    price = _normalise_kwd_price(raw_price, currency)
                    previous_close = None
                    if len(closes) >= 2:
                        previous_close = _normalise_kwd_price(float(closes.iloc[-2]), currency)

                    # Fetch P/E ratio from ticker info
                    pe_ratio = None
                    try:
                        info = ticker.info
                        pe_val = info.get("trailingPE") or info.get("forwardPE")
                        if pe_val is not None:
                            pe_ratio = round(float(pe_val), 2)
                    except Exception as pe_exc:
                        logger.debug("P/E fetch failed for %s: %s", yahoo_sym, pe_exc)

                    # Cache result for 5 minutes (TTL set on price_cache instance)
                    set_cached(price_cache, yahoo_sym, {
                        "price": price,
                        "previous_close": previous_close,
                        "pe_ratio": pe_ratio,
                    })

                cur.execute(
                    """
                    UPDATE stocks
                    SET current_price = ?,
                        last_updated  = ?,
                        price_source  = 'YAHOO',
                        pe_ratio      = ?,
                        previous_close = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    (round(price, 6), int(time.time()), pe_ratio, previous_close, stock_id, user_id),
                )
                conn.commit()

                result.updated += 1
                result.details.append({
                    "symbol": symbol,
                    "yahoo": yahoo_sym,
                    "price": round(price, 6),
                    "currency": currency,
                    "status": "ok",
                })
                logger.info("✅ %s → %s %.6f %s", symbol, yahoo_sym, price, currency)

            except Exception as exc:
                result.failed += 1
                result.errors.append({"symbol": symbol, "error": str(exc)})
                logger.warning("❌ %s: %s", symbol, exc)

    finally:
        conn.close()

    result.elapsed_sec = time.time() - t0
    logger.info(
        "Price update complete: %d updated, %d failed, %d skipped (%.1fs)",
        result.updated, result.failed, result.skipped, result.elapsed_sec,
    )
    return result

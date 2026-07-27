"""
Price Service — stock price update logic.

Migrated from the legacy cron handler in ui.py (lines 1-125).
Handles:
    - TickerChart-first live price fetches
    - Yahoo Finance fallback when TickerChart is unavailable
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

from app.core.cache import cache_key, price_cache
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
        "source": "yahoo",
    }


async def _fetch_snapshot_from_tickerchart(symbol: str, currency: str) -> dict:
    """Fetch a live price snapshot from TickerChart's OHLCV endpoint.

    Returns the same shape as _fetch_price_snapshot_sync.
    KSE prices are in fils → divided by 1000 to get KWD.
    """
    from datetime import date, timedelta
    from app.services import tickerchart_service as tc

    sym_upper = symbol.strip().upper()

    # Map currency to TickerChart market abbreviation
    if currency == "KWD":
        market_abb = "KSE"
    elif currency == "USD":
        market_abb = "USA"
    else:
        raise ValueError(f"Unsupported currency for TickerChart snapshot: {currency}")

    # Request the last 7 calendar days to guarantee at least 2 trading-day closes
    to_d = date.today()
    from_d = to_d - timedelta(days=7)

    # 5-second hard cap so a slow/unreachable TickerChart host doesn't stall
    # all parallel holdings requests (Yahoo Finance fallback fires immediately).
    rows = await asyncio.wait_for(
        tc.fetch_ohlcv(sym_upper, market_abb, from_d=from_d, to_d=to_d, interval="day"),
        timeout=5.0,
    )
    if not rows:
        raise ValueError(f"No TickerChart data for {sym_upper}.{market_abb}")

    # TickerChart parser returns rows in ASC order (oldest first)
    latest = rows[-1]
    raw_price = float(latest["close"])
    price = _normalise_kwd_price(raw_price, currency)

    previous_close: Optional[float] = None
    if len(rows) >= 2:
        raw_prev = float(rows[-2]["close"])
        previous_close = _normalise_kwd_price(raw_prev, currency)

    pe_ratio = tc.read_quotes_snapshot_pe(sym_upper, market_abb)
    if pe_ratio is None and price > 0:
        try:
            ltm_eps = await tc.fetch_ltm_eps(sym_upper, market_abb)
        except Exception as exc:
            logger.debug("TickerChart EPS fetch failed for %s.%s: %s", sym_upper, market_abb, exc)
        else:
            if ltm_eps is not None and ltm_eps > 0:
                pe_ratio = round(price / ltm_eps, 2)
    elif pe_ratio is not None:
        pe_ratio = round(pe_ratio, 2)

    return {
        "symbol": symbol,
        "price": round(price, 6),
        "previous_close": round(previous_close, 6) if previous_close is not None else None,
        "pe_ratio": pe_ratio,
        "currency": currency,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "tickerchart",
    }


async def get_price_snapshot(symbol: str, currency: str = "KWD", force_refresh: bool = False) -> dict:
    """Return a cached live quote, degrading gracefully on upstream failures.

    Tries TickerChart first (accurate KSE / US exchange data), then falls
    back to Yahoo Finance if TickerChart is unavailable or returns no data.
    """
    key = cache_key("price", symbol.strip().upper(), currency.upper())
    cached = None if force_refresh else price_cache.get(key)
    if cached is not None:
        return cached

    # --- Primary: TickerChart ---
    try:
        result = await _fetch_snapshot_from_tickerchart(symbol, currency)
        price_cache[key] = result
        return result
    except Exception as tc_exc:
        logger.debug(
            "TickerChart snapshot failed for %s (%s): %s — falling back to Yahoo Finance",
            symbol, currency, tc_exc,
        )

    # --- Fallback: Yahoo Finance ---
    try:
        result = await asyncio.to_thread(_fetch_price_snapshot_sync, symbol, currency)
        price_cache[key] = result
        return result
    except Exception as exc:
        return {
            "symbol": symbol,
            "price": None,
            "previous_close": None,
            "pe_ratio": None,
            "currency": currency,
            "error": str(exc),
        }


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
    used_full_scan_fallback: bool = False

    def to_dict(self) -> dict:
        return {
            "stocks_found": self.stocks_found,
            "updated": self.updated,
            "failed": self.failed,
            "skipped": self.skipped,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "details": self.details,
            "errors": self.errors,
            "used_full_scan_fallback": self.used_full_scan_fallback,
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
    t0 = time.time()
    result = PriceUpdateResult()

    conn = get_conn()
    cur = conn.cursor()

    try:
        # Ensure additive columns exist across SQLite/PostgreSQL before reads
        add_column_if_missing("stocks", "pe_ratio", "REAL")
        add_column_if_missing("stocks", "previous_close", "REAL")

        def _select_all_stocks() -> list:
            cur.execute(
                """
                SELECT s.id, s.symbol, s.currency, s.yf_ticker, s.pe_ratio, 0 AS net_shares
                FROM stocks s
                WHERE s.user_id = ?
                  AND s.symbol IS NOT NULL AND s.symbol != ''
                """,
                (user_id,),
            )
            return cur.fetchall()

        # ── Fetch eligible stocks ────────────────────────────────────
        stocks = []
        if only_with_holdings:
            cur.execute(
                """
                SELECT s.id, s.symbol, s.currency, s.yf_ticker, s.pe_ratio,
                    COALESCE(
                        SUM(CASE
                            WHEN UPPER(TRIM(t.txn_type)) = 'BUY'
                                THEN COALESCE(t.shares, 0) + COALESCE(t.bonus_shares, 0)
                            WHEN UPPER(TRIM(t.txn_type)) = 'SELL'
                                THEN -COALESCE(t.shares, 0)
                            ELSE COALESCE(t.bonus_shares, 0)
                        END),
                    0) AS net_shares
                FROM stocks s
                LEFT JOIN transactions t
                    ON UPPER(TRIM(s.symbol)) = UPPER(TRIM(t.stock_symbol))
                   AND s.user_id = t.user_id
                   AND COALESCE(NULLIF(TRIM(s.portfolio), ''), 'KFH') = COALESCE(NULLIF(TRIM(t.portfolio), ''), 'KFH')
                   AND COALESCE(t.category, 'portfolio') = 'portfolio'
                   AND COALESCE(t.is_deleted, 0) = 0
                WHERE s.user_id = ?
                  AND s.symbol IS NOT NULL AND s.symbol != ''
                GROUP BY s.id, s.symbol, s.currency, s.yf_ticker, s.pe_ratio
                HAVING COALESCE(
                    SUM(CASE
                        WHEN UPPER(TRIM(t.txn_type)) = 'BUY'
                            THEN COALESCE(t.shares, 0) + COALESCE(t.bonus_shares, 0)
                        WHEN UPPER(TRIM(t.txn_type)) = 'SELL'
                            THEN -COALESCE(t.shares, 0)
                        ELSE COALESCE(t.bonus_shares, 0)
                    END),
                0) > 0.001
                """,
                (user_id,),
            )
            stocks = cur.fetchall()
            if not stocks:
                result.used_full_scan_fallback = True
                logger.warning(
                    "Price updater holdings-filter matched 0 stocks for user_id=%s; falling back to full stock scan",
                    user_id,
                )
                stocks = _select_all_stocks()
        else:
            stocks = _select_all_stocks()

        result.stocks_found = len(stocks)
        logger.info("Price updater: found %d stocks to update", len(stocks))

        # ── Fetch & write prices ─────────────────────────────────────
        for stock_id, symbol, currency, stored_yf_ticker, existing_pe_ratio, _ in stocks:
            try:
                snapshot = asyncio.run(get_price_snapshot(symbol, currency or "KWD", force_refresh=True))
                price = snapshot.get("price")
                if price is None:
                    logger.warning("No price data for %s: %s", symbol, snapshot.get("error") or "no_data")
                    result.skipped += 1
                    result.details.append({"symbol": symbol, "status": "no_data", "error": snapshot.get("error")})
                    continue

                previous_close = snapshot.get("previous_close")
                pe_ratio = snapshot.get("pe_ratio") if snapshot.get("pe_ratio") is not None else existing_pe_ratio
                price_source = str(snapshot.get("source") or "yahoo").upper()

                cur.execute(
                    """
                    UPDATE stocks
                    SET current_price = ?,
                        last_updated  = ?,
                        price_source  = ?,
                        pe_ratio      = ?,
                        previous_close = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    (round(float(price), 6), int(time.time()), price_source, pe_ratio, previous_close, stock_id, user_id),
                )

                result.updated += 1
                result.details.append({
                    "symbol": symbol,
                    "source": price_source,
                    "price": round(float(price), 6),
                    "currency": currency,
                    "status": "ok",
                })
                logger.info("✅ %s → %s %.6f %s", symbol, price_source, float(price), currency)

            except Exception as exc:
                result.failed += 1
                result.errors.append({"symbol": symbol, "error": str(exc)})
                logger.warning("❌ %s: %s", symbol, exc)

        conn.commit()

    finally:
        conn.close()

    result.elapsed_sec = time.time() - t0
    logger.info(
        "Price update complete: %d updated, %d failed, %d skipped (%.1fs)",
        result.updated, result.failed, result.skipped, result.elapsed_sec,
    )
    return result

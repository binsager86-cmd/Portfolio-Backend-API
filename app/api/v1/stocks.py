"""
Stocks API v1 — CRUD for stock records (price tracking, metadata).

The ``stocks`` table stores per-user stock definitions with current prices,
currencies, and optional metadata (sector, industry, TradingView symbol).

Includes stock-list browse (Kuwait / US hardcoded reference lists) and
single-ticker yfinance price fetch for use at stock-creation time.
"""

import time
import asyncio
import logging
import dataclasses
import threading
import json
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.core.exceptions import NotFoundError, BadRequestError, ConflictError
from app.core.database import query_df, query_one, query_val, exec_sql, add_column_if_missing
from app.data.stock_lists import KUWAIT_STOCKS, US_STOCKS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stocks", tags=["Stocks"])


_US_STOCKS_CACHE_TTL_SEC = 24 * 60 * 60
_US_STOCKS_CACHE: dict = {
    "expires_at": 0.0,
    "stocks": [],
}
_US_STOCKS_CACHE_LOCK = threading.Lock()
_US_STOCKS_DISK_CACHE_PATH = Path(__file__).resolve().parents[2] / "cache" / "us_stock_universe.json"
_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PortfolioApp/1.0)",
    "Accept": "text/plain,application/json;q=0.9,*/*;q=0.8",
}


def _normalize_us_symbol(raw_symbol: str) -> str:
    return raw_symbol.strip().upper().replace(".", "-")


def _append_us_entry(target: list, seen: set, symbol: str, name: str) -> None:
    sym = _normalize_us_symbol(symbol)
    if not sym:
        return
    if any(ch in sym for ch in ("=", ":", "^", "/")):
        return
    if sym in seen:
        return

    display_name = (name or "").strip() or sym
    target.append({
        "symbol": sym,
        "name": display_name,
        "yf_ticker": sym,
    })
    seen.add(sym)


def _build_cached_us_universe() -> list:
    merged: list = []
    seen: set[str] = set()

    for s in US_STOCKS:
        _append_us_entry(merged, seen, s.get("symbol", ""), s.get("name", ""))

    def _fetch_text(url: str) -> Optional[str]:
        import requests

        try:
            resp = requests.get(url, timeout=20, headers=_HTTP_HEADERS)
            resp.raise_for_status()
            return resp.text
        except Exception:
            return None

    def _persist_universe(stocks: list) -> None:
        try:
            _US_STOCKS_DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": int(time.time()),
                "count": len(stocks),
                "stocks": stocks,
            }
            _US_STOCKS_DISK_CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")
        except Exception as e:
            logger.debug("Failed to persist US stock universe cache: %s", e)

    def _load_persisted_universe() -> list:
        try:
            if not _US_STOCKS_DISK_CACHE_PATH.exists():
                return []
            payload = json.loads(_US_STOCKS_DISK_CACHE_PATH.read_text(encoding="utf-8"))
            rows = payload.get("stocks") if isinstance(payload, dict) else []
            if not isinstance(rows, list):
                return []
            restored: list = []
            restored_seen: set[str] = set()
            for row in rows:
                if not isinstance(row, dict):
                    continue
                _append_us_entry(
                    restored,
                    restored_seen,
                    str(row.get("symbol", "")),
                    str(row.get("name", "")),
                )
            return restored
        except Exception as e:
            logger.debug("Failed to load persisted US stock universe cache: %s", e)
            return []

    external_added = 0

    try:
        feeds = [
            {
                "urls": [
                    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                    "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt",
                ],
                "symbol_key": "Symbol",
                "name_key": "Security Name",
                "test_key": "Test Issue",
            },
            {
                "urls": [
                    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                    "https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt",
                ],
                "symbol_key": "ACT Symbol",
                "name_key": "Security Name",
                "test_key": "Test Issue",
            },
        ]

        for feed in feeds:
            text = None
            for url in feed["urls"]:
                text = _fetch_text(url)
                if text:
                    break
            if not text:
                continue

            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not lines:
                continue

            headers = lines[0].split("|")
            idx = {h: i for i, h in enumerate(headers)}
            if feed["symbol_key"] not in idx or feed["name_key"] not in idx:
                continue

            symbol_i = idx[feed["symbol_key"]]
            name_i = idx[feed["name_key"]]
            test_i = idx.get(feed["test_key"])

            for ln in lines[1:]:
                if ln.startswith("File Creation Time"):
                    continue
                parts = ln.split("|")
                if len(parts) <= max(symbol_i, name_i):
                    continue
                if test_i is not None and len(parts) > test_i and parts[test_i].strip().upper() == "Y":
                    continue

                symbol_val = parts[symbol_i].strip()
                name_val = parts[name_i].strip()
                before = len(seen)
                _append_us_entry(merged, seen, symbol_val, name_val)
                if len(seen) > before:
                    external_added += 1
    except Exception as e:
        logger.warning("US universe expansion (NASDAQ feeds) failed: %s", e)

    try:
        sec_text = _fetch_text("https://www.sec.gov/files/company_tickers_exchange.json")
        if sec_text:
            payload = json.loads(sec_text)
            fields = payload.get("fields") if isinstance(payload, dict) else []
            data = payload.get("data") if isinstance(payload, dict) else []
            if isinstance(fields, list) and isinstance(data, list):
                field_map = {str(name).strip().lower(): i for i, name in enumerate(fields)}
                ticker_i = field_map.get("ticker")
                name_i = field_map.get("name")
                exchange_i = field_map.get("exchange")
                allowed = {
                    "nasdaq", "nyse", "nyse american", "nyse arca", "nyse mkt",
                    "cboe", "amex",
                }
                if ticker_i is not None and name_i is not None:
                    for row in data:
                        if not isinstance(row, list):
                            continue
                        if len(row) <= max(ticker_i, name_i):
                            continue
                        exchange_val = ""
                        if exchange_i is not None and len(row) > exchange_i:
                            exchange_val = str(row[exchange_i] or "").strip().lower()
                        if exchange_val and exchange_val not in allowed:
                            continue
                        symbol_val = str(row[ticker_i] or "").strip()
                        name_val = str(row[name_i] or "").strip()
                        before = len(seen)
                        _append_us_entry(merged, seen, symbol_val, name_val)
                        if len(seen) > before:
                            external_added += 1
    except Exception as e:
        logger.warning("US universe expansion (SEC feed) failed: %s", e)

    if external_added > 0:
        _persist_universe(merged)
        return merged

    restored = _load_persisted_universe()
    if restored:
        for row in restored:
            _append_us_entry(merged, seen, row.get("symbol", ""), row.get("name", ""))
        logger.warning(
            "US universe live feeds unavailable; restored %d symbols from disk cache",
            len(merged),
        )

    return merged


def _get_cached_us_universe() -> list:
    now = time.time()
    if _US_STOCKS_CACHE["stocks"] and now < float(_US_STOCKS_CACHE["expires_at"]):
        return _US_STOCKS_CACHE["stocks"]

    with _US_STOCKS_CACHE_LOCK:
        now = time.time()
        if _US_STOCKS_CACHE["stocks"] and now < float(_US_STOCKS_CACHE["expires_at"]):
            return _US_STOCKS_CACHE["stocks"]

        expanded = _build_cached_us_universe()
        _US_STOCKS_CACHE["stocks"] = expanded
        _US_STOCKS_CACHE["expires_at"] = now + _US_STOCKS_CACHE_TTL_SEC
        logger.info("US stock-list cache refreshed: %d symbols", len(expanded))
        return expanded


def _augment_us_stock_search(stocks: list[dict], search: str) -> list[dict]:
    """Run blocking yfinance lookups off the event loop."""
    stocks = list(stocks)
    existing_symbols = {s["symbol"].upper() for s in stocks}
    try:
        import yfinance as yf
        yf.screen(
            yf.EquityQuery("is", ["exchange", "NMS", "NYQ", "NGM", "PCX", "BTS", "ASE"]),
            size=25,
            sortField="intradaymarketcap",
            sortAsc=False,
        )
    except Exception:
        pass

    try:
        import yfinance as yf
        sym_upper = search.strip().upper()
        if sym_upper not in existing_symbols:
            tk = yf.Ticker(sym_upper)
            info = tk.info or {}
            name = info.get("shortName") or info.get("longName")
            if name and info.get("regularMarketPrice"):
                stocks.append({"symbol": sym_upper, "name": name, "yf_ticker": sym_upper})
                existing_symbols.add(sym_upper)
    except Exception:
        pass

    try:
        import yfinance as yf
        results = yf.search(search.strip(), max_results=10)
        if results and "quotes" in results:
            for qt in results["quotes"]:
                sym = qt.get("symbol", "")
                name = qt.get("shortname") or qt.get("longname") or ""
                qtype = qt.get("quoteType", "")
                if sym and sym.upper() not in existing_symbols and qtype in ("EQUITY", "ETF") and not any(c in sym for c in ".:"):
                    stocks.append({"symbol": sym.upper(), "name": name, "yf_ticker": sym.upper()})
                    existing_symbols.add(sym.upper())
    except Exception:
        pass

    return stocks


# ── Schemas ──────────────────────────────────────────────────────────

class StockCreate(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=50)
    name: Optional[str] = Field(None, max_length=200)
    portfolio: str = Field(..., description="KFH, BBYN, or USA")
    currency: str = Field("KWD", max_length=10)
    current_price: Optional[float] = Field(None, ge=0)
    yf_ticker: Optional[str] = Field(None, max_length=50, description="Yahoo Finance ticker, e.g. KFH.KW or AAPL")
    tradingview_symbol: Optional[str] = Field(None, max_length=100)
    tradingview_exchange: Optional[str] = Field(None, max_length=100)
    price_source: Optional[str] = Field(None, max_length=50)


class StockUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=200)
    current_price: Optional[float] = Field(None, ge=0)
    currency: Optional[str] = Field(None, max_length=10)
    portfolio: Optional[str] = Field(None, max_length=20)
    yf_ticker: Optional[str] = Field(None, max_length=50)
    tradingview_symbol: Optional[str] = Field(None, max_length=100)
    tradingview_exchange: Optional[str] = Field(None, max_length=100)
    price_source: Optional[str] = Field(None, max_length=50)


# ── List stocks ──────────────────────────────────────────────────────

@router.get("")
async def list_stocks(
    portfolio: Optional[str] = Query(None),
    search: Optional[str] = Query(None, description="Search symbol or name"),
    current_user: TokenData = Depends(get_current_user),
):
    """List all stocks for the current user, optionally filtered."""
    conditions = ["user_id = ?"]
    params: list = [current_user.user_id]

    if portfolio:
        conditions.append("portfolio = ?")
        params.append(portfolio)
    if search:
        conditions.append("(symbol LIKE ? OR name LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])

    where = " AND ".join(conditions)
    df = query_df(
        f"""
        SELECT id, symbol, name, portfolio, currency, current_price,
               tradingview_symbol, tradingview_exchange, price_source,
               last_updated
        FROM stocks
        WHERE {where}
        ORDER BY portfolio, symbol
        """,
        tuple(params),
    )

    records = df.to_dict(orient="records") if not df.empty else []
    return {"status": "ok", "data": {"stocks": records, "count": len(records)}}


# ── Stock reference list (Kuwait / US) ───────────────────────────────

@router.get("/stock-list")
async def get_stock_list(
    market: str = Query("Kuwait", description="'Kuwait' or 'US'"),
    search: Optional[str] = Query(None, description="Filter by symbol or name"),
):
    """
    Return the hardcoded reference stock list for a given market.
    No auth required — this is public reference data.
    Each entry has: symbol, name, yf_ticker.

    For US market: when search is provided and the hardcoded list has < 5
    results, augment with live yfinance search results.
    """
    is_us = not market.lower().startswith("k")
    base_stocks = KUWAIT_STOCKS if not is_us else await asyncio.to_thread(_get_cached_us_universe)
    stocks = [dict(s) for s in base_stocks]

    if search:
        q = search.upper()
        stocks = [
            s for s in stocks
            if q in s["symbol"].upper() or q in s["name"].upper()
        ]

    # For US market: augment with live yfinance search when few hardcoded matches
    if is_us and search and len(stocks) < 5:
        stocks = await asyncio.to_thread(_augment_us_stock_search, stocks, search)

    return {
        "status": "ok",
        "data": {
            "stocks": stocks,
            "count": len(stocks),
            "market": "Kuwait" if not is_us else "US",
        },
    }


# ── Fetch price via yfinance ─────────────────────────────────────────

class FetchPriceRequest(BaseModel):
    yf_ticker: str = Field(..., description="Yahoo Finance ticker, e.g. KFH.KW or AAPL")
    currency: str = Field("KWD", description="Currency for auto fils→KWD conversion")


@router.post("/fetch-price")
async def fetch_price(
    body: FetchPriceRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Fetch the latest closing price for a single ticker via yfinance.
    Returns price (auto-corrects Kuwait fils→KWD if > 50).
    """
    import yfinance as yf

    ticker = body.yf_ticker.strip()
    if not ticker:
        raise BadRequestError("yf_ticker is required")

    try:
        data = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
        if data.empty:
            return {"status": "ok", "data": {"price": None, "ticker": ticker, "message": "No data returned"}}

        # Get last closing price
        close_col = "Close"
        if close_col not in data.columns:
            # Multi-level column index from yfinance
            for col in data.columns:
                if "Close" in str(col):
                    close_col = col
                    break

        price = float(data[close_col].dropna().iloc[-1])

        # Auto-correct Kuwait fils → KWD (always divide for KWD)
        if body.currency == "KWD":
            price = round(price / 1000.0, 6)

        return {
            "status": "ok",
            "data": {"price": round(price, 4), "ticker": ticker},
        }
    except Exception as e:
        logger.warning(f"yfinance fetch failed for {ticker}: {e}")
        return {
            "status": "ok",
            "data": {"price": None, "ticker": ticker, "message": str(e)},
        }


# ── Get single stock ─────────────────────────────────────────────────

@router.get("/{stock_id}")
async def get_stock(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Get a single stock by its database ID."""
    row = query_one(
        "SELECT * FROM stocks WHERE id = ? AND user_id = ?",
        (stock_id, current_user.user_id),
    )
    if not row:
        raise NotFoundError("Stock", stock_id)

    return {"status": "ok", "data": dict(row)}


# ── Get stock by symbol ──────────────────────────────────────────────

@router.get("/by-symbol/{symbol}")
async def get_stock_by_symbol(
    symbol: str,
    current_user: TokenData = Depends(get_current_user),
):
    """Get a stock by its symbol."""
    row = query_one(
        "SELECT * FROM stocks WHERE TRIM(symbol) = ? AND user_id = ?",
        (symbol.strip(), current_user.user_id),
    )
    if not row:
        raise NotFoundError("Stock", symbol)

    return {"status": "ok", "data": dict(row)}


# ── Create stock ─────────────────────────────────────────────────────

@router.post("", status_code=201)
async def create_stock(
    body: StockCreate,
    current_user: TokenData = Depends(get_current_user),
):
    """Create a new stock entry."""
    uid = current_user.user_id
    symbol = body.symbol.strip().upper()

    # Ensure yf_ticker column exists (additive migration)
    add_column_if_missing("stocks", "yf_ticker", "TEXT")

    # Check for duplicate symbol per user and portfolio. The same symbol can exist in multiple markets.
    existing = query_val(
        "SELECT id FROM stocks WHERE UPPER(TRIM(symbol)) = ? AND user_id = ? AND COALESCE(NULLIF(TRIM(portfolio), ''), '') = ?",
        (symbol, uid, body.portfolio),
    )
    if existing:
        raise ConflictError(f"Stock '{symbol}' already exists in portfolio '{body.portfolio}'")

    now = int(time.time())
    exec_sql(
        """INSERT INTO stocks
           (user_id, symbol, name, portfolio, currency, current_price,
            yf_ticker, tradingview_symbol, tradingview_exchange, price_source, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            uid, symbol, body.name or symbol, body.portfolio,
            body.currency, body.current_price or 0.0,
            body.yf_ticker, body.tradingview_symbol, body.tradingview_exchange,
            body.price_source, now,
        ),
    )

    new_id = query_val(
        "SELECT id FROM stocks WHERE UPPER(TRIM(symbol)) = ? AND user_id = ? AND COALESCE(NULLIF(TRIM(portfolio), ''), '') = ? ORDER BY id DESC LIMIT 1",
        (symbol, uid, body.portfolio),
    )

    return {
        "status": "ok",
        "data": {"id": new_id, "symbol": symbol, "message": "Stock created"},
    }


# ── Update stock ─────────────────────────────────────────────────────

@router.put("/{stock_id}")
async def update_stock(
    stock_id: int,
    body: StockUpdate,
    current_user: TokenData = Depends(get_current_user),
):
    """Update a stock's metadata or price."""
    existing = query_one(
        "SELECT id FROM stocks WHERE id = ? AND user_id = ?",
        (stock_id, current_user.user_id),
    )
    if not existing:
        raise NotFoundError("Stock", stock_id)

    updates = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if not updates:
        raise BadRequestError("No valid fields to update")

    # Auto-set last_updated if price changed
    if "current_price" in updates:
        updates["last_updated"] = int(time.time())

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    params = list(updates.values()) + [stock_id, current_user.user_id]

    exec_sql(
        f"UPDATE stocks SET {set_clause} WHERE id = ? AND user_id = ?",
        tuple(params),
    )

    return {"status": "ok", "data": {"id": stock_id, "message": "Stock updated"}}


# ── Delete stock ─────────────────────────────────────────────────────

@router.delete("/{stock_id}")
async def delete_stock(
    stock_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Delete a stock entry (hard delete)."""
    existing = query_one(
        "SELECT id, symbol FROM stocks WHERE id = ? AND user_id = ?",
        (stock_id, current_user.user_id),
    )
    if not existing:
        raise NotFoundError("Stock", stock_id)

    exec_sql(
        "DELETE FROM stocks WHERE id = ? AND user_id = ?",
        (stock_id, current_user.user_id),
    )

    return {"status": "ok", "data": {"id": stock_id, "message": "Stock deleted"}}


# ── Merge / unify two stock records ──────────────────────────────────

class StockMergeRequest(BaseModel):
    source_stock_id: int = Field(..., description="Stock to merge FROM (will be deleted)")
    target_stock_id: int = Field(..., description="Stock to merge INTO (will be kept)")


@router.post("/merge")
async def merge_stocks(
    body: StockMergeRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Merge two stock records: reassign all transactions from the source
    stock to the target stock, then delete the source stock record.

    Use this to unify duplicate entries (e.g. GFH + GFH.KW → GFH).
    """
    uid = current_user.user_id

    source = query_one(
        "SELECT id, symbol, portfolio, currency FROM stocks WHERE id = ? AND user_id = ?",
        (body.source_stock_id, uid),
    )
    if not source:
        raise NotFoundError("Source stock", body.source_stock_id)

    target = query_one(
        "SELECT id, symbol, portfolio, currency FROM stocks WHERE id = ? AND user_id = ?",
        (body.target_stock_id, uid),
    )
    if not target:
        raise NotFoundError("Target stock", body.target_stock_id)

    if source["id"] == target["id"]:
        raise BadRequestError("Source and target stock cannot be the same")

    source_sym = source["symbol"].strip()
    target_sym = target["symbol"].strip()

    # Count transactions that will be moved
    moved = query_val(
        "SELECT COUNT(*) FROM transactions WHERE stock_symbol = ? AND user_id = ?",
        (source_sym, uid),
    ) or 0

    # Reassign transactions from source symbol to target symbol
    exec_sql(
        "UPDATE transactions SET stock_symbol = ? WHERE stock_symbol = ? AND user_id = ?",
        (target_sym, source_sym, uid),
    )

    # Also update portfolio assignment on moved transactions if needed
    target_portfolio = target["portfolio"]
    exec_sql(
        "UPDATE transactions SET portfolio = ? WHERE stock_symbol = ? AND user_id = ? AND (portfolio IS NULL OR portfolio = ?)",
        (target_portfolio, target_sym, uid, source["portfolio"]),
    )

    # Delete the source stock record
    exec_sql(
        "DELETE FROM stocks WHERE id = ? AND user_id = ?",
        (body.source_stock_id, uid),
    )

    logger.info(
        "Merged stock %s (id=%s) into %s (id=%s) for user %s — %s transactions moved",
        source_sym, source["id"], target_sym, target["id"], uid, moved,
    )

    return {
        "status": "ok",
        "data": {
            "message": f"Merged {source_sym} into {target_sym}",
            "source_symbol": source_sym,
            "target_symbol": target_sym,
            "transactions_moved": moved,
        },
    }


# ── Bulk price update (manual) ───────────────────────────────────────

@router.post("/update-prices")
async def manual_price_update(
    current_user: TokenData = Depends(get_current_user),
):
    """
    Trigger a price update for all stocks owned by the current user.
    Uses the shared price_service.
    """
    from app.services.price_service import update_all_prices

    try:
        result = await run_in_threadpool(
            lambda: update_all_prices(
                user_id=current_user.user_id,
                only_with_holdings=True,
            )
        )

        # Normalize result objects across Pydantic/dataclass/plain return types.
        if hasattr(result, "model_dump") and callable(result.model_dump):
            data = result.model_dump()
        elif hasattr(result, "dict") and callable(result.dict):
            data = result.dict()
        elif dataclasses.is_dataclass(result):
            data = dataclasses.asdict(result)
        elif hasattr(result, "to_dict") and callable(result.to_dict):
            data = result.to_dict()
        else:
            data = {}

        if not isinstance(data, dict):
            data = {}

        updated_count = data.get("updated", 0)
        stocks_found = data.get("stocks_found", 0)
        data["message"] = f"Updated {updated_count} of {stocks_found} stocks"
        data["updated_count"] = updated_count
        data["updatedCount"] = updated_count

        return {"status": "ok", "data": data}
    except Exception as e:
        logger.exception("Price update failed for user %s", current_user.user_id)
        return {
            "status": "error",
            "data": {"message": f"Price update failed: {str(e)}", "updated": 0},
        }

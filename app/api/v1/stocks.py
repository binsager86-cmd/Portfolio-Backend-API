"""
Stocks API v1 — CRUD for stock records (price tracking, metadata).

The ``stocks`` table stores per-user stock definitions with current prices,
currencies, and optional metadata (sector, industry, TradingView symbol).

Includes stock-list browse (Kuwait / US hardcoded reference lists) and
single-ticker yfinance price fetch for use at stock-creation time.
"""

import time
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.core.exceptions import NotFoundError, BadRequestError, ConflictError
from app.core.database import query_df, query_one, query_val, exec_sql, add_column_if_missing
from app.data.stock_lists import KUWAIT_STOCKS, US_STOCKS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stocks", tags=["Stocks"])


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
    """
    stocks = KUWAIT_STOCKS if market.lower().startswith("k") else US_STOCKS

    if search:
        q = search.upper()
        stocks = [
            s for s in stocks
            if q in s["symbol"].upper() or q in s["name"].upper()
        ]

    return {
        "status": "ok",
        "data": {
            "stocks": stocks,
            "count": len(stocks),
            "market": "Kuwait" if market.lower().startswith("k") else "US",
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

        # Auto-correct Kuwait fils → KWD
        if body.currency == "KWD" and price > 50:
            price = round(price / 1000.0, 3)

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

    # Check for duplicate symbol per user
    existing = query_val(
        "SELECT id FROM stocks WHERE TRIM(symbol) = ? AND user_id = ?",
        (symbol, uid),
    )
    if existing:
        raise ConflictError(f"Stock '{symbol}' already exists")

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
        "SELECT id FROM stocks WHERE symbol = ? AND user_id = ? ORDER BY id DESC LIMIT 1",
        (symbol, uid),
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

    result = update_all_prices(
        user_id=current_user.user_id,
        only_with_holdings=True,
    )

    return {
        "status": "ok",
        "data": result.to_dict() if hasattr(result, "to_dict") else {"message": "Prices updated"},
    }

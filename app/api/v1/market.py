"""
Market API v1 — Boursa Kuwait market summary data.

Endpoints:
  GET /summary     — full market summary (indices, sectors, gainers/losers)
  GET /refresh     — force refresh cached data
  GET /history     — historical market snapshots for a date range
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_current_user
from app.core.cache import cache_key, price_cache, get_cached, set_cached
from app.core.security import TokenData
from app.services.price_service import get_price_snapshot
from app.services.market_service import get_market_data, get_market_history

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["Market"])


@router.get("/overview")
async def market_overview(
    live: bool = Query(False, description="Set to true to trigger a background refresh; response still returns the latest cached snapshot immediately"),
    include_quotes: bool = Query(True, description="Include cached live quote snapshots for symbols visible in the market overview"),
    current_user: TokenData = Depends(get_current_user),
):
    """
    Return the latest market data snapshot from the database.

    In production, prices are never fetched on-demand; they are refreshed by
    the cron scheduler.  Use ``?live=true`` to enqueue a background refresh
    while the stale snapshot is still returned immediately.
    """
    overview_cache_key = cache_key("market", "overview")
    cached = get_cached(price_cache, overview_cache_key)
    if cached is not None and not live:
        return {"data": cached, "live": False, "status": "cached"}

    # Serve the most recent DB snapshot (no scraping)
    try:
        snapshot = await asyncio.to_thread(get_market_data, False)
    except Exception as e:
        logger.error("Market overview snapshot failed: %s", e)
        raise HTTPException(status_code=502, detail="Market snapshot unavailable")

    if include_quotes:
        symbols: dict[str, str] = {}
        for section in ("top_gainers", "top_losers", "top_value"):
            for item in snapshot.get(section, []) or []:
                symbol = str(item.get("symbol", "")).strip()
                if symbol:
                    symbols.setdefault(symbol, "KWD")

        if symbols:
            quote_tasks = [get_price_snapshot(symbol, currency) for symbol, currency in symbols.items()]
            quote_results = await asyncio.gather(*quote_tasks)
            snapshot["live_quotes"] = {
                symbol: result for (symbol, _currency), result in zip(symbols.items(), quote_results)
            }

    set_cached(price_cache, overview_cache_key, snapshot)

    if live:
        # Fire-and-forget background refresh; caller gets the stale snapshot now
        async def _bg_refresh():
            try:
                fresh = await asyncio.to_thread(get_market_data, True)
                set_cached(price_cache, overview_cache_key, fresh)
            except Exception as _e:
                logger.warning("Background market refresh failed: %s", _e)

        asyncio.create_task(_bg_refresh())
        return {"data": snapshot, "live": True, "status": "refreshing"}

    return {"data": snapshot, "live": False, "status": "ok"}


@router.get("/summary")
async def market_summary(
    current_user: TokenData = Depends(get_current_user),
):
    """Return cached market data for today (auto-scrapes if stale)."""
    try:
        data = await asyncio.to_thread(get_market_data)
        return {"status": "ok", "data": data}
    except Exception as e:
        logger.error("Market summary failed: %s", e)
        raise HTTPException(status_code=502, detail="Market data unavailable")


@router.get("/refresh")
async def market_refresh(
    current_user: TokenData = Depends(get_current_user),
):
    """Force re-scrape market data (bypasses cache)."""
    try:
        data = await asyncio.to_thread(get_market_data, True)
        return {"status": "ok", "data": data}
    except Exception as e:
        logger.error("Market refresh failed: %s", e)
        raise HTTPException(status_code=502, detail="Market data scrape failed")


@router.get("/history")
async def market_history(
    current_user: TokenData = Depends(get_current_user),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    limit: int = Query(30, ge=1, le=365, description="Max snapshots to return"),
):
    """Return historical market snapshots (one per trade date, most recent first)."""
    try:
        rows = await asyncio.to_thread(get_market_history, start_date, end_date, limit)
        return {"status": "ok", "data": rows}
    except Exception as e:
        logger.error("Market history failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve market history")

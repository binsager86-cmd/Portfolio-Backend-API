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
from app.core.security import TokenData
from app.services.market_service import get_market_data, get_market_history

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["Market"])


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

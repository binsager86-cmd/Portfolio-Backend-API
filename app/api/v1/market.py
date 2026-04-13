"""
Market API v1 — Boursa Kuwait market summary data.

Endpoints:
  GET /summary     — full market summary (indices, sectors, gainers/losers)
  GET /refresh     — force refresh cached data
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.services.market_service import get_market_data

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

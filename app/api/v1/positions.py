"""Positions API v1 — exit signal monitor endpoints."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_current_user
from app.core.security import TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/positions", tags=["positions"])


async def get_ohlcv_rows(
    symbol: str,
    exchange: str = Query(default="KSE"),
    country: Optional[str] = Query(default=None),
) -> list[dict]:
    """Resolve symbol and fetch indicator-enriched OHLCV rows from TickerChart."""
    from app.services import tickerchart_service as tc
    from app.services.indicators_service import attach_indicators
    from app.services.signal_engine.data.preprocessing import forward_fill_gaps

    parsed = tc.split_symbol(symbol, exchange, country)
    if parsed is None:
        raise HTTPException(status_code=400, detail=f"Cannot resolve symbol '{symbol}' to a TickerChart market")

    base, market = parsed
    fetch_from = date.today() - timedelta(days=730)

    try:
        rows = await tc.fetch_ohlcv(base, market, from_d=fetch_from, to_d=None)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        logger.warning("TickerChart request failed for %s.%s: %s", base, market, exc)
        raise HTTPException(status_code=502, detail="Failed to reach TickerChart data provider") from exc

    if not rows:
        raise HTTPException(status_code=404, detail=f"No price data returned for {symbol}")

    rows = forward_fill_gaps(rows)
    rows = attach_indicators(rows)
    return rows


@router.get("/{symbol}/exit-signal")
async def get_exit_signal(
    symbol: str,
    entry_price: float = Query(..., gt=0),
    bars_held: int = Query(0, ge=0),
    rows: list[dict] = Depends(get_ohlcv_rows),
    current_user: TokenData = Depends(get_current_user),
):
    del current_user
    from app.services.signal_engine.engine.exit_signal_engine import generate_exit_signal

    signal = generate_exit_signal(
        symbol=symbol.strip().upper(),
        rows=rows,
        entry_price=entry_price,
        bars_held=bars_held,
    )
    return {"status": "ok", "data": signal}

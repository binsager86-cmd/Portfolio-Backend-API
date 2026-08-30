"""
Trend-Hold Book API.

Read-only endpoints over the virtual-money paper-trading ledger that
mechanically fills trend_hold_engine decisions (see
app/services/eagle_eye_v2/trend_hold_book.py). Fully independent of the
real portfolio (app/api/v1/portfolio.py, app/api/v1/trading.py) and of the
unrelated eagle_eye_v2/simulator/ system.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.schemas.trend_hold_book import (
    TrendHoldBookPortfolio,
    TrendHoldBookPosition,
    TrendHoldBookPositionsResponse,
    TrendHoldBookTrade,
    TrendHoldBookTradesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trend-hold-book", tags=["Trend-Hold Book"])


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _mark_positions(positions: dict, trend_hold_map: dict) -> list[dict]:
    """Attach latest close + unrealized P&L to each open position."""
    marked = []
    for ticker, pos in positions.items():
        row = trend_hold_map.get(ticker) or {}
        latest_close = _safe_float(row.get("close"))
        quantity = _safe_float(pos.get("quantity")) or 0.0
        avg_cost = _safe_float(pos.get("avg_cost")) or 0.0
        market_value = quantity * latest_close if latest_close is not None else None
        unrealized_pnl = (
            quantity * (latest_close - avg_cost) if latest_close is not None else None
        )
        marked.append(
            {
                "ticker": ticker,
                "quantity": quantity,
                "avg_cost": avg_cost,
                "latest_close": latest_close,
                "market_value_kwd": market_value,
                "unrealized_pnl_kwd": unrealized_pnl,
                "opened_date": pos.get("opened_date"),
            }
        )
    marked.sort(key=lambda p: p["ticker"])
    return marked


@router.get("/portfolio", response_model=TrendHoldBookPortfolio)
async def get_trend_hold_book_portfolio(
    _user: TokenData = Depends(get_current_user),
):
    """Return the Trend-Hold Book's current cash, mark-to-market equity, and total return."""
    from app.services.eagle_eye.store import load_all_trend_hold_state
    from app.services.eagle_eye_v2 import trend_hold_book_store as book

    book.ensure_trend_hold_book_tables()
    state = book.load_book_state()
    positions = book.load_all_positions()
    trend_hold_map = load_all_trend_hold_state()

    cash = _safe_float(state.get("cash_kwd")) or 0.0
    starting_capital = _safe_float(state.get("starting_capital_kwd")) or 0.0
    marked = _mark_positions(positions, trend_hold_map)
    equity = cash + sum(p["market_value_kwd"] or (p["quantity"] * p["avg_cost"]) for p in marked)
    total_return_pct = ((equity / starting_capital) - 1.0) * 100.0 if starting_capital > 0 else 0.0

    return TrendHoldBookPortfolio(
        cash_kwd=round(cash, 3),
        starting_capital_kwd=round(starting_capital, 3),
        equity_kwd=round(equity, 3),
        total_return_pct=round(total_return_pct, 3),
        open_position_count=len(marked),
        as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


@router.get("/positions", response_model=TrendHoldBookPositionsResponse)
async def get_trend_hold_book_positions(
    _user: TokenData = Depends(get_current_user),
):
    """Return every currently open Trend-Hold Book paper position, marked to market."""
    from app.services.eagle_eye.store import load_all_trend_hold_state
    from app.services.eagle_eye_v2 import trend_hold_book_store as book

    book.ensure_trend_hold_book_tables()
    positions = book.load_all_positions()
    trend_hold_map = load_all_trend_hold_state()
    marked = _mark_positions(positions, trend_hold_map)

    return TrendHoldBookPositionsResponse(
        positions=[TrendHoldBookPosition(**p) for p in marked]
    )


@router.get("/trades", response_model=TrendHoldBookTradesResponse)
async def get_trend_hold_book_trades(
    limit: int = Query(default=300, ge=1, le=1000),
    _user: TokenData = Depends(get_current_user),
):
    """Return the Trend-Hold Book's trade ledger, newest first."""
    from app.services.eagle_eye_v2 import trend_hold_book_store as book

    book.ensure_trend_hold_book_tables()
    rows = book.load_recent_trades(limit=limit)

    trades = [
        TrendHoldBookTrade(
            id=int(r["id"]),
            ticker=r["ticker"],
            side=r["side"],
            trade_date=r["trade_date"],
            quantity=_safe_float(r.get("quantity")) or 0.0,
            price=_safe_float(r.get("price")) or 0.0,
            gross_kwd=_safe_float(r.get("gross_kwd")) or 0.0,
            commission_kwd=_safe_float(r.get("commission_kwd")) or 0.0,
            realized_pnl_kwd=_safe_float(r.get("realized_pnl_kwd")),
            reason=r.get("reason"),
        )
        for r in rows
    ]
    return TrendHoldBookTradesResponse(trades=trades)

"""
Trend-Hold Book API.

Read-only endpoints over the virtual-money paper-trading ledger that
mechanically fills trend_hold_engine decisions (see
app/services/eagle_eye_v2/trend_hold_book.py). Fully independent of the
real portfolio (app/api/v1/portfolio.py, app/api/v1/trading.py), of the
unrelated eagle_eye_v2/simulator/ system, and of the V1 Rating Book
(app/api/v1/v1_rating_book.py) -- a second, independent paper book run
side by side for comparison. Response-building logic is shared with that
router via _paper_book_common.py; only book_id and the price source differ.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_current_user
from app.api.v1._paper_book_common import (
    build_lessons_response,
    build_lessons_summary_response,
    build_nav_history_response,
    build_performance_response,
    build_portfolio_response,
    build_positions_response,
    build_trades_response,
    safe_float,
)
from app.core.security import TokenData
from app.schemas.trend_hold_book import (
    TrendHoldBookLessonsResponse,
    TrendHoldBookLessonsSummary,
    TrendHoldBookNavHistoryResponse,
    TrendHoldBookPerformance,
    TrendHoldBookPortfolio,
    TrendHoldBookPositionsResponse,
    TrendHoldBookTradesResponse,
    TrendHoldDecisionLogEntry,
    TrendHoldDecisionLogResponse,
)
from app.services.eagle_eye_v2.trend_hold_book import BOOK_ID

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trend-hold-book", tags=["Trend-Hold Book"])


def _price_map() -> dict:
    """
    Base map is the once-daily trend_hold_engine close for the full
    universe; live_prices overlays a fresher TickerChart quote for whichever
    tickers the book currently holds (see
    trend_hold_book.py::run_open_position_price_refresh), so open positions'
    displayed value/P&L can move between full scans.
    """
    from app.services.eagle_eye.store import load_all_trend_hold_state
    from app.services.eagle_eye_v2 import paper_book_store as book

    price_map = {t: safe_float(row.get("close")) for t, row in load_all_trend_hold_state().items()}
    price_map.update(book.load_live_prices(BOOK_ID))
    return price_map


@router.get("/portfolio", response_model=TrendHoldBookPortfolio)
async def get_trend_hold_book_portfolio(_user: TokenData = Depends(get_current_user)):
    """Return the Trend-Hold Book's current cash, mark-to-market equity, and total return."""
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()
    return build_portfolio_response(BOOK_ID, _price_map())


@router.post("/refresh-prices")
async def refresh_trend_hold_book_prices(_user: TokenData = Depends(get_current_user)):
    """
    Manually fetch a fresh TickerChart price for whatever the Trend-Hold
    Book currently holds -- the same intraday refresh the scheduler runs
    every 15 minutes during the KSE session (see
    run_open_position_price_refresh()), triggered on demand from the
    "Fetch Price" button. Only marks open positions to market; never
    touches ee_trend_hold_state or triggers a BUY/SCALE_OUT/SELL_SIGNAL
    decision.
    """
    from app.services.eagle_eye_v2.trend_hold_book import run_open_position_price_refresh

    result = await run_in_threadpool(run_open_position_price_refresh)
    return {
        "status": "ok",
        "updated": result.get("updated", 0),
        "positions": result.get("positions", 0),
        "errors": result.get("errors", 0),
    }


@router.get("/positions", response_model=TrendHoldBookPositionsResponse)
async def get_trend_hold_book_positions(_user: TokenData = Depends(get_current_user)):
    """Return every currently open Trend-Hold Book paper position, marked to market."""
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()
    return build_positions_response(BOOK_ID, _price_map())


@router.get("/trades", response_model=TrendHoldBookTradesResponse)
async def get_trend_hold_book_trades(
    limit: int = Query(default=300, ge=1, le=1000),
    _user: TokenData = Depends(get_current_user),
):
    """Return the Trend-Hold Book's trade ledger, newest first."""
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()
    return build_trades_response(BOOK_ID, limit)


@router.get("/nav-history", response_model=TrendHoldBookNavHistoryResponse)
async def get_trend_hold_book_nav_history(
    days: int = Query(default=180, ge=1, le=1000),
    _user: TokenData = Depends(get_current_user),
):
    """Return the Trend-Hold Book's daily equity history, oldest first -- powers the equity curve chart."""
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()
    return build_nav_history_response(BOOK_ID, days)


@router.get("/decision-log", response_model=TrendHoldDecisionLogResponse)
async def get_trend_hold_decision_log(
    limit: int = Query(default=200, ge=1, le=1000),
    include_wait: bool = Query(default=False),
    ticker: Optional[str] = Query(default=None),
    _user: TokenData = Depends(get_current_user),
):
    """
    Return the trend-hold engine's decision history log -- what it decided
    and why, for every scanned ticker across every session. This is
    independent of the Trend-Hold Book's trade ledger: it includes every
    decision the engine made (BUY/HOLD/SCALE_OUT/SELL_SIGNAL, and WAIT when
    include_wait=true), not just the ones the book actually acted on. No V1
    equivalent exists yet -- this endpoint is trend-hold-specific.
    """
    from app.services.eagle_eye.store import (
        load_trend_hold_decision_log,
        load_trend_hold_decision_log_for_ticker,
    )

    if ticker:
        rows = load_trend_hold_decision_log_for_ticker(ticker.upper(), limit=limit)
    else:
        rows = load_trend_hold_decision_log(limit=limit, include_wait=include_wait)

    entries = [
        TrendHoldDecisionLogEntry(
            ticker=r["ticker"],
            trade_date=r["trade_date"],
            decision=r.get("decision") or "WAIT",
            reason=r.get("reason"),
            position_state=r.get("position_state"),
            close=safe_float(r.get("close")),
            structural_stop=safe_float(r.get("structural_stop")),
            confidence=safe_float(r.get("confidence")),
        )
        for r in rows
    ]
    return TrendHoldDecisionLogResponse(entries=entries)


@router.get("/lessons", response_model=TrendHoldBookLessonsResponse)
async def get_trend_hold_book_lessons(
    limit: int = Query(default=200, ge=1, le=1000),
    _user: TokenData = Depends(get_current_user),
):
    """
    Return the Trend-Hold Book's post-trade "autopsy" log -- one entry per
    closed leg (SCALE_OUT/EXIT), explaining what happened and why using the
    realized price path (MAE/MFE, holding period, single-session vs
    grinding decline), plus an enhancement suggestion. See
    trend_hold_lessons.py for the (fully rule-based, auditable) classifier.
    """
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()
    return build_lessons_response(BOOK_ID, limit)


@router.get("/lessons/summary", response_model=TrendHoldBookLessonsSummary)
async def get_trend_hold_book_lessons_summary(_user: TokenData = Depends(get_current_user)):
    """
    Aggregate rollup of the lessons log -- counts per classification/
    outcome and average excursion metrics. This is the evidence a human
    would want before deciding whether a trend_hold_engine.py parameter
    (CHANDELIER_ATR_MULT, SCALE_OUT_GAIN_PCT, ...) actually needs to change.
    """
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()
    return build_lessons_summary_response(BOOK_ID)


@router.get("/performance", response_model=TrendHoldBookPerformance)
async def get_trend_hold_book_performance(_user: TokenData = Depends(get_current_user)):
    """
    Standard trading performance scorecard: win/loss counts and rate,
    total realized P&L, best/worst single trade, average win/loss,
    profit factor, and per-trade expectancy -- computed directly from
    the trade ledger's realized_pnl_kwd, independent of the lessons
    classifier (populated as soon as any trade closes, lessons or not).
    """
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()
    return build_performance_response(BOOK_ID)

"""
V1 Rating Book API.

Read-only endpoints over the virtual-money paper-trading ledger that
mechanically fills the V1 rating engine's decisions (see
app/services/eagle_eye_v2/v1_rating_book.py). Fully independent of the
real portfolio, the unrelated eagle_eye_v2/simulator/ system, and of the
Trend-Hold Book (app/api/v1/trend_hold_book.py) -- run side by side so the
two strategies' performance can be compared directly. Response-building
logic is shared with that router via _paper_book_common.py; only book_id
and the price source differ.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

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
    BookComparisonResponse,
    TrendHoldBookLessonsResponse,
    TrendHoldBookLessonsSummary,
    TrendHoldBookNavHistoryResponse,
    TrendHoldBookPerformance,
    TrendHoldBookPortfolio,
    TrendHoldBookPositionsResponse,
    TrendHoldBookTradesResponse,
)
from app.services.eagle_eye_v2.v1_rating_book import BOOK_ID

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1-rating-book", tags=["V1 Rating Book"])


def _price_map() -> dict:
    from app.services.eagle_eye.store import load_all_ratings

    return {
        str(r["ticker"]).upper(): safe_float(r.get("last_price"))
        for r in load_all_ratings(limit=1000)
    }


@router.get("/portfolio", response_model=TrendHoldBookPortfolio)
async def get_v1_rating_book_portfolio(_user: TokenData = Depends(get_current_user)):
    """Return the V1 Rating Book's current cash, mark-to-market equity, and total return."""
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()
    return build_portfolio_response(BOOK_ID, _price_map())


@router.get("/positions", response_model=TrendHoldBookPositionsResponse)
async def get_v1_rating_book_positions(_user: TokenData = Depends(get_current_user)):
    """Return every currently open V1 Rating Book paper position, marked to market."""
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()
    return build_positions_response(BOOK_ID, _price_map())


@router.get("/trades", response_model=TrendHoldBookTradesResponse)
async def get_v1_rating_book_trades(
    limit: int = Query(default=300, ge=1, le=1000),
    _user: TokenData = Depends(get_current_user),
):
    """Return the V1 Rating Book's trade ledger, newest first."""
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()
    return build_trades_response(BOOK_ID, limit)


@router.get("/nav-history", response_model=TrendHoldBookNavHistoryResponse)
async def get_v1_rating_book_nav_history(
    days: int = Query(default=180, ge=1, le=1000),
    _user: TokenData = Depends(get_current_user),
):
    """Return the V1 Rating Book's daily equity history, oldest first -- powers the equity curve chart."""
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()
    return build_nav_history_response(BOOK_ID, days)


@router.get("/lessons", response_model=TrendHoldBookLessonsResponse)
async def get_v1_rating_book_lessons(
    limit: int = Query(default=200, ge=1, le=1000),
    _user: TokenData = Depends(get_current_user),
):
    """
    Return the V1 Rating Book's post-trade "autopsy" log -- one entry per
    closed leg, using the same rule-based classifier as the Trend-Hold
    Book (trend_hold_lessons.py -- fully signal-source-agnostic).
    """
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()
    return build_lessons_response(BOOK_ID, limit)


@router.get("/lessons/summary", response_model=TrendHoldBookLessonsSummary)
async def get_v1_rating_book_lessons_summary(_user: TokenData = Depends(get_current_user)):
    """Aggregate rollup of the V1 Rating Book's lessons log."""
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()
    return build_lessons_summary_response(BOOK_ID)


@router.get("/performance", response_model=TrendHoldBookPerformance)
async def get_v1_rating_book_performance(_user: TokenData = Depends(get_current_user)):
    """
    Standard trading performance scorecard for the V1 Rating Book -- see
    /trend-hold-book/performance for field definitions. Compare the two
    directly via GET /v1-rating-book/compare.
    """
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()
    return build_performance_response(BOOK_ID)


@router.get("/compare", response_model=BookComparisonResponse)
async def compare_paper_books(_user: TokenData = Depends(get_current_user)):
    """
    Both paper books' performance scorecards side by side -- the direct
    "which one is best" answer, without two separate round trips. Includes
    each book's portfolio view (realized/unrealized/net P&L) alongside the
    realized-only performance scorecard, so the comparison strip isn't
    limited to closed-trade numbers.
    """
    from app.api.v1.trend_hold_book import _price_map as _trend_hold_price_map
    from app.services.eagle_eye_v2 import paper_book_store as book
    from app.services.eagle_eye_v2.trend_hold_book import BOOK_ID as TREND_HOLD_BOOK_ID

    book.ensure_paper_book_tables()
    return BookComparisonResponse(
        trend_hold=build_performance_response(TREND_HOLD_BOOK_ID),
        v1_rating=build_performance_response(BOOK_ID),
        trend_hold_portfolio=build_portfolio_response(TREND_HOLD_BOOK_ID, _trend_hold_price_map()),
        v1_rating_portfolio=build_portfolio_response(BOOK_ID, _price_map()),
    )

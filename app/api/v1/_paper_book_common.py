"""
Paper Book — shared response-building logic for both books' API routers.

Both books (Trend-Hold Book at /trend-hold-book, V1 Rating Book at
/v1-rating-book) expose the same 7-endpoint shape over the same
paper_book_store.py tables, differing only in book_id and in where each
sources "current price per ticker" from (trend_hold_engine's `close` vs
V1's `last_price`). Every builder here is book-agnostic: pass a book_id and
(where needed) a plain {ticker: price} map, get back the same response
models both routers already return.

Not a router itself -- no routes are registered here, only helpers imported
by app/api/v1/trend_hold_book.py and app/api/v1/v1_rating_book.py.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, Optional

from app.schemas.trend_hold_book import (
    TrendHoldBookLesson,
    TrendHoldBookLessonsResponse,
    TrendHoldBookLessonsSummary,
    TrendHoldBookNavHistoryResponse,
    TrendHoldBookNavPoint,
    TrendHoldBookPerformance,
    TrendHoldBookPortfolio,
    TrendHoldBookPosition,
    TrendHoldBookPositionsResponse,
    TrendHoldBookTrade,
    TrendHoldBookTradesResponse,
)


def safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _mark_positions(positions: dict, price_map: Dict[str, float]) -> list[dict]:
    """Attach latest price + unrealized P&L to each open position."""
    marked = []
    for ticker, pos in positions.items():
        latest_close = price_map.get(ticker)
        quantity = safe_float(pos.get("quantity")) or 0.0
        avg_cost = safe_float(pos.get("avg_cost")) or 0.0
        market_value = quantity * latest_close if latest_close is not None else None
        unrealized_pnl = quantity * (latest_close - avg_cost) if latest_close is not None else None
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


def build_portfolio_response(book_id: str, price_map: Dict[str, float]) -> TrendHoldBookPortfolio:
    from app.services.eagle_eye_v2 import paper_book_store as book

    state = book.load_book_state(book_id)
    positions = book.load_all_positions(book_id)

    cash = safe_float(state.get("cash_kwd")) or 0.0
    starting_capital = safe_float(state.get("starting_capital_kwd")) or 0.0
    marked = _mark_positions(positions, price_map)
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


def build_positions_response(book_id: str, price_map: Dict[str, float]) -> TrendHoldBookPositionsResponse:
    from app.services.eagle_eye_v2 import paper_book_store as book

    positions = book.load_all_positions(book_id)
    marked = _mark_positions(positions, price_map)
    return TrendHoldBookPositionsResponse(positions=[TrendHoldBookPosition(**p) for p in marked])


def build_trades_response(book_id: str, limit: int) -> TrendHoldBookTradesResponse:
    from app.services.eagle_eye_v2 import paper_book_store as book

    rows = book.load_recent_trades(book_id, limit=limit)
    trades = [
        TrendHoldBookTrade(
            id=int(r["id"]),
            ticker=r["ticker"],
            side=r["side"],
            trade_date=r["trade_date"],
            quantity=safe_float(r.get("quantity")) or 0.0,
            price=safe_float(r.get("price")) or 0.0,
            gross_kwd=safe_float(r.get("gross_kwd")) or 0.0,
            commission_kwd=safe_float(r.get("commission_kwd")) or 0.0,
            realized_pnl_kwd=safe_float(r.get("realized_pnl_kwd")),
            reason=r.get("reason"),
            confidence=safe_float(r.get("confidence")),
        )
        for r in rows
    ]
    return TrendHoldBookTradesResponse(trades=trades)


def build_nav_history_response(book_id: str, days: int) -> TrendHoldBookNavHistoryResponse:
    from app.services.eagle_eye_v2 import paper_book_store as book

    rows = book.load_nav_history(book_id, days=days)
    points = [
        TrendHoldBookNavPoint(
            nav_date=r["nav_date"],
            cash_kwd=safe_float(r.get("cash_kwd")) or 0.0,
            equity_kwd=safe_float(r.get("equity_kwd")) or 0.0,
            open_position_count=int(r.get("open_position_count") or 0),
        )
        for r in rows
    ]
    return TrendHoldBookNavHistoryResponse(points=points)


def build_lessons_response(book_id: str, limit: int) -> TrendHoldBookLessonsResponse:
    from app.services.eagle_eye_v2 import paper_book_store as book

    rows = book.load_lessons(book_id, limit=limit)
    lessons = [
        TrendHoldBookLesson(
            ticker=r["ticker"],
            trade_date=r["trade_date"],
            side=r["side"],
            classification=r["classification"],
            outcome=r["outcome"],
            mae_pct=safe_float(r.get("mae_pct")),
            mfe_pct=safe_float(r.get("mfe_pct")),
            giveback_pct=safe_float(r.get("giveback_pct")),
            holding_days=r.get("holding_days"),
            reason=r.get("reason") or "",
            enhancement=r.get("enhancement") or "",
        )
        for r in rows
    ]
    return TrendHoldBookLessonsResponse(lessons=lessons)


def build_lessons_summary_response(book_id: str) -> TrendHoldBookLessonsSummary:
    from app.services.eagle_eye_v2 import paper_book_store as book

    summary = book.load_lessons_summary(book_id)
    return TrendHoldBookLessonsSummary(**summary)


def build_performance_response(book_id: str) -> TrendHoldBookPerformance:
    from app.services.eagle_eye_v2 import paper_book_store as book

    stats = book.load_performance_stats(book_id)
    return TrendHoldBookPerformance(**stats)

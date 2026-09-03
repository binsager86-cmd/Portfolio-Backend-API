"""
Trend-Hold Book — execution engine.

Mechanically turns each ticker's latest ee_trend_hold_state decision into a
virtual fill against the "trend_hold" paper book (paper_book_store.py, a
multi-book ledger shared with v1_rating_book.py -- see that module's own
docstring for the V1-driven comparison book). Adds no new signal logic of
its own -- it only reacts to BUY / SCALE_OUT / SELL_SIGNAL decisions the
existing trend_hold_batch.py job already wrote.

Position sizing / commission constants live in paper_book_store.py and are
shared with the V1 rating book, so the two books' performance scorecards
are directly, fairly comparable.
"""
from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any, Dict, Optional

from app.services.eagle_eye_v2.trend_hold_engine import SCALE_OUT_FRACTION

logger = logging.getLogger(__name__)

BOOK_ID = "trend_hold"

_ACTIONABLE_DECISIONS = {"BUY", "SCALE_OUT", "SELL_SIGNAL"}


def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _build_lesson(
    ticker: str, side: str, entry_date: Optional[str], entry_price: Optional[float],
    exit_date: str, exit_price: float,
) -> Optional[dict]:
    """
    Best-effort post-trade autopsy for one closing leg. Never allowed to
    block or fail the actual trade fill -- a lesson-analysis error just
    means that trade's row has no lesson yet, not a failed trade.
    """
    try:
        from app.services.eagle_eye.store import load_ohlcv
        from app.services.eagle_eye_v2.trend_hold_lessons import analyze_trade

        ohlcv = load_ohlcv(ticker)
        lesson = analyze_trade(
            side=side,
            entry_date=entry_date,
            entry_price=entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            ohlcv=ohlcv,
        )
        return {
            "classification": lesson.classification,
            "outcome": lesson.outcome,
            "mae_pct": lesson.mae_pct,
            "mfe_pct": lesson.mfe_pct,
            "giveback_pct": lesson.giveback_pct,
            "holding_days": lesson.holding_days,
            "reason": lesson.reason,
            "enhancement": lesson.enhancement,
        }
    except Exception as exc:
        logger.warning("trend_hold_book: lesson analysis failed for %s: %s", ticker, exc)
        return None


def run_trend_hold_book_step() -> Dict[str, Any]:
    from app.services.eagle_eye.store import load_all_trend_hold_state
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()

    trend_hold_map = load_all_trend_hold_state()
    book_state = book.load_book_state(BOOK_ID)
    positions = book.load_all_positions(BOOK_ID)
    cash = float(book_state["cash_kwd"])

    stats = {"bought": 0, "scaled_out": 0, "exited": 0, "skipped": 0, "errors": 0}

    def equity() -> float:
        total = cash
        for ticker, pos in positions.items():
            row = trend_hold_map.get(ticker)
            price = _f(row.get("close")) if row else None
            if price is None:
                price = pos.get("avg_cost") or 0.0
            total += (pos.get("quantity") or 0.0) * price
        return total

    for ticker in sorted(trend_hold_map.keys()):
        row = trend_hold_map[ticker]
        decision = row.get("decision")
        trade_date = row.get("trade_date")
        price = _f(row.get("close"))
        reason = row.get("reason")

        if decision not in _ACTIONABLE_DECISIONS:
            continue
        if not trade_date or price is None or price <= 0:
            stats["skipped"] += 1
            continue
        if book.trade_exists(BOOK_ID, ticker, trade_date):
            # Already actioned this ticker's signal for this session -- the
            # idempotency guard that makes a scheduler re-run safe.
            stats["skipped"] += 1
            continue

        try:
            if decision == "BUY":
                if ticker in positions:
                    logger.warning(
                        "trend_hold_book: BUY for %s but a paper position is already open -- skipping (state drift)",
                        ticker,
                    )
                    stats["skipped"] += 1
                    continue
                if len(positions) >= book.MAX_CONCURRENT_POSITIONS:
                    stats["skipped"] += 1
                    continue

                spend = min(book.POSITION_SIZE_FRACTION * equity(), cash)
                if spend <= 0:
                    stats["skipped"] += 1
                    continue

                quantity = spend / (price * (1 + book.COMMISSION_RATE))
                gross = quantity * price
                commission = gross * book.COMMISSION_RATE
                cash -= (gross + commission)

                book.record_buy_fill(
                    book_id=BOOK_ID,
                    ticker=ticker,
                    trade_date=trade_date,
                    quantity=quantity,
                    price=price,
                    gross_kwd=gross,
                    commission_kwd=commission,
                    reason=reason,
                    cash_kwd=cash,
                    confidence=_f(row.get("confidence")),
                )
                positions[ticker] = {
                    "ticker": ticker,
                    "quantity": quantity,
                    "avg_cost": price,
                    "entry_commission_kwd": commission,
                    "opened_date": trade_date,
                }
                stats["bought"] += 1

            elif decision == "SCALE_OUT":
                pos = positions.get(ticker)
                qty_before = float(pos["quantity"]) if pos else 0.0
                if pos is None or qty_before <= 0:
                    logger.warning(
                        "trend_hold_book: SCALE_OUT for %s but no open paper position -- skipping", ticker
                    )
                    stats["skipped"] += 1
                    continue

                sell_qty = qty_before * SCALE_OUT_FRACTION
                gross = sell_qty * price
                commission = gross * book.COMMISSION_RATE
                entry_commission_share = float(pos["entry_commission_kwd"] or 0.0) * (sell_qty / qty_before)
                realized_pnl = sell_qty * (price - float(pos["avg_cost"])) - commission - entry_commission_share
                cash += (gross - commission)

                remaining_qty = qty_before - sell_qty
                remaining_entry_commission = float(pos["entry_commission_kwd"] or 0.0) - entry_commission_share

                lesson = _build_lesson(
                    ticker, "SCALE_OUT", pos.get("opened_date"), float(pos["avg_cost"]), trade_date, price,
                )
                book.record_scale_out_fill(
                    book_id=BOOK_ID,
                    ticker=ticker,
                    trade_date=trade_date,
                    sell_quantity=sell_qty,
                    price=price,
                    gross_kwd=gross,
                    commission_kwd=commission,
                    realized_pnl_kwd=realized_pnl,
                    reason=reason,
                    cash_kwd=cash,
                    remaining_quantity=remaining_qty,
                    remaining_entry_commission_kwd=remaining_entry_commission,
                    avg_cost=float(pos["avg_cost"]),
                    opened_date=pos["opened_date"],
                    lesson=lesson,
                )
                positions[ticker]["quantity"] = remaining_qty
                positions[ticker]["entry_commission_kwd"] = remaining_entry_commission
                stats["scaled_out"] += 1

            elif decision == "SELL_SIGNAL":
                pos = positions.get(ticker)
                qty_before = float(pos["quantity"]) if pos else 0.0
                if pos is None or qty_before <= 0:
                    logger.warning(
                        "trend_hold_book: SELL_SIGNAL for %s but no open paper position -- skipping", ticker
                    )
                    stats["skipped"] += 1
                    continue

                sell_qty = qty_before
                gross = sell_qty * price
                commission = gross * book.COMMISSION_RATE
                realized_pnl = (
                    sell_qty * (price - float(pos["avg_cost"]))
                    - commission
                    - float(pos["entry_commission_kwd"] or 0.0)
                )
                cash += (gross - commission)

                lesson = _build_lesson(
                    ticker, "EXIT", pos.get("opened_date"), float(pos["avg_cost"]), trade_date, price,
                )
                book.record_exit_fill(
                    book_id=BOOK_ID,
                    ticker=ticker,
                    trade_date=trade_date,
                    sell_quantity=sell_qty,
                    price=price,
                    gross_kwd=gross,
                    commission_kwd=commission,
                    realized_pnl_kwd=realized_pnl,
                    reason=reason,
                    cash_kwd=cash,
                    lesson=lesson,
                    confidence=_f(row.get("confidence")),
                )
                del positions[ticker]
                stats["exited"] += 1

        except Exception as exc:
            logger.warning("trend_hold_book: error processing %s (%s): %s", ticker, decision, exc)
            stats["errors"] += 1

    final_equity = equity()
    session_date = max((row.get("trade_date") for row in trend_hold_map.values() if row.get("trade_date")), default=None)
    book.save_nav_snapshot(
        book_id=BOOK_ID,
        nav_date=session_date or date.today().isoformat(),
        cash_kwd=cash,
        equity_kwd=final_equity,
        open_position_count=len(positions),
    )

    stats["cash_kwd"] = round(cash, 3)
    stats["equity_kwd"] = round(final_equity, 3)
    stats["open_positions"] = len(positions)
    return stats


def run_open_position_price_refresh() -> Dict[str, Any]:
    """
    Intraday mark-to-market refresh for the Trend-Hold Book's currently open
    positions ONLY -- not a universe scan and not a decision-engine run.

    Fetches each open position's latest TickerChart close via fetch_ohlcv
    and caches it in ee_trend_hold_book_live_prices (paper_book_store.py),
    which the API layer (_price_map() in app/api/v1/trend_hold_book.py)
    overlays on top of the once-daily trend_hold_engine close so the book's
    displayed equity/positions can move during the session. Never writes to
    ee_trend_hold_state and never influences a BUY/SCALE_OUT/SELL_SIGNAL
    decision -- those stay exclusively the once-daily scan+book jobs'.
    """
    import asyncio
    import threading
    from datetime import date, timedelta

    from app.services import tickerchart_service as tc
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()
    positions = book.load_all_positions(BOOK_ID)
    if not positions:
        book.save_live_prices(BOOK_ID, {})
        return {"positions": 0, "updated": 0, "errors": 0}

    tickers = sorted(positions.keys())
    today = date.today()
    from_d = today - timedelta(days=7)

    async def _fetch_all() -> Dict[str, float]:
        async def _one(ticker: str) -> tuple[str, Optional[float]]:
            try:
                rows = await tc.fetch_ohlcv(ticker, "KSE", from_d=from_d, to_d=today, interval="day")
                if not rows:
                    return ticker, None
                latest = sorted(rows, key=lambda r: r["date"])[-1]
                return ticker, _f(latest.get("close"))
            except Exception as exc:
                logger.warning("trend_hold_book: live price fetch failed for %s: %s", ticker, exc)
                return ticker, None

        results = await asyncio.gather(*[_one(t) for t in tickers])
        return {t: p for t, p in results if p is not None and p > 0}

    result_box: list = []
    exc_box: list = []

    def _target() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box.append(loop.run_until_complete(_fetch_all()))
        except Exception as exc:  # noqa: BLE001
            exc_box.append(exc)
        finally:
            loop.close()

    # Run in a fresh event loop on a background thread -- safe whether this
    # is called from plain sync code (APScheduler) or from within a running
    # event loop (a FastAPI request handler), where asyncio.run() would raise.
    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()

    if exc_box:
        logger.warning("trend_hold_book: intraday price refresh failed: %s", exc_box[0])
        return {"positions": len(tickers), "updated": 0, "errors": len(tickers)}

    prices = result_box[0] if result_box else {}
    book.save_live_prices(BOOK_ID, prices)
    return {"positions": len(tickers), "updated": len(prices), "errors": len(tickers) - len(prices)}

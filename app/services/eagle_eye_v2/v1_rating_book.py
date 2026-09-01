"""
V1 Rating Book — execution engine.

Mechanically turns each ticker's latest V1 rating engine output
(ee_ratings_cache -- rating_engine.py, the original BUY/WATCHLIST/HOLD/
NEUTRAL/REDUCE/SELL/AVOID classification) into a virtual fill against the
"v1_rating" paper book (paper_book_store.py, the same multi-book ledger
trend_hold_book.py's "trend_hold" book already uses). The two books share
identical starting capital, position sizing, and commission (see
paper_book_store.py) so their performance scorecards are directly, fairly
comparable -- the whole point of running them side by side.

Unlike trend_hold_engine.py, V1's rating engine has no entry/exit state
machine of its own -- it just reclassifies every ticker fresh every night,
with no memory of whether a paper position is currently open. This module
supplies that missing trade-lifecycle layer, using only V1's own already-
computed numbers (no invented logic):

  BUY:  rating is BUY/STRONG_BUY and no open v1_rating position -> buy at
        last_price.
  SELL: an open position exists and EITHER rating has left BUY/STRONG_BUY,
        OR last_price has broken that day's freshly-computed stop_loss --
        whichever fires first.
  (no SCALE_OUT-equivalent -- V1 has no partial-profit milestone the way
  trend_hold_engine's +20%-gain rule does, so this book is a simpler
  straight buy/hold/sell system by design.)

V1's own confidence score (ee_ratings_cache.confidence, already 0-100) is
reused directly as this book's trade confidence -- no new scoring needed.
"""
from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

BOOK_ID = "v1_rating"

_BUY_RATINGS = {"BUY", "STRONG_BUY"}


def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _build_lesson(
    ticker: str, entry_date: Optional[str], entry_price: Optional[float],
    exit_date: str, exit_price: float,
) -> Optional[dict]:
    """Same rule-based trade autopsy the trend-hold book uses -- signal-source-agnostic."""
    try:
        from app.services.eagle_eye.store import load_ohlcv
        from app.services.eagle_eye_v2.trend_hold_lessons import analyze_trade

        ohlcv = load_ohlcv(ticker)
        lesson = analyze_trade(
            side="EXIT",
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
        logger.warning("v1_rating_book: lesson analysis failed for %s: %s", ticker, exc)
        return None


def run_v1_rating_book_step() -> Dict[str, Any]:
    from app.services.eagle_eye.store import load_all_ratings
    from app.services.eagle_eye_v2 import paper_book_store as book

    book.ensure_paper_book_tables()

    rating_rows = load_all_ratings(limit=1000)
    ratings_map = {str(r["ticker"]).upper(): dict(r) for r in rating_rows}

    book_state = book.load_book_state(BOOK_ID)
    positions = book.load_all_positions(BOOK_ID)
    cash = float(book_state["cash_kwd"])

    stats = {"bought": 0, "exited": 0, "skipped": 0, "errors": 0}

    def equity() -> float:
        total = cash
        for ticker, pos in positions.items():
            row = ratings_map.get(ticker)
            price = _f(row.get("last_price")) if row else None
            if price is None:
                price = pos.get("avg_cost") or 0.0
            total += (pos.get("quantity") or 0.0) * price
        return total

    for ticker in sorted(ratings_map.keys()):
        row = ratings_map[ticker]
        rating = str(row.get("rating") or "").upper()
        trade_date = row.get("computed_date") or (row.get("computed_at") or "")[:10]
        price = _f(row.get("last_price"))
        stop_loss = _f(row.get("stop_loss"))

        if not trade_date or price is None or price <= 0:
            stats["skipped"] += 1
            continue
        if book.trade_exists(BOOK_ID, ticker, trade_date):
            # Already actioned this ticker for this session -- idempotency
            # guard that makes a scheduler re-run safe.
            stats["skipped"] += 1
            continue

        pos = positions.get(ticker)

        try:
            if pos is None:
                if rating not in _BUY_RATINGS:
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
                    reason=f"V1 rating is {rating}",
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

            else:
                downgraded = rating not in _BUY_RATINGS
                stop_broken = stop_loss is not None and stop_loss > 0 and price < stop_loss
                if not (downgraded or stop_broken):
                    continue

                qty_before = float(pos["quantity"])
                if qty_before <= 0:
                    stats["skipped"] += 1
                    continue

                reason = (
                    f"V1 rating downgraded to {rating}" if downgraded
                    else f"V1 stop-loss breached at {stop_loss:.3f}"
                )
                gross = qty_before * price
                commission = gross * book.COMMISSION_RATE
                realized_pnl = (
                    qty_before * (price - float(pos["avg_cost"]))
                    - commission
                    - float(pos["entry_commission_kwd"] or 0.0)
                )
                cash += (gross - commission)

                lesson = _build_lesson(ticker, pos.get("opened_date"), float(pos["avg_cost"]), trade_date, price)
                book.record_exit_fill(
                    book_id=BOOK_ID,
                    ticker=ticker,
                    trade_date=trade_date,
                    sell_quantity=qty_before,
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
            logger.warning("v1_rating_book: error processing %s: %s", ticker, exc)
            stats["errors"] += 1

    final_equity = equity()
    session_date = max((r.get("computed_date") for r in ratings_map.values() if r.get("computed_date")), default=None)
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

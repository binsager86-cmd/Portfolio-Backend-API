"""
Trend-Hold Book — execution engine.

Mechanically turns each ticker's latest ee_trend_hold_state decision into a
virtual fill against the Trend-Hold Book ledger (trend_hold_book_store.py).
Adds no new signal logic of its own -- it only reacts to BUY / SCALE_OUT /
SELL_SIGNAL decisions the existing trend_hold_batch.py job already wrote.

Position sizing / commission constants are deliberately the same ones
already established in this app for this exact market by the (unrelated)
eagle_eye_v2/simulator/ backtest engine
(app/services/eagle_eye_v2/simulator/constants.py).
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

from app.services.eagle_eye_v2.trend_hold_engine import SCALE_OUT_FRACTION

logger = logging.getLogger(__name__)

POSITION_SIZE_FRACTION = 0.10
MAX_CONCURRENT_POSITIONS = 10
COMMISSION_RATE = 0.00325

_ACTIONABLE_DECISIONS = {"BUY", "SCALE_OUT", "SELL_SIGNAL"}


def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def run_trend_hold_book_step() -> Dict[str, Any]:
    from app.services.eagle_eye.store import load_all_trend_hold_state
    from app.services.eagle_eye_v2 import trend_hold_book_store as book

    book.ensure_trend_hold_book_tables()

    trend_hold_map = load_all_trend_hold_state()
    book_state = book.load_book_state()
    positions = book.load_all_positions()
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
        if book.trade_exists(ticker, trade_date):
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
                if len(positions) >= MAX_CONCURRENT_POSITIONS:
                    stats["skipped"] += 1
                    continue

                spend = min(POSITION_SIZE_FRACTION * equity(), cash)
                if spend <= 0:
                    stats["skipped"] += 1
                    continue

                quantity = spend / (price * (1 + COMMISSION_RATE))
                gross = quantity * price
                commission = gross * COMMISSION_RATE
                cash -= (gross + commission)

                book.record_buy_fill(
                    ticker=ticker,
                    trade_date=trade_date,
                    quantity=quantity,
                    price=price,
                    gross_kwd=gross,
                    commission_kwd=commission,
                    reason=reason,
                    cash_kwd=cash,
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
                commission = gross * COMMISSION_RATE
                entry_commission_share = float(pos["entry_commission_kwd"] or 0.0) * (sell_qty / qty_before)
                realized_pnl = sell_qty * (price - float(pos["avg_cost"])) - commission - entry_commission_share
                cash += (gross - commission)

                remaining_qty = qty_before - sell_qty
                remaining_entry_commission = float(pos["entry_commission_kwd"] or 0.0) - entry_commission_share

                book.record_scale_out_fill(
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
                commission = gross * COMMISSION_RATE
                realized_pnl = (
                    sell_qty * (price - float(pos["avg_cost"]))
                    - commission
                    - float(pos["entry_commission_kwd"] or 0.0)
                )
                cash += (gross - commission)

                book.record_exit_fill(
                    ticker=ticker,
                    trade_date=trade_date,
                    sell_quantity=sell_qty,
                    price=price,
                    gross_kwd=gross,
                    commission_kwd=commission,
                    realized_pnl_kwd=realized_pnl,
                    reason=reason,
                    cash_kwd=cash,
                )
                del positions[ticker]
                stats["exited"] += 1

        except Exception as exc:
            logger.warning("trend_hold_book: error processing %s (%s): %s", ticker, decision, exc)
            stats["errors"] += 1

    stats["cash_kwd"] = round(cash, 3)
    stats["open_positions"] = len(positions)
    return stats

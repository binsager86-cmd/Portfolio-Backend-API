"""Corporate actions fetcher for the Kuwait Signal Engine.

Handles price adjustments for dividends, stock splits, and rights issues
so that historical indicator calculations are not distorted by non-price
events.  Also provides signal suppression logic for the ex-dividend window.

Kuwait-specific notes
---------------------
* Boursa Kuwait dividends are typically declared in fils per share (cash
  dividends) or as stock bonus percentages (bonus shares / splits).
* The ex-dividend date is the first trading day on which the buyer is NOT
  entitled to the upcoming dividend.
* Signals should be suppressed ±3 trading days around the ex-dividend date
  because price action is dominated by the mechanics of dividend pricing,
  not by technical supply/demand.

Corporate Action Event Schema
-----------------------------
Each event is a plain dict with the following fields:

    {
        "date":        str          # YYYY-MM-DD — ex-date for dividends, effective date for splits
        "type":        str          # "DIVIDEND" | "SPLIT" | "RIGHTS" | "BONUS"
        "value":       float        # fils per share (dividend) OR split ratio (split/bonus)
        "stock_code":  str          # KSE stock code
    }
"""
from __future__ import annotations

from datetime import date, timedelta
from enum import Enum
from typing import Any


class CorporateActionFlag(str, Enum):
    NONE = "NONE"
    DIVIDEND = "DIVIDEND"
    SPLIT = "SPLIT"
    RIGHTS = "RIGHTS"
    BONUS = "BONUS"


# Days to suppress signals before and after the ex-dividend date
EX_DIV_BUFFER_DAYS: int = 3


# ── Price adjustment: dividends ───────────────────────────────────────────────

def adjust_for_dividends(
    rows: list[dict[str, Any]],
    dividend_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Back-adjust historical OHLCV prices for cash dividends.

    For each dividend event, subtract the dividend amount (in fils) from all
    prices *before* the ex-dividend date.  This ensures that indicators
    computed on the adjusted series reflect only genuine price trends, not
    artificial drops caused by dividend payments.

    The adjustment is applied to: open, high, low, close.
    Volume is not adjusted.

    Args:
        rows:             OHLCV rows sorted ascending by date.
        dividend_events:  List of dividend event dicts (see module docstring).

    Returns:
        New list of adjusted rows (original rows are NOT mutated).
    """
    if not rows or not dividend_events:
        return rows

    adjusted = [dict(r) for r in rows]   # shallow copy each row

    # Sort events ascending by date — apply oldest adjustment first
    sorted_events = sorted(
        (e for e in dividend_events if e.get("type") in {"DIVIDEND"}),
        key=lambda e: e["date"],
    )

    for event in sorted_events:
        ex_date = event["date"]
        div_fils = float(event.get("value") or 0.0)
        if div_fils <= 0:
            continue
        # Adjust all bars BEFORE the ex-date
        for row in adjusted:
            if str(row.get("date", "")) < ex_date:
                for price_key in ("open", "high", "low", "close"):
                    v = row.get(price_key)
                    if v is not None:
                        row[price_key] = round(max(0.0, float(v) - div_fils), 2)

    return adjusted


def adjust_for_splits(
    rows: list[dict[str, Any]],
    split_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Back-adjust historical OHLCV prices and volumes for stock splits / bonus shares.

    For a split ratio R (new shares per old share, e.g. 2.0 for 2-for-1),
    prices *before* the effective date are divided by R and volumes are
    multiplied by R to keep total market value constant.

    Args:
        rows:          OHLCV rows sorted ascending by date.
        split_events:  List of event dicts with type="SPLIT" or "BONUS".
                       "value" field is the split ratio (new/old).

    Returns:
        New list of adjusted rows.
    """
    if not rows or not split_events:
        return rows

    adjusted = [dict(r) for r in rows]

    sorted_events = sorted(
        (e for e in split_events if e.get("type") in {"SPLIT", "BONUS"}),
        key=lambda e: e["date"],
    )

    for event in sorted_events:
        eff_date = event["date"]
        ratio = float(event.get("value") or 1.0)
        if ratio <= 0 or abs(ratio - 1.0) < 1e-9:
            continue
        for row in adjusted:
            if str(row.get("date", "")) < eff_date:
                for price_key in ("open", "high", "low", "close"):
                    v = row.get(price_key)
                    if v is not None:
                        row[price_key] = round(float(v) / ratio, 2)
                vol = row.get("volume")
                if vol is not None:
                    row["volume"] = int(float(vol) * ratio)

    return adjusted


# ── Signal suppression: ex-dividend window ────────────────────────────────────

def is_near_ex_dividend(
    signal_date: str,
    ex_div_dates: list[str],
    buffer_days: int = EX_DIV_BUFFER_DAYS,
) -> bool:
    """Return True if signal_date is within ±buffer_days of any ex-dividend date.

    Signals should be suppressed during this window because the price action
    is distorted by dividend mechanics, not by genuine supply/demand.

    Args:
        signal_date:  Date of the signal in YYYY-MM-DD format.
        ex_div_dates: List of ex-dividend dates in YYYY-MM-DD format.
        buffer_days:  Days before AND after the ex-date to suppress (default 3).

    Returns:
        True if the signal should be suppressed.
    """
    if not ex_div_dates:
        return False

    try:
        sig = date.fromisoformat(signal_date)
    except ValueError:
        return False

    delta = timedelta(days=buffer_days)
    for ex_str in ex_div_dates:
        try:
            ex = date.fromisoformat(ex_str)
        except ValueError:
            continue
        if abs((sig - ex).days) <= buffer_days:
            return True
    return False


# ── Corporate action flag ─────────────────────────────────────────────────────

def get_corporate_action_flag(
    signal_date: str,
    events: list[dict[str, Any]],
    buffer_days: int = EX_DIV_BUFFER_DAYS,
) -> CorporateActionFlag:
    """Return the most relevant corporate action flag for a given date.

    If multiple events fall within the buffer window the priority order is:
        SPLIT > RIGHTS > BONUS > DIVIDEND > NONE

    Args:
        signal_date: Date to check in YYYY-MM-DD format.
        events:      List of corporate action event dicts.
        buffer_days: Days before/after the event date to flag.

    Returns:
        CorporateActionFlag enum value.
    """
    if not events:
        return CorporateActionFlag.NONE

    try:
        sig = date.fromisoformat(signal_date)
    except ValueError:
        return CorporateActionFlag.NONE

    priority_order: list[str] = ["SPLIT", "RIGHTS", "BONUS", "DIVIDEND"]
    matched_types: set[str] = set()

    for event in events:
        ev_type = str(event.get("type", "")).upper()
        if ev_type not in priority_order:
            continue
        try:
            ev_date = date.fromisoformat(str(event.get("date", "")))
        except ValueError:
            continue
        if abs((sig - ev_date).days) <= buffer_days:
            matched_types.add(ev_type)

    for action in priority_order:
        if action in matched_types:
            return CorporateActionFlag(action)
    return CorporateActionFlag.NONE


# ── Combined adjustment pipeline ──────────────────────────────────────────────

def apply_all_adjustments(
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply all corporate action adjustments in the correct order.

    Order: splits first (ratio changes), then dividends (additive).

    Args:
        rows:   Raw OHLCV rows.
        events: All corporate action events for this stock.

    Returns:
        Fully adjusted rows.
    """
    rows = adjust_for_splits(rows, events)
    rows = adjust_for_dividends(rows, events)
    return rows

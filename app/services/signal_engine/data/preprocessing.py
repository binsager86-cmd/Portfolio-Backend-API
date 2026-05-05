"""OHLCV pre-processing utilities for the Kuwait Signal Engine.

These functions operate on raw OHLCV rows BEFORE indicator computation
so that TA-Lib sees a gap-free series.
"""
from __future__ import annotations

from datetime import date as _date, timedelta
from typing import Any

from app.services.signal_engine.config.kuwait_constants import TRADING_WEEKDAYS
from app.services.signal_engine.config.model_params import MAX_FORWARD_FILL_DAYS


def forward_fill_gaps(
    rows: list[dict[str, Any]],
    max_days: int = MAX_FORWARD_FILL_DAYS,
) -> list[dict[str, Any]]:
    """Carry-forward the previous close for short gaps in the OHLCV series.

    If a stock has no trade data for 1 – max_days consecutive Kuwait trading
    sessions (Sun–Thu), synthetic rows are inserted with:
        open = high = low = close = previous close, volume = 0, value = 0.

    Gaps longer than max_days are left unfilled — the HMM and indicators will
    encounter the discontinuity, which is the correct behaviour for a stock
    that was halted or suspended for an extended period.

    The synthetic rows have a ``_forward_filled = True`` flag so downstream
    code can identify and exclude them from certain analytics (e.g. ADTV
    calculations should ignore zero-volume fill rows).

    Args:
        rows:     OHLCV rows sorted ascending by ISO date strings.
        max_days: Maximum consecutive missing sessions to fill (default 3).

    Returns:
        New list with synthetic rows inserted at their correct positions.
        Input list is not mutated.
    """
    if len(rows) < 2:
        return list(rows)

    filled: list[dict[str, Any]] = [rows[0]]

    for i in range(1, len(rows)):
        prev = filled[-1]
        curr = rows[i]

        try:
            prev_d = _date.fromisoformat(str(prev["date"]))
            curr_d = _date.fromisoformat(str(curr["date"]))
        except (ValueError, TypeError, KeyError):
            filled.append(curr)
            continue

        # Enumerate missing Kuwait trading sessions between the two dates
        missing: list[_date] = []
        d = prev_d + timedelta(days=1)
        while d < curr_d:
            if d.weekday() in TRADING_WEEKDAYS:
                missing.append(d)
            d += timedelta(days=1)

        if 0 < len(missing) <= max_days:
            prev_close = float(prev.get("close") or 0.0)
            for gap_date in missing:
                filled.append({
                    "date": gap_date.isoformat(),
                    "open": prev_close,
                    "high": prev_close,
                    "low": prev_close,
                    "close": prev_close,
                    "volume": 0.0,
                    "value": 0.0,
                    "_forward_filled": True,
                })
        # Gap > max_days: leave unfilled (skip insertion)

        filled.append(curr)

    return filled


def detect_corporate_action_gap(
    rows: list[dict[str, Any]],
    gap_threshold: float = 0.12,
) -> bool:
    """Return True if the most recent bar shows a suspicious overnight price gap.

    An overnight gap exceeding ``gap_threshold`` (default 12 %) is likely caused
    by an unadjusted corporate action (ex-dividend, bonus issue, stock split)
    rather than genuine intraday price movement.  Kuwait circuit breakers cap
    single-session moves at +10 % / −5 %, so a ≥ 12 % open-vs-prior-close gap
    cannot be explained by normal price action.

    Forward-filled synthetic bars (``_forward_filled = True``) are skipped
    because they carry the previous close as their open — no real gap exists.

    Args:
        rows:          OHLCV rows sorted ascending by date, ≥ 2 elements.
        gap_threshold: Fractional gap magnitude that triggers the flag (0.12 = 12 %).

    Returns:
        True when the current bar's open differs from the prior bar's close by
        more than ``gap_threshold``; False otherwise.
    """
    if len(rows) < 2:
        return False
    last = rows[-1]
    # Ignore synthetic forward-fill bars — they replicate the prior close as open
    if last.get("_forward_filled"):
        return False
    prev_close = float(rows[-2].get("close") or 0.0)
    curr_open  = float(last.get("open") or 0.0)
    if prev_close <= 0 or curr_open <= 0:
        return False
    gap = abs(curr_open - prev_close) / prev_close
    return gap > gap_threshold

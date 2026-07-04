from __future__ import annotations

from typing import Any

from app.core.database import query_all, query_val


def liquidity_filter(symbol: str, min_daily_value_kwd: float = 100000.0) -> tuple[bool, dict[str, Any]]:
    return liquidity_filter_at(symbol, None, min_daily_value_kwd)


def liquidity_filter_at(
    symbol: str,
    trade_date: int | None,
    min_daily_value_kwd: float = 100000.0,
) -> tuple[bool, dict[str, Any]]:
    if trade_date is None:
        rows = query_all(
            """
            SELECT value_kwd, close, volume
            FROM ee_ohlcv
            WHERE symbol = ?
            ORDER BY trade_date DESC
            LIMIT 60
            """,
            (symbol,),
        )
    else:
        rows = query_all(
            """
            SELECT value_kwd, close, volume
            FROM ee_ohlcv
            WHERE symbol = ? AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT 60
            """,
            (symbol, trade_date),
        )
    if not rows:
        return False, {"reason": "missing_ohlcv"}

    values = [float(r.get("value_kwd") or 0.0) for r in rows]
    closes = [float(r.get("close") or 0.0) for r in rows]
    vols = [float(r.get("volume") or 0.0) for r in rows]

    values_sorted = sorted(values)
    median_val = values_sorted[len(values_sorted) // 2] if values_sorted else 0.0
    min_price = min(closes) if closes else 0.0
    zero_vol = sum(1 for v in vols if v <= 0)

    ok = median_val >= float(min_daily_value_kwd) and min_price >= 50.0 and zero_vol <= 3
    return ok, {
        "median_daily_value_kwd_20": median_val,
        "min_price_fils_60": min_price,
        "zero_volume_sessions_60": zero_vol,
    }


def compute_position_size(
    equity: float,
    entry: float,
    stop: float,
    median_daily_value_kwd_20: float,
    risk_per_trade: float = 0.01,
) -> dict[str, Any]:
    risk_per_share = max(0.000001, entry - stop)
    risk_budget = max(0.0, equity * risk_per_trade)
    raw_units = risk_budget / risk_per_share
    notional = raw_units * entry
    liquidity_cap = median_daily_value_kwd_20 * 0.05
    capped_notional = min(notional, liquidity_cap)
    units = 0.0 if entry <= 0 else capped_notional / entry
    return {
        "units": round(units, 4),
        "notional_kwd": round(capped_notional, 2),
        "risk_budget_kwd": round(risk_budget, 2),
        "liquidity_cap_kwd": round(liquidity_cap, 2),
    }


def can_open_new_position(score: float, max_positions: int = 8) -> tuple[bool, str]:
    open_positions = int(query_val("SELECT COUNT(1) FROM ee_positions WHERE status = 'open'", ()) or 0)
    if open_positions >= max_positions:
        return False, "max_positions_reached"
    if score < 60:
        return False, "score_below_minimum"
    return True, "ok"

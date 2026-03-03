"""
TWR (Time-Weighted Return) Calculator.

Implements the Modified Dietz method for TWR calculation and
simple MWRR (Money-Weighted Rate of Return) via IRR approximation.

Extracted from the legacy ``modified_twr_calculator.py`` and
``calculate_real_twr.py`` scripts.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from app.core.database import query_df

logger = logging.getLogger(__name__)


def compute_twr(
    snapshots: pd.DataFrame,
    date_col: str = "snapshot_date",
    value_col: str = "portfolio_value",
    cashflow_col: str = "deposit_cash",
) -> Optional[float]:
    """
    Calculate Time-Weighted Return using the Modified Dietz method.

    Parameters:
        snapshots: DataFrame with columns [date_col, value_col, cashflow_col]
                   sorted by date ascending.
        date_col: Column name for snapshot dates (ISO format).
        value_col: Column name for portfolio value at each date.
        cashflow_col: Column name for cumulative external cash flows.

    Returns:
        TWR as a percentage, or None if insufficient data.
    """
    if snapshots.empty or len(snapshots) < 2:
        return None

    df = snapshots.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Chain-link sub-period returns
    cumulative_return = 1.0

    for i in range(1, len(df)):
        v_begin = float(df.loc[i - 1, value_col] or 0)
        v_end = float(df.loc[i, value_col] or 0)
        cf_begin = float(df.loc[i - 1, cashflow_col] or 0)
        cf_end = float(df.loc[i, cashflow_col] or 0)

        # Net cash flow for this sub-period
        net_cf = cf_end - cf_begin

        # Modified Dietz: adjust beginning value by half the cash flow
        adjusted_begin = v_begin + net_cf * 0.5

        if adjusted_begin <= 0:
            continue

        sub_return = (v_end - v_begin - net_cf) / adjusted_begin
        cumulative_return *= (1 + sub_return)

    return (cumulative_return - 1) * 100  # percentage


def compute_mwrr(
    snapshots: pd.DataFrame,
    date_col: str = "snapshot_date",
    value_col: str = "portfolio_value",
    cashflow_col: str = "deposit_cash",
) -> Optional[float]:
    """
    Approximate Money-Weighted Rate of Return (MWRR) using Newton's method.

    This is a simplified IRR calculation based on snapshot cash flows.

    Returns:
        MWRR as an annualized percentage, or None if insufficient data.
    """
    if snapshots.empty or len(snapshots) < 2:
        return None

    df = snapshots.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Build cash flow series: negative = outflow, positive = inflow
    first_value = float(df.iloc[0][value_col] or 0)
    last_value = float(df.iloc[-1][value_col] or 0)

    cash_flows: List[Tuple[datetime, float]] = []

    # Initial investment
    cf_first = float(df.iloc[0][cashflow_col] or 0)
    cash_flows.append((df.iloc[0][date_col].to_pydatetime(), -cf_first if cf_first > 0 else -first_value))

    # Intermediate cash flows
    for i in range(1, len(df)):
        cf_curr = float(df.iloc[i][cashflow_col] or 0)
        cf_prev = float(df.iloc[i - 1][cashflow_col] or 0)
        delta_cf = cf_curr - cf_prev
        if abs(delta_cf) > 0.01:
            cash_flows.append((df.iloc[i][date_col].to_pydatetime(), -delta_cf))

    # Terminal value (portfolio value at end)
    cash_flows.append((df.iloc[-1][date_col].to_pydatetime(), last_value))

    if len(cash_flows) < 2:
        return None

    # Simple IRR via Newton's method
    base_date = cash_flows[0][0]
    total_days = (cash_flows[-1][0] - base_date).days
    if total_days <= 0:
        return None

    def npv(rate: float) -> float:
        total = 0.0
        for dt, cf in cash_flows:
            t = (dt - base_date).days / 365.25
            total += cf / ((1 + rate) ** t)
        return total

    def npv_prime(rate: float) -> float:
        total = 0.0
        for dt, cf in cash_flows:
            t = (dt - base_date).days / 365.25
            if t > 0:
                total -= t * cf / ((1 + rate) ** (t + 1))
        return total

    # Newton's method
    rate = 0.1  # initial guess: 10%
    for _ in range(100):
        f = npv(rate)
        fp = npv_prime(rate)
        if abs(fp) < 1e-12:
            break
        rate -= f / fp
        if abs(f) < 1e-8:
            break

    return rate * 100  # percentage


def calculate_performance(
    user_id: int,
    portfolio: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """
    Calculate TWR and MWRR for a user's portfolio over a date range.

    Returns dict with twr_percent, mwrr_percent, and supporting data.
    """
    conditions = ["user_id = ?"]
    params: list = [user_id]

    if portfolio:
        conditions.append("portfolio = ?")
        params.append(portfolio)
    if start_date:
        conditions.append("snapshot_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("snapshot_date <= ?")
        params.append(end_date)

    where = " AND ".join(conditions)
    df = query_df(
        f"""
        SELECT snapshot_date, portfolio_value, deposit_cash
        FROM portfolio_snapshots
        WHERE {where}
        ORDER BY snapshot_date ASC
        """,
        tuple(params),
    )

    twr = compute_twr(df)
    mwrr = compute_mwrr(df)

    return {
        "twr_percent": round(twr, 4) if twr is not None else None,
        "mwrr_percent": round(mwrr, 4) if mwrr is not None else None,
        "snapshots_used": len(df),
        "start_date": str(df.iloc[0]["snapshot_date"]) if not df.empty else None,
        "end_date": str(df.iloc[-1]["snapshot_date"]) if not df.empty else None,
    }

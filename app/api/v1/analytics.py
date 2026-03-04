"""
Analytics API v1 — TWR, MWRR, Sharpe, Sortino, realized profit, cash reconciliation.

Provides performance metrics using the PortfolioService class.
"""

import time
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.core.exceptions import BadRequestError
from app.core.database import query_df, query_val, exec_sql
from app.services.fx_service import PORTFOLIO_CCY
from app.services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/performance")
async def performance_metrics(
    portfolio: Optional[str] = Query(None),
    period: str = Query("all", description="all, ytd, 1y, 6m, 3m, 1m"),
    current_user: TokenData = Depends(get_current_user),
):
    """
    Portfolio performance metrics — TWR, MWRR, ROI.

    Now computed via PortfolioService.calculate_performance() with full
    GIPS-compliant TWR and XIRR MWRR (Newton-Raphson + bisection fallback).
    """
    if portfolio and portfolio not in PORTFOLIO_CCY:
        raise BadRequestError(f"Unknown portfolio '{portfolio}'")

    svc = PortfolioService(current_user.user_id)
    result = svc.calculate_performance(period=period, portfolio=portfolio)

    return {"status": "ok", "data": result}


@router.get("/risk-metrics")
async def risk_metrics(
    rf_rate: float = Query(..., description="Annual risk-free rate for Sharpe (user must set manually)"),
    mar: float = Query(0.0, description="Minimum acceptable return for Sortino (default 0%)"),
    current_user: TokenData = Depends(get_current_user),
):
    """
    Risk-adjusted return metrics — Sharpe and Sortino ratios.

    rf_rate is required — user sets it manually via the CBK Rate editor.
    No server-side default; the stored value is sent from the client.

    Sharpe = mean(Rp − Rf) / std(Rp − Rf) × √N
    Sortino = mean(Rp − MAR) / downside_std × √N
    """
    svc = PortfolioService(current_user.user_id)
    sharpe = svc.calculate_sharpe_ratio(rf_rate=rf_rate)
    sortino = svc.calculate_sortino_ratio(mar=mar)

    return {
        "status": "ok",
        "data": {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "rf_rate": rf_rate,
            "mar": mar,
        },
    }


@router.get("/settings/rf-rate")
async def get_rf_rate(
    current_user: TokenData = Depends(get_current_user),
):
    """Get the stored risk-free rate for the current user."""
    val = query_val(
        "SELECT setting_value FROM user_settings WHERE user_id = ? AND setting_key = 'rf_rate'",
        (current_user.user_id,),
    )
    return {
        "status": "ok",
        "data": {"rf_rate": float(val) if val is not None else None},
    }


@router.put("/settings/rf-rate")
async def set_rf_rate(
    rf_rate: float = Query(..., description="Risk-free rate as percentage, e.g. 4.25"),
    current_user: TokenData = Depends(get_current_user),
):
    """Store the user's risk-free rate (as percentage, e.g. 4.25 for 4.25%)."""
    import time
    exec_sql(
        "INSERT INTO user_settings (user_id, setting_key, setting_value, updated_at) "
        "VALUES (?, 'rf_rate', ?, ?) "
        "ON CONFLICT (user_id, setting_key) DO UPDATE "
        "SET setting_value = EXCLUDED.setting_value, updated_at = EXCLUDED.updated_at",
        (current_user.user_id, str(rf_rate), int(time.time())),
    )
    return {
        "status": "ok",
        "data": {"rf_rate": rf_rate},
    }


@router.get("/realized-profit")
async def realized_profit(
    current_user: TokenData = Depends(get_current_user),
):
    """
    Realized profit breakdown — dual-path WAC.

    Returns total realized P&L (KWD), split into profits and losses,
    with per-transaction details.
    """
    svc = PortfolioService(current_user.user_id)
    result = svc.calculate_realized_profit_details()

    return {"status": "ok", "data": result}


@router.get("/cash-balances")
async def cash_balances(
    force: bool = Query(False, description="Override manual_override flags"),
    current_user: TokenData = Depends(get_current_user),
):
    """
    Cash reconciliation via 5-source UNION ALL.

    Returns per-portfolio cash balance with currency and manual_override flag.
    Respects manual_override unless force=true.
    """
    uid = current_user.user_id
    svc = PortfolioService(uid)
    balances = svc.recalc_portfolio_cash(force_override=force)

    # Enrich with currency and override flag from portfolio_cash table
    enriched = {}
    for pf, bal in balances.items():
        ccy = PORTFOLIO_CCY.get(pf, "KWD")
        override = False
        try:
            row = query_val(
                "SELECT manual_override FROM portfolio_cash WHERE user_id = ? AND portfolio = ?",
                (uid, pf),
            )
            override = bool(row) if row else False
        except Exception:
            pass
        enriched[pf] = {
            "balance": bal,
            "currency": ccy,
            "manual_override": override,
        }

    # Also include portfolios with manual override that aren't in the computed set
    try:
        from app.core.database import query_df as _qdf
        df = _qdf(
            "SELECT portfolio, balance, currency FROM portfolio_cash "
            "WHERE user_id = ? AND manual_override = 1",
            (uid,),
        )
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                pf = row["portfolio"]
                if pf not in enriched:
                    enriched[pf] = {
                        "balance": float(row["balance"]),
                        "currency": row.get("currency", "KWD"),
                        "manual_override": True,
                    }
    except Exception:
        pass

    return {"status": "ok", "data": enriched}


# ── Manual cash override ─────────────────────────────────────────────

class CashOverridePayload(BaseModel):
    balance: float = Field(..., description="Manual cash balance")
    currency: str = Field("KWD", description="Currency code")


@router.put("/cash-balances/{portfolio}")
async def set_cash_override(
    portfolio: str,
    payload: CashOverridePayload,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Manually set/override the cash balance for a portfolio.

    Sets ``manual_override = 1`` so the automatic reconciliation
    skips this portfolio until the override is cleared.
    """
    if portfolio not in PORTFOLIO_CCY:
        raise BadRequestError(
            f"Unknown portfolio '{portfolio}'. Valid: {list(PORTFOLIO_CCY.keys())}"
        )

    now = int(time.time())
    uid = current_user.user_id

    # Upsert into portfolio_cash with manual_override = 1
    existing = query_val(
        "SELECT 1 FROM portfolio_cash WHERE user_id = ? AND portfolio = ?",
        (uid, portfolio),
    )
    if existing:
        exec_sql(
            "UPDATE portfolio_cash SET balance=?, currency=?, manual_override=1, last_updated=? "
            "WHERE user_id=? AND portfolio=?",
            (payload.balance, payload.currency, now, uid, portfolio),
        )
    else:
        exec_sql(
            "INSERT INTO portfolio_cash "
            "(user_id, portfolio, balance, currency, last_updated, manual_override) "
            "VALUES (?,?,?,?,?,1)",
            (uid, portfolio, payload.balance, payload.currency, now),
        )

    return {
        "status": "ok",
        "data": {
            "portfolio": portfolio,
            "balance": payload.balance,
            "currency": payload.currency,
            "manual_override": True,
        },
    }


@router.delete("/cash-balances/{portfolio}/override")
async def clear_cash_override(
    portfolio: str,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Clear manual override for a portfolio cash balance.

    Resets ``manual_override = 0`` so automatic reconciliation
    resumes for this portfolio.
    """
    if portfolio not in PORTFOLIO_CCY:
        raise BadRequestError(
            f"Unknown portfolio '{portfolio}'. Valid: {list(PORTFOLIO_CCY.keys())}"
        )

    uid = current_user.user_id

    exec_sql(
        "UPDATE portfolio_cash SET manual_override=0 WHERE user_id=? AND portfolio=?",
        (uid, portfolio),
    )

    # Re-calc the balance now
    svc = PortfolioService(uid)
    balances = svc.recalc_portfolio_cash()

    return {
        "status": "ok",
        "data": {
            "portfolio": portfolio,
            "balance": balances.get(portfolio, 0.0),
            "manual_override": False,
        },
    }


@router.get("/snapshots")
async def list_snapshots(
    portfolio: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    current_user: TokenData = Depends(get_current_user),
):
    """List portfolio snapshots with optional date range filter."""
    conditions = ["user_id = ?"]
    params: list = [current_user.user_id]

    # Note: portfolio_snapshots is per-user (no portfolio column).
    # The portfolio parameter is accepted but ignored for backward compat.
    if start_date:
        conditions.append("snapshot_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("snapshot_date <= ?")
        params.append(end_date)

    where = " AND ".join(conditions)
    df = query_df(
        f"""
        SELECT id, snapshot_date, portfolio_value, daily_movement,
               beginning_difference, deposit_cash, accumulated_cash,
               net_gain, change_percent, roi_percent,
               twr_percent, mwrr_percent, created_at
        FROM portfolio_snapshots
        WHERE {where}
        ORDER BY snapshot_date DESC
        """,
        tuple(params),
    )

    records = df.fillna(0).to_dict(orient="records") if not df.empty else []

    return {
        "status": "ok",
        "data": {
            "snapshots": records,
            "count": len(records),
        },
    }


@router.get("/position-snapshots")
async def list_position_snapshots(
    stock_symbol: Optional[str] = Query(None),
    portfolio: Optional[str] = Query(None),
    current_user: TokenData = Depends(get_current_user),
):
    """List position-level snapshots (per-stock tracking)."""
    conditions = ["user_id = ?"]
    params: list = [current_user.user_id]

    if stock_symbol:
        conditions.append("stock_symbol = ?")
        params.append(stock_symbol)
    if portfolio:
        conditions.append("portfolio_id IN (SELECT id FROM portfolios WHERE name = ? AND user_id = ?)")
        params.extend([portfolio, current_user.user_id])

    where = " AND ".join(conditions)

    df = query_df(
        f"""
        SELECT id, stock_id, stock_symbol, portfolio_id, snapshot_date,
               total_shares, total_cost, avg_cost, realized_pnl,
               cash_dividends_received, status
        FROM position_snapshots
        WHERE {where}
        ORDER BY snapshot_date DESC
        """,
        tuple(params),
    )

    records = df.fillna(0).to_dict(orient="records") if not df.empty else []

    return {
        "status": "ok",
        "data": {
            "snapshots": records,
            "count": len(records),
        },
    }

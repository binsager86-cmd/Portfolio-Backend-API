"""
Portfolio API routes — overview, holdings, per-portfolio tables.

All values that involve USD positions are pre-converted to KWD in the response
so the frontend never needs to do currency math.
"""

from fastapi import APIRouter, Depends, Query
from typing import Optional

from app.services.auth_service import get_current_user, TokenData
from app.services.portfolio_service import (
    get_complete_overview,
    get_portfolio_overview,
    get_portfolio_value,
    get_current_holdings,
    build_portfolio_table,
    get_account_balances,
    get_total_portfolio_value,
)
from app.services.fx_service import PORTFOLIO_CCY, get_usd_kwd_rate, convert_to_kwd

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])


# ── Overview ─────────────────────────────────────────────────────────

@router.get("/overview")
async def portfolio_overview(
    current_user: TokenData = Depends(get_current_user),
):
    """
    Complete portfolio overview — transaction aggregates, market values,
    cash balances, and calculated metrics.  All monetary values in KWD.
    """
    data = get_complete_overview(current_user.user_id)
    return {"status": "ok", "data": data}


# ── Holdings ──────────────────────────────────────────────────────────

@router.get("/holdings")
async def portfolio_holdings(
    portfolio: Optional[str] = Query(None, description="Filter by portfolio name (KFH, BBYN, USA)"),
    current_user: TokenData = Depends(get_current_user),
):
    """
    Current stock holdings with KWD-converted market values and P&L.
    Optionally filter by portfolio name.
    """
    if portfolio and portfolio not in PORTFOLIO_CCY:
        return {"status": "error", "detail": f"Unknown portfolio '{portfolio}'. Valid: {list(PORTFOLIO_CCY.keys())}"}

    portfolios_to_query = [portfolio] if portfolio else list(PORTFOLIO_CCY.keys())

    all_holdings = []
    totals = {
        "total_market_value_kwd": 0.0,
        "total_cost_kwd": 0.0,
        "total_unrealized_pnl_kwd": 0.0,
        "total_realized_pnl_kwd": 0.0,
        "total_pnl_kwd": 0.0,
        "total_dividends_kwd": 0.0,
    }

    for pname in portfolios_to_query:
        df = build_portfolio_table(pname, current_user.user_id)
        if df.empty:
            continue

        for _, row in df.iterrows():
            holding = row.to_dict()
            all_holdings.append(holding)

            totals["total_market_value_kwd"] += float(holding.get("market_value_kwd", 0))
            totals["total_cost_kwd"] += float(holding.get("total_cost_kwd", 0))
            totals["total_unrealized_pnl_kwd"] += float(holding.get("unrealized_pnl_kwd", 0) or 0)
            totals["total_realized_pnl_kwd"] += convert_to_kwd(
                float(holding.get("realized_pnl", 0)), holding.get("currency", "KWD")
            )
            totals["total_pnl_kwd"] += float(holding.get("total_pnl_kwd", 0))
            totals["total_dividends_kwd"] += convert_to_kwd(
                float(holding.get("cash_dividends", 0)), holding.get("currency", "KWD")
            )

    # Include total portfolio value (stocks + cash) from unified source
    unified = get_total_portfolio_value(current_user.user_id)
    total_portfolio_value_kwd = unified["total_value_kwd"]
    cash_balance_kwd = unified["cash_kwd"]

    # Recalculate allocation: MV_kwd / total_portfolio_value_kwd
    if total_portfolio_value_kwd > 0:
        for h in all_holdings:
            h["weight_by_cost"] = float(h.get("market_value_kwd", 0)) / total_portfolio_value_kwd

    return {
        "status": "ok",
        "data": {
            "holdings": all_holdings,
            "totals": totals,
            "total_portfolio_value_kwd": total_portfolio_value_kwd,
            "cash_balance_kwd": cash_balance_kwd,
            "usd_kwd_rate": get_usd_kwd_rate(),
            "count": len(all_holdings),
        },
    }


# ── Per-portfolio table ──────────────────────────────────────────────

@router.get("/table/{portfolio_name}")
async def portfolio_table(
    portfolio_name: str,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Full holdings table for a single portfolio (KFH, BBYN, USA).
    Returns the same data as the legacy Streamlit portfolio tab.
    """
    if portfolio_name not in PORTFOLIO_CCY:
        return {"status": "error", "detail": f"Unknown portfolio '{portfolio_name}'. Valid: {list(PORTFOLIO_CCY.keys())}"}

    df = build_portfolio_table(portfolio_name, current_user.user_id)
    records = df.to_dict(orient="records") if not df.empty else []

    return {
        "status": "ok",
        "data": {
            "portfolio": portfolio_name,
            "currency": PORTFOLIO_CCY[portfolio_name],
            "holdings": records,
            "count": len(records),
            "usd_kwd_rate": get_usd_kwd_rate(),
        },
    }


# ── Account cash balances ────────────────────────────────────────────

@router.get("/accounts")
async def account_balances(
    current_user: TokenData = Depends(get_current_user),
):
    """External account cash balances."""
    data = get_account_balances(current_user.user_id)
    return {"status": "ok", "data": data}


# ── FX rate info ─────────────────────────────────────────────────────

@router.get("/fx-rate")
async def fx_rate(
    current_user: TokenData = Depends(get_current_user),
):
    """Current USD→KWD exchange rate (cached for 1 hour)."""
    rate = get_usd_kwd_rate()
    return {
        "status": "ok",
        "data": {
            "usd_kwd": rate,
            "source": "yfinance (cached)",
        },
    }

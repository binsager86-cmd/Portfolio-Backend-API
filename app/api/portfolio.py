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

    When no portfolio filter is applied ("All"), holdings for the same
    stock symbol across different portfolios are **merged** into a single
    row with aggregated quantities, costs, and P&L.
    """
    if portfolio and portfolio not in PORTFOLIO_CCY:
        return {"status": "error", "detail": f"Unknown portfolio '{portfolio}'. Valid: {list(PORTFOLIO_CCY.keys())}"}

    portfolios_to_query = [portfolio] if portfolio else list(PORTFOLIO_CCY.keys())

    all_holdings = []

    for pname in portfolios_to_query:
        df = build_portfolio_table(pname, current_user.user_id)
        if df.empty:
            continue

        for _, row in df.iterrows():
            holding = row.to_dict()
            holding["_portfolio"] = pname          # track source portfolio
            all_holdings.append(holding)

    # ── Aggregate duplicate symbols ─────────────────────────────────
    # When "All" is selected (no filter), merge rows that share the
    # same symbol so each stock appears once with summed quantities.
    if not portfolio and all_holdings:
        merged: dict = {}   # symbol → aggregated row
        for h in all_holdings:
            sym = h.get("symbol", "").strip()
            if sym in merged:
                m = merged[sym]
                m["shares_qty"] += float(h.get("shares_qty", 0))
                m["total_cost"] += float(h.get("total_cost", 0))
                m["market_value"] += float(h.get("market_value", 0))
                m["unrealized_pnl"] += float(h.get("unrealized_pnl", 0))
                m["realized_pnl"] += float(h.get("realized_pnl", 0))
                m["cash_dividends"] += float(h.get("cash_dividends", 0))
                m["reinvested_dividends"] += float(h.get("reinvested_dividends", 0))
                m["bonus_dividend_shares"] += float(h.get("bonus_dividend_shares", 0))
                m["bonus_share_value"] += float(h.get("bonus_share_value", 0))
                m["total_pnl"] += float(h.get("total_pnl", 0))
                m["market_value_kwd"] += float(h.get("market_value_kwd", 0))
                m["unrealized_pnl_kwd"] += float(h.get("unrealized_pnl_kwd", 0))
                m["total_pnl_kwd"] += float(h.get("total_pnl_kwd", 0))
                m["total_cost_kwd"] += float(h.get("total_cost_kwd", 0))
                # Track source portfolios
                m["_portfolios"].append(h.get("_portfolio", ""))
            else:
                merged[sym] = {
                    **h,
                    "shares_qty": float(h.get("shares_qty", 0)),
                    "total_cost": float(h.get("total_cost", 0)),
                    "market_value": float(h.get("market_value", 0)),
                    "unrealized_pnl": float(h.get("unrealized_pnl", 0)),
                    "realized_pnl": float(h.get("realized_pnl", 0)),
                    "cash_dividends": float(h.get("cash_dividends", 0)),
                    "reinvested_dividends": float(h.get("reinvested_dividends", 0)),
                    "bonus_dividend_shares": float(h.get("bonus_dividend_shares", 0)),
                    "bonus_share_value": float(h.get("bonus_share_value", 0)),
                    "total_pnl": float(h.get("total_pnl", 0)),
                    "market_value_kwd": float(h.get("market_value_kwd", 0)),
                    "unrealized_pnl_kwd": float(h.get("unrealized_pnl_kwd", 0)),
                    "total_pnl_kwd": float(h.get("total_pnl_kwd", 0)),
                    "total_cost_kwd": float(h.get("total_cost_kwd", 0)),
                    "_portfolios": [h.get("_portfolio", "")],
                }

        # Recalculate derived fields after aggregation
        for m in merged.values():
            qty = m["shares_qty"]
            tc = m["total_cost"]
            mp = m.get("market_price", 0) or 0
            m["avg_cost"] = round(tc / qty, 6) if qty > 0 else 0.0
            m["market_value"] = round(qty * mp, 3)
            m["unrealized_pnl"] = round((mp - m["avg_cost"]) * qty, 3) if qty > 0 and mp > 0 else 0.0
            # Recalculate KWD market value from fresh qty * price
            ccy = m.get("currency", "KWD")
            m["market_value_kwd"] = convert_to_kwd(m["market_value"], ccy)
            m["unrealized_pnl_kwd"] = convert_to_kwd(m["unrealized_pnl"], ccy)
            m["total_pnl"] = round(m["unrealized_pnl"] + m["realized_pnl"] + m["cash_dividends"], 3)
            m["total_pnl_kwd"] = convert_to_kwd(m["total_pnl"], ccy)
            denom = tc + abs(m["realized_pnl"])
            m["pnl_pct"] = (m["total_pnl"] / denom) if denom > 0 else 0.0
            m["dividend_yield_on_cost_pct"] = (m["cash_dividends"] / tc) if tc > 0 else 0.0
            # Clean up internal fields
            m.pop("_portfolio", None)
            m.pop("_portfolios", None)

        all_holdings = list(merged.values())

    else:
        # Single portfolio selected — remove internal tracking field
        for h in all_holdings:
            h.pop("_portfolio", None)

    # ── Compute totals ──────────────────────────────────────────────
    totals = {
        "total_market_value_kwd": 0.0,
        "total_cost_kwd": 0.0,
        "total_unrealized_pnl_kwd": 0.0,
        "total_realized_pnl_kwd": 0.0,
        "total_pnl_kwd": 0.0,
        "total_dividends_kwd": 0.0,
    }
    for h in all_holdings:
        totals["total_market_value_kwd"] += float(h.get("market_value_kwd", 0))
        totals["total_cost_kwd"] += float(h.get("total_cost_kwd", 0))
        totals["total_unrealized_pnl_kwd"] += float(h.get("unrealized_pnl_kwd", 0) or 0)
        totals["total_realized_pnl_kwd"] += convert_to_kwd(
            float(h.get("realized_pnl", 0)), h.get("currency", "KWD")
        )
        totals["total_pnl_kwd"] += float(h.get("total_pnl_kwd", 0))
        totals["total_dividends_kwd"] += convert_to_kwd(
            float(h.get("cash_dividends", 0)), h.get("currency", "KWD")
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

"""
Portfolio API v1 — overview, holdings, per-portfolio tables.

All monetary values involving USD positions are pre-converted to KWD
so the frontend never needs to do currency math.
"""

import io
import time
import logging
from typing import Optional
from datetime import date

import pandas as pd
from fastapi import APIRouter, Depends, Query, Request
from starlette.responses import StreamingResponse

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.core.exceptions import NotFoundError, BadRequestError
from app.core.database import query_df, query_one, exec_sql, column_exists
from app.services.portfolio_service import (
    PortfolioService,
    get_complete_overview,
    get_portfolio_overview,
    get_portfolio_value,
    get_total_portfolio_value,
    get_current_holdings,
    build_portfolio_table,
    get_account_balances,
)
from app.services.fx_service import PORTFOLIO_CCY, get_usd_kwd_rate, convert_to_kwd
from app.services.audit_service import (
    log_event, TXN_CREATE, TXN_UPDATE, TXN_DELETE, TXN_RESTORE,
    ADMIN_ACTION,
)
from app.schemas.portfolio import TransactionCreate, TransactionUpdate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


def _txn_cash_delta(txn_type: str, purchase_cost: float, sell_value: float,
                    cash_dividend: float, fees: float) -> float:
    """
    Compute the net cash effect of a transaction.

    Matches Streamlit's formula:
        Buy  → cash -= (cost + fees)    → negative delta
        Sell → cash += (proceeds - fees) → positive delta (net of fees)
        Dividend → cash += cash_dividend → positive delta

    Fees for Buys are already reflected in purchase_cost in most cases,
    but the 5-source UNION ALL subtracts fees separately, so the delta
    here mirrors that: Buy delta = -cost, Sell delta = +sell_value,
    Dividend delta = +cash_dividend, Fee delta = -fees.
    Combined: delta = -cost + sell_value + cash_dividend - fees.
    """
    delta = 0.0
    if txn_type == "Buy":
        delta -= purchase_cost
    if txn_type == "Sell":
        delta += sell_value
    delta += cash_dividend
    delta -= fees
    return delta


# ── Overview ─────────────────────────────────────────────────────────

@router.get("/overview")
async def portfolio_overview(current_user: TokenData = Depends(get_current_user)):
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
        raise BadRequestError(
            f"Unknown portfolio '{portfolio}'. Valid: {list(PORTFOLIO_CCY.keys())}"
        )

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

    # ── Recalculate allocation: MV_kwd / total_portfolio_value_kwd ────
    # Allocation = each stock's market value as a percentage of the
    # full portfolio (stocks + cash).  Cash takes the remainder.
    # Stored as allocation_pct (separate from weight_by_cost which is cost-based).
    if total_portfolio_value_kwd > 0:
        for h in all_holdings:
            h["allocation_pct"] = float(h.get("market_value_kwd", 0)) / total_portfolio_value_kwd
    else:
        for h in all_holdings:
            h["allocation_pct"] = 0.0

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
        raise BadRequestError(
            f"Unknown portfolio '{portfolio_name}'. Valid: {list(PORTFOLIO_CCY.keys())}"
        )

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
async def account_balances(current_user: TokenData = Depends(get_current_user)):
    """External account cash balances."""
    data = get_account_balances(current_user.user_id)
    return {"status": "ok", "data": data}


# ── FX rate info ─────────────────────────────────────────────────────

@router.get("/fx-rate")
async def fx_rate(current_user: TokenData = Depends(get_current_user)):
    """Current USD→KWD exchange rate (cached for 1 hour)."""
    rate = get_usd_kwd_rate()
    return {
        "status": "ok",
        "data": {"usd_kwd": rate, "source": "yfinance (cached)"},
    }


# ── Transaction CRUD ─────────────────────────────────────────────────

@router.get("/transactions")
async def list_transactions(
    portfolio: Optional[str] = Query(None),
    stock_symbol: Optional[str] = Query(None),
    txn_type: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    current_user: TokenData = Depends(get_current_user),
):
    """List transactions with optional filters and pagination."""
    conditions = ["user_id = ?", "COALESCE(is_deleted, 0) = 0"]
    params: list = [current_user.user_id]

    if portfolio:
        conditions.append("portfolio = ?")
        params.append(portfolio)
    if stock_symbol:
        conditions.append("TRIM(stock_symbol) = ?")
        params.append(stock_symbol.strip())
    if txn_type:
        conditions.append("txn_type = ?")
        params.append(txn_type)

    where = " AND ".join(conditions)

    # Count total
    from app.core.database import query_val
    total = query_val(f"SELECT COUNT(*) FROM transactions WHERE {where}", tuple(params))

    # Fetch page
    offset = (page - 1) * page_size
    sql = f"""
        SELECT id, user_id, portfolio, stock_symbol, txn_date, txn_type,
               shares, purchase_cost, sell_value, bonus_shares, cash_dividend,
               reinvested_dividend, fees, price_override, planned_cum_shares,
               broker, reference, notes, category, is_deleted, created_at
        FROM transactions
        WHERE {where}
        ORDER BY txn_date DESC, created_at DESC, id DESC
        LIMIT ? OFFSET ?
    """
    params.extend([page_size, offset])

    df = query_df(sql, tuple(params))
    records = df.to_dict(orient="records") if not df.empty else []

    total_pages = max(1, (total + page_size - 1) // page_size)

    return {
        "status": "ok",
        "data": {
            "transactions": records,
            "count": len(records),
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total,
                "total_pages": total_pages,
            },
        },
    }


@router.get("/transactions/{txn_id}")
async def get_transaction(
    txn_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Get a single transaction by ID."""
    row = query_one(
        "SELECT * FROM transactions WHERE id = ? AND user_id = ? AND COALESCE(is_deleted, 0) = 0",
        (txn_id, current_user.user_id),
    )
    if not row:
        raise NotFoundError("Transaction", txn_id)

    return {"status": "ok", "data": dict(row)}


@router.post("/transactions", status_code=201)
async def create_transaction(
    request: Request,
    body: TransactionCreate,
    current_user: TokenData = Depends(get_current_user),
):
    """Create a new transaction."""
    txn = body

    ALLOWED_TYPES = ("Buy", "Sell", "DIVIDEND_ONLY")
    if txn.txn_type not in ALLOWED_TYPES:
        raise BadRequestError(f"txn_type must be one of {ALLOWED_TYPES}")
    if txn.txn_type == "Buy" and not txn.purchase_cost:
        raise BadRequestError("purchase_cost required for Buy transactions")
    if txn.txn_type == "Sell" and not txn.sell_value:
        raise BadRequestError("sell_value required for Sell transactions")
    if txn.txn_type == "Buy" and txn.shares <= 0:
        raise BadRequestError("shares must be > 0 for Buy transactions")
    if txn.txn_type == "Sell" and txn.shares <= 0:
        raise BadRequestError("shares must be > 0 for Sell transactions")
    if txn.txn_type == "DIVIDEND_ONLY":
        has_any = (txn.cash_dividend or 0) > 0 or (txn.reinvested_dividend or 0) > 0 or (txn.bonus_shares or 0) > 0
        if not has_any:
            raise BadRequestError("DIVIDEND_ONLY requires at least one of: cash_dividend, reinvested_dividend, bonus_shares")

    now = int(time.time())

    # Capture FX rate at transaction time (matches Streamlit's get_current_fx_rate())
    try:
        current_fx = get_usd_kwd_rate()
    except Exception:
        current_fx = None

    exec_sql(
        """INSERT INTO transactions
           (user_id, portfolio, stock_symbol, txn_date, txn_type, shares,
            purchase_cost, sell_value, bonus_shares, cash_dividend,
            reinvested_dividend, fees, price_override, planned_cum_shares,
            broker, reference, notes, category, fx_rate_at_txn,
            source, is_deleted, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'portfolio', ?,
                   'MANUAL', 0, ?)""",
        (
            current_user.user_id, txn.portfolio, txn.stock_symbol,
            txn.txn_date, txn.txn_type, txn.shares,
            txn.purchase_cost or 0.0, txn.sell_value or 0.0, txn.bonus_shares or 0,
            txn.cash_dividend or 0.0, txn.reinvested_dividend or 0.0, txn.fees or 0.0,
            txn.price_override, txn.planned_cum_shares,
            txn.broker, txn.reference, txn.notes, current_fx, now,
        ),
    )

    # ── Auto-create stock record if missing (so price updater can find it)
    from app.core.database import query_val as _qv
    sym_upper = txn.stock_symbol.strip().upper()
    existing_stock = _qv(
        "SELECT id FROM stocks WHERE TRIM(symbol) = ? AND user_id = ?",
        (sym_upper, current_user.user_id),
    )
    if not existing_stock and txn.txn_type in ("Buy", "Sell"):
        ccy = "USD" if txn.portfolio == "USA" else "KWD"
        # Resolve yf_ticker from reference lists
        from app.services.price_service import _yahoo_symbol
        yf_ticker = _yahoo_symbol(sym_upper, ccy)
        exec_sql(
            """INSERT INTO stocks
               (user_id, symbol, name, portfolio, currency, current_price,
                yf_ticker, price_source, created_at)
               VALUES (?, ?, ?, ?, ?, 0.0, ?, 'AUTO', ?)""",
            (current_user.user_id, sym_upper, sym_upper, txn.portfolio,
             ccy, yf_ticker, int(time.time())),
        )
        logger.info("Auto-created stock record for %s (yf: %s)", sym_upper, yf_ticker)

    # Return the created transaction
    new_id = _qv(
        "SELECT id FROM transactions WHERE user_id = ? ORDER BY id DESC LIMIT 1",
        (current_user.user_id,),
    )

    log_event(
        TXN_CREATE,
        user_id=current_user.user_id,
        resource_type="transaction",
        resource_id=new_id,
        details={"symbol": txn.stock_symbol, "type": txn.txn_type, "shares": txn.shares},
        request=request,
    )

    # ── Ledger: recalculate portfolio cash (respects manual_override — matches Streamlit)
    # Streamlit: Buy → cash -= (cost+fees), Sell → cash += (proceeds-fees),
    #            Dividend → cash += cash_dividend
    delta = _txn_cash_delta(
        txn.txn_type,
        txn.purchase_cost or 0.0,
        txn.sell_value or 0.0,
        txn.cash_dividend or 0.0,
        txn.fees or 0.0,
    )
    svc = PortfolioService(current_user.user_id)
    svc.recalc_portfolio_cash(
        deposit_delta=delta, delta_portfolio=txn.portfolio,
    )

    # Return fresh cash balance so frontend can update immediately
    unified = svc.get_total_portfolio_value()

    return {
        "status": "ok",
        "data": {
            "id": new_id,
            "message": "Transaction created",
            "cash_balance": unified["cash_kwd"],
            "total_value": unified["total_value_kwd"],
        },
    }


@router.put("/transactions/{txn_id}")
async def update_transaction(
    txn_id: int,
    request: Request,
    body: TransactionUpdate,
    current_user: TokenData = Depends(get_current_user),
):
    """Update an existing transaction."""
    # Read old values so we can compute the delta (old → new) for manual-override cash
    existing = query_one(
        "SELECT id, portfolio, txn_type, purchase_cost, sell_value, "
        "       cash_dividend, fees "
        "FROM transactions WHERE id = ? AND user_id = ? AND COALESCE(is_deleted, 0) = 0",
        (txn_id, current_user.user_id),
    )
    if not existing:
        raise NotFoundError("Transaction", txn_id)

    old_portfolio = existing["portfolio"]
    old_delta = _txn_cash_delta(
        existing["txn_type"],
        float(existing["purchase_cost"] or 0),
        float(existing["sell_value"] or 0),
        float(existing["cash_dividend"] or 0),
        float(existing["fees"] or 0),
    )

    # Build SET clause from provided fields (only non-None)
    updates = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if not updates:
        raise BadRequestError("No valid fields to update")

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    params = list(updates.values()) + [txn_id, current_user.user_id]

    exec_sql(
        f"UPDATE transactions SET {set_clause} WHERE id = ? AND user_id = ?",
        tuple(params),
    )

    log_event(
        TXN_UPDATE,
        user_id=current_user.user_id,
        resource_type="transaction",
        resource_id=txn_id,
        details={"updated_fields": list(updates.keys())},
        request=request,
    )

    # ── Ledger: recalculate portfolio cash (respects manual_override — matches Streamlit)
    # Compute new delta from the updated fields (fall back to old values)
    new_txn_type = updates.get("txn_type", existing["txn_type"])
    new_portfolio = updates.get("portfolio", old_portfolio)
    new_delta = _txn_cash_delta(
        new_txn_type,
        float(updates.get("purchase_cost", existing["purchase_cost"] or 0)),
        float(updates.get("sell_value", existing["sell_value"] or 0)),
        float(updates.get("cash_dividend", existing["cash_dividend"] or 0)),
        float(updates.get("fees", existing["fees"] or 0)),
    )

    svc = PortfolioService(current_user.user_id)
    if old_portfolio != new_portfolio:
        # Portfolio changed: reverse old delta on old portfolio, apply new delta on new portfolio
        svc.recalc_portfolio_cash(deposit_delta=-old_delta, delta_portfolio=old_portfolio)
        svc.recalc_portfolio_cash(deposit_delta=new_delta, delta_portfolio=new_portfolio)
    else:
        # Same portfolio: apply the difference
        svc.recalc_portfolio_cash(
            deposit_delta=new_delta - old_delta, delta_portfolio=new_portfolio,
        )

    unified = svc.get_total_portfolio_value()

    return {
        "status": "ok",
        "data": {
            "id": txn_id,
            "message": "Transaction updated",
            "cash_balance": unified["cash_kwd"],
            "total_value": unified["total_value_kwd"],
        },
    }


@router.delete("/transactions/{txn_id}")
async def delete_transaction(
    txn_id: int,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
):
    """Soft-delete a transaction."""
    existing = query_one(
        "SELECT id, portfolio, txn_type, purchase_cost, sell_value, "
        "       cash_dividend, fees "
        "FROM transactions WHERE id = ? AND user_id = ? AND COALESCE(is_deleted, 0) = 0",
        (txn_id, current_user.user_id),
    )
    if not existing:
        raise NotFoundError("Transaction", txn_id)

    # Compute the delta this transaction was contributing, then reverse it
    del_delta = _txn_cash_delta(
        existing["txn_type"],
        float(existing["purchase_cost"] or 0),
        float(existing["sell_value"] or 0),
        float(existing["cash_dividend"] or 0),
        float(existing["fees"] or 0),
    )

    now = int(time.time())
    exec_sql(
        "UPDATE transactions SET is_deleted = 1, deleted_at = ? WHERE id = ? AND user_id = ?",
        (now, txn_id, current_user.user_id),
    )

    log_event(
        TXN_DELETE,
        user_id=current_user.user_id,
        resource_type="transaction",
        resource_id=txn_id,
        request=request,
    )

    # ── Ledger: recalculate portfolio cash (respects manual_override — matches Streamlit)
    # Reverse the cash effect of the deleted transaction
    svc = PortfolioService(current_user.user_id)
    svc.recalc_portfolio_cash(
        deposit_delta=-del_delta, delta_portfolio=existing["portfolio"],
    )

    unified = svc.get_total_portfolio_value()

    return {
        "status": "ok",
        "data": {
            "id": txn_id,
            "message": "Transaction deleted",
            "cash_balance": unified["cash_kwd"],
            "total_value": unified["total_value_kwd"],
        },
    }


@router.post("/transactions/{txn_id}/restore")
async def restore_transaction(
    txn_id: int,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
):
    """Restore a soft-deleted transaction."""
    existing = query_one(
        "SELECT id, portfolio, txn_type, purchase_cost, sell_value, "
        "       cash_dividend, fees "
        "FROM transactions WHERE id = ? AND user_id = ? AND is_deleted = 1",
        (txn_id, current_user.user_id),
    )
    if not existing:
        raise NotFoundError("Transaction", txn_id)

    # Compute the cash effect to re-apply
    restore_delta = _txn_cash_delta(
        existing["txn_type"],
        float(existing["purchase_cost"] or 0),
        float(existing["sell_value"] or 0),
        float(existing["cash_dividend"] or 0),
        float(existing["fees"] or 0),
    )

    exec_sql(
        "UPDATE transactions SET is_deleted = 0, deleted_at = NULL WHERE id = ? AND user_id = ?",
        (txn_id, current_user.user_id),
    )

    log_event(
        TXN_RESTORE,
        user_id=current_user.user_id,
        resource_type="transaction",
        resource_id=txn_id,
        request=request,
    )

    # ── Ledger: recalculate portfolio cash (respects manual_override — matches Streamlit)
    # Re-apply the cash effect of the restored transaction
    svc = PortfolioService(current_user.user_id)
    svc.recalc_portfolio_cash(
        deposit_delta=restore_delta, delta_portfolio=existing["portfolio"],
    )

    unified = svc.get_total_portfolio_value()

    return {
        "status": "ok",
        "data": {
            "id": txn_id,
            "message": "Transaction restored",
            "cash_balance": unified["cash_kwd"],
            "total_value": unified["total_value_kwd"],
        },
    }


@router.delete("/transactions")
async def delete_all_transactions(
    request: Request,
    portfolio: Optional[str] = Query(None, description="Filter by portfolio (optional)"),
    current_user: TokenData = Depends(get_current_user),
):
    """Soft-delete all transactions for the user (optionally filtered by portfolio)."""
    now = int(time.time())

    if portfolio:
        if portfolio not in PORTFOLIO_CCY:
            raise BadRequestError(
                f"Unknown portfolio '{portfolio}'. Valid: {list(PORTFOLIO_CCY.keys())}"
            )
        count_val = query_one(
            "SELECT COUNT(*) FROM transactions WHERE user_id = ? AND portfolio = ? AND COALESCE(is_deleted, 0) = 0",
            (current_user.user_id, portfolio),
        )
        exec_sql(
            "UPDATE transactions SET is_deleted = 1, deleted_at = ? "
            "WHERE user_id = ? AND portfolio = ? AND COALESCE(is_deleted, 0) = 0",
            (now, current_user.user_id, portfolio),
        )
    else:
        count_val = query_one(
            "SELECT COUNT(*) FROM transactions WHERE user_id = ? AND COALESCE(is_deleted, 0) = 0",
            (current_user.user_id,),
        )
        exec_sql(
            "UPDATE transactions SET is_deleted = 1, deleted_at = ? "
            "WHERE user_id = ? AND COALESCE(is_deleted, 0) = 0",
            (now, current_user.user_id),
        )

    deleted_count = count_val[0] if count_val else 0

    log_event(
        TXN_DELETE,
        user_id=current_user.user_id,
        resource_type="transaction",
        resource_id=0,
        details={"bulk_delete": True, "portfolio": portfolio, "count": deleted_count},
        request=request,
    )

    # ── Ledger: recalculate portfolio cash (respects manual_override — matches Streamlit)
    svc = PortfolioService(current_user.user_id)
    svc.recalc_portfolio_cash()  # force_override=False

    return {
        "status": "ok",
        "data": {"deleted_count": deleted_count, "message": f"Deleted {deleted_count} transactions"},
    }


# ── Holdings export ──────────────────────────────────────────────────

@router.get("/holdings-export")
async def holdings_export(
    portfolio: Optional[str] = Query(None),
    current_user: TokenData = Depends(get_current_user),
):
    """
    Export current holdings as Excel (.xlsx).
    """
    portfolios_to_query = (
        [portfolio] if portfolio and portfolio in PORTFOLIO_CCY
        else list(PORTFOLIO_CCY.keys())
    )

    rows = []
    for pname in portfolios_to_query:
        df = build_portfolio_table(pname, current_user.user_id)
        if df.empty:
            continue
        for _, row in df.iterrows():
            h = row.to_dict()
            rows.append({
                "Portfolio": pname,
                "Company": h.get("company", ""),
                "Symbol": h.get("symbol", ""),
                "Quantity": h.get("shares_qty", 0),
                "Avg Cost": round(h.get("avg_cost", 0), 3),
                "Total Cost": round(h.get("total_cost", 0), 2),
                "Total Cost (KWD)": round(h.get("total_cost_kwd", 0), 2),
                "Market Price": round(h.get("market_price", 0), 3),
                "Market Value": round(h.get("market_value", 0), 2),
                "Market Value (KWD)": round(h.get("market_value_kwd", 0), 2),
                "Unrealized P/L": round(h.get("unrealized_pnl", 0), 2),
                "Unrealized P/L (KWD)": round(h.get("unrealized_pnl_kwd", 0), 2),
                "Cash Dividends": round(h.get("cash_dividends", 0), 2),
                "Reinvested Dividends": round(h.get("reinvested_dividends", 0), 2),
                "Bonus Shares": h.get("bonus_dividend_shares", 0),
                "Bonus Value": round(h.get("bonus_share_value", 0), 2),
                "Allocation %": round(h.get("weight_by_cost", 0) * 100, 2),
                "Dividend Yield %": round(h.get("dividend_yield_on_cost_pct", 0), 2),
                "Currency": h.get("currency", "KWD"),
                "P/E Ratio": h.get("pe_ratio") or "",
            })

    out = pd.DataFrame(rows)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="Holdings", index=False)
    buf.seek(0)

    today = date.today().isoformat()
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="holdings_{today}.xlsx"'},
    )


# ── Reset Account (delete all data) ─────────────────────────────────

@router.post("/reset-account")
async def reset_account(
    request: Request,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Hard-delete ALL portfolio data for the current user, essentially
    resetting the account to a fresh state.  The user record itself
    is preserved so they can log in again.

    Tables cleared (in dependency order):
      pfm_assets, pfm_liabilities, pfm_income_expenses, pfm_snapshots,
      position_snapshots, portfolio_snapshots, portfolio_cash,
      portfolio_transactions, ledger_entries, external_accounts,
      cash_deposits, transactions, stocks, portfolios, user_settings,
      securities_master, security_aliases
    """
    uid = current_user.user_id

    # Order matters: child rows first to avoid FK issues if any exist
    tables = [
        "pfm_assets",
        "pfm_liabilities",
        "pfm_income_expenses",
        "pfm_snapshots",
        "position_snapshots",
        "portfolio_snapshots",
        "portfolio_cash",
        "portfolio_transactions",
        "ledger_entries",
        "external_accounts",
        "cash_deposits",
        "transactions",
        "stocks",
        "portfolios",
        "user_settings",
        "securities_master",
        "security_aliases",
    ]

    deleted = {}
    for table in tables:
        try:
            exec_sql(f"DELETE FROM {table} WHERE user_id = ?", (uid,))
            deleted[table] = "cleared"
        except Exception as exc:
            logger.warning("reset-account: could not clear %s: %s", table, exc)
            deleted[table] = "skipped"

    log_event(
        ADMIN_ACTION,
        user_id=uid,
        resource_type="account",
        details={"action": "reset_account", "deleted": deleted},
        request=request,
    )

    logger.info("🗑️  Account reset for user %s — %s", uid, deleted)

    return {
        "status": "ok",
        "data": {
            "message": "Account reset successfully",
            "deleted": deleted,
        },
    }

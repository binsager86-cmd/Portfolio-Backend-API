"""
Cash Deposits API v1 — CRUD for cash deposits/withdrawals.
"""

import io
import time
import logging
from datetime import date
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, Query, Request
from starlette.responses import StreamingResponse

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.core.exceptions import NotFoundError, BadRequestError
from app.core.database import query_df, query_one, query_val, exec_sql
from app.services.fx_service import convert_to_kwd, PORTFOLIO_CCY
from app.services.audit_service import (
    log_event, CASH_CREATE, CASH_UPDATE, CASH_DELETE, CASH_RESTORE,
)
from app.schemas.cash import CashDepositCreate, CashDepositUpdate
from app.services.portfolio_service import PortfolioService
from app.api.v1.tracker import sync_deposit_to_snapshot

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cash", tags=["Cash Deposits"])


@router.get("/deposits")
async def list_deposits(
    portfolio: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    current_user: TokenData = Depends(get_current_user),
):
    """List cash deposits with optional portfolio filter and pagination."""
    conditions = ["user_id = ?", "COALESCE(is_deleted, 0) = 0"]
    params: list = [current_user.user_id]

    if portfolio:
        if portfolio not in PORTFOLIO_CCY:
            raise BadRequestError(f"Unknown portfolio '{portfolio}'")
        conditions.append("portfolio = ?")
        params.append(portfolio)

    where = " AND ".join(conditions)
    total = query_val(f"SELECT COUNT(*) FROM cash_deposits WHERE {where}", tuple(params))

    offset = (page - 1) * page_size
    sql = f"""
        SELECT id, user_id, portfolio, deposit_date, amount, currency,
               bank_name, source, notes, is_deleted, created_at
        FROM cash_deposits
        WHERE {where}
        ORDER BY deposit_date DESC, created_at DESC
        LIMIT ? OFFSET ?
    """
    params.extend([page_size, offset])

    df = query_df(sql, tuple(params))
    records = df.to_dict(orient="records") if not df.empty else []

    # Calculate KWD total
    total_kwd = sum(
        convert_to_kwd(float(r.get("amount", 0)), r.get("currency", "KWD"))
        for r in records
    )
    total_pages = max(1, (total + page_size - 1) // page_size)

    return {
        "status": "ok",
        "data": {
            "deposits": records,
            "count": len(records),
            "total_kwd": round(total_kwd, 3),
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total,
                "total_pages": total_pages,
            },
        },
    }


@router.get("/deposits/{deposit_id}")
async def get_deposit(
    deposit_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Get a single cash deposit by ID."""
    row = query_one(
        "SELECT * FROM cash_deposits WHERE id = ? AND user_id = ? AND COALESCE(is_deleted, 0) = 0",
        (deposit_id, current_user.user_id),
    )
    if not row:
        raise NotFoundError("CashDeposit", deposit_id)

    return {"status": "ok", "data": dict(row)}


@router.post("/deposits", status_code=201)
async def create_deposit(
    request: Request,
    body: CashDepositCreate,
    current_user: TokenData = Depends(get_current_user),
):
    """Create a new cash deposit/withdrawal."""
    dep = body

    if dep.portfolio not in PORTFOLIO_CCY:
        raise BadRequestError(f"Unknown portfolio '{dep.portfolio}'")

    now = int(time.time())

    # Auto-populate FX rate if not provided (matches Streamlit's get_current_fx_rate())
    fx_rate = dep.fx_rate_at_deposit
    if fx_rate is None:
        try:
            from app.services.fx_service import get_usd_kwd_rate
            fx_rate = get_usd_kwd_rate()
        except Exception:
            fx_rate = None

    exec_sql(
        """INSERT INTO cash_deposits
           (user_id, portfolio, deposit_date, amount, currency,
            bank_name, source, notes, description, comments,
            include_in_analysis, fx_rate_at_deposit, is_deleted, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
        (
            current_user.user_id, dep.portfolio, dep.deposit_date,
            dep.amount, dep.currency, dep.bank_name,
            dep.source, dep.notes, dep.description, dep.comments,
            dep.include_in_analysis, fx_rate, now,
        ),
    )

    new_id = query_val(
        "SELECT id FROM cash_deposits WHERE user_id = ? ORDER BY id DESC LIMIT 1",
        (current_user.user_id,),
    )

    log_event(
        CASH_CREATE,
        user_id=current_user.user_id,
        resource_type="cash_deposit",
        resource_id=new_id,
        details={"portfolio": dep.portfolio, "amount": dep.amount, "source": dep.source},
        request=request,
    )

    # Recalculate portfolio cash — deposit_delta ensures manual overrides
    # are incremented by the deposit amount instead of being skipped.
    svc = PortfolioService(current_user.user_id)
    svc.recalc_portfolio_cash(
        deposit_delta=dep.amount, delta_portfolio=dep.portfolio,
    )

    # Return fresh overview totals so frontend can update immediately
    unified = svc.get_total_portfolio_value()

    # Sync deposit to tracker snapshot for this date
    try:
        sync_deposit_to_snapshot(current_user.user_id, dep.deposit_date)
    except Exception as exc:
        logger.warning("snapshot sync after deposit create failed: %s", exc)

    return {
        "status": "ok",
        "data": {
            "id": new_id,
            "message": "Deposit created",
            "cash_balance": unified["cash_kwd"],
            "total_value": unified["total_value_kwd"],
        },
    }


@router.put("/deposits/{deposit_id}")
async def update_deposit(
    deposit_id: int,
    request: Request,
    body: CashDepositUpdate,
    current_user: TokenData = Depends(get_current_user),
):
    """Update a cash deposit."""
    # Read old amount + portfolio to compute delta for manual-override update
    old_row = query_one(
        "SELECT id, amount, portfolio FROM cash_deposits WHERE id = ? AND user_id = ? AND COALESCE(is_deleted, 0) = 0",
        (deposit_id, current_user.user_id),
    )
    if not old_row:
        raise NotFoundError("CashDeposit", deposit_id)

    old_amount = float(old_row["amount"] or 0)
    old_portfolio = old_row["portfolio"]

    updates = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if not updates:
        raise BadRequestError("No valid fields to update")

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    params = list(updates.values()) + [deposit_id, current_user.user_id]

    exec_sql(
        f"UPDATE cash_deposits SET {set_clause} WHERE id = ? AND user_id = ?",
        tuple(params),
    )

    new_amount = float(updates.get("amount", old_amount))
    new_portfolio = updates.get("portfolio", old_portfolio)

    log_event(
        CASH_UPDATE,
        user_id=current_user.user_id,
        resource_type="cash_deposit",
        resource_id=deposit_id,
        details={"updated_fields": list(updates.keys())},
        request=request,
    )

    # Recalculate portfolio cash — pass delta so manual overrides are updated
    svc = PortfolioService(current_user.user_id)
    if old_portfolio != new_portfolio:
        # Portfolio changed: subtract from old, add to new
        svc.recalc_portfolio_cash(deposit_delta=-old_amount, delta_portfolio=old_portfolio)
        svc.recalc_portfolio_cash(deposit_delta=new_amount, delta_portfolio=new_portfolio)
    else:
        # Same portfolio: delta = new_amount - old_amount
        svc.recalc_portfolio_cash(
            deposit_delta=new_amount - old_amount, delta_portfolio=new_portfolio,
        )

    # Return fresh overview totals so frontend can update immediately
    unified = svc.get_total_portfolio_value()

    # Sync deposit to tracker snapshot — re-read the date from DB
    try:
        dep_row = query_one(
            "SELECT deposit_date FROM cash_deposits WHERE id = ? AND user_id = ?",
            (deposit_id, current_user.user_id),
        )
        if dep_row:
            sync_deposit_to_snapshot(current_user.user_id, dep_row["deposit_date"])
    except Exception as exc:
        logger.warning("snapshot sync after deposit update failed: %s", exc)

    return {
        "status": "ok",
        "data": {
            "id": deposit_id,
            "message": "Deposit updated",
            "cash_balance": unified["cash_kwd"],
            "total_value": unified["total_value_kwd"],
        },
    }


@router.delete("/deposits/{deposit_id}")
async def delete_deposit(
    deposit_id: int,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
):
    """Soft-delete a cash deposit."""
    # Read amount + portfolio before deleting so we can subtract from manual override
    existing = query_one(
        "SELECT id, amount, portfolio FROM cash_deposits WHERE id = ? AND user_id = ? AND COALESCE(is_deleted, 0) = 0",
        (deposit_id, current_user.user_id),
    )
    if not existing:
        raise NotFoundError("CashDeposit", deposit_id)

    del_amount = float(existing["amount"] or 0)
    del_portfolio = existing["portfolio"]

    now = int(time.time())
    exec_sql(
        "UPDATE cash_deposits SET is_deleted = 1, deleted_at = ? WHERE id = ? AND user_id = ?",
        (now, deposit_id, current_user.user_id),
    )

    log_event(
        CASH_DELETE,
        user_id=current_user.user_id,
        resource_type="cash_deposit",
        resource_id=deposit_id,
        request=request,
    )

    # Recalculate portfolio cash — subtract deleted amount from manual override
    svc = PortfolioService(current_user.user_id)
    svc.recalc_portfolio_cash(
        deposit_delta=-del_amount, delta_portfolio=del_portfolio,
    )

    # Return fresh overview totals so frontend can update immediately
    unified = svc.get_total_portfolio_value()

    # Sync deposit to tracker snapshot — re-read the date from DB
    try:
        dep_row = query_one(
            "SELECT deposit_date FROM cash_deposits WHERE id = ? AND user_id = ?",
            (deposit_id, current_user.user_id),
        )
        if dep_row:
            sync_deposit_to_snapshot(current_user.user_id, dep_row["deposit_date"])
    except Exception as exc:
        logger.warning("snapshot sync after deposit delete failed: %s", exc)

    return {
        "status": "ok",
        "data": {
            "id": deposit_id,
            "message": "Deposit deleted",
            "cash_balance": unified["cash_kwd"],
            "total_value": unified["total_value_kwd"],
        },
    }


@router.post("/deposits/{deposit_id}/restore")
async def restore_deposit(
    deposit_id: int,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
):
    """Restore a soft-deleted deposit."""
    # Read amount + portfolio so we can add back to manual override
    existing = query_one(
        "SELECT id, amount, portfolio FROM cash_deposits WHERE id = ? AND user_id = ? AND is_deleted = 1",
        (deposit_id, current_user.user_id),
    )
    if not existing:
        raise NotFoundError("CashDeposit", deposit_id)

    restore_amount = float(existing["amount"] or 0)
    restore_portfolio = existing["portfolio"]

    exec_sql(
        "UPDATE cash_deposits SET is_deleted = 0, deleted_at = NULL WHERE id = ? AND user_id = ?",
        (deposit_id, current_user.user_id),
    )

    log_event(
        CASH_RESTORE,
        user_id=current_user.user_id,
        resource_type="cash_deposit",
        resource_id=deposit_id,
        request=request,
    )

    # Recalculate portfolio cash — add restored amount back to manual override
    svc = PortfolioService(current_user.user_id)
    svc.recalc_portfolio_cash(
        deposit_delta=restore_amount, delta_portfolio=restore_portfolio,
    )

    # Return fresh overview totals so frontend can update immediately
    unified = svc.get_total_portfolio_value()

    # Sync deposit to tracker snapshot — re-read the date from DB
    try:
        dep_row = query_one(
            "SELECT deposit_date FROM cash_deposits WHERE id = ? AND user_id = ?",
            (deposit_id, current_user.user_id),
        )
        if dep_row:
            sync_deposit_to_snapshot(current_user.user_id, dep_row["deposit_date"])
    except Exception as exc:
        logger.warning("snapshot sync after deposit restore failed: %s", exc)

    return {
        "status": "ok",
        "data": {
            "id": deposit_id,
            "message": "Deposit restored",
            "cash_balance": unified["cash_kwd"],
            "total_value": unified["total_value_kwd"],
        },
    }


# ── Export endpoint ──────────────────────────────────────────────────

@router.get("/deposits-export")
async def deposits_export(current_user: TokenData = Depends(get_current_user)):
    """
    Export all cash deposits/withdrawals as Excel (.xlsx).
    """
    user_id = current_user.user_id

    sql = """
        SELECT
            d.id,
            d.deposit_date AS date,
            COALESCE(d.portfolio, 'KFH') AS portfolio,
            COALESCE(d.source, 'deposit') AS type,
            d.amount,
            COALESCE(d.currency, 'KWD') AS currency,
            d.bank_name,
            d.notes
        FROM cash_deposits d
        WHERE d.user_id = ?
          AND COALESCE(d.is_deleted, 0) = 0
        ORDER BY d.deposit_date DESC, d.id DESC
    """
    df = query_df(sql, (user_id,))

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Cash Deposits", index=False)
    buf.seek(0)

    today = date.today().isoformat()
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="deposits_{today}.xlsx"'},
    )

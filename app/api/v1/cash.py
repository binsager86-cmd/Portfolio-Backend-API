"""
Cash Deposits API v1 — CRUD for cash deposits/withdrawals.
"""

import io
import math
import time
import logging
from datetime import date
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from pydantic import ValidationError
from starlette.responses import StreamingResponse

from app.api.deps import get_current_user
from app.core.database import get_db, query_df, query_one, query_val, exec_sql, exec_sql_returning_id, transaction
from app.core.exceptions import NotFoundError, BadRequestError
from app.core.repositories.cash import CashDepositRepository
from app.core.security import TokenData
from app.schemas.cash import CashDepositCreate, CashDepositUpdate
from app.services.audit_service import (
    log_event, CASH_CREATE, CASH_UPDATE, CASH_DELETE, CASH_RESTORE,
)
from app.services.fx_service import DEFAULT_USD_TO_KWD, convert_to_kwd, PORTFOLIO_CCY
from app.services.portfolio_service import PortfolioService
from app.api.v1.tracker import recalculate_all_snapshots, sync_deposit_to_snapshot
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cash", tags=["Cash Deposits"])


def _stored_fx(row_or_fx) -> float:
    try:
        fx = float(row_or_fx or 0)
    except (TypeError, ValueError):
        fx = 0.0
    return fx if fx > 0 else DEFAULT_USD_TO_KWD


def _cash_amount_in_portfolio_currency(
    amount: float,
    currency: str,
    portfolio: str,
    source: str,
    fx_rate_at_deposit: Optional[float],
) -> float:
    sign = -1.0 if (source or "deposit").lower() == "withdrawal" else 1.0
    src_ccy = (currency or "KWD").upper()
    dst_ccy = PORTFOLIO_CCY.get(portfolio, "KWD").upper()
    amount = float(amount or 0)
    fx = _stored_fx(fx_rate_at_deposit)
    if src_ccy == dst_ccy:
        converted = amount
    elif src_ccy == "USD" and dst_ccy == "KWD":
        converted = amount * fx
    elif src_ccy == "KWD" and dst_ccy == "USD":
        converted = amount / fx
    else:
        converted = amount
    return sign * converted


def _cash_amount_kwd(amount: float, currency: str, source: str, fx_rate_at_deposit: Optional[float]) -> float:
    sign = -1.0 if (source or "deposit").lower() == "withdrawal" else 1.0
    if (currency or "KWD").upper() == "USD":
        return sign * float(amount or 0) * _stored_fx(fx_rate_at_deposit)
    return sign * convert_to_kwd(float(amount or 0), currency or "KWD")


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

    totals_df = query_df(
        f"""SELECT amount, currency, source, fx_rate_at_deposit
            FROM cash_deposits
            WHERE {where}""",
        tuple(params[:-2]),
    )
    total_deposits_kwd = 0.0
    total_withdrawals_kwd = 0.0
    if not totals_df.empty:
        for row in totals_df.to_dict(orient="records"):
            signed_kwd = _cash_amount_kwd(
                float(row.get("amount") or 0),
                row.get("currency") or "KWD",
                row.get("source") or "deposit",
                row.get("fx_rate_at_deposit"),
            )
            if signed_kwd < 0:
                total_withdrawals_kwd += signed_kwd
            else:
                total_deposits_kwd += signed_kwd
    total_kwd = total_deposits_kwd + total_withdrawals_kwd
    total_pages = max(1, (total + page_size - 1) // page_size)

    return {
        "status": "ok",
        "data": {
            "deposits": records,
            "count": len(records),
            "total_kwd": round(total_kwd, 3),
            "total_deposits_kwd": round(total_deposits_kwd, 3),
            "total_withdrawals_kwd": round(total_withdrawals_kwd, 3),
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
    db: Session = Depends(get_db),
):
    """Get a single cash deposit by ID."""
    # [B-2] Use ORM repository instead of raw query_one()
    repo = CashDepositRepository(db)
    deposit = repo.get_active(deposit_id, current_user.user_id)
    if not deposit:
        raise NotFoundError("CashDeposit", deposit_id)
    return {
        "status": "ok",
        "data": {
            c.name: getattr(deposit, c.name)
            for c in deposit.__table__.columns
        },
    }


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

    # Recalculate portfolio cash — deposit_delta ensures manual overrides
    # are incremented by the deposit amount instead of being skipped.
    # Withdrawals subtract from cash, deposits add.
    effective_delta = _cash_amount_in_portfolio_currency(
        dep.amount, dep.currency, dep.portfolio, dep.source or "deposit", fx_rate,
    )
    svc = PortfolioService(current_user.user_id)
    with transaction() as conn:
        new_id = exec_sql_returning_id(
            """INSERT INTO cash_deposits
               (user_id, portfolio, deposit_date, amount, currency, bank_name,
                source, notes, description, comments, include_in_analysis,
                fx_rate_at_deposit, is_deleted, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
            (current_user.user_id, dep.portfolio, dep.deposit_date, dep.amount,
             dep.currency, dep.bank_name, dep.source, dep.notes, dep.description,
             dep.comments, dep.include_in_analysis, fx_rate, now),
        )

        log_event(
            CASH_CREATE,
            user_id=current_user.user_id,
            resource_type="cash_deposit",
            resource_id=new_id,
            details={"portfolio": dep.portfolio, "amount": dep.amount, "source": dep.source},
            request=request,
        )

        svc.recalc_portfolio_cash(
            deposit_delta=effective_delta, delta_portfolio=dep.portfolio, conn=conn,
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
    deposit = query_one(
        "SELECT * FROM cash_deposits WHERE id = ? AND user_id = ? AND COALESCE(is_deleted, 0) = 0",
        (deposit_id, current_user.user_id),
    )
    if not deposit:
        raise NotFoundError("CashDeposit", deposit_id)

    old_amount = float(deposit["amount"] or 0)
    old_portfolio = deposit["portfolio"]
    old_source = (deposit["source"] or "deposit").lower()
    old_currency = deposit["currency"] or "KWD"
    old_fx = deposit["fx_rate_at_deposit"]
    old_date = deposit["deposit_date"]

    updates = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if not updates:
        raise BadRequestError("No valid fields to update")
    if "portfolio" in updates and updates["portfolio"] not in PORTFOLIO_CCY:
        raise BadRequestError(f"Unknown portfolio '{updates['portfolio']}'")

    new_amount = float(updates.get("amount", old_amount))
    new_portfolio = updates.get("portfolio", old_portfolio)
    new_source = (updates.get("source", old_source) or "deposit").lower()
    new_currency = updates.get("currency", old_currency)
    new_fx = updates.get("fx_rate_at_deposit", old_fx)
    new_date = updates.get("deposit_date", old_date)

    # Compute effective cash amounts (withdrawals are negative)
    old_effective = _cash_amount_in_portfolio_currency(
        old_amount, old_currency, old_portfolio, old_source, old_fx,
    )
    new_effective = _cash_amount_in_portfolio_currency(
        new_amount, new_currency, new_portfolio, new_source, new_fx,
    )

    # Recalculate portfolio cash — pass delta so manual overrides are updated
    svc = PortfolioService(current_user.user_id)
    set_clause = ", ".join(f"{field} = ?" for field in updates)
    with transaction() as conn:
        exec_sql(
            f"UPDATE cash_deposits SET {set_clause} WHERE id = ? AND user_id = ? AND COALESCE(is_deleted, 0) = 0",
            tuple(updates.values()) + (deposit_id, current_user.user_id),
        )
        log_event(
            CASH_UPDATE,
            user_id=current_user.user_id,
            resource_type="cash_deposit",
            resource_id=deposit_id,
            details={"updated_fields": list(updates.keys())},
            request=request,
        )
        if old_portfolio != new_portfolio:
            # Portfolio changed: reverse old effect, apply new effect
            svc.recalc_portfolio_cash(deposit_delta=-old_effective, delta_portfolio=old_portfolio, conn=conn)
            svc.recalc_portfolio_cash(deposit_delta=new_effective, delta_portfolio=new_portfolio, conn=conn)
        else:
            # Same portfolio: delta = new effective - old effective
            svc.recalc_portfolio_cash(
                deposit_delta=new_effective - old_effective, delta_portfolio=new_portfolio, conn=conn,
            )
    # Return fresh overview totals so frontend can update immediately
    unified = svc.get_total_portfolio_value()

    # Sync deposit to tracker snapshot using ORM-fetched date
    try:
        for dep_date in sorted({old_date, new_date}):
            sync_deposit_to_snapshot(current_user.user_id, dep_date)
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
    existing = query_one(
        "SELECT * FROM cash_deposits WHERE id = ? AND user_id = ? AND COALESCE(is_deleted, 0) = 0",
        (deposit_id, current_user.user_id),
    )
    if not existing:
        raise NotFoundError("CashDeposit", deposit_id)

    del_amount = float(existing["amount"] or 0)
    del_portfolio = existing["portfolio"]
    del_source = (existing["source"] or "deposit").lower()
    del_currency = existing["currency"] or "KWD"
    del_fx = existing["fx_rate_at_deposit"]
    del_date = existing["deposit_date"]

    now = int(time.time())
    del_delta = -_cash_amount_in_portfolio_currency(
        del_amount, del_currency, del_portfolio, del_source, del_fx,
    )
    svc = PortfolioService(current_user.user_id)
    with transaction() as conn:
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

        # Deleting a deposit subtracts; deleting a withdrawal adds back.
        svc.recalc_portfolio_cash(
            deposit_delta=del_delta, delta_portfolio=del_portfolio, conn=conn,
        )

    # Return fresh overview totals so frontend can update immediately
    unified = svc.get_total_portfolio_value()

    # Sync deposit to tracker snapshot using ORM-fetched date
    try:
        sync_deposit_to_snapshot(current_user.user_id, del_date)
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
    existing = query_one(
        "SELECT * FROM cash_deposits WHERE id = ? AND user_id = ? AND is_deleted = 1",
        (deposit_id, current_user.user_id),
    )
    if not existing:
        raise NotFoundError("CashDeposit", deposit_id)

    restore_amount = float(existing["amount"] or 0)
    restore_portfolio = existing["portfolio"]
    restore_source = (existing["source"] or "deposit").lower()
    restore_currency = existing["currency"] or "KWD"
    restore_fx = existing["fx_rate_at_deposit"]
    restore_date = existing["deposit_date"]

    restore_delta = _cash_amount_in_portfolio_currency(
        restore_amount, restore_currency, restore_portfolio, restore_source, restore_fx,
    )
    svc = PortfolioService(current_user.user_id)
    with transaction() as conn:
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

        # Restoring a deposit adds; restoring a withdrawal subtracts.
        svc.recalc_portfolio_cash(
            deposit_delta=restore_delta, delta_portfolio=restore_portfolio, conn=conn,
        )

    # Return fresh overview totals so frontend can update immediately
    unified = svc.get_total_portfolio_value()

    # Sync deposit to tracker snapshot using ORM-fetched date
    try:
        sync_deposit_to_snapshot(current_user.user_id, restore_date)
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


# ── Upload / Import endpoint ────────────────────────────────────────

@router.post("/deposits-import")
async def deposits_import(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Query("merge", description="'merge' (append) or 'replace' (delete existing first)"),
    current_user: TokenData = Depends(get_current_user),
):
    """
    Bulk import cash deposits from an Excel file (.xlsx).

    Expected columns (case-insensitive, underscores/spaces accepted):
      - **deposit_date** / date  (required, YYYY-MM-DD or Excel date)
      - **amount**  (required, positive number)
      - **currency**  (optional, default KWD)
      - **portfolio**  (optional, default KFH)
      - **source**  (optional, 'deposit' or 'withdrawal', default 'deposit')
      - **bank_name**  (optional)
      - **description**  (optional)
      - **comments**  (optional)
      - **notes**  (optional)
      - **include_in_analysis**  (optional, Yes/No/1/0, default Yes)

    Modes:
      - merge: Append new deposits alongside existing ones.
      - replace: Delete all existing deposits first, then import.

    Returns per-row summary of imported / skipped / errors.
    """
    if mode not in ("merge", "replace"):
        raise BadRequestError("mode must be 'merge' or 'replace'")

    if not file.filename or not file.filename.lower().endswith(".xlsx"):
        raise BadRequestError("File must be an Excel file (.xlsx)")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise BadRequestError("File too large (max 10 MB)")

    user_id = current_user.user_id

    try:
        xls = pd.ExcelFile(io.BytesIO(contents))
        # Use first sheet
        sheet = xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    except Exception as exc:
        raise BadRequestError(f"Failed to read Excel file: {exc}")

    # Normalize column names
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Validate required columns
    date_col = next(
        (c for c in ("deposit_date", "date", "trade_date") if c in df.columns), None
    )
    amount_col = "amount" if "amount" in df.columns else None

    if not date_col or not amount_col:
        raise BadRequestError(
            "Excel must contain at least 'deposit_date' (or 'date') and 'amount' columns. "
            f"Found columns: {list(df.columns)}"
        )

    now = int(time.time())
    imported = 0
    skipped = 0
    errors: list = []
    affected_dates: set[str] = set()
    snapshot_warning: Optional[str] = None

    # Auto FX rate
    fx_rate = None
    try:
        from app.services.fx_service import get_usd_kwd_rate
        fx_rate = get_usd_kwd_rate()
    except Exception:
        pass

    def _process_rows() -> None:
        nonlocal imported, skipped, errors, affected_dates
        for idx, row in df.iterrows():
            try:
                # Parse date
                raw_date = row.get(date_col)
                if raw_date is None or (isinstance(raw_date, float) and pd.isna(raw_date)):
                    skipped += 1
                    errors.append({"row": int(idx) + 2, "error": "Empty date"})
                    continue

                if hasattr(raw_date, "strftime"):
                    dep_date = raw_date.strftime("%Y-%m-%d")
                else:
                    s = str(raw_date).strip()
                    if not s or s.lower() in ("nan", "nat"):
                        skipped += 1
                        continue
                    try:
                        dep_date = pd.to_datetime(s).strftime("%Y-%m-%d")
                    except Exception:
                        dep_date = s

                # Parse amount
                raw_amount = row.get(amount_col)
                if raw_amount is None or (isinstance(raw_amount, float) and pd.isna(raw_amount)):
                    skipped += 1
                    errors.append({"row": int(idx) + 2, "error": "Empty amount"})
                    continue
                amount = float(raw_amount)
                if not math.isfinite(amount):
                    skipped += 1
                    errors.append({"row": int(idx) + 2, "error": "Amount must be finite"})
                    continue
                if amount == 0:
                    skipped += 1
                    continue

                # Optional fields with sensible defaults
                def _cell_str(row_data: pd.Series, col: str, default: str = "") -> str:
                    val = row_data.get(col)
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        return default
                    s = str(val).strip()
                    return default if s.lower() in ("nan", "nat") else s

                portfolio = _cell_str(row, "portfolio", "KFH")
                if portfolio not in PORTFOLIO_CCY:
                    skipped += 1
                    errors.append({"row": int(idx) + 2, "error": f"Unsupported portfolio: {portfolio}"})
                    continue
                currency = _cell_str(row, "currency", "KWD").upper()
                if currency not in set(PORTFOLIO_CCY.values()):
                    skipped += 1
                    errors.append({"row": int(idx) + 2, "error": f"Unsupported currency: {currency}"})
                    continue
                source = _cell_str(row, "source", "deposit").lower()
                if source not in ("deposit", "withdrawal"):
                    skipped += 1
                    errors.append({"row": int(idx) + 2, "error": f"Unsupported source: {source}"})
                    continue
                bank_name = _cell_str(row, "bank_name")
                description = _cell_str(row, "description")
                comments = _cell_str(row, "comments")
                notes = _cell_str(row, "notes")

                include_raw = _cell_str(row, "include_in_analysis", "1")
                include_in_analysis = 0 if include_raw.lower() in ("0", "no", "false", "record") else 1

                row_fx = row.get("fx_rate_at_deposit") if "fx_rate_at_deposit" in df.columns else fx_rate
                fx_for_row = None if row_fx is None or (isinstance(row_fx, float) and pd.isna(row_fx)) else float(row_fx)
                if fx_for_row is not None and (not math.isfinite(fx_for_row) or fx_for_row <= 0):
                    skipped += 1
                    errors.append({"row": int(idx) + 2, "error": "FX rate must be positive and finite"})
                    continue

                try:
                    CashDepositCreate(
                        portfolio=portfolio,
                        deposit_date=dep_date,
                        amount=amount,
                        currency=currency,
                        bank_name=bank_name or None,
                        source=source,
                        notes=notes or None,
                        description=description or None,
                        comments=comments or None,
                        include_in_analysis=include_in_analysis,
                        fx_rate_at_deposit=fx_for_row,
                    )
                except (ValidationError, ValueError) as exc:
                    skipped += 1
                    errors.append({"row": int(idx) + 2, "error": str(exc).splitlines()[0][:120]})
                    continue

                exec_sql(
                    """INSERT INTO cash_deposits
                       (user_id, portfolio, deposit_date, amount, currency,
                        bank_name, source, deposit_type, notes,
                        description, comments, include_in_analysis,
                        fx_rate_at_deposit, is_deleted, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                    (
                        user_id, portfolio, dep_date, amount, currency,
                        bank_name, source, source, notes,
                        description, comments, include_in_analysis,
                        fx_for_row, now,
                    ),
                )
                imported += 1

                if include_in_analysis:
                    affected_dates.add(dep_date)

            except Exception as exc:
                skipped += 1
                errors.append({"row": int(idx) + 2, "error": str(exc)[:120]})

    if mode == "replace":
        with transaction():
            exec_sql("DELETE FROM cash_deposits WHERE user_id = ?", (user_id,))
            _process_rows()
            if errors:
                raise BadRequestError("Replace import failed; existing deposits were preserved.")
            if imported <= 0:
                raise BadRequestError("Replace import contained no importable deposits; existing deposits were preserved.")
    else:
        _process_rows()

    # Recalculate portfolio cash
    if imported > 0:
        try:
            svc = PortfolioService(user_id)
            svc.recalc_portfolio_cash()
        except Exception as exc:
            logger.warning("recalc_portfolio_cash after deposit import: %s", exc)

        try:
            if mode == "replace":
                recalculate_all_snapshots(user_id)
            else:
                for dep_date in sorted(affected_dates):
                    sync_deposit_to_snapshot(user_id, dep_date)
        except Exception as exc:
            snapshot_warning = "Deposits imported, but tracker recalculation must be retried"
            logger.warning("snapshot sync after deposit import failed: %s", exc)

    log_event(
        CASH_CREATE,
        user_id=user_id,
        resource_type="cash_deposit_import",
        resource_id=0,
        details={"imported": imported, "skipped": skipped, "mode": mode},
        request=request,
    )

    return {
        "status": "ok",
        "data": {
            "imported": imported,
            "skipped": skipped,
            "total_rows": len(df),
            "errors": errors[:50],
            "mode": mode,
            "snapshot_sync": "failed" if snapshot_warning else "ok",
            "warning": snapshot_warning,
        },
    }


# ── Download sample template ────────────────────────────────────────

@router.get("/deposits-template")
async def deposits_template():
    """
    Download a sample Excel template for cash deposit uploads.
    """
    sample = pd.DataFrame([
        {
            "deposit_date": date.today().isoformat(),
            "amount": 1000.0,
            "currency": "KWD",
            "portfolio": "KFH",
            "source": "deposit",
            "bank_name": "KFH Bank",
            "description": "Monthly Salary",
            "comments": "",
            "notes": "",
            "include_in_analysis": "Yes",
        },
        {
            "deposit_date": date.today().isoformat(),
            "amount": 500.0,
            "currency": "USD",
            "portfolio": "USA",
            "source": "deposit",
            "bank_name": "US Bank",
            "description": "Transfer",
            "comments": "",
            "notes": "",
            "include_in_analysis": "Yes",
        },
        {
            "deposit_date": date.today().isoformat(),
            "amount": 200.0,
            "currency": "KWD",
            "portfolio": "KFH",
            "source": "withdrawal",
            "bank_name": "KFH Bank",
            "description": "Cash withdrawal",
            "comments": "",
            "notes": "",
            "include_in_analysis": "Yes",
        },
    ])

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        sample.to_excel(writer, sheet_name="Cash Deposits", index=False)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="cash_deposits_template.xlsx"'},
    )

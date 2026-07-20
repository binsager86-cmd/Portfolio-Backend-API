"""
PFM (Personal Finance Management) API v1 — net-worth snapshots.
"""

import time
import logging
from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.core.exceptions import NotFoundError, BadRequestError
from app.core.database import query_df, query_one, query_val, exec_sql, exec_sql_returning_id, get_connection, transaction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pfm", tags=["PFM"])


class PFMAssetCreate(BaseModel):
    asset_type: str = Field(..., min_length=1, max_length=80)
    category: str = Field(..., min_length=1, max_length=80)
    name: str = Field(..., min_length=1, max_length=160)
    quantity: Optional[float] = Field(None, ge=0, allow_inf_nan=False)
    price: Optional[float] = Field(None, ge=0, allow_inf_nan=False)
    currency: str = Field("KWD", max_length=10)
    value_kwd: float = Field(ge=0, allow_inf_nan=False)


class PFMLiabilityCreate(BaseModel):
    category: str = Field(..., min_length=1, max_length=100)
    amount_kwd: float = Field(ge=0, allow_inf_nan=False)
    is_current: bool = False
    is_long_term: bool = False


class PFMIncomeExpenseCreate(BaseModel):
    kind: Literal["income", "expense"]
    category: str = Field(..., min_length=1, max_length=100)
    monthly_amount: float = Field(ge=0, allow_inf_nan=False)
    is_finance_cost: bool = False
    is_gna: bool = False
    sort_order: int = Field(0, ge=0, le=10_000)


class PFMSnapshotCreate(BaseModel):
    snapshot_date: date
    notes: Optional[str] = Field(None, max_length=2000)
    assets: list[PFMAssetCreate] = Field(default_factory=list, max_length=500)
    liabilities: list[PFMLiabilityCreate] = Field(default_factory=list, max_length=500)
    income_expenses: list[PFMIncomeExpenseCreate] = Field(default_factory=list, max_length=500)


@router.get("/snapshots")
async def list_pfm_snapshots(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: TokenData = Depends(get_current_user),
):
    """List PFM net-worth snapshots (paginated)."""
    total = query_val(
        "SELECT COUNT(*) FROM pfm_snapshots WHERE user_id = ?",
        (current_user.user_id,),
    )

    offset = (page - 1) * page_size
    df = query_df(
        """
        SELECT id, snapshot_date, notes, total_assets,
               total_liabilities, net_worth, created_at
        FROM pfm_snapshots
        WHERE user_id = ?
        ORDER BY snapshot_date DESC
        LIMIT ? OFFSET ?
        """,
        (current_user.user_id, page_size, offset),
    )

    records = df.to_dict(orient="records") if not df.empty else []
    total_pages = max(1, (total + page_size - 1) // page_size)

    return {
        "status": "ok",
        "data": {
            "snapshots": records,
            "count": len(records),
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total,
                "total_pages": total_pages,
            },
        },
    }


@router.get("/snapshots/{snapshot_id}")
async def get_pfm_snapshot(
    snapshot_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Get a PFM snapshot with full detail (assets, liabilities, income/expenses)."""
    row = query_one(
        "SELECT * FROM pfm_snapshots WHERE id = ? AND user_id = ?",
        (snapshot_id, current_user.user_id),
    )
    if not row:
        raise NotFoundError("PFM Snapshot", snapshot_id)

    snapshot = dict(row)

    # Fetch related items
    assets_df = query_df(
        "SELECT * FROM pfm_assets WHERE snapshot_id = ? AND user_id = ?",
        (snapshot_id, current_user.user_id),
    )
    liabilities_df = query_df(
        "SELECT * FROM pfm_liabilities WHERE snapshot_id = ? AND user_id = ?",
        (snapshot_id, current_user.user_id),
    )
    ie_df = query_df(
        "SELECT * FROM pfm_income_expenses WHERE snapshot_id = ? AND user_id = ?",
        (snapshot_id, current_user.user_id),
    )

    snapshot["assets"] = assets_df.to_dict(orient="records") if not assets_df.empty else []
    snapshot["liabilities"] = liabilities_df.to_dict(orient="records") if not liabilities_df.empty else []
    snapshot["income_expenses"] = ie_df.to_dict(orient="records") if not ie_df.empty else []

    return {"status": "ok", "data": snapshot}


@router.post("/snapshots", status_code=201)
async def create_pfm_snapshot(
    body: PFMSnapshotCreate,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Create a new PFM net-worth snapshot with assets, liabilities, and income/expenses.
    """
    snapshot_date = body.snapshot_date.isoformat()
    assets = body.assets
    liabilities = body.liabilities
    income_expenses = body.income_expenses

    total_assets = sum(float(a.value_kwd) for a in assets)
    total_liabilities = sum(float(l.amount_kwd) for l in liabilities)
    net_worth = total_assets - total_liabilities

    now = int(time.time())

    with transaction():
        # Insert snapshot
        snapshot_id = exec_sql_returning_id(
            """INSERT INTO pfm_snapshots
               (user_id, snapshot_date, notes, total_assets, total_liabilities, net_worth, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (current_user.user_id, snapshot_date, body.notes,
             total_assets, total_liabilities, net_worth, now),
        )

        # Insert assets
        for a in assets:
            exec_sql(
                """INSERT INTO pfm_assets
                   (snapshot_id, user_id, asset_type, category, name,
                    quantity, price, currency, value_kwd, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (snapshot_id, current_user.user_id,
                 a.asset_type, a.category, a.name,
                 a.quantity, a.price, a.currency,
                 a.value_kwd, now),
            )

        # Insert liabilities
        for l in liabilities:
            exec_sql(
                """INSERT INTO pfm_liabilities
                   (snapshot_id, user_id, category, amount_kwd,
                    is_current, is_long_term, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (snapshot_id, current_user.user_id,
                 l.category, l.amount_kwd,
                 int(l.is_current),
                 int(l.is_long_term), now),
            )

        # Insert income/expenses
        for ie in income_expenses:
            exec_sql(
                """INSERT INTO pfm_income_expenses
                   (snapshot_id, user_id, kind, category, monthly_amount,
                    is_finance_cost, is_gna, sort_order, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (snapshot_id, current_user.user_id,
                 ie.kind, ie.category,
                 ie.monthly_amount,
                 int(ie.is_finance_cost),
                 int(ie.is_gna),
                 ie.sort_order, now),
            )

    return {
        "status": "ok",
        "data": {
            "id": snapshot_id,
            "net_worth": net_worth,
            "message": "PFM snapshot created",
        },
    }


@router.delete("/snapshots/{snapshot_id}")
async def delete_pfm_snapshot(
    snapshot_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """Delete a PFM snapshot and its child records."""
    existing = query_one(
        "SELECT id FROM pfm_snapshots WHERE id = ? AND user_id = ?",
        (snapshot_id, current_user.user_id),
    )
    if not existing:
        raise NotFoundError("PFM Snapshot", snapshot_id)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM pfm_income_expenses WHERE snapshot_id = ? AND user_id = ?",
                     (snapshot_id, current_user.user_id))
        cur.execute("DELETE FROM pfm_liabilities WHERE snapshot_id = ? AND user_id = ?",
                     (snapshot_id, current_user.user_id))
        cur.execute("DELETE FROM pfm_assets WHERE snapshot_id = ? AND user_id = ?",
                     (snapshot_id, current_user.user_id))
        cur.execute("DELETE FROM pfm_snapshots WHERE id = ? AND user_id = ?",
                     (snapshot_id, current_user.user_id))
        conn.commit()

    return {"status": "ok", "data": {"id": snapshot_id, "message": "PFM snapshot deleted"}}

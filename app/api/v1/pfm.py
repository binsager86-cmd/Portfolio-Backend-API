"""
PFM (Personal Finance Management) API v1 — net-worth snapshots.
"""

import time
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.core.exceptions import NotFoundError, BadRequestError
from app.core.database import query_df, query_one, query_val, exec_sql, get_connection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pfm", tags=["PFM"])


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
    body: dict,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Create a new PFM net-worth snapshot with assets, liabilities, and income/expenses.
    """
    snapshot_date = body.get("snapshot_date")
    if not snapshot_date:
        raise BadRequestError("snapshot_date is required")

    notes = body.get("notes")
    assets = body.get("assets", [])
    liabilities = body.get("liabilities", [])
    income_expenses = body.get("income_expenses", [])

    total_assets = sum(float(a.get("value_kwd", 0)) for a in assets)
    total_liabilities = sum(float(l.get("amount_kwd", 0)) for l in liabilities)
    net_worth = total_assets - total_liabilities

    now = int(time.time())

    with get_connection() as conn:
        cur = conn.cursor()

        # Insert snapshot
        cur.execute(
            """INSERT INTO pfm_snapshots
               (user_id, snapshot_date, notes, total_assets, total_liabilities, net_worth, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (current_user.user_id, snapshot_date, notes,
             total_assets, total_liabilities, net_worth, now),
        )
        snapshot_id = cur.lastrowid

        # Insert assets
        for a in assets:
            cur.execute(
                """INSERT INTO pfm_assets
                   (snapshot_id, user_id, asset_type, category, name,
                    quantity, price, currency, value_kwd, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (snapshot_id, current_user.user_id,
                 a.get("asset_type"), a.get("category"), a.get("name"),
                 a.get("quantity"), a.get("price"), a.get("currency", "KWD"),
                 a.get("value_kwd", 0), now),
            )

        # Insert liabilities
        for l in liabilities:
            cur.execute(
                """INSERT INTO pfm_liabilities
                   (snapshot_id, user_id, category, amount_kwd,
                    is_current, is_long_term, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (snapshot_id, current_user.user_id,
                 l.get("category"), l.get("amount_kwd", 0),
                 int(l.get("is_current", False)),
                 int(l.get("is_long_term", False)), now),
            )

        # Insert income/expenses
        for ie in income_expenses:
            cur.execute(
                """INSERT INTO pfm_income_expenses
                   (snapshot_id, user_id, kind, category, monthly_amount,
                    is_finance_cost, is_gna, sort_order, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (snapshot_id, current_user.user_id,
                 ie.get("kind"), ie.get("category"),
                 ie.get("monthly_amount", 0),
                 int(ie.get("is_finance_cost", False)),
                 int(ie.get("is_gna", False)),
                 ie.get("sort_order", 0), now),
            )

        conn.commit()

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

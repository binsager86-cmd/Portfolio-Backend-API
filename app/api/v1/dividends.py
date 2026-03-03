"""
Dividends Tracker API v1 — dividend summary, per-stock breakdown, yield calculation.

Mirrors the Streamlit ``ui_dividends_tracker()`` logic:
  - All dividends come from the *transactions* table
    (cash_dividend, bonus_shares, reinvested_dividend columns).
  - Cash amounts are converted to KWD using the FX service.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.core.database import query_df
from app.services.fx_service import convert_to_kwd

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dividends", tags=["Dividends"])


# ── All dividends (flat list) ────────────────────────────────────────

@router.get("")
async def list_dividends(
    stock_symbol: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    current_user: TokenData = Depends(get_current_user),
):
    """
    List all dividend entries (transactions with cash_dividend > 0,
    bonus_shares > 0, or reinvested_dividend > 0).

    Returns KWD-converted totals.
    """
    conditions = [
        "t.user_id = ?",
        "COALESCE(t.is_deleted, 0) = 0",
        """(
            COALESCE(t.cash_dividend, 0) > 0
            OR COALESCE(t.bonus_shares, 0) > 0
            OR COALESCE(t.reinvested_dividend, 0) > 0
        )""",
    ]
    params: list = [current_user.user_id]

    if stock_symbol:
        conditions.append("TRIM(t.stock_symbol) = ?")
        params.append(stock_symbol.strip())

    where = " AND ".join(conditions)

    # Full query (with stock currency join)
    sql = f"""
        SELECT
            t.id,
            t.stock_symbol,
            t.portfolio,
            t.txn_date,
            COALESCE(t.cash_dividend, 0)        AS cash_dividend,
            COALESCE(t.bonus_shares, 0)          AS bonus_shares,
            COALESCE(t.reinvested_dividend, 0)   AS reinvested_dividend,
            COALESCE(s.currency, 'KWD')          AS currency,
            t.notes
        FROM transactions t
        LEFT JOIN stocks s
            ON TRIM(t.stock_symbol) = TRIM(s.symbol)
           AND s.user_id = t.user_id
        WHERE {where}
        ORDER BY t.txn_date DESC, t.stock_symbol
    """
    df = query_df(sql, tuple(params))

    if df.empty:
        return {
            "status": "ok",
            "data": {
                "dividends": [],
                "count": 0,
                "totals": {
                    "total_cash_dividend_kwd": 0.0,
                    "total_bonus_shares": 0.0,
                    "total_reinvested_kwd": 0.0,
                    "unique_stocks": 0,
                },
            },
        }

    # KWD conversions
    df["cash_dividend_kwd"] = df.apply(
        lambda r: round(convert_to_kwd(float(r["cash_dividend"]), r["currency"]), 3),
        axis=1,
    )
    df["reinvested_kwd"] = df.apply(
        lambda r: round(convert_to_kwd(float(r["reinvested_dividend"]), r["currency"]), 3),
        axis=1,
    )

    total_records = len(df)
    offset = (page - 1) * page_size
    page_df = df.iloc[offset : offset + page_size]
    records = page_df.to_dict(orient="records")
    total_pages = max(1, (total_records + page_size - 1) // page_size)

    totals = {
        "total_cash_dividend_kwd": round(float(df["cash_dividend_kwd"].sum()), 3),
        "total_bonus_shares": float(df["bonus_shares"].sum()),
        "total_reinvested_kwd": round(float(df["reinvested_kwd"].sum()), 3),
        "unique_stocks": int(df["stock_symbol"].nunique()),
    }

    return {
        "status": "ok",
        "data": {
            "dividends": records,
            "count": len(records),
            "totals": totals,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total_records,
                "total_pages": total_pages,
            },
        },
    }


# ── Summary by stock ─────────────────────────────────────────────────

@router.get("/by-stock")
async def dividends_by_stock(
    current_user: TokenData = Depends(get_current_user),
):
    """
    Dividend totals grouped by stock — cash dividends (KWD), bonus shares,
    reinvested (KWD), dividend count, and yield-on-cost.
    """
    uid = current_user.user_id

    # Dividend rows
    div_df = query_df(
        """
        SELECT
            t.stock_symbol,
            COALESCE(t.cash_dividend, 0)        AS cash_dividend,
            COALESCE(t.bonus_shares, 0)          AS bonus_shares,
            COALESCE(t.reinvested_dividend, 0)   AS reinvested_dividend,
            COALESCE(s.currency, 'KWD')          AS currency
        FROM transactions t
        LEFT JOIN stocks s
            ON TRIM(t.stock_symbol) = TRIM(s.symbol)
           AND s.user_id = t.user_id
        WHERE t.user_id = ?
          AND COALESCE(t.is_deleted, 0) = 0
          AND (
              COALESCE(t.cash_dividend, 0) > 0
              OR COALESCE(t.bonus_shares, 0) > 0
              OR COALESCE(t.reinvested_dividend, 0) > 0
          )
        """,
        (uid,),
    )

    if div_df.empty:
        return {"status": "ok", "data": {"stocks": [], "count": 0}}

    # KWD conversion
    div_df["cash_dividend_kwd"] = div_df.apply(
        lambda r: convert_to_kwd(float(r["cash_dividend"]), r["currency"]),
        axis=1,
    )
    div_df["reinvested_kwd"] = div_df.apply(
        lambda r: convert_to_kwd(float(r["reinvested_dividend"]), r["currency"]),
        axis=1,
    )

    # Cost basis per stock
    cost_df = query_df(
        """
        SELECT stock_symbol,
               SUM(CASE WHEN txn_type = 'Buy' THEN COALESCE(purchase_cost, 0) ELSE 0 END) AS total_cost
        FROM transactions
        WHERE user_id = ? AND COALESCE(is_deleted, 0) = 0
        GROUP BY stock_symbol
        """,
        (uid,),
    )

    # Aggregate per stock
    summary = (
        div_df.groupby("stock_symbol")
        .agg(
            total_cash_dividend_kwd=("cash_dividend_kwd", "sum"),
            total_bonus_shares=("bonus_shares", "sum"),
            total_reinvested_kwd=("reinvested_kwd", "sum"),
            dividend_count=("cash_dividend", "count"),
        )
        .reset_index()
    )

    # Merge cost
    summary = summary.merge(cost_df, on="stock_symbol", how="left")
    summary["total_cost"] = summary["total_cost"].fillna(0)
    summary["yield_on_cost_pct"] = summary.apply(
        lambda r: round(r["total_cash_dividend_kwd"] / r["total_cost"] * 100, 2)
        if r["total_cost"] > 0
        else 0.0,
        axis=1,
    )

    records = summary.to_dict(orient="records")

    return {
        "status": "ok",
        "data": {
            "stocks": records,
            "count": len(records),
        },
    }


# ── Bonus shares history ─────────────────────────────────────────────

@router.get("/bonus-shares")
async def bonus_shares_list(
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    current_user: TokenData = Depends(get_current_user),
):
    """
    All transactions with bonus_shares > 0 — flat list + by-stock summary.
    Mirrors the Streamlit Bonus Shares tab.
    """
    uid = current_user.user_id

    sql = """
        SELECT
            t.id,
            t.stock_symbol,
            t.portfolio,
            t.txn_date,
            COALESCE(t.bonus_shares, 0)  AS bonus_shares,
            COALESCE(t.shares, 0)        AS shares,
            COALESCE(s.currency, 'KWD')  AS currency,
            t.notes
        FROM transactions t
        LEFT JOIN stocks s
            ON TRIM(t.stock_symbol) = TRIM(s.symbol)
           AND s.user_id = t.user_id
        WHERE t.user_id = ?
          AND COALESCE(t.is_deleted, 0) = 0
          AND COALESCE(t.bonus_shares, 0) > 0
        ORDER BY t.txn_date DESC, t.stock_symbol
    """
    df = query_df(sql, (uid,))

    if df.empty:
        return {
            "status": "ok",
            "data": {
                "records": [],
                "count": 0,
                "total_bonus_shares": 0,
                "by_stock": [],
            },
        }

    total_records = len(df)
    offset = (page - 1) * page_size
    page_df = df.iloc[offset : offset + page_size]
    records = page_df.to_dict(orient="records")
    total_pages = max(1, (total_records + page_size - 1) // page_size)

    # By-stock summary
    by_stock = (
        df.groupby("stock_symbol")
        .agg(
            total_bonus_shares=("bonus_shares", "sum"),
            bonus_count=("bonus_shares", "count"),
        )
        .reset_index()
        .sort_values("total_bonus_shares", ascending=False)
        .to_dict(orient="records")
    )

    return {
        "status": "ok",
        "data": {
            "records": records,
            "count": len(records),
            "total_bonus_shares": float(df["bonus_shares"].sum()),
            "by_stock": by_stock,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total_records,
                "total_pages": total_pages,
            },
        },
    }


# ── Delete a dividend record ─────────────────────────────────────────

@router.delete("/{dividend_id}")
async def delete_dividend(
    dividend_id: int,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Soft-delete a dividend record (sets is_deleted = 1 on the transaction row).
    Only deletes if the transaction belongs to the user and has dividend data.
    """
    from app.core.database import query_one, exec_sql
    import time

    row = query_one(
        """SELECT id FROM transactions
           WHERE id = ? AND user_id = ? AND COALESCE(is_deleted, 0) = 0
             AND (COALESCE(cash_dividend, 0) > 0 OR COALESCE(bonus_shares, 0) > 0
                  OR COALESCE(reinvested_dividend, 0) > 0)""",
        (dividend_id, current_user.user_id),
    )
    if not row:
        from app.core.exceptions import NotFoundError
        raise NotFoundError("Dividend", dividend_id)

    now = int(time.time())
    exec_sql(
        "UPDATE transactions SET is_deleted = 1, deleted_at = ? WHERE id = ? AND user_id = ?",
        (now, dividend_id, current_user.user_id),
    )

    return {"status": "ok", "data": {"id": dividend_id, "message": "Dividend record deleted"}}

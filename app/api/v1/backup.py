"""
Backup & Restore API v1 — Excel export / import.

Mirrors the Streamlit ``ui_backup_restore()`` logic.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.api.deps import get_current_user
from app.core.security import TokenData
from app.core.exceptions import BadRequestError
from app.services.backup_service import (
    export_portfolio_excel,
    import_transactions_excel,
)
from app.services.fx_service import PORTFOLIO_CCY

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backup", tags=["Backup & Restore"])


# ── Export ────────────────────────────────────────────────────────────

@router.get("/export")
async def export_backup(
    current_user: TokenData = Depends(get_current_user),
):
    """
    Download a complete Excel backup of all user data:
    Transactions, Cash Deposits, Stocks, Portfolio Snapshots.
    """
    buffer = export_portfolio_excel(current_user.user_id)

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename=portfolio_backup_{current_user.user_id}.xlsx"
        },
    )


# ── Import ────────────────────────────────────────────────────────────

@router.post("/import")
async def import_backup(
    file: UploadFile = File(...),
    portfolio: Optional[str] = Query(None, description="Target portfolio (KFH, BBYN, USA). Omit for bulk import — portfolio is read from each row."),
    sheet_name: Optional[str] = Query("Transactions", description="Sheet name to import"),
    mode: str = Query("merge", description="Import mode: 'merge' (append) or 'replace' (delete existing first)"),
    current_user: TokenData = Depends(get_current_user),
):
    """
    Full backup restore from an uploaded Excel file.

    Modes:
      - merge: Append new data to existing tables.
      - replace: Hard-delete ALL user data in all restored tables first, then import.

    Imports all sheets found: Transactions, Stocks, Cash Deposits, Portfolio Snapshots.
    If *portfolio* is omitted the value is read from each row's 'portfolio' column.
    """
    if portfolio is not None and portfolio not in PORTFOLIO_CCY:
        raise BadRequestError(f"Unknown portfolio '{portfolio}'. Valid: {list(PORTFOLIO_CCY.keys())}")

    if mode not in ("merge", "replace"):
        raise BadRequestError("mode must be 'merge' or 'replace'")

    if not file.filename or not file.filename.endswith((".xlsx", ".xls")):
        raise BadRequestError("File must be an Excel file (.xlsx or .xls)")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10 MB limit
        raise BadRequestError("File too large (max 10 MB)")

    try:
        # Replace mode: hard-delete ALL user data in the restored tables
        if mode == "replace":
            from app.core.database import exec_sql
            # Order matters: snapshots → cash → transactions → stocks (FK-safe)
            for table in ["portfolio_snapshots", "cash_deposits", "transactions", "stocks"]:
                exec_sql(f"DELETE FROM {table} WHERE user_id = ?", (current_user.user_id,))

        result = import_transactions_excel(
            user_id=current_user.user_id,
            file_bytes=contents,
            portfolio=portfolio,
            sheet_name=sheet_name,
        )
        result["mode"] = mode

        return {
            "status": "ok",
            "data": result,
        }
    except Exception as exc:
        logger.exception("Backup import failed for user %s: %s", current_user.user_id, exc)
        raise BadRequestError(f"Import failed: {exc}")


# ── Data ownership check / migrate ────────────────────────────────────

@router.get("/data-check")
async def data_ownership_check(
    current_user: TokenData = Depends(get_current_user),
):
    """
    Diagnose data ownership — show record counts per user_id for each table.
    Helps debug production issues where imported data may be under a different user.
    """
    uid = current_user.user_id
    tables = ["stocks", "transactions", "cash_deposits", "portfolio_snapshots"]
    result = {"current_user_id": uid, "tables": {}}

    for table in tables:
        try:
            from app.core.database import query_df as _qdf
            df = _qdf(f"SELECT user_id, COUNT(*) as cnt FROM {table} GROUP BY user_id", ())
            rows = df.to_dict(orient="records") if not df.empty else []
            result["tables"][table] = {
                "by_user": {int(r["user_id"]): int(r["cnt"]) for r in rows},
                "your_records": next((int(r["cnt"]) for r in rows if int(r["user_id"]) == uid), 0),
            }
        except Exception as exc:
            result["tables"][table] = {"error": str(exc)}

    return {"status": "ok", "data": result}


@router.post("/claim-data")
async def claim_orphaned_data(
    source_user_id: int = Query(..., description="The user_id whose data to claim"),
    current_user: TokenData = Depends(get_current_user),
):
    """
    Reassign ALL data from *source_user_id* to the current authenticated user.
    Use after diagnosing a user_id mismatch (e.g., backup imported under user 1,
    but Google OAuth created user 2).

    This is a one-time migration — call it once after confirming the mismatch
    via /data-check.
    """
    uid = current_user.user_id
    if source_user_id == uid:
        raise BadRequestError("Source and target user IDs are the same.")

    from app.core.database import exec_sql as _exec, query_val as _qv

    tables = ["stocks", "transactions", "cash_deposits", "portfolio_snapshots"]
    migrated = {}

    for table in tables:
        count = _qv(
            f"SELECT COUNT(*) FROM {table} WHERE user_id = ?",
            (source_user_id,),
        ) or 0
        if count > 0:
            _exec(
                f"UPDATE {table} SET user_id = ? WHERE user_id = ?",
                (uid, source_user_id),
            )
        migrated[table] = count

    logger.info(
        "Claimed data from user %d → user %d: %s",
        source_user_id, uid, migrated,
    )

    return {
        "status": "ok",
        "data": {
            "source_user_id": source_user_id,
            "target_user_id": uid,
            "migrated": migrated,
            "message": f"Migrated data from user {source_user_id} to user {uid}",
        },
    }

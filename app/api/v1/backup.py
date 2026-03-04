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

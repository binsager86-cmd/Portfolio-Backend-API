"""
Compliance API — SOC2 audit log export and retention management.

All endpoints require admin privileges.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.responses import StreamingResponse

from app.api.deps import require_admin
from app.services.compliance_service import enforce_data_retention, stream_audit_csv

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compliance", tags=["compliance"])


@router.get(
    "/audit-export",
    summary="Stream audit log as CSV (admin only)",
    response_description="CSV file with PII redacted",
)
async def export_audit_log(
    start: datetime = Query(..., description="ISO 8601 start datetime (UTC)"),
    end: datetime = Query(..., description="ISO 8601 end datetime (UTC)"),
    _admin=Depends(require_admin),
):
    """
    Download a date-ranged slice of audit_events as a streaming CSV.

    - Max window: 90 days per request.
    - PII fields (email, ip_address, token, …) are replaced with ``[REDACTED]``.
    - Yields rows lazily — safe for large datasets.
    """
    if end <= start:
        raise HTTPException(status_code=400, detail="end must be after start")

    try:
        generator = stream_audit_csv(start, end)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    filename = f"audit_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        generator,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "/enforce-retention",
    summary="Manually trigger data retention policy (admin only)",
)
async def trigger_retention(
    retention_days: int = Query(365, ge=30, le=3650, description="Retention window in days"),
    _admin=Depends(require_admin),
):
    """
    Delete audit events older than *retention_days*.

    The nightly scheduler calls this automatically at 03:00 Asia/Kuwait.
    Use this endpoint to run it on-demand (e.g. before audits).
    """
    deleted = enforce_data_retention(retention_days)
    return {"status": "ok", "deleted": deleted, "retention_days": retention_days}

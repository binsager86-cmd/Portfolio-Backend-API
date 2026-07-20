"""
Compliance & Audit Service — SOC2-ready audit exports and data retention.

Provides:
  - PII redaction (fields in PII_FIELDS are replaced with "[REDACTED]")
    - Streaming CSV export of audit_log (max 90-day window)
  - Automated data retention enforcement (hard-delete old events)
"""

import csv
import io
import logging
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

from app.core.database import exec_sql, query_all

logger = logging.getLogger(__name__)

# Fields that must never appear in an export
PII_FIELDS = {"email", "phone", "ip_address", "token", "password_hash"}
MASK = "[REDACTED]"

# Maximum allowed export window (SOC2 recommendation: 90 days per request)
MAX_EXPORT_DAYS = 90


def _ensure_audit_log_indexes() -> None:
    """Ensure indexes used by audit export and retention exist on audit_log."""
    exec_sql("CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at)")


def redact_pii(row: dict) -> dict:
    """Replace PII field values with MASK, cast everything else to str."""
    return {k: (MASK if k in PII_FIELDS else str(v) if v is not None else "") for k, v in row.items()}


async def stream_audit_csv(
    start: datetime,
    end: datetime,
) -> AsyncGenerator[str, None]:
    """
    Yield audit log rows as CSV chunks with PII masking.

    Args:
        start: Range start (UTC-aware or naive datetime).
        end:   Range end (UTC-aware or naive datetime).

    Raises:
        ValueError: If the requested window exceeds MAX_EXPORT_DAYS.
    """
    # Normalise to epoch seconds; audit_log.created_at is written as int(time.time()).
    if start.tzinfo is not None:
        start = start.astimezone(timezone.utc).replace(tzinfo=None)
    if end.tzinfo is not None:
        end = end.astimezone(timezone.utc).replace(tzinfo=None)

    if (end - start) > timedelta(days=MAX_EXPORT_DAYS):
        raise ValueError(f"Max export range is {MAX_EXPORT_DAYS} days")

    _ensure_audit_log_indexes()
    start_ts = int(start.replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(end.replace(tzinfo=timezone.utc).timestamp())

    rows = query_all(
        "SELECT id, user_id, action, resource_type, details, ip_address, created_at "
        "FROM audit_log "
        "WHERE created_at BETWEEN ? AND ? "
        "ORDER BY created_at",
        (start_ts, end_ts),
    )

    # Yield CSV header
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "user_id", "category", "action", "details", "ip_address", "created_at"])
    yield output.getvalue()

    # Yield one row at a time to keep memory bounded for large exports
    for raw_row in rows:
        if isinstance(raw_row, (list, tuple)):
            row_dict = dict(zip(
                ["id", "user_id", "action", "resource_type", "details", "ip_address", "created_at"],
                raw_row,
            ))
        else:
            row_dict = dict(raw_row)
        action = str(row_dict.get("action") or "")
        category = action.split(".", 1)[0] if "." in action else (row_dict.get("resource_type") or "general")
        row_dict = {
            "id": row_dict.get("id"),
            "user_id": row_dict.get("user_id"),
            "category": category,
            "action": row_dict.get("action"),
            "details": row_dict.get("details"),
            "ip_address": row_dict.get("ip_address"),
            "created_at": row_dict.get("created_at"),
        }

        output.seek(0)
        output.truncate(0)
        writer.writerow(list(redact_pii(row_dict).values()))
        yield output.getvalue()


def enforce_data_retention(retention_days: int = 365) -> int:
    """
    Hard-delete audit_log rows older than *retention_days*.

    Called nightly by the APScheduler job (03:00 Asia/Kuwait).

    Returns:
        Number of rows deleted.
    """
    _ensure_audit_log_indexes()
    cutoff = int((datetime.utcnow() - timedelta(days=retention_days)).replace(tzinfo=timezone.utc).timestamp())

    # Count first (SQLite does not support RETURNING on older versions)
    old_rows = query_all(
        "SELECT id FROM audit_log WHERE created_at < ?",
        (cutoff,),
    )
    count = len(old_rows)

    if count:
        exec_sql(
            "DELETE FROM audit_log WHERE created_at < ?",
            (cutoff,),
        )
        logger.info(
            "Retention policy applied: %d audit event(s) purged (older than %d days)",
            count,
            retention_days,
        )
    else:
        logger.debug("Retention sweep: no audit events older than %d days", retention_days)

    return count

"""
Compliance & Audit Service — SOC2-ready audit exports and data retention.

Provides:
  - PII redaction (fields in PII_FIELDS are replaced with "[REDACTED]")
  - Streaming CSV export of audit_events (max 90-day window)
  - Automated data retention enforcement (hard-delete old events)

The audit_events table is created on first use if absent.
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


def _ensure_audit_table() -> None:
    """Create audit_events table if it does not exist."""
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS audit_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER,
            category    TEXT    NOT NULL DEFAULT 'general',
            action      TEXT    NOT NULL,
            details     TEXT,
            ip_address  TEXT,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    # Index for fast range queries (needed for export + retention)
    exec_sql(
        "CREATE INDEX IF NOT EXISTS ix_audit_events_created_at "
        "ON audit_events(created_at)"
    )


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
    # Normalise to naive UTC strings for SQLite comparison
    if start.tzinfo is not None:
        start = start.astimezone(timezone.utc).replace(tzinfo=None)
    if end.tzinfo is not None:
        end = end.astimezone(timezone.utc).replace(tzinfo=None)

    if (end - start) > timedelta(days=MAX_EXPORT_DAYS):
        raise ValueError(f"Max export range is {MAX_EXPORT_DAYS} days")

    _ensure_audit_table()

    rows = query_all(
        "SELECT id, user_id, category, action, details, ip_address, created_at "
        "FROM audit_events "
        "WHERE created_at BETWEEN ? AND ? "
        "ORDER BY created_at",
        (start.isoformat(sep=" "), end.isoformat(sep=" ")),
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
                ["id", "user_id", "category", "action", "details", "ip_address", "created_at"],
                raw_row,
            ))
        else:
            row_dict = dict(raw_row)

        output.seek(0)
        output.truncate(0)
        writer.writerow(list(redact_pii(row_dict).values()))
        yield output.getvalue()


def enforce_data_retention(retention_days: int = 365) -> int:
    """
    Hard-delete audit_events older than *retention_days*.

    Called nightly by the APScheduler job (03:00 Asia/Kuwait).

    Returns:
        Number of rows deleted.
    """
    _ensure_audit_table()
    cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat(sep=" ")

    # Count first (SQLite does not support RETURNING on older versions)
    old_rows = query_all(
        "SELECT id FROM audit_events WHERE created_at < ?",
        (cutoff,),
    )
    count = len(old_rows)

    if count:
        exec_sql(
            "DELETE FROM audit_events WHERE created_at < ?",
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

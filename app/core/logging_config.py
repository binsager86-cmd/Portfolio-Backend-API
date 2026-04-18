"""
Structured JSON Logging — production-ready log configuration.

In development: human-readable colored format with correlation ID.
In production: JSON-per-line for log aggregation (Datadog, CloudWatch, etc.)

Features:
  - Automatic correlation_id injection via contextvars (set by middleware)
  - JSON output includes arbitrary extra fields for structured queries
  - Dev output includes correlation_id prefix when present

Usage:
    from app.core.logging_config import setup_logging
    setup_logging()  # call once at startup
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

from app.core.config import get_settings

# ── Context variable for request-scoped correlation ID ───────────────
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

# Extra structured fields that any code path can attach to logs
_EXTRA_KEYS = frozenset({
    "method", "path", "status_code", "duration_ms", "user_id",
    "client_ip", "user_agent", "error_code", "query_count",
})


class ContextFilter(logging.Filter):
    """Inject correlation_id from contextvar into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "correlation_id", None):
            record.correlation_id = correlation_id_var.get("")  # type: ignore[attr-defined]
        return True


class JSONFormatter(logging.Formatter):
    """Emit one JSON object per log line with structured extras."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Correlation ID
        cid = getattr(record, "correlation_id", "")
        if cid:
            log_entry["correlation_id"] = cid

        # Structured extras (method, path, status_code, duration_ms, etc.)
        for key in _EXTRA_KEYS:
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        # Exception info
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class DevFormatter(logging.Formatter):
    """Human-readable format for development — includes correlation ID."""

    def format(self, record: logging.LogRecord) -> str:
        cid = getattr(record, "correlation_id", "") or ""
        prefix = f"[{cid[:8]}] " if cid else ""
        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        base = f"{ts}  {record.levelname:<8}  {prefix}{record.name}  {record.getMessage()}"

        # Append structured extras in dev for visibility
        extras = {k: getattr(record, k, None) for k in _EXTRA_KEYS if getattr(record, k, None) is not None}
        if extras:
            base += f"  {extras}"

        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging() -> None:
    """Configure root logger based on environment."""
    settings = get_settings()

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicate output
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(ContextFilter())

    if settings.is_production:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(DevFormatter())

    root.addHandler(handler)

    # Quiet noisy libraries
    for name in ("urllib3", "httpcore", "httpx", "watchfiles"):
        logging.getLogger(name).setLevel(logging.WARNING)

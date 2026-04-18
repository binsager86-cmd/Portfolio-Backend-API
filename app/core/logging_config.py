"""
Structured JSON Logging — production-ready log configuration.

In development: human-readable colored format.
In production: JSON-per-line for log aggregation (Datadog, CloudWatch, etc.)

Usage:
    from app.core.logging_config import setup_logging
    setup_logging()  # call once at startup
"""

import json
import logging
import sys
from datetime import datetime, timezone

from app.core.config import get_settings


class JSONFormatter(logging.Formatter):
    """Emit one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach correlation ID if present
        cid = getattr(record, "correlation_id", None)
        if cid:
            log_entry["correlation_id"] = cid

        # Attach extra fields from LogRecord
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class DevFormatter(logging.Formatter):
    """Human-readable format for development."""

    FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"

    def __init__(self):
        super().__init__(self.FMT)


def setup_logging() -> None:
    """Configure root logger based on environment."""
    settings = get_settings()

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicate output
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if settings.is_production:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(DevFormatter())

    root.addHandler(handler)

    # Quiet noisy libraries
    for name in ("urllib3", "httpcore", "httpx", "watchfiles"):
        logging.getLogger(name).setLevel(logging.WARNING)

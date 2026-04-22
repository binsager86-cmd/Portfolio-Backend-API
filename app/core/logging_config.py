"""
Structured logging configuration with correlation ID context support.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

from app.core.config import get_settings

# Request-scoped correlation ID injected by middleware
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

_EXTRA_KEYS = frozenset({
    "method",
    "path",
    "status_code",
    "duration_ms",
    "user_id",
    "client_ip",
    "user_agent",
    "error_code",
    "query_count",
})


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "correlation_id", None):
            record.correlation_id = correlation_id_var.get("")
        return True


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        cid = getattr(record, "correlation_id", "")
        if cid:
            payload["correlation_id"] = cid

        for key in _EXTRA_KEYS:
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val

        if record.exc_info and record.exc_info[1]:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class DevFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        cid = getattr(record, "correlation_id", "") or ""
        prefix = f"[{cid[:8]}] " if cid else ""
        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        msg = f"{ts}  {record.levelname:<8}  {prefix}{record.name}  {record.getMessage()}"

        extras = {
            k: getattr(record, k, None)
            for k in _EXTRA_KEYS
            if getattr(record, k, None) is not None
        }
        if extras:
            msg += f"  {extras}"

        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        return msg


def setup_logging() -> None:
    settings = get_settings()

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(ContextFilter())
    handler.setFormatter(JSONFormatter() if settings.is_production else DevFormatter())
    root.addHandler(handler)

    for name in ("urllib3", "httpcore", "httpx", "watchfiles"):
        logging.getLogger(name).setLevel(logging.WARNING)

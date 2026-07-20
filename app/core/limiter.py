"""
Rate limiter — shared slowapi instance.

Kept in its own module to avoid circular imports between main.py and routers.
"""

import os

from slowapi import Limiter

from app.core.client_ip import get_client_ip
from app.core.config import get_settings

_enabled = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() != "false"
_settings = get_settings()
_redis_url = (_settings.REDIS_URL or os.environ.get("REDIS_URL") or "").strip()

limiter = Limiter(
    key_func=lambda request: get_client_ip(request) or "unknown",
    default_limits=["120/minute"],          # global default
    storage_uri=_redis_url or "memory://",
    enabled=_enabled,
)

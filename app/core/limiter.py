"""
Rate limiter — shared slowapi instance.

Kept in its own module to avoid circular imports between main.py and routers.
"""

import os
import logging

from slowapi import Limiter

from app.core.client_ip import get_client_ip
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_enabled = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() != "false"
_require_redis = os.environ.get("RATE_LIMIT_REQUIRE_REDIS", "false").lower() == "true"
_settings = get_settings()
_redis_url = (_settings.REDIS_URL or os.environ.get("REDIS_URL") or "").strip()

if _enabled and _settings.ENVIRONMENT.lower() == "production" and not _redis_url:
    if _require_redis:
        raise RuntimeError("RATE_LIMIT_REQUIRE_REDIS=true requires REDIS_URL in production")
    logger.warning("REDIS_URL is not configured; using in-memory rate limiting")

limiter = Limiter(
    key_func=lambda request: get_client_ip(request) or "unknown",
    default_limits=["120/minute"],          # global default
    storage_uri=_redis_url or "memory://",
    enabled=_enabled,
)

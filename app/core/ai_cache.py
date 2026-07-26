"""
Async Redis AI Cache Layer — DB 1 (separate from Celery's DB 0).

All operations are fail-open: a Redis outage degrades to cold-path AI calls,
never raises to the caller.
"""

import json
import logging
import os
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import redis.asyncio as redis

logger = logging.getLogger(__name__)

# DB 1 keeps AI results separate from Celery broker/results on DB 0
_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def _redis_url_for_db(url: str, db: int) -> str:
    parsed = urlsplit(url)
    query_items = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k != "db"]
    query = urlencode(query_items)
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{db}", query, parsed.fragment))


# Force DB 1 regardless of what REDIS_URL points to, preserving scheme/auth/query params.
_AI_REDIS_URL = _redis_url_for_db(_REDIS_URL, 1)

ai_cache: redis.Redis = redis.Redis.from_url(
    _AI_REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=3,
    retry_on_timeout=True,
)


async def get_ai_cache(key: str) -> Optional[dict[str, Any]]:
    """Fetch a cached value by key. Returns None on miss or Redis failure."""
    try:
        raw = await ai_cache.get(key)
        return json.loads(raw) if raw else None
    except Exception as exc:
        logger.warning("Redis AI cache read failed (fail-open): %s", exc)
        return None


async def set_ai_cache(key: str, value: dict[str, Any], ttl: int = 86400) -> bool:
    """Persist a value with TTL (seconds). Returns False on Redis failure."""
    try:
        await ai_cache.setex(key, ttl, json.dumps(value, separators=(",", ":")))
        return True
    except Exception as exc:
        logger.warning("Redis AI cache write failed: %s", exc)
        return False


async def invalidate_ai_pattern(pattern: str) -> int:
    """
    Delete all keys matching *pattern* (e.g. ``"ai:42:*"``).

    Returns the number of keys deleted, or 0 on failure.

    .. warning::
        Uses ``KEYS`` — safe for low-cardinality patterns (per-user invalidation).
        Do not call with broad wildcards on large keyspaces.
    """
    try:
        keys = await ai_cache.keys(pattern)
        if keys:
            return await ai_cache.delete(*keys)
        return 0
    except Exception as exc:
        logger.warning("Redis AI cache invalidation failed: %s", exc)
        return 0

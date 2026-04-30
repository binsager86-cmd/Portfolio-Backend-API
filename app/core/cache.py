"""Shared in-memory TTL caches for price and news data."""

import hashlib
from typing import Any, Hashable

from cachetools import TTLCache

# ── Cache instances ───────────────────────────────────────────────────────────

# Stock prices (Yahoo Finance tickers) — 5-minute TTL
price_cache: TTLCache = TTLCache(maxsize=2000, ttl=300)

# Boursa Kuwait API responses — 10-minute TTL
news_cache: TTLCache = TTLCache(maxsize=1000, ttl=600)


# ── Helpers ───────────────────────────────────────────────────────────────────

def cache_key(*args: Any, **kwargs: Any) -> str:
    """Build a stable cache key from arbitrary arguments."""
    raw = f"{args}:{sorted(kwargs.items())}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(cache: TTLCache, key: Hashable) -> Any:
    """Return cached value for *key*, or ``None`` on miss / expiry."""
    return cache.get(key)


def set_cached(cache: TTLCache, key: Hashable, value: Any) -> None:
    """Store *value* in *cache* under *key*."""
    cache[key] = value

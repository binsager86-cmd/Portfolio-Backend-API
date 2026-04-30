"""
Shared async HTTP client — single httpx.AsyncClient singleton with connection
pooling, timeouts, and a tenacity-backed retry wrapper.

Usage
-----
    from app.core.http_client import fetch_with_retry

    response = await fetch_with_retry("https://example.com/api", params={...})
    data = response.json()

The module-level `async_client` is re-used across requests (connection pool).
Call `close_client()` in the app lifespan shutdown hook if needed.
"""

import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _build_async_client() -> httpx.AsyncClient:
    """Create the shared async client, preferring HTTP/2 when available."""
    common_kwargs = {
        "timeout": httpx.Timeout(connect=3.0, read=8.0, write=3.0, pool=10.0),
        "limits": httpx.Limits(max_keepalive_connections=20, max_connections=50),
        "headers": {"User-Agent": "PortfolioTracker/1.0 (+https://yourdomain.com)"},
        "follow_redirects": True,
    }
    try:
        return httpx.AsyncClient(http2=True, **common_kwargs)
    except ImportError:
        logger.warning("HTTP/2 support unavailable; falling back to HTTP/1.1 client")
        return httpx.AsyncClient(**common_kwargs)


# ── Shared connection pool ────────────────────────────────────────────────────
# Created once at import time; reused for all outbound requests so TCP
# connections are kept alive across calls.
async_client = _build_async_client()


async def close_client() -> None:
    """Gracefully close the shared client (call from app shutdown lifespan)."""
    await async_client.aclose()


# ── Retry-wrapped GET ─────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type(
        (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError)
    ),
    reraise=True,
)
async def fetch_with_retry(url: str, **kwargs) -> httpx.Response:
    """
    GET *url* using the shared async client, retrying up to 3 times on
    network/timeout/HTTP errors with exponential back-off (1 s → 2 s → 4 s).

    Raises
    ------
    httpx.HTTPStatusError
        After all retry attempts are exhausted (4xx/5xx status codes).
    httpx.ConnectError / httpx.TimeoutException
        After all retry attempts on network-level failures.
    """
    resp = await async_client.get(url, **kwargs)
    resp.raise_for_status()
    return resp

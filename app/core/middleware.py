"""
Middleware — security headers, request size limits, correlation ID, timing.

Includes:
  1. Security headers (CSP, HSTS, X-Frame-Options, etc.)
  2. Request body size limit (DoS prevention)
  3. Correlation ID (X-Correlation-ID) for request tracing + contextvar logging
  4. Request timing (X-Response-Time-Ms) with structured per-request latency logs
"""

import logging
import hashlib
import time
import uuid

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.config import get_settings
from app.core.logging_config import correlation_id_var

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds OWASP-recommended security headers to every response.

    Headers set:
      - X-Content-Type-Options: nosniff
      - X-Frame-Options: DENY
      - X-XSS-Protection: 1; mode=block
      - Referrer-Policy: strict-origin-when-cross-origin
      - Content-Security-Policy: default-src 'self'  (API-only, no inline scripts needed)
      - Permissions-Policy: interest-cohort=()  (disable FLoC)
      - Strict-Transport-Security (production only, after HTTPS termination)
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)

        settings = get_settings()

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "interest-cohort=()"
        response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"

        # HSTS only in production (assumes TLS termination by reverse proxy)
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )

        return response


class PrivateNetworkAccessMiddleware(BaseHTTPMiddleware):
    """
    Handle Chrome's Private Network Access (PNA) preflight requests.

    When a page at localhost:PORT makes a cross-origin request to 127.0.0.1:PORT,
    Chrome sends a preflight with Access-Control-Request-Private-Network: true.
    The server must respond with Access-Control-Allow-Private-Network: true or
    Chrome silently blocks the request (Axios sees a network error).
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)

        # If the preflight asked for private network access, grant it
        if request.headers.get("access-control-request-private-network") == "true":
            response.headers["Access-Control-Allow-Private-Network"] = "true"

        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Rejects requests with Content-Length exceeding the configured maximum.

    Catches oversized payloads early before they consume memory.
    Does NOT read the body — just checks Content-Length header.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        settings = get_settings()
        max_bytes = settings.MAX_REQUEST_BODY_BYTES

        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_bytes:
            logger.warning(
                "Request too large: %s bytes from %s %s",
                content_length,
                request.method,
                request.url.path,
            )
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "status": "error",
                    "detail": f"Request body too large (max {max_bytes} bytes)",
                    "error_code": "PAYLOAD_TOO_LARGE",
                },
            )

        return await call_next(request)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Generates a unique correlation ID for every request.

    - Reuses X-Correlation-ID from the client if provided.
    - Stores the ID in request.state for downstream access.
    - Sets the correlation_id contextvar so all log lines include it.
    - Echoes it back in the response header.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        cid = request.headers.get("x-correlation-id") or uuid.uuid4().hex[:16]
        request.state.correlation_id = cid
        token = correlation_id_var.set(cid)

        try:
            response = await call_next(request)
            response.headers["X-Correlation-ID"] = cid
            return response
        finally:
            correlation_id_var.reset(token)


_timing_logger = logging.getLogger("app.request")

# Paths to skip in access logs (noisy health checks, static files)
_SKIP_LOG_PATHS = frozenset({"/health", "/api/health", "/favicon.ico", "/openapi.json"})

LEGACY_PREFIXES = ("/api/auth/", "/api/portfolio/", "/api/cron/")
SUNSET_DATE = "2025-12-31T23:59:59Z"


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """
    Measures request processing time and adds X-Response-Time-Ms header.

    Emits structured per-request access log with method, path, status,
    duration, and client info. Warns on slow requests (>1s).
    """

    SLOW_THRESHOLD_MS = 1000

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"

        path = request.url.path
        if path not in _SKIP_LOG_PATHS:
            level = logging.WARNING if elapsed_ms > self.SLOW_THRESHOLD_MS else logging.INFO
            _timing_logger.log(
                level,
                "%s %s %s %.0fms",
                request.method,
                path,
                response.status_code,
                elapsed_ms,
                extra={
                    "method": request.method,
                    "path": path,
                    "status_code": response.status_code,
                    "duration_ms": round(elapsed_ms, 1),
                    "client_ip": request.client.host if request.client else "",
                    "user_agent": (request.headers.get("user-agent") or "")[:120],
                },
            )

        return response


class LegacyDeprecationMiddleware(BaseHTTPMiddleware):
    """
    Adds deprecation headers to legacy unversioned API routes.

    Legacy clients continue to work, but every response under the old
    `/api/{auth,portfolio,cron}` prefixes advertises the `/api/v1/...`
    successor so callers can migrate gradually.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        path = request.url.path

        if any(path.startswith(prefix) for prefix in LEGACY_PREFIXES):
            response.headers["Deprecation"] = "true"
            response.headers["Sunset"] = SUNSET_DATE
            successor = path.replace("/api/", "/api/v1/", 1)
            response.headers["Link"] = f'<{successor}>; rel="successor-version"'

        return response


class AssetCacheMiddleware(BaseHTTPMiddleware):
    """Apply long-lived immutable cache headers for static assets."""

    _STATIC_SUFFIXES = (".js", ".css", ".png", ".webp", ".woff2")

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        path = request.url.path

        if path.endswith(self._STATIC_SUFFIXES):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            response.headers["ETag"] = hashlib.md5(path.encode()).hexdigest()[:8]

        return response

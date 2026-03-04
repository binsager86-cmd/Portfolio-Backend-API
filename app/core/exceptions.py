"""
Custom HTTP Exceptions — consistent error handling across the API.

Usage:
    from app.core.exceptions import NotFoundError, BadRequestError

    raise NotFoundError("Transaction", transaction_id)
    raise BadRequestError("shares must be positive")
"""

from typing import Any, Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse


# ── Base API Error ───────────────────────────────────────────────────

class APIError(HTTPException):
    """Base exception for all API errors with structured body."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        self.error_code = error_code or self._default_code(status_code)
        super().__init__(
            status_code=status_code,
            detail=detail,
            headers=headers,
        )

    @staticmethod
    def _default_code(status_code: int) -> str:
        return {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            409: "CONFLICT",
            422: "VALIDATION_ERROR",
            429: "RATE_LIMITED",
            500: "INTERNAL_ERROR",
            503: "SERVICE_UNAVAILABLE",
        }.get(status_code, "ERROR")


# ── Concrete exceptions ─────────────────────────────────────────────

class NotFoundError(APIError):
    """Resource not found (404)."""

    def __init__(self, resource: str = "Resource", resource_id: Any = None):
        detail = f"{resource} not found"
        if resource_id is not None:
            detail = f"{resource} with id={resource_id} not found"
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            error_code="NOT_FOUND",
        )


class BadRequestError(APIError):
    """Invalid request data (400)."""

    def __init__(self, detail: str = "Bad request"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="BAD_REQUEST",
        )


class UnauthorizedError(APIError):
    """Authentication required or failed (401)."""

    def __init__(self, detail: str = "Not authenticated"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="UNAUTHORIZED",
            headers={"WWW-Authenticate": "Bearer"},
        )


class ForbiddenError(APIError):
    """Authenticated but insufficient permissions (403)."""

    def __init__(self, detail: str = "Permission denied"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="FORBIDDEN",
        )


class ConflictError(APIError):
    """Resource conflict, e.g. duplicate entry (409)."""

    def __init__(self, detail: str = "Resource conflict"):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code="CONFLICT",
        )


class ServiceUnavailableError(APIError):
    """External service unavailable (503)."""

    def __init__(self, detail: str = "Service temporarily unavailable"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code="SERVICE_UNAVAILABLE",
        )


# ── Exception handlers (register in main.py) ────────────────────────

def _cors_headers(request: Request) -> dict[str, str]:
    """Build CORS headers for error responses.

    The CORSMiddleware sometimes doesn't add headers to exception-handler
    responses (BaseHTTPMiddleware + exception-handler interaction).  Adding
    them explicitly ensures browsers can always read the error body.
    """
    origin = request.headers.get("origin", "")
    if origin:
        return {
            "access-control-allow-origin": origin,
            "access-control-allow-credentials": "true",
            "vary": "Origin",
        }
    return {}


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Structured JSON error response for all APIError subclasses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_code": exc.error_code,
            "detail": exc.detail,
        },
        headers=_cors_headers(request),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions in production."""
    import traceback
    import logging
    logger = logging.getLogger("unhandled")
    tb = traceback.format_exc()
    logger.error("Unhandled %s on %s %s:\n%s", type(exc).__name__, request.method, request.url.path, tb)

    from app.core.config import get_settings
    settings = get_settings()

    # Include exception class + message (but not full traceback) so the
    # frontend / curl shows enough detail to diagnose 500s.
    detail = f"{type(exc).__name__}: {exc}"
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_code": "INTERNAL_ERROR",
            "detail": detail,
        },
        headers=_cors_headers(request),
    )

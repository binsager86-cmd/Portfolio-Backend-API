"""
Unit tests for custom exceptions and error handling.

Covers:
  - APIError base class and status code mapping
  - NotFoundError, BadRequestError, UnauthorizedError, etc.
  - Error response format consistency
"""

import pytest
from fastapi import status

from app.core.exceptions import (
    APIError,
    NotFoundError,
    BadRequestError,
    UnauthorizedError,
    ForbiddenError,
    ConflictError,
    ServiceUnavailableError,
)


class TestAPIError:
    """Tests for the base APIError class."""

    def test_default_error_codes(self):
        """Each status code maps to the correct default error code."""
        err = APIError(400, "test")
        assert err.error_code == "BAD_REQUEST"

        err = APIError(401, "test")
        assert err.error_code == "UNAUTHORIZED"

        err = APIError(403, "test")
        assert err.error_code == "FORBIDDEN"

        err = APIError(404, "test")
        assert err.error_code == "NOT_FOUND"

        err = APIError(409, "test")
        assert err.error_code == "CONFLICT"

        err = APIError(429, "test")
        assert err.error_code == "RATE_LIMITED"

        err = APIError(500, "test")
        assert err.error_code == "INTERNAL_ERROR"

        err = APIError(503, "test")
        assert err.error_code == "SERVICE_UNAVAILABLE"

    def test_custom_error_code(self):
        """Custom error_code overrides the default."""
        err = APIError(400, "test", error_code="CUSTOM_ERROR")
        assert err.error_code == "CUSTOM_ERROR"

    def test_unknown_status_code(self):
        """Unknown status codes get generic 'ERROR' code."""
        err = APIError(418, "I'm a teapot")
        assert err.error_code == "ERROR"
        assert err.detail == "I'm a teapot"


class TestNotFoundError:
    """Tests for NotFoundError."""

    def test_basic_not_found(self):
        err = NotFoundError()
        assert err.status_code == 404
        assert err.error_code == "NOT_FOUND"
        assert "Resource" in err.detail

    def test_with_resource_name(self):
        err = NotFoundError("Transaction")
        assert "Transaction" in err.detail

    def test_with_resource_id(self):
        err = NotFoundError("Transaction", 42)
        assert err.status_code == 404
        assert "42" in err.detail
        assert "Transaction" in err.detail


class TestBadRequestError:
    """Tests for BadRequestError."""

    def test_default_message(self):
        err = BadRequestError()
        assert err.status_code == 400
        assert err.detail == "Bad request"

    def test_custom_message(self):
        err = BadRequestError("shares must be positive")
        assert err.detail == "shares must be positive"


class TestUnauthorizedError:
    """Tests for UnauthorizedError."""

    def test_default_message(self):
        err = UnauthorizedError()
        assert err.status_code == 401
        assert err.error_code == "UNAUTHORIZED"

    def test_has_www_authenticate_header(self):
        """Unauthorized errors should include WWW-Authenticate header."""
        err = UnauthorizedError()
        assert err.headers is not None
        assert "WWW-Authenticate" in err.headers
        assert err.headers["WWW-Authenticate"] == "Bearer"


class TestForbiddenError:
    """Tests for ForbiddenError."""

    def test_default_message(self):
        err = ForbiddenError()
        assert err.status_code == 403
        assert err.error_code == "FORBIDDEN"


class TestConflictError:
    """Tests for ConflictError."""

    def test_default_message(self):
        err = ConflictError()
        assert err.status_code == 409
        assert err.error_code == "CONFLICT"


class TestServiceUnavailableError:
    """Tests for ServiceUnavailableError."""

    def test_default_message(self):
        err = ServiceUnavailableError()
        assert err.status_code == 503
        assert err.error_code == "SERVICE_UNAVAILABLE"


class TestExceptionInheritance:
    """Verify all exceptions inherit from APIError and HTTPException."""

    @pytest.mark.parametrize("exc_class", [
        NotFoundError,
        BadRequestError,
        UnauthorizedError,
        ForbiddenError,
        ConflictError,
        ServiceUnavailableError,
    ])
    def test_inherits_api_error(self, exc_class):
        from fastapi import HTTPException
        assert issubclass(exc_class, APIError)
        assert issubclass(exc_class, HTTPException)

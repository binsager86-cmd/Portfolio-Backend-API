"""
Common schemas — response wrappers, pagination.

All API responses follow a consistent envelope:
    {"status": "ok", "data": {...}}
    {"status": "error", "error_code": "...", "detail": "..."}
"""

from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel):
    """Standard API response envelope."""
    status: str = "ok"
    data: Optional[Any] = None


class PaginationMeta(BaseModel):
    """Pagination metadata for list endpoints."""
    page: int = 1
    page_size: int = 50
    total_items: int = 0
    total_pages: int = 0


class PaginatedResponse(BaseModel):
    """Standard paginated list response."""
    status: str = "ok"
    data: Any = None
    pagination: Optional[PaginationMeta] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    status: str = "error"
    error_code: str
    detail: str

"""
Pydantic Schemas — package-level exports.
"""

from app.schemas.common import ApiResponse, PaginationMeta, PaginatedResponse
from app.schemas.user import (
    LoginRequest,
    RegisterRequest,
    ChangePasswordRequest,
    UserInfo,
    TokenResponse,
)
from app.schemas.portfolio import (
    HoldingRow,
    HoldingsResponse,
    PortfolioOverview,
    PortfolioTableResponse,
    TransactionCreate,
    TransactionUpdate,
    TransactionResponse,
    TransactionListResponse,
)
from app.schemas.cash import (
    CashDepositCreate,
    CashDepositUpdate,
    CashDepositResponse,
    CashDepositListResponse,
)
from app.schemas.analytics import PerformanceMetrics
from app.schemas.eagle_eye_audit import (
    AuditEventCreate,
    ChangeRequestCreate,
    ChangeReviewRequest,
    ChangeTransitionRequest,
)

__all__ = [
    "ApiResponse",
    "PaginationMeta",
    "PaginatedResponse",
    "LoginRequest",
    "RegisterRequest",
    "ChangePasswordRequest",
    "UserInfo",
    "TokenResponse",
    "HoldingRow",
    "HoldingsResponse",
    "PortfolioOverview",
    "PortfolioTableResponse",
    "TransactionCreate",
    "TransactionUpdate",
    "TransactionResponse",
    "TransactionListResponse",
    "CashDepositCreate",
    "CashDepositUpdate",
    "CashDepositResponse",
    "CashDepositListResponse",
    "PerformanceMetrics",
    "AuditEventCreate",
    "ChangeRequestCreate",
    "ChangeReviewRequest",
    "ChangeTransitionRequest",
]

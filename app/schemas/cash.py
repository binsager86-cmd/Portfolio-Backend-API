"""
Cash deposit schemas — CRUD request/response models.
"""

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def _normalize_source(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in {"deposit", "withdrawal"}:
        raise ValueError("source must be deposit or withdrawal")
    return normalized


def _validate_iso_date_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    date.fromisoformat(value)
    return value


class CashDepositCreate(BaseModel):
    """Create a new cash deposit."""
    portfolio: str = Field(..., description="Portfolio name: KFH, BBYN, or USA")
    deposit_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    amount: float = Field(..., gt=0)
    currency: str = Field("KWD", max_length=10)
    bank_name: Optional[str] = Field(None, max_length=100)
    source: str = Field("deposit", description="deposit or withdrawal")
    notes: Optional[str] = None
    description: Optional[str] = Field(None, max_length=255)
    comments: Optional[str] = None
    include_in_analysis: int = Field(1, ge=0, le=1, description="1=include, 0=exclude")
    fx_rate_at_deposit: Optional[float] = None

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        return _normalize_source(value) or "deposit"

    @field_validator("deposit_date")
    @classmethod
    def valid_deposit_date(cls, value: str) -> str:
        return _validate_iso_date_string(value) or value


class CashDepositUpdate(BaseModel):
    """Update a cash deposit (partial)."""
    portfolio: Optional[str] = Field(None, description="Portfolio name: KFH, BBYN, or USA")
    deposit_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    amount: Optional[float] = Field(None, gt=0)
    currency: Optional[str] = Field(None, max_length=10)
    bank_name: Optional[str] = Field(None, max_length=100)
    source: Optional[str] = None
    notes: Optional[str] = None
    description: Optional[str] = None
    comments: Optional[str] = None
    include_in_analysis: Optional[int] = Field(None, ge=0, le=1)
    fx_rate_at_deposit: Optional[float] = None

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_source(value)

    @field_validator("deposit_date")
    @classmethod
    def valid_deposit_date(cls, value: Optional[str]) -> Optional[str]:
        return _validate_iso_date_string(value)


class CashDepositResponse(BaseModel):
    """Single cash deposit record."""
    id: int
    user_id: int
    portfolio: str
    deposit_date: str
    amount: float
    currency: str
    bank_name: Optional[str] = None
    source: Optional[str] = "deposit"
    notes: Optional[str] = None
    is_deleted: Optional[int] = 0
    created_at: Optional[int] = None


class CashDepositListResponse(BaseModel):
    """Cash deposit list."""
    status: str = "ok"
    data: Optional[Dict[str, Any]] = None  # {deposits, count, total_kwd}


class CashBalanceResponse(BaseModel):
    """Account balance summary."""
    status: str = "ok"
    data: Optional[Dict[str, Any]] = None  # {total_cash_kwd, accounts}

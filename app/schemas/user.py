"""
User schemas — auth request/response models.
"""

from typing import Optional
import re

from pydantic import BaseModel, Field, field_validator


def _validate_strong_password(v: str) -> str:
    """Enforce 8+ chars, uppercase, digit, special character."""
    if len(v) < 8:
        raise ValueError("Password must be at least 8 characters")
    if not re.search(r"[A-Z]", v):
        raise ValueError("Password must contain at least one uppercase letter")
    if not re.search(r"[0-9]", v):
        raise ValueError("Password must contain at least one number")
    if not re.search(r"[^A-Za-z0-9]", v):
        raise ValueError("Password must contain at least one special character")
    return v


class LoginRequest(BaseModel):
    """JSON login body."""
    username: str = Field(..., min_length=1, max_length=200)
    password: str = Field(..., min_length=1, max_length=200)


class RegisterRequest(BaseModel):
    """New user registration."""
    username: str = Field(..., min_length=3, max_length=200)
    password: str = Field(..., min_length=8, max_length=200)
    name: Optional[str] = Field(None, max_length=200)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return _validate_strong_password(v)


class ChangePasswordRequest(BaseModel):
    """Password change."""
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=200)

    @field_validator("new_password")
    @classmethod
    def new_password_strength(cls, v: str) -> str:
        return _validate_strong_password(v)


class UserInfo(BaseModel):
    """Authenticated user info (returned by /me)."""
    user_id: int
    username: str
    name: Optional[str] = None


class TokenResponse(BaseModel):
    """Returned by login endpoints — includes access + refresh tokens."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int = 86400  # seconds (24h default, overridden at runtime)
    user_id: int
    username: str
    name: Optional[str] = None


class RefreshRequest(BaseModel):
    """Refresh token exchange."""
    refresh_token: str


class RefreshResponse(BaseModel):
    """Returned when a refresh token is exchanged — includes rotated refresh token."""
    access_token: str
    refresh_token: Optional[str] = None  # New rotated refresh token
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 min default

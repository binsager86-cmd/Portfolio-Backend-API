"""
User schemas — auth request/response models.
"""

from typing import Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """JSON login body."""
    username: str = Field(..., min_length=1, max_length=200)
    password: str = Field(..., min_length=1, max_length=200)


class RegisterRequest(BaseModel):
    """New user registration."""
    username: str = Field(..., min_length=3, max_length=200)
    password: str = Field(..., min_length=6, max_length=200)
    name: Optional[str] = Field(None, max_length=200)


class ChangePasswordRequest(BaseModel):
    """Password change."""
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6, max_length=200)


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

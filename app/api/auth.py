"""
Auth API routes (legacy flat prefix) — login, user info.

These endpoints are kept for backward compatibility on ``/api/auth/*``.
New clients should use ``/api/v1/auth/*`` instead.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing import Optional

from app.services.auth_service import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    get_current_user,
    TokenData,
    TokenResponse,
)
from app.core.config import get_settings

router = APIRouter(prefix="/api/auth", tags=["Authentication"])
_settings = get_settings()


# ── Request / Response models ────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=200)
    password: str = Field(..., min_length=1, max_length=200)


class UserInfo(BaseModel):
    user_id: int
    username: str
    name: Optional[str] = None


# ── Rate limiter import ───────────────────────────────────────────────
from app.core.limiter import limiter


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(request: Request, form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form.username, form.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access = create_access_token(user["id"], user["username"])
    refresh = create_refresh_token(user["id"], user["username"])
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=_settings.JWT_EXPIRE_MINUTES * 60,
        user_id=user["id"],
        username=user["username"],
        name=user.get("name"),
    )


@router.post("/login/json", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login_json(request: Request, body: LoginRequest):
    user = authenticate_user(body.username, body.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    access = create_access_token(user["id"], user["username"])
    refresh = create_refresh_token(user["id"], user["username"])
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=_settings.JWT_EXPIRE_MINUTES * 60,
        user_id=user["id"],
        username=user["username"],
        name=user.get("name"),
    )


@router.get("/me", response_model=UserInfo)
async def me(current_user: TokenData = Depends(get_current_user)):
    return UserInfo(
        user_id=current_user.user_id,
        username=current_user.username,
        name=None,
    )

"""
FastAPI Dependencies — shared across all v1 routes.

Provides:
  get_current_user  — extracts + validates JWT from Authorization header
  get_db            — yields a SQLAlchemy session (re-exported from database)
"""

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import (
    oauth2_scheme,
    decode_access_token,
    TokenData,
)
from app.core.database import query_one, query_val, get_db as _get_db  # noqa: F401


# Re-export get_db so routes can import from deps
get_db = _get_db


def _is_access_token_revoked_for_user(user_id: int, issued_at: int | None, issued_at_ms: int | None = None) -> bool:
    row = query_one(
        "SELECT COALESCE(access_tokens_revoked_at_ms, 0), COALESCE(access_tokens_revoked_at, 0) FROM users WHERE id = ?",
        (user_id,),
    )
    revoked_at_ms = row[0] if row else 0
    revoked_at = row[1] if row else 0
    if revoked_at_ms and issued_at_ms:
        return int(issued_at_ms) <= int(revoked_at_ms)
    if not revoked_at:
        return False
    if not issued_at:
        return True
    return int(issued_at) <= int(revoked_at)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    """
    Dependency that extracts & validates the JWT from the Authorization header.

    Only accepts tokens with ``type: "access"``.
    Refresh tokens are rejected (use /auth/refresh instead).
    Verifies that the user still exists in the database.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token_data = decode_access_token(token)
    except Exception:
        raise credentials_exception

    # Verify user still exists in DB
    exists = query_val("SELECT id FROM users WHERE id = ?", (token_data.user_id,))
    if not exists:
        raise credentials_exception

    if _is_access_token_revoked_for_user(token_data.user_id, token_data.iat, token_data.auth_iat_ms):
        raise credentials_exception

    return token_data


async def require_admin(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """Dependency that ensures the current user is an admin."""
    is_admin = query_val(
        "SELECT COALESCE(is_admin, 0) FROM users WHERE id = ?",
        (current_user.user_id,),
    )
    if not bool(is_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user

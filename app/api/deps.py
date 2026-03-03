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
from app.core.database import query_val, get_db as _get_db  # noqa: F401


# Re-export get_db so routes can import from deps
get_db = _get_db


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

    return token_data

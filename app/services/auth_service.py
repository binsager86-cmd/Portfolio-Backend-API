"""
Authentication Service — high-level auth logic.

The crypto primitives (JWT, password hashing) live in ``core/security.py``.
The FastAPI dependency (get_current_user) lives in ``api/deps.py``.
This module provides the business-logic layer: authenticate_user().
"""

import logging
from typing import Optional

from app.core.database import SessionLocal
from app.core.repositories.user import UserRepository
from app.core.security import (             # re-export for backward compat
    verify_password,
    hash_password,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    TokenData,
    TokenResponse,
    RefreshRequest,
    RefreshResponse,
)

# Re-export get_current_user for files that still import from here
from app.api.deps import get_current_user   # noqa: F401

logger = logging.getLogger(__name__)


# ── Login helper ─────────────────────────────────────────────────────

def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    [B-2] Verify credentials against the users table via ORM.

    Returns a dict with id, username, name, is_admin on success, else None.
    Uses UserRepository instead of raw query_one() so auth goes through
    the SQLAlchemy session (audit trail, connection pool, type safety).
    """
    db = SessionLocal()
    try:
        repo = UserRepository(db)
        user = repo.get_by_username(username)
        if user is None:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return {
            "id": user.id,
            "username": user.username,
            "name": user.name,
            "is_admin": bool(user.is_admin),
        }
    finally:
        db.close()



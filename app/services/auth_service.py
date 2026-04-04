"""
Authentication Service — high-level auth logic.

The crypto primitives (JWT, password hashing) live in ``core/security.py``.
The FastAPI dependency (get_current_user) lives in ``api/deps.py``.
This module provides the business-logic layer: authenticate_user().
"""

import logging
from typing import Optional

from app.core.database import query_one
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
    Verify credentials against the users table.
    Returns a dict with id, username, name, is_admin on success, else None.
    """
    row = query_one(
        "SELECT id, username, password_hash, name, COALESCE(is_admin, 0) FROM users WHERE username = ?",
        (username,),
    )
    if row is None:
        return None

    uid, uname, pw_hash, name, is_admin = row
    if not verify_password(password, pw_hash):
        return None

    return {"id": uid, "username": uname, "name": name, "is_admin": bool(is_admin)}


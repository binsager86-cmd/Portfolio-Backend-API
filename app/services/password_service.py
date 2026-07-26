"""Shared password-change workflows."""

import time
from typing import Any, Optional

from fastapi import Request

from app.core.database import exec_sql, transaction
from app.core.security import hash_password
from app.schemas.user import _validate_strong_password
from app.services.audit_service import AUTH_PASSWORD_CHANGE, log_event


def apply_user_password_change(
    user_id: int,
    new_password: str,
    *,
    request: Optional[Request] = None,
    actor_user_id: Optional[int] = None,
    audit_action: str = AUTH_PASSWORD_CHANGE,
    audit_details: Optional[dict[str, Any]] = None,
    access_issued_at_ms: Optional[int] = None,
) -> None:
    """Validate and change a user's password using the active transaction."""
    _validate_strong_password(new_password)
    details = dict(audit_details or {})
    if actor_user_id is not None:
        details["actor_user_id"] = actor_user_id

    now = int(time.time())
    now_ms = int(time.time() * 1000)
    access_cutoff_ms = max(now_ms, int(access_issued_at_ms) if access_issued_at_ms is not None else now_ms)
    exec_sql(
        """UPDATE users
           SET password_hash = ?,
               refresh_tokens_revoked_at = ?,
               access_tokens_revoked_at = ?,
               access_tokens_revoked_at_ms = ?,
               failed_login_attempts = 0,
               locked_until = NULL,
               last_failed_login = NULL
           WHERE id = ?""",
        (hash_password(new_password), now, now, access_cutoff_ms, user_id),
    )
    log_event(
        audit_action,
        user_id=user_id,
        resource_type="user",
        resource_id=user_id,
        details=details or None,
        request=request,
    )


def change_user_password(
    user_id: int,
    new_password: str,
    *,
    request: Optional[Request] = None,
    actor_user_id: Optional[int] = None,
    audit_action: str = AUTH_PASSWORD_CHANGE,
    audit_details: Optional[dict[str, Any]] = None,
    access_issued_at_ms: Optional[int] = None,
) -> None:
    """Validate and change a user's password, revoking sessions atomically."""
    with transaction():
        apply_user_password_change(
            user_id,
            new_password,
            request=request,
            actor_user_id=actor_user_id,
            audit_action=audit_action,
            audit_details=audit_details,
            access_issued_at_ms=access_issued_at_ms,
        )

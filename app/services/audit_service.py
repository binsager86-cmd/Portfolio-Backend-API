"""
Audit Logging Service — records all security-relevant events.

Categories:
  AUTH   — login, logout, failed_login, register, password_change, token_refresh
  TXN    — transaction_create, transaction_update, transaction_delete, transaction_restore
  CASH   — deposit_create, deposit_update, deposit_delete, deposit_restore
  ADMIN  — user_lockout, user_unlock, config_change
"""

import json
import time
import logging
from typing import Any, Optional

from fastapi import Request

from app.core.database import exec_sql

logger = logging.getLogger(__name__)

# ── Action constants ─────────────────────────────────────────────────

# Auth
AUTH_LOGIN = "auth.login"
AUTH_LOGIN_FAILED = "auth.login_failed"
AUTH_REGISTER = "auth.register"
AUTH_LOGOUT = "auth.logout"
AUTH_PASSWORD_CHANGE = "auth.password_change"
AUTH_TOKEN_REFRESH = "auth.token_refresh"
AUTH_LOCKOUT = "auth.lockout"
AUTH_UNLOCK = "auth.unlock"

# Financial
TXN_CREATE = "transaction.create"
TXN_UPDATE = "transaction.update"
TXN_DELETE = "transaction.delete"
TXN_RESTORE = "transaction.restore"

CASH_CREATE = "cash_deposit.create"
CASH_UPDATE = "cash_deposit.update"
CASH_DELETE = "cash_deposit.delete"
CASH_RESTORE = "cash_deposit.restore"

# Admin
ADMIN_ACTION = "admin.action"


def _get_ip(request: Optional[Request] = None) -> Optional[str]:
    if request is None:
        return None
    # Respect X-Forwarded-For behind reverse proxy
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


def _get_ua(request: Optional[Request] = None) -> Optional[str]:
    if request is None:
        return None
    return request.headers.get("user-agent", "")[:500]


def log_event(
    action: str,
    *,
    user_id: Optional[int] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[int] = None,
    details: Optional[dict[str, Any]] = None,
    request: Optional[Request] = None,
) -> None:
    """
    Record an audit event in the audit_log table.

    This is fire-and-forget; failures are logged but never raise.
    """
    try:
        exec_sql(
            """INSERT INTO audit_log
               (user_id, action, resource_type, resource_id, details,
                ip_address, user_agent, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                action,
                resource_type,
                resource_id,
                json.dumps(details) if details else None,
                _get_ip(request),
                _get_ua(request),
                int(time.time()),
            ),
        )
    except Exception as exc:
        # Never let audit logging break the request
        logger.error("Audit log write failed: %s — action=%s user=%s", exc, action, user_id)

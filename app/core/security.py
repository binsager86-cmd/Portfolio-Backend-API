"""
Security Module — JWT tokens + password hashing.

Provides:
  - Password hashing/verification (bcrypt via raw ``bcrypt`` library)
  - Access token creation (short-lived, type="access")
  - Refresh token creation (long-lived, type="refresh")
  - Token decoding with type enforcement

Extracted from auth_service.py for clean separation:
  core/security.py          →  crypto primitives, token encode/decode
  api/deps.py               →  FastAPI dependencies (get_current_user)
  services/auth_service.py  →  high-level auth logic (authenticate_user)
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid

import bcrypt as _bc
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

# ── OAuth2 scheme (used by deps.py) ─────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login/form")

# Token type constants
TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"

# Refresh token lifetime (from config)
REFRESH_TOKEN_EXPIRE_DAYS = _settings.REFRESH_TOKEN_EXPIRE_DAYS


# ── Token data models ────────────────────────────────────────────────

class TokenData(BaseModel):
    """Decoded JWT payload — carried through request lifecycle."""
    user_id: int
    username: str
    token_type: str = TOKEN_TYPE_ACCESS
    jti: Optional[str] = None
    exp: Optional[int] = None
    is_admin: bool = False


class TokenResponse(BaseModel):
    """Returned by login endpoints — now includes refresh_token."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int = _settings.JWT_EXPIRE_MINUTES * 60  # seconds
    user_id: int
    username: str
    name: Optional[str] = None
    is_admin: bool = False


class RefreshRequest(BaseModel):
    """Body for the /auth/refresh endpoint."""
    refresh_token: str


class RefreshResponse(BaseModel):
    """Returned by the token refresh endpoint."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = _settings.JWT_EXPIRE_MINUTES * 60


# ── Password helpers ─────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt (configurable rounds from settings)."""
    return _bc.hashpw(
        password.encode("utf-8"),
        _bc.gensalt(rounds=_settings.BCRYPT_ROUNDS),
    ).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify a plain password against a stored bcrypt hash.

    The legacy plaintext fallback is gated behind LEGACY_PLAINTEXT_LOGIN.
    When enabled it auto-upgrades the stored value to bcrypt on match.
    """
    # 1. bcrypt hash comparison (standard path)
    try:
        h = hashed.encode("utf-8") if isinstance(hashed, str) else hashed
        if _bc.checkpw(plain.encode("utf-8"), h):
            return True
    except Exception:
        pass

    # 2. plaintext fallback — only if explicitly enabled (migration aid)
    if _settings.LEGACY_PLAINTEXT_LOGIN and plain == hashed:
        logger.warning("Plaintext password matched — auto-upgrading to bcrypt hash")
        _auto_upgrade_password(plain, hashed)
        return True

    return False


def _auto_upgrade_password(plain: str, old_hash: str) -> None:
    """Replace a plaintext/weak password in the DB with a bcrypt hash."""
    try:
        from app.core.database import exec_sql

        new_hash = _bc.hashpw(
            plain.encode("utf-8"),
            _bc.gensalt(rounds=_settings.BCRYPT_ROUNDS),
        ).decode("utf-8")
        exec_sql(
            "UPDATE users SET password_hash = ? WHERE password_hash = ?",
            (new_hash, old_hash),
        )
        logger.info("Password auto-upgraded to bcrypt for legacy user")
    except Exception as exc:
        logger.error("Failed to auto-upgrade password: %s", exc)


# ── JWT helpers ──────────────────────────────────────────────────────

def create_access_token(
    user_id: int,
    username: str,
    expires_delta: Optional[timedelta] = None,
    is_admin: bool = False,
) -> str:
    """Create a signed JWT access token (short-lived)."""
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=_settings.JWT_EXPIRE_MINUTES)
    )
    payload = {
        "sub": str(user_id),
        "username": username,
        "type": TOKEN_TYPE_ACCESS,
        "exp": expire,
        "jti": uuid.uuid4().hex,
        "is_admin": is_admin,
    }
    return jwt.encode(payload, _settings.SECRET_KEY, algorithm=_settings.JWT_ALGORITHM)


def create_refresh_token(
    user_id: int,
    username: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT refresh token (long-lived).

    Each refresh token has a unique JTI so it can be revoked
    on rotation (old token blacklisted when new one is issued).
    """
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    )
    payload = {
        "sub": str(user_id),
        "username": username,
        "type": TOKEN_TYPE_REFRESH,
        "exp": expire,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, _settings.SECRET_KEY, algorithm=_settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> TokenData:
    """
    Decode and validate a JWT **access** token.

    Raises ``JWTError`` if:
      - signature is invalid
      - token is expired
      - ``sub`` (user_id) is missing
      - ``type`` is not "access"
    """
    payload = jwt.decode(
        token, _settings.SECRET_KEY, algorithms=[_settings.JWT_ALGORITHM]
    )
    user_id = int(payload.get("sub", 0))
    username = payload.get("username", "")
    token_type = payload.get("type", TOKEN_TYPE_ACCESS)
    jti = payload.get("jti")
    exp = payload.get("exp")
    is_admin = bool(payload.get("is_admin", False))

    if not user_id:
        raise JWTError("Missing subject")
    if token_type != TOKEN_TYPE_ACCESS:
        raise JWTError(f"Expected access token, got {token_type}")

    return TokenData(user_id=user_id, username=username, token_type=token_type, jti=jti, exp=exp, is_admin=is_admin)


def decode_refresh_token(token: str) -> TokenData:
    """
    Decode and validate a JWT **refresh** token.

    Raises ``JWTError`` if:
      - signature is invalid
      - token is expired
      - ``sub`` (user_id) is missing
      - ``type`` is not "refresh"
    """
    payload = jwt.decode(
        token, _settings.SECRET_KEY, algorithms=[_settings.JWT_ALGORITHM]
    )
    user_id = int(payload.get("sub", 0))
    username = payload.get("username", "")
    token_type = payload.get("type", "")
    jti = payload.get("jti")
    exp = payload.get("exp")

    if not user_id:
        raise JWTError("Missing subject")
    if token_type != TOKEN_TYPE_REFRESH:
        raise JWTError(f"Expected refresh token, got {token_type}")

    return TokenData(user_id=user_id, username=username, token_type=token_type, jti=jti, exp=exp)

    return TokenData(user_id=user_id, username=username, token_type=token_type)

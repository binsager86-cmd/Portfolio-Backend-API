"""
Auth API v1 — login, register, token refresh, user info, change password.

Security features:
  - Account lockout after N failed login attempts
  - Refresh token rotation (old token blacklisted on each refresh)
  - Audit logging on all auth events
"""

import time
import logging

from fastapi import APIRouter, Depends, Request, status
from fastapi.security import OAuth2PasswordRequestForm

from app.core.limiter import limiter
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    hash_password,
    verify_password,
)
from app.core.config import get_settings
from app.core.exceptions import UnauthorizedError, ConflictError, BadRequestError
from pydantic import BaseModel
from app.core.database import query_one, query_val, exec_sql, column_exists
from app.api.deps import get_current_user
from app.schemas.user import (
    LoginRequest,
    RegisterRequest,
    ChangePasswordRequest,
    UserInfo,
    TokenResponse,
    RefreshRequest,
    RefreshResponse,
)
from app.services.auth_service import authenticate_user
from app.services.audit_service import (
    log_event,
    AUTH_LOGIN,
    AUTH_LOGIN_FAILED,
    AUTH_REGISTER,
    AUTH_GOOGLE_LOGIN,
    AUTH_PASSWORD_CHANGE,
    AUTH_TOKEN_REFRESH,
    AUTH_LOCKOUT,
)

logger = logging.getLogger(__name__)
_settings = get_settings()

router = APIRouter(prefix="/auth", tags=["Auth"])


# ── Lockout helpers ──────────────────────────────────────────────────

def _check_lockout(username: str) -> None:
    """Raise 400 if the account is currently locked."""
    row = query_one(
        "SELECT locked_until, failed_login_attempts FROM users WHERE username = ?",
        (username,),
    )
    if row is None:
        return  # user doesn't exist — handled later by authenticate_user
    locked_until = row[0]
    if locked_until and locked_until > int(time.time()):
        remaining = locked_until - int(time.time())
        raise BadRequestError(
            f"Account locked due to too many failed attempts. "
            f"Try again in {remaining} seconds."
        )


def _record_failed_attempt(username: str, request: Request) -> None:
    """Increment failure counter; lock if threshold reached."""
    now = int(time.time())
    exec_sql(
        "UPDATE users SET failed_login_attempts = COALESCE(failed_login_attempts, 0) + 1, "
        "last_failed_login = ? WHERE username = ?",
        (now, username),
    )
    row = query_one(
        "SELECT id, failed_login_attempts FROM users WHERE username = ?",
        (username,),
    )
    if row and row[1] >= _settings.ACCOUNT_LOCKOUT_ATTEMPTS:
        lockout_until = now + _settings.ACCOUNT_LOCKOUT_MINUTES * 60
        exec_sql(
            "UPDATE users SET locked_until = ? WHERE username = ?",
            (lockout_until, username),
        )
        log_event(
            AUTH_LOCKOUT,
            user_id=row[0],
            details={"reason": "max_failed_attempts", "attempts": row[1]},
            request=request,
        )
        logger.warning("Account '%s' locked until %s", username, lockout_until)


def _reset_lockout(user_id: int) -> None:
    """Clear lockout counters on successful login."""
    exec_sql(
        "UPDATE users SET failed_login_attempts = 0, locked_until = NULL, "
        "last_failed_login = NULL WHERE id = ?",
        (user_id,),
    )


# ── Token blacklist helpers ──────────────────────────────────────────

def _blacklist_token(jti: str, user_id: int, expires_at: int) -> None:
    """Add a refresh token's JTI to the blacklist."""
    if not jti:
        return
    try:
        exec_sql(
            "INSERT INTO token_blacklist (jti, user_id, blacklisted_at, expires_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            (jti, user_id, int(time.time()), expires_at),
        )
    except Exception as exc:
        logger.error("Token blacklist write failed: %s", exc)


def _is_token_blacklisted(jti: str) -> bool:
    """Check if a refresh token JTI has been revoked."""
    if not jti:
        return False
    val = query_val(
        "SELECT 1 FROM token_blacklist WHERE jti = ?", (jti,)
    )
    return val is not None


# ── Helper: build token response ─────────────────────────────────────

def _build_token_response(user: dict) -> TokenResponse:
    """Create access + refresh tokens and wrap in a TokenResponse."""
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


# ── Login (JSON body) ────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login_json(request: Request, body: LoginRequest):
    """
    JSON-body login — friendlier for mobile / React Native clients.
    Accepts {"username": "...", "password": "..."} and returns a Bearer JWT
    plus a long-lived refresh token.
    """
    _check_lockout(body.username)

    user = authenticate_user(body.username, body.password)
    if user is None:
        _record_failed_attempt(body.username, request)
        log_event(AUTH_LOGIN_FAILED, details={"username": body.username}, request=request)
        raise UnauthorizedError("Invalid username or password")

    _reset_lockout(user["id"])
    log_event(AUTH_LOGIN, user_id=user["id"], request=request)

    return _build_token_response(user)


# ── Login (OAuth2 form — for Swagger UI) ─────────────────────────────

@router.post("/login/form", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login_form(request: Request, form: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 password-grant login.
    Use Swagger UI "Authorize" button or POST form-encoded
    username + password. Returns a Bearer JWT token + refresh token.
    """
    _check_lockout(form.username)

    user = authenticate_user(form.username, form.password)
    if user is None:
        _record_failed_attempt(form.username, request)
        log_event(AUTH_LOGIN_FAILED, details={"username": form.username}, request=request)
        raise UnauthorizedError("Invalid username or password")

    _reset_lockout(user["id"])
    log_event(AUTH_LOGIN, user_id=user["id"], request=request)

    return _build_token_response(user)


# ── Token refresh ────────────────────────────────────────────────────

@router.post("/refresh", response_model=RefreshResponse)
@limiter.limit("10/minute")
async def refresh_token(request: Request, body: RefreshRequest):
    """
    Exchange a valid refresh token for new access + refresh tokens.

    **Rotation:** The old refresh token's JTI is blacklisted and a fresh
    refresh token is issued.  Reuse of a revoked token is detected and
    rejected (replay protection).
    """
    try:
        token_data = decode_refresh_token(body.refresh_token)
    except Exception:
        raise UnauthorizedError("Invalid or expired refresh token")

    # Replay detection — reject blacklisted tokens
    if token_data.jti and _is_token_blacklisted(token_data.jti):
        logger.warning(
            "Replay detected: blacklisted refresh token jti=%s user=%s",
            token_data.jti, token_data.user_id,
        )
        raise UnauthorizedError("Refresh token has been revoked")

    # Verify user still exists
    exists = query_val("SELECT id FROM users WHERE id = ?", (token_data.user_id,))
    if not exists:
        raise UnauthorizedError("User not found")

    # Blacklist the old refresh token
    _blacklist_token(
        jti=token_data.jti or "",
        user_id=token_data.user_id,
        expires_at=token_data.exp or 0,
    )

    # Issue rotated tokens
    new_access = create_access_token(token_data.user_id, token_data.username)
    new_refresh = create_refresh_token(token_data.user_id, token_data.username)

    log_event(AUTH_TOKEN_REFRESH, user_id=token_data.user_id, request=request)

    return RefreshResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        expires_in=_settings.JWT_EXPIRE_MINUTES * 60,
    )


# ── Current user info ────────────────────────────────────────────────

@router.get("/me", response_model=UserInfo)
async def me(current_user=Depends(get_current_user)):
    """Return info about the authenticated user."""
    # Fetch full name from DB
    row = query_one(
        "SELECT name FROM users WHERE id = ?", (current_user.user_id,)
    )
    name = row[0] if row else None
    return UserInfo(
        user_id=current_user.user_id,
        username=current_user.username,
        name=name,
    )


# ── Register ─────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("3/minute")
async def register(request: Request, body: RegisterRequest):
    """Create a new user account and return JWT + refresh token."""
    # Check if username already exists
    existing = query_val(
        "SELECT id FROM users WHERE username = ?", (body.username,)
    )
    if existing:
        raise ConflictError(f"Username '{body.username}' already taken")

    hashed = hash_password(body.password)
    now = int(time.time())

    exec_sql(
        "INSERT INTO users (username, password_hash, name, created_at, failed_login_attempts) "
        "VALUES (?, ?, ?, ?, 0)",
        (body.username, hashed, body.name, now),
    )

    # Fetch the new user's ID
    user_id = query_val(
        "SELECT id FROM users WHERE username = ?", (body.username,)
    )

    log_event(AUTH_REGISTER, user_id=user_id, request=request)

    return _build_token_response({
        "id": user_id,
        "username": body.username,
        "name": body.name,
    })


# ── Google Sign-In ───────────────────────────────────────────────────

class GoogleSignInRequest(BaseModel):
    id_token: str


@router.post("/google", response_model=TokenResponse)
@limiter.limit("10/minute")
async def google_sign_in(request: Request, body: GoogleSignInRequest):
    """
    Validate a Google token, create or find the user, and return JWT tokens.

    Accepts either:
      - A Google ID token (from native @react-native-google-signin)
      - A Google access token (from web expo-auth-session OAuth flow)

    We try ID-token verification first; if that fails we try the
    access_token userinfo endpoint as a fallback.
    """
    import httpx

    google_data = None
    token = body.id_token

    async with httpx.AsyncClient(timeout=10) as client:
        # ── Attempt 1: verify as id_token ────────────────────────
        try:
            resp = await client.get(
                f"https://oauth2.googleapis.com/tokeninfo?id_token={token}"
            )
            if resp.status_code == 200:
                google_data = resp.json()
        except httpx.HTTPError:
            pass  # fall through to access_token attempt

        # ── Attempt 2: verify as access_token via userinfo ───────
        if google_data is None or "email" not in google_data:
            try:
                resp = await client.get(
                    "https://www.googleapis.com/oauth2/v3/userinfo",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if resp.status_code == 200:
                    google_data = resp.json()
            except httpx.HTTPError:
                pass

    if not google_data:
        raise UnauthorizedError("Invalid Google token — could not verify with Google.")

    email = google_data.get("email")
    if not email:
        raise BadRequestError("Google account does not have an email address")

    google_name = google_data.get("name", "")
    google_sub = google_data.get("sub", "")  # Google user ID

    # Look up user by email (username) or google_sub
    existing = query_one(
        "SELECT id, username, name FROM users WHERE username = ?",
        (email,),
    )

    if existing:
        # Existing user — log in
        user = {"id": existing[0], "username": existing[1], "name": existing[2]}
        _reset_lockout(user["id"])
        log_event(AUTH_GOOGLE_LOGIN, user_id=user["id"], details={"google_sub": google_sub}, request=request)
        return _build_token_response(user)

    # New user — create account (no password needed)
    import secrets
    random_pw_hash = hash_password(secrets.token_urlsafe(32))
    now = int(time.time())

    exec_sql(
        "INSERT INTO users (username, password_hash, name, created_at, failed_login_attempts) "
        "VALUES (?, ?, ?, ?, 0)",
        (email, random_pw_hash, google_name or email.split("@")[0], now),
    )

    user_id = query_val("SELECT id FROM users WHERE username = ?", (email,))
    user = {"id": user_id, "username": email, "name": google_name or email.split("@")[0]}

    log_event(AUTH_GOOGLE_LOGIN, user_id=user_id, details={"google_sub": google_sub, "new_account": True}, request=request)
    return _build_token_response(user)


# ── Change password ──────────────────────────────────────────────────

@router.put("/change-password")
async def change_password(
    request: Request,
    body: ChangePasswordRequest,
    current_user=Depends(get_current_user),
):
    """Change the authenticated user's password."""
    # Verify current password
    row = query_one(
        "SELECT password_hash FROM users WHERE id = ?",
        (current_user.user_id,),
    )
    if row is None:
        raise UnauthorizedError("User not found")

    if not verify_password(body.current_password, row[0]):
        raise BadRequestError("Current password is incorrect")

    new_hash = hash_password(body.new_password)
    exec_sql(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (new_hash, current_user.user_id),
    )

    log_event(AUTH_PASSWORD_CHANGE, user_id=current_user.user_id, request=request)

    return {"status": "ok", "message": "Password changed successfully"}


# ── API Key management ───────────────────────────────────────────────

def _ensure_api_key_column():
    """Additive migration: add gemini_api_key column if missing."""
    if not column_exists("users", "gemini_api_key"):
        exec_sql("ALTER TABLE users ADD COLUMN gemini_api_key TEXT DEFAULT ''")


class ApiKeyRequest(BaseModel):
    api_key: str


@router.put("/api-key")
async def save_api_key(
    body: ApiKeyRequest,
    current_user=Depends(get_current_user),
):
    """Save user's Gemini API key."""
    _ensure_api_key_column()
    exec_sql(
        "UPDATE users SET gemini_api_key = ? WHERE id = ?",
        (body.api_key.strip(), current_user.user_id),
    )
    return {"status": "ok", "data": {"message": "API key saved"}}


@router.get("/api-key")
async def get_api_key(current_user=Depends(get_current_user)):
    """Get user's saved API key (masked)."""
    _ensure_api_key_column()
    row = query_one(
        "SELECT gemini_api_key FROM users WHERE id = ?",
        (current_user.user_id,),
    )
    key = row[0] if row and row[0] else ""
    has_key = bool(key)
    masked = key[:4] + "..." + key[-4:] if len(key) > 8 else ("****" if key else None)
    return {"status": "ok", "data": {"has_key": has_key, "masked_key": masked}}

"""
Unit & integration tests for the authentication system.

Covers:
  - Password hashing / verification
  - Access token creation, decoding, type enforcement
  - Refresh token creation, decoding, type enforcement
  - Refresh token cannot be used as access token (and vice-versa)
  - /auth/refresh endpoint (happy path + error cases)
  - /auth/register endpoint
  - Login returns both access_token and refresh_token
"""

from datetime import timedelta

import pytest
from jose import jwt


# ── Unit: password helpers ────────────────────────────────────────────

def test_hash_password_produces_bcrypt():
    from app.core.security import hash_password

    hashed = hash_password("secret123")
    assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
    assert len(hashed) == 60


def test_verify_password_correct():
    from app.core.security import hash_password, verify_password

    hashed = hash_password("mypassword")
    assert verify_password("mypassword", hashed) is True


def test_verify_password_wrong():
    from app.core.security import hash_password, verify_password

    hashed = hash_password("mypassword")
    assert verify_password("wrongpass", hashed) is False


# ── Unit: access token ───────────────────────────────────────────────

def test_create_access_token_has_type_claim():
    from app.core.security import create_access_token
    from app.core.config import get_settings

    settings = get_settings()
    token = create_access_token(42, "alice")
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])

    assert payload["sub"] == "42"
    assert payload["username"] == "alice"
    assert payload["type"] == "access"


def test_decode_access_token_roundtrip():
    from app.core.security import create_access_token, decode_access_token

    token = create_access_token(7, "bob")
    data = decode_access_token(token)
    assert data.user_id == 7
    assert data.username == "bob"
    assert data.token_type == "access"


def test_decode_access_token_rejects_refresh():
    """A refresh token must NOT be accepted by decode_access_token."""
    from app.core.security import create_refresh_token, decode_access_token
    from jose import JWTError

    refresh = create_refresh_token(1, "alice")
    with pytest.raises(JWTError, match="Expected access token"):
        decode_access_token(refresh)


# ── Unit: refresh token ──────────────────────────────────────────────

def test_create_refresh_token_has_type_claim():
    from app.core.security import create_refresh_token
    from app.core.config import get_settings

    settings = get_settings()
    token = create_refresh_token(42, "alice")
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])

    assert payload["sub"] == "42"
    assert payload["type"] == "refresh"


def test_decode_refresh_token_roundtrip():
    from app.core.security import create_refresh_token, decode_refresh_token

    token = create_refresh_token(9, "carol")
    data = decode_refresh_token(token)
    assert data.user_id == 9
    assert data.username == "carol"
    assert data.token_type == "refresh"


def test_decode_refresh_token_rejects_access():
    """An access token must NOT be accepted by decode_refresh_token."""
    from app.core.security import create_access_token, decode_refresh_token
    from jose import JWTError

    access = create_access_token(1, "alice")
    with pytest.raises(JWTError, match="Expected refresh token"):
        decode_refresh_token(access)


def test_access_token_custom_expiry():
    from app.core.security import create_access_token
    from app.core.config import get_settings

    settings = get_settings()
    token = create_access_token(1, "alice", expires_delta=timedelta(minutes=5))
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    assert payload["type"] == "access"


# ── Integration: login returns refresh token ─────────────────────────

def test_login_returns_refresh_token(test_client):
    """Login should return both access_token and refresh_token."""
    resp = test_client.post(
        "/api/v1/auth/login",
        json={"username": "testuser", "password": "testpass123"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert "refresh_token" in body
    assert body["refresh_token"] is not None
    assert body["token_type"] == "bearer"
    assert "expires_in" in body
    assert body["expires_in"] > 0


# ── Integration: /auth/refresh endpoint ──────────────────────────────

def test_refresh_endpoint_happy_path(test_client):
    """Valid refresh token returns a new access token."""
    # First login to get a refresh token
    login_resp = test_client.post(
        "/api/v1/auth/login",
        json={"username": "testuser", "password": "testpass123"},
    )
    refresh_token = login_resp.json()["refresh_token"]

    # Exchange it for a new access token
    resp = test_client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0


def test_refresh_endpoint_rejects_access_token(test_client):
    """The refresh endpoint must reject access tokens."""
    from app.core.security import create_access_token

    # Create an access token directly (avoids rate limit)
    access_token = create_access_token(1, "testuser")

    resp = test_client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": access_token},
    )
    assert resp.status_code == 401


def test_refresh_endpoint_rejects_garbage(test_client):
    """The refresh endpoint rejects invalid tokens."""
    resp = test_client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": "not-a-real-token"},
    )
    assert resp.status_code == 401


def test_refreshed_access_token_works(test_client):
    """A refreshed access token should work on protected endpoints."""
    from app.core.security import create_refresh_token

    # Create a refresh token directly (avoids rate limit)
    refresh_token = create_refresh_token(1, "testuser")

    # Refresh → get new access token
    refresh_resp = test_client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refresh_resp.status_code == 200
    new_access = refresh_resp.json()["access_token"]

    # Use new access token on /me
    me_resp = test_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {new_access}"},
    )
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == "testuser"


# ── Integration: refresh token rejected on protected endpoints ───────

def test_refresh_token_rejected_on_protected_endpoint(test_client):
    """Refresh tokens must NOT work on normal protected endpoints."""
    from app.core.security import create_refresh_token

    # Create a refresh token directly (avoids rate limit)
    refresh_token = create_refresh_token(1, "testuser")

    # Try to use refresh token as Authorization bearer
    resp = test_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {refresh_token}"},
    )
    assert resp.status_code == 401


# ── Integration: register ────────────────────────────────────────────

def test_register_returns_tokens(test_client):
    """Registration should return both access and refresh tokens."""
    import time

    resp = test_client.post(
        "/api/v1/auth/register",
        json={
            "username": f"newuser_{int(time.time())}",
            "password": "securepass123",
            "name": "New User",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "access_token" in body
    assert "refresh_token" in body
    assert body["refresh_token"] is not None
    assert body["name"] == "New User"


def test_register_duplicate_username(test_client):
    """Registering with an existing username returns 409."""
    resp = test_client.post(
        "/api/v1/auth/register",
        json={
            "username": "testuser",
            "password": "anotherpass123",
        },
    )
    assert resp.status_code == 409

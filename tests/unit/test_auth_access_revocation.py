from datetime import datetime, timedelta, timezone

from jose import jwt

from app.core.config import get_settings
from app.core.database import exec_sql, query_val
from app.core.security import hash_password
from tests.helpers import ensure_user2


def _set_user_password(user_id: int, password: str) -> None:
    exec_sql(
        "UPDATE users SET password_hash = ?, access_tokens_revoked_at = 0, access_tokens_revoked_at_ms = 0, refresh_tokens_revoked_at = 0 WHERE id = ?",
        (hash_password(password), user_id),
    )


def test_password_change_revokes_existing_access_token(test_client):
    user2 = ensure_user2(test_client)
    _set_user_password(user2["user_id"], "user2pass789")
    user2 = ensure_user2(test_client)
    old_token = user2["headers"]["Authorization"].split(" ", 1)[1]

    response = test_client.put(
        "/api/v1/auth/change-password",
        headers=user2["headers"],
        json={"current_password": "user2pass789", "new_password": "Newpass123!"},
    )
    assert response.status_code == 200, response.text

    revoked_at = query_val("SELECT access_tokens_revoked_at FROM users WHERE id = ?", (user2["user_id"],))
    assert revoked_at is not None
    assert int(revoked_at) > 0
    revoked_at_ms = query_val("SELECT access_tokens_revoked_at_ms FROM users WHERE id = ?", (user2["user_id"],))
    assert revoked_at_ms is not None
    assert int(revoked_at_ms) > 0

    old_response = test_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {old_token}"},
    )
    assert old_response.status_code == 401

    relogin = test_client.post(
        "/api/v1/auth/login",
        json={"username": "user2", "password": "Newpass123!"},
    )
    assert relogin.status_code == 200, relogin.text
    me_response = test_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {relogin.json()['access_token']}"},
    )
    assert me_response.status_code == 200, me_response.text


def test_logout_revokes_current_access_token(test_client):
    user2 = ensure_user2(test_client)
    _set_user_password(user2["user_id"], "Newpass123!")
    login = test_client.post(
        "/api/v1/auth/login",
        json={"username": "user2", "password": "Newpass123!"},
    )
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    response = test_client.post("/api/v1/auth/logout", headers=headers)
    assert response.status_code == 200, response.text

    old_response = test_client.get("/api/v1/auth/me", headers=headers)
    assert old_response.status_code == 401

    relogin = test_client.post(
        "/api/v1/auth/login",
        json={"username": "user2", "password": "Newpass123!"},
    )
    assert relogin.status_code == 200, relogin.text
    me_response = test_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {relogin.json()['access_token']}"},
    )
    assert me_response.status_code == 200, me_response.text


def test_access_token_without_iat_is_rejected_after_revocation(test_client):
    settings = get_settings()
    token = jwt.encode(
        {
            "sub": "1",
            "username": "testuser",
            "type": "access",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    exec_sql("UPDATE users SET access_tokens_revoked_at = ? WHERE id = ?", (int(datetime.now(timezone.utc).timestamp()), 1))

    try:
        response = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 401
    finally:
        exec_sql("UPDATE users SET access_tokens_revoked_at = 0, access_tokens_revoked_at_ms = 0 WHERE id = ?", (1,))

"""
Integration tests for API endpoints.

Uses the test_client and auth_headers fixtures from conftest.py.
"""


def test_health(test_client):
    """Health endpoint requires no auth."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_login_success(test_client):
    """Login with valid credentials returns a JWT."""
    resp = test_client.post(
        "/api/v1/auth/login",
        json={"username": "testuser", "password": "testpass123"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert body["username"] == "testuser"


def test_login_failure(test_client):
    """Login with wrong password returns 401."""
    resp = test_client.post(
        "/api/v1/auth/login",
        json={"username": "testuser", "password": "wrongpass"},
    )
    assert resp.status_code == 401


def test_me_requires_auth(test_client):
    """The /me endpoint requires authentication."""
    resp = test_client.get("/api/v1/auth/me")
    assert resp.status_code in (401, 403)


def test_me_with_auth(test_client, auth_headers):
    """The /me endpoint returns user info when authenticated."""
    resp = test_client.get("/api/v1/auth/me", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["username"] == "testuser"


def test_portfolio_overview_requires_auth(test_client):
    """Portfolio overview requires auth."""
    resp = test_client.get("/api/v1/portfolio/overview")
    assert resp.status_code in (401, 403)


def test_portfolio_overview(test_client, auth_headers):
    """Portfolio overview returns structured data."""
    resp = test_client.get("/api/v1/portfolio/overview", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "data" in body


def test_cron_status(test_client):
    """Cron status endpoint is public."""
    resp = test_client.get("/api/v1/cron/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_fx_rate(test_client, auth_headers):
    """FX rate endpoint returns a rate."""
    resp = test_client.get("/api/v1/portfolio/fx-rate", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "usd_kwd" in body.get("data", {})

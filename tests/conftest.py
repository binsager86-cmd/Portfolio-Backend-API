"""
Test fixtures — shared across all test modules.

Provides:
  - test_client: FastAPI TestClient with auth header
  - test_db: In-memory SQLite database for isolation
  - auth_headers: Valid JWT authorization header
"""

import os
import sqlite3
import tempfile
import time

import pandas as pd
pd.set_option("future.no_silent_downcasting", True)

import pytest
from fastapi.testclient import TestClient

# Create a temporary DB file BEFORE importing app modules
_test_db_fd, _test_db_path = tempfile.mkstemp(suffix=".db", prefix="test_portfolio_")
os.close(_test_db_fd)

os.environ["DATABASE_PATH"] = _test_db_path
os.environ["ENVIRONMENT"] = "development"
os.environ["SECRET_KEY"] = "test-secret-key-for-unit-tests"
os.environ["CRON_SECRET_KEY"] = "test-cron-key"
os.environ["PRICE_UPDATE_ENABLED"] = "false"
os.environ["RATE_LIMIT_ENABLED"] = "false"
os.environ.setdefault("GEMINI_API_KEY", "test-fake-gemini-key")


@pytest.fixture(scope="session")
def _init_test_db():
    """Create the test database using the app's own schema initialization, then seed test data."""
    db_path = _test_db_path

    # Let the app's schema module create all tables with the correct columns.
    # This avoids manual schema duplication drifting out of sync.
    from app.core.schema import ensure_all_tables
    ensure_all_tables()

    # Also run the fundamental schema setup (creates analysis tables)
    try:
        from app.api.v1.fundamental import _ensure_schema as _ensure_fundamental_schema
        _ensure_fundamental_schema()
    except Exception:
        pass

    # Seed a test user (pre-computed bcrypt hash for "testpass123")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()

    _test_hash = "$2b$12$drYtGzFmYlnMLLvZdo5nauyYZUN0slnBha1iCtgLqGghj/OfBHuwm"
    cur.execute(
        "INSERT INTO users (username, password_hash, name, created_at) VALUES (?, ?, ?, ?)",
        ("testuser", _test_hash, "Test User", int(time.time())),
    )

    # Seed portfolios
    for pname, ccy in [("KFH", "KWD"), ("BBYN", "KWD"), ("USA", "USD")]:
        cur.execute(
            "INSERT INTO portfolios (user_id, name, currency, created_at) VALUES (1, ?, ?, ?)",
            (pname, ccy, int(time.time())),
        )

    conn.commit()
    conn.close()

    yield

    # Cleanup temp DB
    try:
        os.unlink(_test_db_path)
    except OSError:
        pass


@pytest.fixture(scope="session")
def test_client(_init_test_db) -> TestClient:
    """FastAPI test client with a seeded in-memory database."""
    from app.main import app
    return TestClient(app)


@pytest.fixture(scope="session")
def auth_headers(test_client: TestClient) -> dict:
    """Get a valid JWT auth header for the test user."""
    resp = test_client.post(
        "/api/v1/auth/login",
        json={"username": "testuser", "password": "testpass123"},
    )
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

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


@pytest.fixture(scope="session")
def _init_test_db():
    """Create the test database schema in a temp file."""
    db_path = _test_db_path

    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()

    # Minimal schema for testing
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT,
            created_at INTEGER,
            failed_login_attempts INTEGER DEFAULT 0,
            locked_until INTEGER,
            last_failed_login INTEGER
        );

        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            currency TEXT DEFAULT 'KWD',
            description TEXT,
            created_at INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            portfolio TEXT NOT NULL,
            stock_symbol TEXT NOT NULL,
            txn_date TEXT,
            txn_type TEXT NOT NULL,
            shares REAL,
            purchase_cost REAL,
            sell_value REAL,
            bonus_shares REAL,
            cash_dividend REAL,
            reinvested_dividend REAL,
            fees REAL,
            price_override REAL,
            planned_cum_shares REAL,
            broker TEXT,
            reference TEXT,
            notes TEXT,
            category TEXT DEFAULT 'portfolio',
            is_deleted INTEGER DEFAULT 0,
            deleted_at INTEGER,
            created_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT,
            portfolio TEXT,
            currency TEXT DEFAULT 'KWD',
            current_price REAL,
            last_updated INTEGER,
            price_source TEXT,
            tradingview_symbol TEXT,
            tradingview_exchange TEXT,
            market_cap REAL,
            sector TEXT,
            industry TEXT
        );

        CREATE TABLE IF NOT EXISTS cash_deposits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            portfolio TEXT NOT NULL,
            deposit_date TEXT NOT NULL,
            amount REAL NOT NULL,
            currency TEXT DEFAULT 'KWD',
            bank_name TEXT,
            deposit_type TEXT DEFAULT 'deposit',
            notes TEXT,
            is_deleted INTEGER DEFAULT 0,
            deleted_at INTEGER,
            created_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS external_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            portfolio_id INTEGER,
            name TEXT NOT NULL,
            account_type TEXT,
            currency TEXT DEFAULT 'KWD',
            current_balance REAL DEFAULT 0,
            last_reconciled_date TEXT,
            notes TEXT,
            created_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            portfolio TEXT,
            snapshot_date TEXT NOT NULL,
            portfolio_value REAL,
            daily_movement REAL,
            beginning_difference REAL,
            deposit_cash REAL,
            accumulated_cash REAL,
            net_gain REAL,
            change_percent REAL,
            roi_percent REAL,
            twr_percent REAL,
            mwrr_percent REAL,
            created_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS portfolio_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            portfolio_id INTEGER NOT NULL,
            account_id INTEGER,
            stock_id INTEGER,
            txn_type TEXT NOT NULL,
            txn_date TEXT NOT NULL,
            amount REAL DEFAULT 0,
            shares REAL,
            price_per_share REAL,
            fees REAL DEFAULT 0,
            currency TEXT DEFAULT 'KWD',
            fx_rate REAL,
            symbol TEXT,
            description TEXT,
            reference TEXT,
            notes TEXT,
            is_deleted INTEGER DEFAULT 0,
            deleted_at INTEGER,
            created_at INTEGER,
            updated_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS pfm_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            snapshot_date TEXT NOT NULL,
            notes TEXT,
            total_assets REAL DEFAULT 0,
            total_liabilities REAL DEFAULT 0,
            net_worth REAL DEFAULT 0,
            created_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS pfm_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            asset_type TEXT NOT NULL,
            category TEXT NOT NULL,
            name TEXT NOT NULL,
            quantity REAL,
            price REAL,
            currency TEXT DEFAULT 'KWD',
            value_kwd REAL DEFAULT 0,
            created_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS pfm_liabilities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            amount_kwd REAL DEFAULT 0,
            is_current INTEGER DEFAULT 0,
            is_long_term INTEGER DEFAULT 0,
            created_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS pfm_income_expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            category TEXT NOT NULL,
            monthly_amount REAL DEFAULT 0,
            is_finance_cost INTEGER DEFAULT 0,
            is_gna INTEGER DEFAULT 0,
            sort_order INTEGER DEFAULT 0,
            created_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS securities_master (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            exchange TEXT,
            currency TEXT DEFAULT 'KWD',
            asset_type TEXT DEFAULT 'EQUITY',
            sector TEXT,
            industry TEXT,
            country TEXT,
            yahoo_symbol TEXT,
            tradingview_symbol TEXT,
            tradingview_exchange TEXT,
            isin TEXT,
            market_cap REAL,
            outstanding_shares REAL,
            is_active INTEGER DEFAULT 1,
            created_at INTEGER,
            updated_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS stocks_master (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            exchange TEXT,
            currency TEXT DEFAULT 'KWD'
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            resource_type TEXT,
            resource_id INTEGER,
            details TEXT,
            ip_address TEXT,
            user_agent TEXT,
            created_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS token_blacklist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            jti TEXT NOT NULL UNIQUE,
            user_id INTEGER,
            blacklisted_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS position_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            stock_id INTEGER,
            stock_symbol TEXT,
            portfolio_id INTEGER,
            snapshot_date TEXT NOT NULL,
            total_shares REAL,
            total_cost REAL,
            avg_cost REAL,
            realized_pnl REAL,
            cash_dividends_received REAL,
            status TEXT DEFAULT 'OPEN'
        );

        CREATE TABLE IF NOT EXISTS portfolio_cash (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            portfolio TEXT NOT NULL,
            balance REAL,
            currency TEXT DEFAULT 'KWD',
            last_updated INTEGER,
            manual_override INTEGER DEFAULT 0
        );
    """)

    # Seed a test user (pre-computed bcrypt hash for "testpass123")
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

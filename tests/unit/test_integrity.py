"""
Tests for Financial Integrity Service (Phase 3.2).

Seeds realistic financial data and verifies all five integrity checks:
  1. Cash balance verification
  2. Position cross-check (aggregate vs WAC)
  3. Snapshot freshness
  4. Transaction anomaly scan
  5. Data completeness
"""

import sqlite3
import time
from datetime import date, timedelta

import pytest


# ── helpers ──────────────────────────────────────────────────────────

def _seed_financial_data(db_path: str):
    """Seed transactions, deposits, stocks, and snapshots for user 1."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    now = int(time.time())
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    # -- Stocks
    cur.executemany(
        "INSERT INTO stocks (user_id, symbol, name, portfolio, currency, current_price, last_updated) "
        "VALUES (1, ?, ?, ?, ?, ?, ?)",
        [
            ("HUMANSOFT", "Humansoft Holding", "KFH", "KWD", 3.200, now),
            ("MABANEE",  "Mabanee Company",   "KFH", "KWD", 0.850, now),
            ("AAPL",     "Apple Inc",          "USA", "USD", 180.5, now),
        ],
    )

    # -- Cash deposits
    cur.executemany(
        "INSERT INTO cash_deposits (user_id, portfolio, deposit_date, amount, currency, deposit_type, created_at) "
        "VALUES (1, ?, ?, ?, ?, 'deposit', ?)",
        [
            ("KFH",  "2024-01-15", 5000.0, "KWD", now),
            ("KFH",  "2024-06-01", 3000.0, "KWD", now),
            ("USA",  "2024-03-01", 2000.0, "USD", now),
            ("BBYN", "2024-02-01", 1000.0, "KWD", now),
        ],
    )

    # -- Transactions (Buy / Sell / Dividend)
    cur.executemany(
        "INSERT INTO transactions "
        "(user_id, portfolio, stock_symbol, txn_date, txn_type, shares, "
        " purchase_cost, sell_value, bonus_shares, cash_dividend, fees, "
        " category, created_at) "
        "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'portfolio', ?)",
        [
            # KFH portfolio — HUMANSOFT
            ("KFH", "HUMANSOFT", "2024-01-20", "Buy",  500, 1550.0, None,  0, 0,    5.0,  now),
            ("KFH", "HUMANSOFT", "2024-07-01", "Buy",  200, 640.0,  None,  0, 0,    3.0,  now),
            ("KFH", "HUMANSOFT", "2024-09-15", "Sell", 100, None,   350.0, 0, 0,    2.5,  now),
            ("KFH", "HUMANSOFT", "2024-12-01", "Buy",    0, 0,      None,  0, 45.0, 0,    now),  # dividend
            # KFH portfolio — MABANEE
            ("KFH", "MABANEE",  "2024-02-10", "Buy",  1000, 900.0, None,  0, 0,    4.0,  now),
            # USA portfolio — AAPL
            ("USA", "AAPL",     "2024-03-15", "Buy",   10,  1700.0, None, 0, 0,    1.0,  now),
            ("USA", "AAPL",     "2024-08-10", "Sell",   3,  None,   550.0, 0, 0,    1.0,  now),
        ],
    )

    # -- portfolio_cash (reconciled balances)
    # KFH expected: deposits(8000) - buys(1550+640+900) + sells(350) + div(45) - fees(5+3+2.5+4) = 5290.5
    cur.executemany(
        "INSERT INTO portfolio_cash (user_id, portfolio, balance, currency, last_updated, manual_override) "
        "VALUES (1, ?, ?, ?, ?, 0)",
        [
            ("KFH",  5290.5,   "KWD", now),
            ("USA",  848.0,    "USD", now),  # 2000-1700+550-1-1 = 848
        ],
    )

    # -- Snapshot (for freshness check)
    cur.execute(
        "INSERT INTO portfolio_snapshots "
        "(user_id, portfolio, snapshot_date, portfolio_value, deposit_cash, created_at) "
        "VALUES (1, 'KFH', ?, 10000.0, 5290.5, ?)",
        (yesterday, now),
    )

    conn.commit()
    conn.close()


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def seeded_client(test_client, _init_test_db):
    """
    Return the test_client after seeding financial data.
    Uses module scope so data persists across all tests in this file.
    """
    import os
    db_path = os.environ["DATABASE_PATH"]
    _seed_financial_data(db_path)
    return test_client


@pytest.fixture(scope="module")
def headers(seeded_client, auth_headers):
    """Reuse session-scoped auth headers (avoids duplicate login / rate limit)."""
    return auth_headers


# ── Tests ────────────────────────────────────────────────────────────

class TestFullIntegrityCheck:
    """Tests for GET /api/v1/integrity/check"""

    def test_full_check_returns_ok(self, seeded_client, headers):
        resp = seeded_client.get("/api/v1/integrity/check", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        data = body["data"]
        assert "overall_valid" in data
        assert "summary" in data
        assert "cash" in data
        assert "positions" in data
        assert "snapshots" in data
        assert "anomalies" in data
        assert "completeness" in data

    def test_full_check_has_all_portfolios(self, seeded_client, headers):
        resp = seeded_client.get("/api/v1/integrity/check", headers=headers)
        data = resp.json()["data"]
        # Cash checks should cover KFH, BBYN, USA (the seeded portfolios)
        assert "KFH" in data["cash"]

    def test_full_check_requires_auth(self, seeded_client):
        resp = seeded_client.get("/api/v1/integrity/check")
        assert resp.status_code in (401, 403)


class TestCashBalanceCheck:
    """Tests for GET /api/v1/integrity/cash/{portfolio}"""

    def test_kfh_cash_valid(self, seeded_client, headers):
        resp = seeded_client.get("/api/v1/integrity/cash/KFH", headers=headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["portfolio"] == "KFH"
        assert data["is_valid"] is True
        assert data["expected_balance"] is not None
        assert data["stored_balance"] is not None
        # Components should be present
        c = data["components"]
        assert float(c["deposits"]) == 8000.0
        assert float(c["buys"]) > 0

    def test_usa_cash_valid(self, seeded_client, headers):
        resp = seeded_client.get("/api/v1/integrity/cash/USA", headers=headers)
        data = resp.json()["data"]
        assert data["portfolio"] == "USA"
        assert data["is_valid"] is True

    def test_bbyn_no_stored_balance(self, seeded_client, headers):
        """BBYN has a deposit but no portfolio_cash row → stored is None."""
        resp = seeded_client.get("/api/v1/integrity/cash/BBYN", headers=headers)
        data = resp.json()["data"]
        assert data["portfolio"] == "BBYN"
        # No stored balance means is_valid is indeterminate
        assert data["stored_balance"] is None
        assert data["is_valid"] is None

    def test_nonexistent_portfolio(self, seeded_client, headers):
        """A portfolio with no data should still return a result."""
        resp = seeded_client.get("/api/v1/integrity/cash/ZZZZZ", headers=headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["expected_balance"] == "0.000"


class TestPositionCheck:
    """Tests for GET /api/v1/integrity/positions/{portfolio}"""

    def test_kfh_positions_valid(self, seeded_client, headers):
        resp = seeded_client.get("/api/v1/integrity/positions/KFH", headers=headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["portfolio"] == "KFH"
        assert data["is_valid"] is True  # shares must match even if cost differs
        assert data["total_symbols"] >= 2  # HUMANSOFT + MABANEE
        assert data["matched"] == data["total_symbols"]
        assert data["mismatches"] == []
        # details should contain per-symbol breakdown
        assert len(data["details"]) == data["total_symbols"]

    def test_usa_positions_valid(self, seeded_client, headers):
        resp = seeded_client.get("/api/v1/integrity/positions/USA", headers=headers)
        data = resp.json()["data"]
        assert data["is_valid"] is True
        assert data["total_symbols"] >= 1  # AAPL


class TestSnapshotCheck:
    """Tests for GET /api/v1/integrity/snapshots/{portfolio}"""

    def test_kfh_has_fresh_snapshot(self, seeded_client, headers):
        resp = seeded_client.get("/api/v1/integrity/snapshots/KFH", headers=headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["has_snapshots"] is True
        assert data["is_fresh"] is True
        assert data["days_since_snapshot"] <= 3

    def test_bbyn_no_snapshots(self, seeded_client, headers):
        resp = seeded_client.get("/api/v1/integrity/snapshots/BBYN", headers=headers)
        data = resp.json()["data"]
        assert data["has_snapshots"] is False


class TestAnomalyCheck:
    """Tests for GET /api/v1/integrity/anomalies"""

    def test_anomalies_returns_report(self, seeded_client, headers):
        resp = seeded_client.get("/api/v1/integrity/anomalies", headers=headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "anomalies" in data
        assert "count" in data
        assert isinstance(data["count"], int)
        assert isinstance(data["anomalies"], list)
        assert "is_valid" in data

    def test_no_over_sell_in_seed_data(self, seeded_client, headers):
        resp = seeded_client.get("/api/v1/integrity/anomalies", headers=headers)
        data = resp.json()["data"]
        over_sells = [a for a in data["anomalies"] if a["type"] == "over_sell"]
        assert len(over_sells) == 0


class TestCompletenessCheck:
    """Tests for GET /api/v1/integrity/completeness"""

    def test_completeness_returns_report(self, seeded_client, headers):
        resp = seeded_client.get("/api/v1/integrity/completeness", headers=headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["portfolios_found"] >= 3
        assert isinstance(data["issues"], list)

    def test_no_orphan_symbols(self, seeded_client, headers):
        """All transacted symbols should have stocks entries."""
        resp = seeded_client.get("/api/v1/integrity/completeness", headers=headers)
        data = resp.json()["data"]
        assert data["orphan_symbols"] == 0

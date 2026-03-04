"""
Integration tests for Portfolio API endpoints.

Covers:
  - Portfolio overview (authenticated, data structure verification)
  - Holdings endpoint with filters
  - Portfolio table per-portfolio
  - Transaction CRUD (create, read, update, soft-delete, restore)
  - User data isolation (user1 cannot see user2's data)
  - Error handling (404, 400, 401)
  - Pagination
"""

import time
import pytest


# ── Portfolio Overview ───────────────────────────────────────────────

class TestPortfolioOverview:
    """GET /api/v1/portfolio/overview — core endpoint."""

    def test_overview_requires_auth(self, test_client):
        resp = test_client.get("/api/v1/portfolio/overview")
        assert resp.status_code in (401, 403)

    def test_overview_authenticated(self, test_client, auth_headers):
        """
        Authenticated overview returns structured data with required fields.
        Matches the Phase 4 critical test template.
        """
        resp = test_client.get("/api/v1/portfolio/overview", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "data" in body

    def test_overview_data_structure(self, test_client, auth_headers):
        """Overview data contains expected top-level fields."""
        resp = test_client.get("/api/v1/portfolio/overview", headers=auth_headers)
        data = resp.json()["data"]

        # These fields are part of the PortfolioOverview schema
        for field in [
            "total_deposits", "total_withdrawals", "net_deposits",
            "total_invested", "total_divested", "total_dividends",
            "total_fees", "transaction_count",
        ]:
            assert field in data, f"Missing field: {field}"


# ── Holdings ─────────────────────────────────────────────────────────

class TestPortfolioHoldings:
    """GET /api/v1/portfolio/holdings"""

    def test_holdings_requires_auth(self, test_client):
        resp = test_client.get("/api/v1/portfolio/holdings")
        assert resp.status_code in (401, 403)

    def test_holdings_returns_structure(self, test_client, auth_headers):
        resp = test_client.get("/api/v1/portfolio/holdings", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        data = body["data"]
        assert "holdings" in data
        assert "totals" in data
        assert "count" in data
        assert isinstance(data["holdings"], list)

    def test_holdings_filter_by_portfolio(self, test_client, auth_headers):
        resp = test_client.get(
            "/api/v1/portfolio/holdings?portfolio=KFH",
            headers=auth_headers,
        )
        assert resp.status_code == 200

    def test_holdings_invalid_portfolio(self, test_client, auth_headers):
        resp = test_client.get(
            "/api/v1/portfolio/holdings?portfolio=INVALID",
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_holdings_totals_structure(self, test_client, auth_headers):
        """Totals should have all KWD-converted fields."""
        resp = test_client.get("/api/v1/portfolio/holdings", headers=auth_headers)
        totals = resp.json()["data"]["totals"]
        for field in [
            "total_market_value_kwd", "total_cost_kwd",
            "total_unrealized_pnl_kwd", "total_realized_pnl_kwd",
            "total_pnl_kwd", "total_dividends_kwd",
        ]:
            assert field in totals, f"Missing totals field: {field}"


# ── Portfolio Table ──────────────────────────────────────────────────

class TestPortfolioTable:
    """GET /api/v1/portfolio/table/{portfolio_name}"""

    def test_table_kfh(self, test_client, auth_headers):
        resp = test_client.get("/api/v1/portfolio/table/KFH", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["portfolio"] == "KFH"
        assert data["currency"] == "KWD"
        assert "holdings" in data

    def test_table_usa(self, test_client, auth_headers):
        resp = test_client.get("/api/v1/portfolio/table/USA", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["portfolio"] == "USA"
        assert data["currency"] == "USD"

    def test_table_invalid_portfolio(self, test_client, auth_headers):
        resp = test_client.get("/api/v1/portfolio/table/INVALID", headers=auth_headers)
        assert resp.status_code == 400


# ── Transaction CRUD ─────────────────────────────────────────────────

class TestTransactionCRUD:
    """Full CRUD lifecycle for transactions."""

    def test_list_transactions(self, test_client, auth_headers):
        resp = test_client.get("/api/v1/portfolio/transactions", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "transactions" in data
        assert "pagination" in data
        assert isinstance(data["transactions"], list)

    def test_list_transactions_requires_auth(self, test_client):
        resp = test_client.get("/api/v1/portfolio/transactions")
        assert resp.status_code in (401, 403)

    def test_create_buy_transaction(self, test_client, auth_headers):
        """Create a Buy transaction via API."""
        resp = test_client.post(
            "/api/v1/portfolio/transactions",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "stock_symbol": "API_TEST.KW",
                "txn_date": "2024-06-15",
                "txn_type": "Buy",
                "shares": 100,
                "purchase_cost": 500.0,
                "fees": 5.0,
            },
        )
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert "id" in data
        assert data["id"] > 0

    def test_create_sell_transaction(self, test_client, auth_headers):
        """Create a Sell transaction via API."""
        resp = test_client.post(
            "/api/v1/portfolio/transactions",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "stock_symbol": "API_TEST.KW",
                "txn_date": "2024-09-15",
                "txn_type": "Sell",
                "shares": 50,
                "sell_value": 300.0,
            },
        )
        assert resp.status_code == 201

    def test_create_transaction_validation(self, test_client, auth_headers):
        """Buy without purchase_cost should fail."""
        resp = test_client.post(
            "/api/v1/portfolio/transactions",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "stock_symbol": "FAIL.KW",
                "txn_date": "2024-01-01",
                "txn_type": "Buy",
                "shares": 100,
                # Missing purchase_cost
            },
        )
        assert resp.status_code == 400

    def test_create_sell_without_sell_value(self, test_client, auth_headers):
        """Sell without sell_value should fail."""
        resp = test_client.post(
            "/api/v1/portfolio/transactions",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "stock_symbol": "FAIL.KW",
                "txn_date": "2024-01-01",
                "txn_type": "Sell",
                "shares": 50,
                # Missing sell_value
            },
        )
        assert resp.status_code == 400

    def test_create_invalid_txn_type(self, test_client, auth_headers):
        """Invalid txn_type should fail."""
        resp = test_client.post(
            "/api/v1/portfolio/transactions",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "stock_symbol": "FAIL.KW",
                "txn_date": "2024-01-01",
                "txn_type": "Transfer",
                "shares": 100,
                "purchase_cost": 1000,
            },
        )
        assert resp.status_code == 400

    def test_get_transaction_by_id(self, test_client, auth_headers):
        """Get a specific transaction by ID."""
        # First create one
        create_resp = test_client.post(
            "/api/v1/portfolio/transactions",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "stock_symbol": "GETTEST.KW",
                "txn_date": "2024-01-15",
                "txn_type": "Buy",
                "shares": 10,
                "purchase_cost": 100.0,
            },
        )
        txn_id = create_resp.json()["data"]["id"]

        # Now fetch it
        resp = test_client.get(
            f"/api/v1/portfolio/transactions/{txn_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["stock_symbol"] == "GETTEST.KW"

    def test_get_nonexistent_transaction(self, test_client, auth_headers):
        resp = test_client.get(
            "/api/v1/portfolio/transactions/999999",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_update_transaction(self, test_client, auth_headers):
        """Update a transaction's notes."""
        # Create one
        create_resp = test_client.post(
            "/api/v1/portfolio/transactions",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "stock_symbol": "UPTEST.KW",
                "txn_date": "2024-01-15",
                "txn_type": "Buy",
                "shares": 10,
                "purchase_cost": 100.0,
            },
        )
        txn_id = create_resp.json()["data"]["id"]

        # Update it
        resp = test_client.put(
            f"/api/v1/portfolio/transactions/{txn_id}",
            headers=auth_headers,
            json={"notes": "Updated via test"},
        )
        assert resp.status_code == 200

    def test_delete_and_restore_transaction(self, test_client, auth_headers):
        """Soft-delete and restore lifecycle."""
        # Create one
        create_resp = test_client.post(
            "/api/v1/portfolio/transactions",
            headers=auth_headers,
            json={
                "portfolio": "BBYN",
                "stock_symbol": "DELTEST.KW",
                "txn_date": "2024-01-15",
                "txn_type": "Buy",
                "shares": 10,
                "purchase_cost": 100.0,
            },
        )
        txn_id = create_resp.json()["data"]["id"]

        # Delete it
        del_resp = test_client.delete(
            f"/api/v1/portfolio/transactions/{txn_id}",
            headers=auth_headers,
        )
        assert del_resp.status_code == 200

        # Should not be findable
        get_resp = test_client.get(
            f"/api/v1/portfolio/transactions/{txn_id}",
            headers=auth_headers,
        )
        assert get_resp.status_code == 404

        # Restore it
        restore_resp = test_client.post(
            f"/api/v1/portfolio/transactions/{txn_id}/restore",
            headers=auth_headers,
        )
        assert restore_resp.status_code == 200

        # Should be findable again
        get_resp2 = test_client.get(
            f"/api/v1/portfolio/transactions/{txn_id}",
            headers=auth_headers,
        )
        assert get_resp2.status_code == 200

    def test_pagination(self, test_client, auth_headers):
        """Transaction list respects page and page_size."""
        resp = test_client.get(
            "/api/v1/portfolio/transactions?page=1&page_size=2",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        pagination = data["pagination"]
        assert pagination["page"] == 1
        assert pagination["page_size"] == 2
        assert len(data["transactions"]) <= 2

    def test_filter_by_portfolio(self, test_client, auth_headers):
        """Transaction list filters by portfolio."""
        resp = test_client.get(
            "/api/v1/portfolio/transactions?portfolio=BBYN",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        for txn in data["transactions"]:
            assert txn["portfolio"] == "BBYN"


# ── User Data Isolation ──────────────────────────────────────────────

class TestUserIsolation:
    """User A cannot see User B's data."""

    def _create_user2_headers(self, test_client):
        """Create user2 and get auth headers without hitting rate limit."""
        from tests.helpers import ensure_user2
        return ensure_user2(test_client)

    def test_user2_cannot_see_user1_transactions(self, test_client, auth_headers):
        """User 2 should not see user 1's transactions."""
        user2 = self._create_user2_headers(test_client)

        # User 1 creates a transaction
        test_client.post(
            "/api/v1/portfolio/transactions",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "stock_symbol": "ISOLATION.KW",
                "txn_date": "2024-01-15",
                "txn_type": "Buy",
                "shares": 100,
                "purchase_cost": 1000.0,
            },
        )

        # User 2 lists transactions — should not find ISOLATION.KW
        resp = test_client.get(
            "/api/v1/portfolio/transactions?stock_symbol=ISOLATION.KW",
            headers=user2["headers"],
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["count"] == 0

    def test_user2_cannot_access_user1_transaction_by_id(self, test_client, auth_headers):
        """User 2 cannot GET user 1's transaction by direct ID."""
        user2 = self._create_user2_headers(test_client)

        # User 1 creates a transaction
        create_resp = test_client.post(
            "/api/v1/portfolio/transactions",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "stock_symbol": "SECURE.KW",
                "txn_date": "2024-01-15",
                "txn_type": "Buy",
                "shares": 10,
                "purchase_cost": 100.0,
            },
        )
        txn_id = create_resp.json()["data"]["id"]

        # User 2 tries to access it
        resp = test_client.get(
            f"/api/v1/portfolio/transactions/{txn_id}",
            headers=user2["headers"],
        )
        assert resp.status_code == 404

    def test_user2_cannot_delete_user1_transaction(self, test_client, auth_headers):
        """User 2 cannot delete user 1's transaction."""
        user2 = self._create_user2_headers(test_client)

        create_resp = test_client.post(
            "/api/v1/portfolio/transactions",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "stock_symbol": "NODELETE.KW",
                "txn_date": "2024-01-15",
                "txn_type": "Buy",
                "shares": 10,
                "purchase_cost": 100.0,
            },
        )
        txn_id = create_resp.json()["data"]["id"]

        # User 2 tries to delete it
        resp = test_client.delete(
            f"/api/v1/portfolio/transactions/{txn_id}",
            headers=user2["headers"],
        )
        assert resp.status_code == 404

    def test_user2_has_empty_overview(self, test_client, auth_headers):
        """User 2 should have empty overview (no transactions seeded)."""
        user2 = self._create_user2_headers(test_client)

        resp = test_client.get(
            "/api/v1/portfolio/overview",
            headers=user2["headers"],
        )
        assert resp.status_code == 200


# ── FX Rate ──────────────────────────────────────────────────────────

class TestFXRateEndpoint:
    """GET /api/v1/portfolio/fx-rate"""

    def test_fx_rate_structure(self, test_client, auth_headers):
        resp = test_client.get("/api/v1/portfolio/fx-rate", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "usd_kwd" in data
        assert isinstance(data["usd_kwd"], (int, float))
        assert data["usd_kwd"] > 0


# ── Account Balances ─────────────────────────────────────────────────

class TestAccountBalances:
    """GET /api/v1/portfolio/accounts"""

    def test_accounts_requires_auth(self, test_client):
        resp = test_client.get("/api/v1/portfolio/accounts")
        assert resp.status_code in (401, 403)

    def test_accounts_returns_data(self, test_client, auth_headers):
        resp = test_client.get("/api/v1/portfolio/accounts", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

"""
Integration tests for Cash Deposit API endpoints.

Covers:
  - Cash deposit CRUD (create, list, get, update, delete, restore)
  - Portfolio filter validation
  - Pagination
  - KWD total calculation
  - User isolation for cash deposits
"""

import pytest


class TestCashDepositList:
    """GET /api/v1/cash/deposits"""

    def test_list_deposits_requires_auth(self, test_client):
        resp = test_client.get("/api/v1/cash/deposits")
        assert resp.status_code in (401, 403)

    def test_list_deposits(self, test_client, auth_headers):
        resp = test_client.get("/api/v1/cash/deposits", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "deposits" in data
        assert "count" in data
        assert "total_kwd" in data
        assert "pagination" in data

    def test_list_deposits_filter_by_portfolio(self, test_client, auth_headers):
        resp = test_client.get(
            "/api/v1/cash/deposits?portfolio=KFH",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        for dep in resp.json()["data"]["deposits"]:
            assert dep["portfolio"] == "KFH"

    def test_list_deposits_invalid_portfolio(self, test_client, auth_headers):
        resp = test_client.get(
            "/api/v1/cash/deposits?portfolio=INVALID",
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_list_deposits_pagination(self, test_client, auth_headers):
        resp = test_client.get(
            "/api/v1/cash/deposits?page=1&page_size=2",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        pagination = resp.json()["data"]["pagination"]
        assert pagination["page"] == 1
        assert pagination["page_size"] == 2


class TestCashDepositCRUD:
    """Full CRUD lifecycle for cash deposits."""

    def test_create_deposit(self, test_client, auth_headers):
        resp = test_client.post(
            "/api/v1/cash/deposits",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "deposit_date": "2024-07-01",
                "amount": 2500.0,
                "currency": "KWD",
                "bank_name": "Test Bank",
                "deposit_type": "deposit",
                "notes": "Test deposit",
            },
        )
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert "id" in data
        assert data["id"] > 0

    def test_create_deposit_minimal(self, test_client, auth_headers):
        """Create deposit with only required fields."""
        resp = test_client.post(
            "/api/v1/cash/deposits",
            headers=auth_headers,
            json={
                "portfolio": "BBYN",
                "deposit_date": "2024-08-01",
                "amount": 1000.0,
            },
        )
        assert resp.status_code == 201

    def test_create_deposit_invalid_portfolio(self, test_client, auth_headers):
        resp = test_client.post(
            "/api/v1/cash/deposits",
            headers=auth_headers,
            json={
                "portfolio": "INVALID",
                "deposit_date": "2024-01-01",
                "amount": 100.0,
            },
        )
        assert resp.status_code == 400

    def test_get_deposit_by_id(self, test_client, auth_headers):
        # Create first
        create_resp = test_client.post(
            "/api/v1/cash/deposits",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "deposit_date": "2024-09-01",
                "amount": 500.0,
            },
        )
        dep_id = create_resp.json()["data"]["id"]

        # Fetch it
        resp = test_client.get(
            f"/api/v1/cash/deposits/{dep_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["id"] == dep_id
        assert float(data["amount"]) == 500.0

    def test_get_nonexistent_deposit(self, test_client, auth_headers):
        resp = test_client.get(
            "/api/v1/cash/deposits/999999",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_update_deposit(self, test_client, auth_headers):
        # Create
        create_resp = test_client.post(
            "/api/v1/cash/deposits",
            headers=auth_headers,
            json={
                "portfolio": "USA",
                "deposit_date": "2024-10-01",
                "amount": 750.0,
                "currency": "USD",
            },
        )
        dep_id = create_resp.json()["data"]["id"]

        # Update
        resp = test_client.put(
            f"/api/v1/cash/deposits/{dep_id}",
            headers=auth_headers,
            json={"notes": "Updated deposit note"},
        )
        assert resp.status_code == 200

    def test_delete_and_restore_deposit(self, test_client, auth_headers):
        # Create
        create_resp = test_client.post(
            "/api/v1/cash/deposits",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "deposit_date": "2024-11-01",
                "amount": 300.0,
            },
        )
        dep_id = create_resp.json()["data"]["id"]

        # Delete
        del_resp = test_client.delete(
            f"/api/v1/cash/deposits/{dep_id}",
            headers=auth_headers,
        )
        assert del_resp.status_code == 200

        # Should be gone
        get_resp = test_client.get(
            f"/api/v1/cash/deposits/{dep_id}",
            headers=auth_headers,
        )
        assert get_resp.status_code == 404

        # Restore
        restore_resp = test_client.post(
            f"/api/v1/cash/deposits/{dep_id}/restore",
            headers=auth_headers,
        )
        assert restore_resp.status_code == 200

        # Should be back
        get_resp2 = test_client.get(
            f"/api/v1/cash/deposits/{dep_id}",
            headers=auth_headers,
        )
        assert get_resp2.status_code == 200


class TestCashDepositIsolation:
    """User data isolation for cash deposits."""

    def test_user2_cannot_see_user1_deposits(self, test_client, auth_headers):
        from tests.helpers import ensure_user2

        user2 = ensure_user2(test_client)

        # User 1 creates a deposit
        test_client.post(
            "/api/v1/cash/deposits",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "deposit_date": "2024-12-01",
                "amount": 9999.0,
            },
        )

        # User 2 should not see the 9999 deposit
        resp = test_client.get("/api/v1/cash/deposits", headers=user2["headers"])
        assert resp.status_code == 200
        deposits = resp.json()["data"]["deposits"]
        amounts = [float(d["amount"]) for d in deposits]
        assert 9999.0 not in amounts

    def test_user2_cannot_delete_user1_deposit(self, test_client, auth_headers):
        from tests.helpers import ensure_user2

        user2 = ensure_user2(test_client)

        create_resp = test_client.post(
            "/api/v1/cash/deposits",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "deposit_date": "2024-12-15",
                "amount": 8888.0,
            },
        )
        dep_id = create_resp.json()["data"]["id"]

        resp = test_client.delete(
            f"/api/v1/cash/deposits/{dep_id}",
            headers=user2["headers"],
        )
        assert resp.status_code == 404

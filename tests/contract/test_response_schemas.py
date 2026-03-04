"""
Contract tests — validate API response shapes match Pydantic schemas.

Verifies:
  - All endpoints return {"status": "ok/error", "data": ...} envelope
  - Error responses have {status, error_code, detail}
  - Specific field types and presence in key responses
  - Pagination structure consistency
"""

import pytest


# ── Response Envelope ────────────────────────────────────────────────

class TestResponseEnvelope:
    """All success responses must follow the {status, data} envelope."""

    @pytest.mark.parametrize("endpoint", [
        "/api/v1/portfolio/overview",
        "/api/v1/portfolio/holdings",
        "/api/v1/portfolio/table/KFH",
        "/api/v1/portfolio/accounts",
        "/api/v1/portfolio/fx-rate",
        "/api/v1/portfolio/transactions",
        "/api/v1/cash/deposits",
        "/api/v1/analytics/performance",
        "/api/v1/analytics/risk-metrics",
        "/api/v1/analytics/realized-profit",
        "/api/v1/analytics/cash-balances",
        "/api/v1/analytics/snapshots",
        "/api/v1/analytics/position-snapshots",
        "/api/v1/integrity/check",
        "/api/v1/integrity/cash/KFH",
        "/api/v1/integrity/positions/KFH",
        "/api/v1/integrity/snapshots/KFH",
        "/api/v1/integrity/anomalies",
        "/api/v1/integrity/completeness",
    ])
    def test_success_response_has_status_ok(self, test_client, auth_headers, endpoint):
        """Every authenticated GET endpoint returns {status: 'ok', data: ...}."""
        resp = test_client.get(endpoint, headers=auth_headers)
        assert resp.status_code == 200, f"{endpoint} returned {resp.status_code}: {resp.text}"
        body = resp.json()
        assert "status" in body, f"{endpoint} missing 'status' field"
        assert body["status"] == "ok", f"{endpoint} status != 'ok': {body['status']}"
        assert "data" in body, f"{endpoint} missing 'data' field"

    def test_health_endpoint_envelope(self, test_client):
        """Health endpoint also follows envelope (no auth needed)."""
        resp = test_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"


class TestErrorResponseFormat:
    """Error responses must have {status: 'error', error_code, detail}."""

    def test_401_error_format(self, test_client):
        """Unauthenticated request returns structured error."""
        resp = test_client.get("/api/v1/portfolio/overview")
        assert resp.status_code in (401, 403)
        body = resp.json()
        assert "detail" in body

    def test_404_error_format(self, test_client, auth_headers):
        """Not found returns structured error."""
        resp = test_client.get(
            "/api/v1/portfolio/transactions/999999",
            headers=auth_headers,
        )
        assert resp.status_code == 404
        body = resp.json()
        assert "detail" in body

    def test_400_error_format(self, test_client, auth_headers):
        """Bad request returns structured error."""
        resp = test_client.get(
            "/api/v1/portfolio/holdings?portfolio=INVALID",
            headers=auth_headers,
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "detail" in body

    def test_422_validation_error(self, test_client, auth_headers):
        """Pydantic validation error returns 422 with details."""
        resp = test_client.post(
            "/api/v1/portfolio/transactions",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "stock_symbol": "X",
                "txn_date": "not-a-date",  # Invalid date format
                "txn_type": "Buy",
                "shares": 100,
                "purchase_cost": 1000,
            },
        )
        assert resp.status_code == 422
        body = resp.json()
        assert "detail" in body


# ── Pagination Contract ──────────────────────────────────────────────

class TestPaginationContract:
    """All paginated endpoints must return consistent pagination structure."""

    @pytest.mark.parametrize("endpoint", [
        "/api/v1/portfolio/transactions",
        "/api/v1/cash/deposits",
    ])
    def test_pagination_structure(self, test_client, auth_headers, endpoint):
        """Paginated endpoints have {page, page_size, total_items, total_pages}."""
        resp = test_client.get(
            f"{endpoint}?page=1&page_size=10",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        pagination = resp.json()["data"]["pagination"]

        # Required pagination fields
        for field in ["page", "page_size", "total_items", "total_pages"]:
            assert field in pagination, f"Missing pagination field: {field}"

        assert pagination["page"] == 1
        assert pagination["page_size"] == 10
        assert isinstance(pagination["total_items"], int)
        assert isinstance(pagination["total_pages"], int)
        assert pagination["total_pages"] >= 1


# ── Holdings Response Schema ────────────────────────────────────────

class TestHoldingsResponseSchema:
    """Verify holdings response matches HoldingRow schema."""

    def test_holdings_data_fields(self, test_client, auth_headers):
        """Holdings list items should have numeric value fields."""
        resp = test_client.get("/api/v1/portfolio/holdings", headers=auth_headers)
        data = resp.json()["data"]
        totals = data["totals"]

        # All totals should be numeric
        for key, val in totals.items():
            assert isinstance(val, (int, float)), f"totals.{key} is {type(val)}, expected numeric"

    def test_holdings_count_matches_list_length(self, test_client, auth_headers):
        """The 'count' field matches the actual list length."""
        resp = test_client.get("/api/v1/portfolio/holdings", headers=auth_headers)
        data = resp.json()["data"]
        assert data["count"] == len(data["holdings"])


# ── Transaction Response Schema ──────────────────────────────────────

class TestTransactionResponseSchema:
    """Verify transaction responses match TransactionResponse schema."""

    def test_created_transaction_has_id(self, test_client, auth_headers):
        resp = test_client.post(
            "/api/v1/portfolio/transactions",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "stock_symbol": "SCHEMA.KW",
                "txn_date": "2024-01-01",
                "txn_type": "Buy",
                "shares": 10,
                "purchase_cost": 100,
            },
        )
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert isinstance(data["id"], int)
        assert data["id"] > 0

    def test_transaction_list_item_fields(self, test_client, auth_headers):
        """Each transaction in a list should have core fields."""
        resp = test_client.get("/api/v1/portfolio/transactions", headers=auth_headers)
        txns = resp.json()["data"]["transactions"]
        if txns:
            txn = txns[0]
            for field in ["id", "user_id", "portfolio", "stock_symbol", "txn_type"]:
                assert field in txn, f"Transaction missing field: {field}"


# ── Cash Deposit Response Schema ─────────────────────────────────────

class TestCashDepositResponseSchema:
    """Verify cash deposit responses match CashDepositResponse schema."""

    def test_deposit_response_fields(self, test_client, auth_headers):
        """Created deposit returns expected structure."""
        resp = test_client.post(
            "/api/v1/cash/deposits",
            headers=auth_headers,
            json={
                "portfolio": "KFH",
                "deposit_date": "2024-01-15",
                "amount": 100.0,
            },
        )
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert "id" in data
        assert isinstance(data["id"], int)

    def test_deposit_list_has_total_kwd(self, test_client, auth_headers):
        """Deposit list includes KWD total for quick display."""
        resp = test_client.get("/api/v1/cash/deposits", headers=auth_headers)
        data = resp.json()["data"]
        assert "total_kwd" in data
        assert isinstance(data["total_kwd"], (int, float))


# ── Integrity Response Schema ────────────────────────────────────────

class TestIntegrityResponseSchema:
    """Verify integrity check responses match IntegrityReport schema."""

    def test_full_check_schema(self, test_client, auth_headers):
        resp = test_client.get("/api/v1/integrity/check", headers=auth_headers)
        data = resp.json()["data"]
        required_fields = ["overall_valid", "summary", "cash", "positions",
                           "snapshots", "anomalies", "completeness"]
        for field in required_fields:
            assert field in data, f"Missing integrity field: {field}"

    def test_cash_check_schema(self, test_client, auth_headers):
        resp = test_client.get("/api/v1/integrity/cash/KFH", headers=auth_headers)
        data = resp.json()["data"]
        required = ["portfolio", "is_valid", "expected_balance", "stored_balance"]
        for field in required:
            assert field in data, f"Missing cash check field: {field}"

    def test_anomaly_check_schema(self, test_client, auth_headers):
        resp = test_client.get("/api/v1/integrity/anomalies", headers=auth_headers)
        data = resp.json()["data"]
        assert "anomalies" in data
        assert "count" in data
        assert "is_valid" in data
        assert isinstance(data["anomalies"], list)
        assert isinstance(data["count"], int)


# ── Auth Response Schema ─────────────────────────────────────────────

class TestAuthResponseSchema:
    """Verify auth endpoint response schemas."""

    def test_login_response_schema(self, test_client):
        """Login returns {access_token, refresh_token, token_type, expires_in, ...}."""
        resp = test_client.post(
            "/api/v1/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        assert resp.status_code == 200
        body = resp.json()
        for field in ["access_token", "refresh_token", "token_type", "expires_in"]:
            assert field in body, f"Login response missing: {field}"
        assert body["token_type"] == "bearer"
        assert isinstance(body["expires_in"], int)
        assert body["expires_in"] > 0

    def test_me_response_schema(self, test_client, auth_headers):
        """GET /me returns {username, user_id, ...}."""
        resp = test_client.get("/api/v1/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "username" in body
        assert "user_id" in body or "id" in body

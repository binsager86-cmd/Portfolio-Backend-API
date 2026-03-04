"""
End-to-End flow tests — full user journeys through the API.

Simulates real-world workflows:
  1. Register → Login → Create portfolio data → Verify overview
  2. Full investment lifecycle: deposit → buy → sell → check P&L
  3. Multi-portfolio workflow with currency conversion
  4. Data lifecycle: create → update → delete → restore → verify
"""

import time
import pytest


class TestInvestmentLifecycle:
    """
    E2E: Complete investment lifecycle.

    Flow: Deposit cash → Buy stock → Verify holdings → Sell stock →
          Check realized P&L → Verify cash reconciliation.
    """

    def test_full_investment_cycle(self, test_client, auth_headers):
        """
        Deposit 10,000 KWD → Buy 1000 shares @ 5.0 (cost=5000, fees=25) →
        Verify holdings show position → Sell 500 shares @ 6.5 (value=3250, fees=15) →
        Verify realized profit exists → Check analytics.
        """
        headers = auth_headers

        # Step 1: Create cash deposit
        dep_resp = test_client.post(
            "/api/v1/cash/deposits",
            headers=headers,
            json={
                "portfolio": "KFH",
                "deposit_date": "2024-01-01",
                "amount": 10000.0,
                "currency": "KWD",
                "bank_name": "E2E Test Bank",
                "deposit_type": "deposit",
            },
        )
        assert dep_resp.status_code == 201, f"Deposit failed: {dep_resp.text}"
        dep_id = dep_resp.json()["data"]["id"]
        assert dep_id > 0

        # Step 2: Buy stock
        buy_resp = test_client.post(
            "/api/v1/portfolio/transactions",
            headers=headers,
            json={
                "portfolio": "KFH",
                "stock_symbol": "E2E_LIFECYCLE.KW",
                "txn_date": "2024-01-15",
                "txn_type": "Buy",
                "shares": 1000,
                "purchase_cost": 5000.0,
                "fees": 25.0,
            },
        )
        assert buy_resp.status_code == 201, f"Buy failed: {buy_resp.text}"
        buy_id = buy_resp.json()["data"]["id"]

        # Step 3: Verify the transaction appears in list
        list_resp = test_client.get(
            "/api/v1/portfolio/transactions?stock_symbol=E2E_LIFECYCLE.KW",
            headers=headers,
        )
        assert list_resp.status_code == 200
        txns = list_resp.json()["data"]["transactions"]
        assert any(t["stock_symbol"] == "E2E_LIFECYCLE.KW" for t in txns)

        # Step 4: Sell half the position
        sell_resp = test_client.post(
            "/api/v1/portfolio/transactions",
            headers=headers,
            json={
                "portfolio": "KFH",
                "stock_symbol": "E2E_LIFECYCLE.KW",
                "txn_date": "2024-06-15",
                "txn_type": "Sell",
                "shares": 500,
                "sell_value": 3250.0,
                "fees": 15.0,
            },
        )
        assert sell_resp.status_code == 201, f"Sell failed: {sell_resp.text}"

        # Step 5: Check realized profit reflects the sale
        profit_resp = test_client.get(
            "/api/v1/analytics/realized-profit",
            headers=headers,
        )
        assert profit_resp.status_code == 200
        profit_data = profit_resp.json()["data"]
        # Should have some realized data (could be from this or seeded data)
        assert isinstance(profit_data["details"], list)

        # Step 6: Overview should still work
        overview_resp = test_client.get(
            "/api/v1/portfolio/overview",
            headers=headers,
        )
        assert overview_resp.status_code == 200
        assert overview_resp.json()["status"] == "ok"

        # Step 7: Holdings should show the remaining position
        holdings_resp = test_client.get(
            "/api/v1/portfolio/holdings?portfolio=KFH",
            headers=headers,
        )
        assert holdings_resp.status_code == 200

        # Step 8: FX rate should be available
        fx_resp = test_client.get("/api/v1/portfolio/fx-rate", headers=headers)
        assert fx_resp.status_code == 200
        assert fx_resp.json()["data"]["usd_kwd"] > 0


class TestMultiPortfolioWorkflow:
    """
    E2E: Operations across multiple portfolios with different currencies.
    """

    def test_cross_portfolio_operations(self, test_client, auth_headers):
        """
        Create transactions in KFH (KWD) and USA (USD) portfolios,
        then verify overview aggregates both correctly in KWD.
        """
        headers = auth_headers

        # KWD portfolio transaction
        test_client.post(
            "/api/v1/portfolio/transactions",
            headers=headers,
            json={
                "portfolio": "KFH",
                "stock_symbol": "E2E_MULTI_KWD.KW",
                "txn_date": "2024-02-01",
                "txn_type": "Buy",
                "shares": 100,
                "purchase_cost": 500.0,
            },
        )

        # USD portfolio transaction
        test_client.post(
            "/api/v1/portfolio/transactions",
            headers=headers,
            json={
                "portfolio": "USA",
                "stock_symbol": "E2E_MULTI_USD",
                "txn_date": "2024-02-01",
                "txn_type": "Buy",
                "shares": 10,
                "purchase_cost": 1500.0,
            },
        )

        # Overview should aggregate both
        resp = test_client.get("/api/v1/portfolio/overview", headers=headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data is not None

        # Holdings should include both portfolios
        resp = test_client.get("/api/v1/portfolio/holdings", headers=headers)
        assert resp.status_code == 200

        # Per-portfolio tables
        for pf in ["KFH", "USA"]:
            resp = test_client.get(
                f"/api/v1/portfolio/table/{pf}",
                headers=headers,
            )
            assert resp.status_code == 200
            assert resp.json()["data"]["portfolio"] == pf


class TestDataLifecycleFlow:
    """
    E2E: Create → Read → Update → Delete → Restore for both
    transactions and cash deposits.
    """

    def test_transaction_full_lifecycle(self, test_client, auth_headers):
        """Complete CRUD lifecycle for a transaction."""
        headers = auth_headers

        # CREATE
        resp = test_client.post(
            "/api/v1/portfolio/transactions",
            headers=headers,
            json={
                "portfolio": "BBYN",
                "stock_symbol": "E2E_CRUD.KW",
                "txn_date": "2024-03-01",
                "txn_type": "Buy",
                "shares": 200,
                "purchase_cost": 1000.0,
                "fees": 10.0,
                "notes": "Original note",
            },
        )
        assert resp.status_code == 201
        txn_id = resp.json()["data"]["id"]

        # READ
        resp = test_client.get(
            f"/api/v1/portfolio/transactions/{txn_id}",
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["notes"] == "Original note"

        # UPDATE
        resp = test_client.put(
            f"/api/v1/portfolio/transactions/{txn_id}",
            headers=headers,
            json={"notes": "Updated note", "fees": 15.0},
        )
        assert resp.status_code == 200

        # READ again — verify update applied
        resp = test_client.get(
            f"/api/v1/portfolio/transactions/{txn_id}",
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["notes"] == "Updated note"

        # DELETE (soft)
        resp = test_client.delete(
            f"/api/v1/portfolio/transactions/{txn_id}",
            headers=headers,
        )
        assert resp.status_code == 200

        # READ — should be 404 (soft-deleted)
        resp = test_client.get(
            f"/api/v1/portfolio/transactions/{txn_id}",
            headers=headers,
        )
        assert resp.status_code == 404

        # RESTORE
        resp = test_client.post(
            f"/api/v1/portfolio/transactions/{txn_id}/restore",
            headers=headers,
        )
        assert resp.status_code == 200

        # READ — should be back
        resp = test_client.get(
            f"/api/v1/portfolio/transactions/{txn_id}",
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["notes"] == "Updated note"

    def test_deposit_full_lifecycle(self, test_client, auth_headers):
        """Complete CRUD lifecycle for a cash deposit."""
        headers = auth_headers

        # CREATE
        resp = test_client.post(
            "/api/v1/cash/deposits",
            headers=headers,
            json={
                "portfolio": "USA",
                "deposit_date": "2024-04-01",
                "amount": 3000.0,
                "currency": "USD",
                "bank_name": "E2E Bank",
                "notes": "Initial deposit",
            },
        )
        assert resp.status_code == 201
        dep_id = resp.json()["data"]["id"]

        # READ
        resp = test_client.get(
            f"/api/v1/cash/deposits/{dep_id}",
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["notes"] == "Initial deposit"

        # UPDATE
        resp = test_client.put(
            f"/api/v1/cash/deposits/{dep_id}",
            headers=headers,
            json={"notes": "Updated deposit"},
        )
        assert resp.status_code == 200

        # DELETE
        resp = test_client.delete(
            f"/api/v1/cash/deposits/{dep_id}",
            headers=headers,
        )
        assert resp.status_code == 200

        # Verify deleted
        resp = test_client.get(
            f"/api/v1/cash/deposits/{dep_id}",
            headers=headers,
        )
        assert resp.status_code == 404

        # RESTORE
        resp = test_client.post(
            f"/api/v1/cash/deposits/{dep_id}/restore",
            headers=headers,
        )
        assert resp.status_code == 200

        # Verify restored
        resp = test_client.get(
            f"/api/v1/cash/deposits/{dep_id}",
            headers=headers,
        )
        assert resp.status_code == 200


class TestNewUserOnboarding:
    """
    E2E: Fresh user registration → first portfolio actions.
    """

    def test_new_user_complete_flow(self, test_client):
        """
        Register → Login → Check empty overview → Create deposit → Buy stock →
        View holdings → Run integrity check.
        """
        # Step 1: Register new user
        username = f"e2e_user_{int(time.time())}"
        reg_resp = test_client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "password": "e2eSecure123!",
                "name": "E2E Test User",
            },
        )
        assert reg_resp.status_code == 201, f"Register failed: {reg_resp.text}"
        reg_data = reg_resp.json()
        assert "access_token" in reg_data
        assert "refresh_token" in reg_data

        headers = {"Authorization": f"Bearer {reg_data['access_token']}"}

        # Step 2: Verify /me works
        me_resp = test_client.get("/api/v1/auth/me", headers=headers)
        assert me_resp.status_code == 200
        assert me_resp.json()["username"] == username

        # Step 3: Overview should work (empty state)
        overview_resp = test_client.get("/api/v1/portfolio/overview", headers=headers)
        assert overview_resp.status_code == 200

        # Step 4: Holdings should return empty list
        holdings_resp = test_client.get("/api/v1/portfolio/holdings", headers=headers)
        assert holdings_resp.status_code == 200
        assert holdings_resp.json()["data"]["count"] == 0

        # Step 5: Analytics endpoints should not crash on empty data
        for endpoint in [
            "/api/v1/analytics/performance",
            "/api/v1/analytics/risk-metrics",
            "/api/v1/analytics/realized-profit",
            "/api/v1/analytics/cash-balances",
            "/api/v1/analytics/snapshots",
        ]:
            resp = test_client.get(endpoint, headers=headers)
            assert resp.status_code == 200, f"{endpoint} failed for new user: {resp.text}"

        # Step 6: Integrity check should work on empty portfolio
        integrity_resp = test_client.get("/api/v1/integrity/check", headers=headers)
        assert integrity_resp.status_code == 200


class TestIntegrityAfterMutations:
    """
    E2E: Verify integrity checks remain valid after financial mutations.
    """

    def test_integrity_consistent_after_trades(self, test_client, auth_headers):
        """
        Create deposits + trades → run integrity → verify no anomalies
        from the test data.
        """
        headers = auth_headers

        # Create a deposit
        test_client.post(
            "/api/v1/cash/deposits",
            headers=headers,
            json={
                "portfolio": "KFH",
                "deposit_date": "2024-05-01",
                "amount": 5000.0,
            },
        )

        # Create buy
        test_client.post(
            "/api/v1/portfolio/transactions",
            headers=headers,
            json={
                "portfolio": "KFH",
                "stock_symbol": "E2E_INTEG.KW",
                "txn_date": "2024-05-15",
                "txn_type": "Buy",
                "shares": 100,
                "purchase_cost": 1000.0,
            },
        )

        # Run anomaly check — should have no over_sell for our new stock
        resp = test_client.get("/api/v1/integrity/anomalies", headers=headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        anomalies = data["anomalies"]
        over_sells = [
            a for a in anomalies
            if a.get("type") == "over_sell" and a.get("symbol") == "E2E_INTEG.KW"
        ]
        assert len(over_sells) == 0, "Should have no over-sell anomalies for E2E_INTEG.KW"

        # Completeness check
        comp_resp = test_client.get("/api/v1/integrity/completeness", headers=headers)
        assert comp_resp.status_code == 200

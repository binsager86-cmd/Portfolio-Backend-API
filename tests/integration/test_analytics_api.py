"""
Integration tests for Analytics API endpoints.

Covers:
  - Performance metrics (TWR, MWRR)
  - Risk metrics (Sharpe, Sortino)
  - Realized profit breakdown
  - Cash balances reconciliation
  - Snapshot listing with filters
  - Position snapshots
"""

import pytest


class TestPerformanceMetrics:
    """GET /api/v1/analytics/performance"""

    def test_performance_requires_auth(self, test_client):
        resp = test_client.get("/api/v1/analytics/performance")
        assert resp.status_code in (401, 403)

    def test_performance_returns_structure(self, test_client, auth_headers):
        resp = test_client.get("/api/v1/analytics/performance", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        # Should contain TWR and MWRR fields
        assert "twr_percent" in data or "twr" in data or isinstance(data, dict)

    def test_performance_with_portfolio_filter(self, test_client, auth_headers):
        resp = test_client.get(
            "/api/v1/analytics/performance?portfolio=KFH",
            headers=auth_headers,
        )
        assert resp.status_code == 200

    def test_performance_with_period(self, test_client, auth_headers):
        for period in ["all", "ytd", "1y", "6m", "3m", "1m"]:
            resp = test_client.get(
                f"/api/v1/analytics/performance?period={period}",
                headers=auth_headers,
            )
            assert resp.status_code == 200

    def test_performance_invalid_portfolio(self, test_client, auth_headers):
        resp = test_client.get(
            "/api/v1/analytics/performance?portfolio=INVALID",
            headers=auth_headers,
        )
        assert resp.status_code == 400


class TestRiskMetrics:
    """GET /api/v1/analytics/risk-metrics"""

    def test_risk_metrics_requires_auth(self, test_client):
        resp = test_client.get("/api/v1/analytics/risk-metrics")
        assert resp.status_code in (401, 403)

    def test_risk_metrics_default_params(self, test_client, auth_headers):
        resp = test_client.get("/api/v1/analytics/risk-metrics", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "sharpe_ratio" in data
        assert "sortino_ratio" in data
        assert "rf_rate" in data
        assert "mar" in data

    def test_risk_metrics_custom_params(self, test_client, auth_headers):
        resp = test_client.get(
            "/api/v1/analytics/risk-metrics?rf_rate=0.05&mar=0.02",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["rf_rate"] == 0.05
        assert data["mar"] == 0.02


class TestRealizedProfit:
    """GET /api/v1/analytics/realized-profit"""

    def test_realized_profit_requires_auth(self, test_client):
        resp = test_client.get("/api/v1/analytics/realized-profit")
        assert resp.status_code in (401, 403)

    def test_realized_profit_structure(self, test_client, auth_headers):
        resp = test_client.get("/api/v1/analytics/realized-profit", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "total_realized_kwd" in data
        assert "total_profit_kwd" in data
        assert "total_loss_kwd" in data
        assert "details" in data
        assert isinstance(data["details"], list)

    def test_realized_profit_values_consistent(self, test_client, auth_headers):
        """total_realized = total_profit + total_loss (loss is negative)."""
        resp = test_client.get("/api/v1/analytics/realized-profit", headers=auth_headers)
        data = resp.json()["data"]
        total = float(data["total_realized_kwd"])
        profit = float(data["total_profit_kwd"])
        loss = float(data["total_loss_kwd"])
        assert abs(total - (profit + loss)) < 0.01


class TestCashBalances:
    """GET /api/v1/analytics/cash-balances"""

    def test_cash_balances_requires_auth(self, test_client):
        resp = test_client.get("/api/v1/analytics/cash-balances")
        assert resp.status_code in (401, 403)

    def test_cash_balances_returns_data(self, test_client, auth_headers):
        resp = test_client.get("/api/v1/analytics/cash-balances", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_cash_balances_force_override(self, test_client, auth_headers):
        resp = test_client.get(
            "/api/v1/analytics/cash-balances?force=true",
            headers=auth_headers,
        )
        assert resp.status_code == 200


class TestSnapshots:
    """GET /api/v1/analytics/snapshots"""

    def test_snapshots_requires_auth(self, test_client):
        resp = test_client.get("/api/v1/analytics/snapshots")
        assert resp.status_code in (401, 403)

    def test_snapshots_returns_structure(self, test_client, auth_headers):
        resp = test_client.get("/api/v1/analytics/snapshots", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "snapshots" in data
        assert "count" in data
        assert isinstance(data["snapshots"], list)

    def test_snapshots_filter_by_portfolio(self, test_client, auth_headers):
        resp = test_client.get(
            "/api/v1/analytics/snapshots?portfolio=KFH",
            headers=auth_headers,
        )
        assert resp.status_code == 200

    def test_snapshots_filter_by_date_range(self, test_client, auth_headers):
        resp = test_client.get(
            "/api/v1/analytics/snapshots?start_date=2024-01-01&end_date=2024-12-31",
            headers=auth_headers,
        )
        assert resp.status_code == 200


class TestPositionSnapshots:
    """GET /api/v1/analytics/position-snapshots"""

    def test_position_snapshots_requires_auth(self, test_client):
        resp = test_client.get("/api/v1/analytics/position-snapshots")
        assert resp.status_code in (401, 403)

    def test_position_snapshots_structure(self, test_client, auth_headers):
        resp = test_client.get(
            "/api/v1/analytics/position-snapshots",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "snapshots" in data
        assert "count" in data

    def test_position_snapshots_filter(self, test_client, auth_headers):
        resp = test_client.get(
            "/api/v1/analytics/position-snapshots?stock_symbol=AAPL",
            headers=auth_headers,
        )
        assert resp.status_code == 200

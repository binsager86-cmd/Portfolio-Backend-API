import pytest

from app.core.database import query_val
from app.services.portfolio_service import PortfolioService
from tests.helpers import create_buy, create_stock


def test_bulk_delete_rolls_back_when_cash_recalc_fails(test_client, auth_headers, monkeypatch):
    txn_id = create_buy(user_id=1, portfolio="KFH", symbol="ATOMIC.KW")

    def fail_recalc(self, *args, **kwargs):
        raise RuntimeError("cash recalc failed")

    monkeypatch.setattr(PortfolioService, "recalc_portfolio_cash", fail_recalc)

    with pytest.raises(RuntimeError, match="cash recalc failed"):
        test_client.delete("/api/v1/portfolio/transactions", headers=auth_headers)

    is_deleted = query_val("SELECT COALESCE(is_deleted, 0) FROM transactions WHERE id = ?", (txn_id,))
    assert int(is_deleted or 0) == 0


def test_transaction_delete_and_restore_are_state_conditional(test_client, auth_headers, monkeypatch):
    txn_id = create_buy(user_id=1, portfolio="KFH", symbol="STATE.KW")
    recalc_calls = []

    def record_recalc(self, *args, **kwargs):
        if "deposit_delta" in kwargs:
            recalc_calls.append(kwargs.get("deposit_delta"))

    monkeypatch.setattr(PortfolioService, "recalc_portfolio_cash", record_recalc)

    first_delete = test_client.delete(f"/api/v1/portfolio/transactions/{txn_id}", headers=auth_headers)
    second_delete = test_client.delete(f"/api/v1/portfolio/transactions/{txn_id}", headers=auth_headers)

    assert first_delete.status_code == 200
    assert second_delete.status_code == 404
    assert recalc_calls == [1000.0]

    first_restore = test_client.post(f"/api/v1/portfolio/transactions/{txn_id}/restore", headers=auth_headers)
    second_restore = test_client.post(f"/api/v1/portfolio/transactions/{txn_id}/restore", headers=auth_headers)

    assert first_restore.status_code == 200
    assert second_restore.status_code == 404
    assert recalc_calls == [1000.0, -1000.0]

    is_deleted = query_val("SELECT COALESCE(is_deleted, 0) FROM transactions WHERE id = ?", (txn_id,))
    assert int(is_deleted or 0) == 0


def test_create_transaction_reuses_existing_user_symbol_stock(test_client, auth_headers, monkeypatch):
    create_stock(user_id=1, portfolio="KFH", symbol="KRE")

    def record_recalc(self, *args, **kwargs):
        return None

    monkeypatch.setattr(PortfolioService, "recalc_portfolio_cash", record_recalc)

    response = test_client.post(
        "/api/v1/portfolio/transactions",
        headers=auth_headers,
        json={
            "portfolio": "BBYN",
            "txn_type": "Buy",
            "stock_symbol": "KRE",
            "txn_date": "2026-07-26",
            "shares": 9172,
            "purchase_cost": 3104.49,
        },
    )

    assert response.status_code == 201
    assert response.json()["data"]["id"]
    stock_count = query_val(
        "SELECT COUNT(*) FROM stocks WHERE user_id = ? AND UPPER(TRIM(symbol)) = ?",
        (1, "KRE"),
    )
    assert int(stock_count or 0) == 1

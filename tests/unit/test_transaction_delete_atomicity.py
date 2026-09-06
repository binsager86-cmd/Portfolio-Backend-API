import time

import pytest

from app.core.database import exec_sql, exec_sql_returning_id, query_val
from app.services.portfolio_service import PortfolioService


def test_transaction_delete_rolls_back_when_cash_recalc_fails(
    test_client, auth_headers, monkeypatch
):
    txn_id = exec_sql_returning_id(
        "INSERT INTO transactions "
        "(user_id, portfolio, stock_symbol, txn_date, txn_type, shares, "
        "purchase_cost, sell_value, cash_dividend, fees, category, is_deleted, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (1, "KFH", "ATOMIC.KW", "2026-09-06", "Buy", 1, 100.0, 0.0, 0.0, 0.0, "portfolio", 0, int(time.time())),
    )

    def fail_recalc(self, *args, **kwargs):
        raise RuntimeError("cash recalc failed")

    monkeypatch.setattr(PortfolioService, "recalc_portfolio_cash", fail_recalc)

    with pytest.raises(RuntimeError, match="cash recalc failed"):
        test_client.delete(
            f"/api/v1/portfolio/transactions/{txn_id}",
            headers=auth_headers,
        )

    is_deleted = query_val(
        "SELECT COALESCE(is_deleted, 0) FROM transactions WHERE id = ?",
        (txn_id,),
    )
    assert int(is_deleted or 0) == 0

    exec_sql("DELETE FROM transactions WHERE id = ?", (txn_id,))
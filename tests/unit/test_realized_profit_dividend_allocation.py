import time

from app.core.database import exec_sql


def test_realized_profit_allocates_dividends_for_legacy_casing_and_blank_portfolio(test_client, auth_headers):
    """Realized-profit endpoint should allocate dividends despite legacy txn formatting drift."""
    user_id = 1
    symbol = "DIVFIX01"
    now = int(time.time())

    exec_sql(
        "DELETE FROM transactions WHERE user_id = ? AND UPPER(TRIM(stock_symbol)) = ?",
        (user_id, symbol),
    )
    exec_sql(
        "DELETE FROM stocks WHERE user_id = ? AND UPPER(TRIM(symbol)) = ?",
        (user_id, symbol),
    )

    exec_sql(
        """
        INSERT INTO stocks (user_id, symbol, name, portfolio, currency, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, symbol, symbol, "USA", "KWD", now),
    )

    # Buy row with non-canonical txn_type casing and spacing.
    exec_sql(
        """
        INSERT INTO transactions (
            user_id, portfolio, stock_symbol, txn_date, txn_type, shares,
            purchase_cost, sell_value, cash_dividend, fees, category, is_deleted, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, "usa ", f" {symbol} ", "2025-01-01", "buy", 10.0,
         100.0, 0.0, 0.0, 0.0, "portfolio", 0, now),
    )

    # Dividend row with blank portfolio and variant type spelling.
    exec_sql(
        """
        INSERT INTO transactions (
            user_id, portfolio, stock_symbol, txn_date, txn_type, shares,
            purchase_cost, sell_value, cash_dividend, fees, category, is_deleted, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, "   ", symbol.lower(), "2025-01-02", "Dividend", 0.0,
         0.0, 0.0, 20.0, 0.0, "portfolio", 0, now),
    )

    # Sell row with uppercase txn_type and stored realized PnL to hit stored path.
    exec_sql(
        """
        INSERT INTO transactions (
            user_id, portfolio, stock_symbol, txn_date, txn_type, shares,
            purchase_cost, sell_value, cash_dividend, fees, category,
            realized_pnl_at_txn, is_deleted, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, "USA", symbol, "2025-01-03", "SELL", 5.0,
         0.0, 55.0, 0.0, 0.0, "portfolio", 15.0, 0, now),
    )

    resp = test_client.get("/api/v1/analytics/realized-profit", headers=auth_headers)
    assert resp.status_code == 200, resp.text

    payload = resp.json()
    assert payload.get("status") == "ok"

    details = payload["data"]["details"]
    symbol_details = [d for d in details if d["symbol"] == symbol]
    assert len(symbol_details) == 1

    trade = symbol_details[0]
    assert trade["portfolio"] == "USA"
    assert trade["dividends_allocated_kwd"] == 10.0
    assert trade["realized_pnl_kwd"] == 15.0
    assert trade["net_pnl_kwd"] == 25.0

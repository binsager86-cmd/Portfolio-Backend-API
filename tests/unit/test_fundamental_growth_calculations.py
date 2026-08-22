import time

from app.core.database import exec_sql, query_val


def _insert_statement(stock_id: int, statement_type: str, fiscal_year: int, fiscal_quarter: int | None, period_end_date: str, *, source_file: str | None = None) -> int:
    now = int(time.time())
    exec_sql(
        """
        INSERT INTO financial_statements (
            stock_id, statement_type, fiscal_year, fiscal_quarter, period_end_date,
            source_file, extracted_by, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (stock_id, statement_type, fiscal_year, fiscal_quarter, period_end_date, source_file, "test", now),
    )
    stmt_id = query_val(
        """
        SELECT id FROM financial_statements
        WHERE stock_id = ? AND statement_type = ? AND period_end_date = ?
        """,
        (stock_id, statement_type, period_end_date),
    )
    return int(stmt_id)


def _insert_line_item(statement_id: int, code: str, amount: float) -> None:
    exec_sql(
        """
        INSERT INTO financial_line_items (
            statement_id, line_item_code, line_item_name, amount, currency, order_index, is_total, manually_edited
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (statement_id, code, code.replace("_", " ").title(), amount, "KWD", 1, 1, 0),
    )


def _seed_stock(symbol: str) -> int:
    now = int(time.time())
    exec_sql(
        """
        INSERT INTO analysis_stocks (
            user_id, symbol, company_name, exchange, currency, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (1, symbol, "Growth Test Co", "KSE", "KWD", now, now),
    )
    stock_id = query_val(
        "SELECT id FROM analysis_stocks WHERE user_id = ? AND symbol = ?",
        (1, symbol),
    )
    return int(stock_id)


def test_growth_skips_sign_flip_percentages(test_client, auth_headers):
    stock_id = _seed_stock("GROWTHSIGN")

    stmt_2023 = _insert_statement(stock_id, "income", 2023, 4, "2023-12-31")
    _insert_line_item(stmt_2023, "NET_INCOME", -100.0)

    stmt_2024 = _insert_statement(stock_id, "income", 2024, 4, "2024-12-31")
    _insert_line_item(stmt_2024, "NET_INCOME", 100.0)

    resp = test_client.get(f"/api/v1/fundamental/stocks/{stock_id}/growth", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    growth = resp.json()["data"]["growth"]

    assert "Net Income Growth" not in growth


def test_growth_uses_same_quarter_yoy_for_interim_total_assets(test_client, auth_headers):
    stock_id = _seed_stock("GROWTHASSET")

    stmt_2024_q1 = _insert_statement(stock_id, "balance", 2024, 1, "2024-03-31", source_file="quarterly-q1")
    _insert_line_item(stmt_2024_q1, "TOTAL_ASSETS", 100.0)

    stmt_2024_q4 = _insert_statement(stock_id, "balance", 2024, 4, "2024-12-31")
    _insert_line_item(stmt_2024_q4, "TOTAL_ASSETS", 500.0)

    stmt_2025_q1 = _insert_statement(stock_id, "balance", 2025, 1, "2025-03-31", source_file="quarterly-q1")
    _insert_line_item(stmt_2025_q1, "TOTAL_ASSETS", 120.0)

    resp = test_client.get(f"/api/v1/fundamental/stocks/{stock_id}/growth", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    growth = resp.json()["data"]["growth"]

    entries = growth["Total Assets Growth"]
    latest = entries[-1]
    assert latest["period"] == "2025-03-31"
    assert latest["prev_period"] == "2024-03-31"
    assert latest["growth"] == 0.2
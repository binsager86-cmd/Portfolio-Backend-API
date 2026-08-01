import time

from app.core.database import exec_sql, query_val


def _seed_stock_with_periods(symbol: str) -> int:
    now = int(time.time())
    exec_sql(
        """
        INSERT INTO analysis_stocks (
            user_id, symbol, company_name, exchange, currency, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (1, symbol, "Metrics Ensure Co", "KSE", "KWD", now, now),
    )
    stock_id = query_val(
        "SELECT id FROM analysis_stocks WHERE user_id = ? AND symbol = ?",
        (1, symbol),
    )

    # Period 1 has statement + line item (calculable)
    exec_sql(
        """
        INSERT INTO financial_statements (
            stock_id, statement_type, fiscal_year, fiscal_quarter, period_end_date,
            extracted_by, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (stock_id, "income", 2025, 1, "2025-03-31", "test", now),
    )
    stmt_id = query_val(
        "SELECT id FROM financial_statements WHERE stock_id = ? AND period_end_date = ? AND statement_type = ?",
        (stock_id, "2025-03-31", "income"),
    )
    exec_sql(
        """
        INSERT INTO financial_line_items (
            statement_id, line_item_code, line_item_name, amount, currency, order_index, is_total, manually_edited
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (stmt_id, "REVENUE", "Revenue", 1000.0, "KWD", 1, 1, 0),
    )

    # Period 2 has statement but no line items (should be skipped, not failed)
    exec_sql(
        """
        INSERT INTO financial_statements (
            stock_id, statement_type, fiscal_year, fiscal_quarter, period_end_date,
            extracted_by, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (stock_id, "income", 2025, 2, "2025-06-30", "test", now),
    )

    return int(stock_id)


def test_metrics_ensure_calculates_valid_and_skips_empty_periods(test_client, auth_headers):
    stock_id = _seed_stock_with_periods("METRICSENS_A")

    resp = test_client.post(f"/api/v1/fundamental/stocks/{stock_id}/metrics/ensure", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()["data"]

    assert body["total_periods"] == 2
    assert body["calculated_periods"] >= 1
    assert body["skipped_periods"] >= 1
    assert body["failed_periods"] == 0

    assert body["calculated_periods"] + body["skipped_periods"] + body["failed_periods"] == body["total_periods"]


def test_metrics_ensure_is_idempotent(test_client, auth_headers):
    stock_id = _seed_stock_with_periods("METRICSENS_B")

    first = test_client.post(f"/api/v1/fundamental/stocks/{stock_id}/metrics/ensure", headers=auth_headers)
    second = test_client.post(f"/api/v1/fundamental/stocks/{stock_id}/metrics/ensure", headers=auth_headers)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert second.json()["data"]["failed_periods"] == 0

"""
Regression tests for the Cash Flow score category's Levered Free Cash Flow
metric.

Two separate bugs were found:
1. stockanalysis.com's KWSE "leveredFCF" field was mis-mapped to the
   generic "free_cash_flow" canonical code instead of its own
   "levered_free_cash_flow" code, so a statement that genuinely reported
   Levered FCF still showed N/A in the score (the value was silently
   absorbed into the unlevered Free Cash Flow bucket instead).
2. When a statement truly doesn't disclose Levered FCF, there was no CFA
   fallback calculation (Free Cash Flow − Mandatory Debt Repayments).
"""
import time

import pytest

from app.api.v1 import fundamental_legacy
from app.api.v1.fundamental_legacy import _SA_FIELD_MAP_CASHFLOW
from app.core.database import exec_sql, query_val


def _seed_stock(symbol: str) -> int:
    now = int(time.time())
    exec_sql(
        """
        INSERT INTO analysis_stocks (
            user_id, symbol, company_name, exchange, currency, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (1, symbol, "Cash Flow Fallback Co", "KSE", "KWD", now, now),
    )
    return int(query_val(
        "SELECT id FROM analysis_stocks WHERE user_id = ? AND symbol = ?",
        (1, symbol),
    ))


def _seed_period(stock_id: int, period_end_date: str, fiscal_year: int, line_items: dict) -> None:
    now = int(time.time())
    exec_sql(
        """
        INSERT INTO financial_statements (
            stock_id, statement_type, fiscal_year, fiscal_quarter, period_end_date,
            extracted_by, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (stock_id, "income", fiscal_year, None, period_end_date, "test", now),
    )
    stmt_id = query_val(
        "SELECT id FROM financial_statements WHERE stock_id = ? AND period_end_date = ? AND statement_type = ?",
        (stock_id, period_end_date, "income"),
    )
    for idx, (code, amount) in enumerate(line_items.items()):
        exec_sql(
            """
            INSERT INTO financial_line_items (
                statement_id, line_item_code, line_item_name, amount, currency, order_index, is_total, manually_edited
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (stmt_id, code, code, amount, "KWD", idx, 0, 0),
        )


def test_stockanalysis_leveredfcf_field_maps_to_its_own_canonical_code():
    """stockanalysis.com's KWSE 'leveredFCF' field must not be folded into
    the generic Free Cash Flow bucket — it needs its own canonical code so
    the Cash Flow score can distinguish the two."""
    canonical_code, display_name = _SA_FIELD_MAP_CASHFLOW["leveredFCF"]
    assert canonical_code == "levered_free_cash_flow"
    assert display_name == "Levered Free Cash Flow"


def test_levered_fcf_retrieved_directly_when_statement_reports_it(test_client):
    stock_id = _seed_stock("LFCFRETRIEVE_A")
    _seed_period(
        stock_id, "2025-12-31", 2025,
        {
            "CASH_FROM_OPERATIONS": 500_000.0,
            "CAPITAL_EXPENDITURES": 100_000.0,
            "LEVERED_FREE_CASH_FLOW": 250_000.0,
        },
    )

    result = fundamental_legacy._calculate_all_metrics(stock_id, "2025-12-31", 2025, None)

    # Reported value wins even though it doesn't equal CFO - CapEx.
    assert result["cashflow"]["Levered Free Cash Flow"] == pytest.approx(250_000.0)


def test_levered_fcf_falls_back_to_fcf_minus_debt_repayment_when_not_reported(test_client):
    """CFA: Levered FCF = Free Cash Flow (CFO - CapEx) - Mandatory Debt
    Repayments, calculated only when the statement doesn't disclose it."""
    stock_id = _seed_stock("LFCFRETRIEVE_B")
    _seed_period(
        stock_id, "2025-12-31", 2025,
        {
            "CASH_FROM_OPERATIONS": 500_000.0,
            "CAPITAL_EXPENDITURES": 100_000.0,
            "DEBT_REPAID": 150_000.0,
            # No LEVERED_FREE_CASH_FLOW line item.
        },
    )

    result = fundamental_legacy._calculate_all_metrics(stock_id, "2025-12-31", 2025, None)

    # FCF = 500,000 - 100,000 = 400,000; LFCF = 400,000 - 150,000 = 250,000
    assert result["cashflow"]["Free Cash Flow"] == pytest.approx(400_000.0)
    assert result["cashflow"]["Levered Free Cash Flow"] == pytest.approx(250_000.0)


def test_levered_fcf_stays_na_without_debt_repayment_data(test_client):
    stock_id = _seed_stock("LFCFRETRIEVE_C")
    _seed_period(
        stock_id, "2025-12-31", 2025,
        {
            "CASH_FROM_OPERATIONS": 500_000.0,
            "CAPITAL_EXPENDITURES": 100_000.0,
            # No LEVERED_FREE_CASH_FLOW and no DEBT_REPAID — can't derive it.
        },
    )

    result = fundamental_legacy._calculate_all_metrics(stock_id, "2025-12-31", 2025, None)

    assert "Levered Free Cash Flow" not in result["cashflow"]

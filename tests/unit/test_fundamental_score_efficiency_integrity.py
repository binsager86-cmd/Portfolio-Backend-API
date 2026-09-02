"""
Regression tests for Efficiency score integrity:

1. Inventory/Payables Turnover must not use Operating Expenses (SG&A-type
   costs) as a stand-in for COGS — that's not a valid CFA proxy and
   silently distorts Days Payable / Cash Conversion Cycle.
2. Cash Conversion Cycle scoring must flag implausible magnitudes (e.g.
   hundreds of days negative) as a data-quality issue instead of awarding
   "excellent cash discipline" points at face value.
"""
import time

import pytest

from app.api.v1 import fundamental_legacy
from app.api.v1.fundamental_legacy import _score_efficiency_detailed
from app.core.database import exec_sql, query_val


def _seed_stock(symbol: str) -> int:
    now = int(time.time())
    exec_sql(
        """
        INSERT INTO analysis_stocks (
            user_id, symbol, company_name, exchange, currency, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (1, symbol, "Efficiency Integrity Co", "KSE", "KWD", now, now),
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


def test_inventory_and_payables_turnover_ignore_operating_expenses_proxy(test_client):
    """When COST_OF_REVENUE/COST_OF_OPERATIONS/PROPERTY_EXPENSES are all
    absent, Inventory/Payables Turnover must come back N/A rather than
    silently using Operating Expenses (SG&A) as a fake COGS."""
    stock_id = _seed_stock("EFFPROXY_A")
    _seed_period(
        stock_id, "2025-12-31", 2025,
        {
            "OPERATING_EXPENSES": 50_000.0,
            "INVENTORY": 20_000.0,
            "ACCOUNTS_PAYABLE": 40_000.0,
            "TOTAL_EQUITY": 100_000.0,  # keeps the period non-empty for persistence
        },
    )

    result = fundamental_legacy._calculate_all_metrics(stock_id, "2025-12-31", 2025, None)

    assert "Inventory Turnover" not in result["efficiency"]
    assert "Payables Turnover" not in result["efficiency"]
    assert "Cash Conversion Cycle" not in result["efficiency"]


def test_inventory_turnover_still_uses_cost_of_operations_proxy(test_client):
    """COST_OF_OPERATIONS remains a valid COGS-equivalent fallback (e.g. for
    REITs / sector filers that don't label it Cost of Revenue)."""
    stock_id = _seed_stock("EFFPROXY_B")
    _seed_period(
        stock_id, "2025-12-31", 2025,
        {
            "COST_OF_OPERATIONS": 80_000.0,
            "INVENTORY": 20_000.0,
        },
    )

    result = fundamental_legacy._calculate_all_metrics(stock_id, "2025-12-31", 2025, None)

    assert result["efficiency"]["Inventory Turnover"] == pytest.approx(4.0)


def test_cash_conversion_cycle_flags_implausible_magnitude_instead_of_scoring_it():
    score, breakdown = _score_efficiency_detailed({"Cash Conversion Cycle": -438.0})

    row = next(r for r in breakdown["metrics"] if r["metric"] == "Cash Conversion Cycle")
    assert row["points"] == 0
    assert "Implausible" in row["reason"]
    # Base 50 + 0 for the flagged CCC, minus N/A-but-zero-point defaults elsewhere.
    assert score == 50.0


def test_cash_conversion_cycle_still_scores_normal_values():
    score, breakdown = _score_efficiency_detailed({"Cash Conversion Cycle": 15.0})

    row = next(r for r in breakdown["metrics"] if r["metric"] == "Cash Conversion Cycle")
    assert row["points"] == 10
    assert "excellent cash discipline" in row["reason"]

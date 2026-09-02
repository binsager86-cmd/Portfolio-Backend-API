"""
Regression tests for the Valuation score category's Earnings Yield and
EV/EBIT metrics: both must fall back to a CFA-standard calculation when the
underlying statement doesn't disclose EPS / EBIT directly, instead of
silently reporting N/A.
"""
import time

import pytest

from app.api.v1 import fundamental_legacy
from app.core.database import exec_sql, query_val


def _seed_stock(symbol: str) -> int:
    now = int(time.time())
    exec_sql(
        """
        INSERT INTO analysis_stocks (
            user_id, symbol, company_name, exchange, currency, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (1, symbol, "Valuation Fallback Co", "KSE", "KWD", now, now),
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


def test_eps_falls_back_to_net_income_over_shares_when_not_reported(test_client):
    """CFA: EPS = Net Income / Diluted Shares Outstanding. Calculate it only
    when the statement has no EPS_DILUTED / EPS_BASIC line item."""
    stock_id = _seed_stock("EPSFALLBACK_A")
    _seed_period(
        stock_id, "2025-12-31", 2025,
        {
            "NET_INCOME": 500_000.0,
            "TOTAL_EQUITY": 2_000_000.0,
            "TOTAL_COMMON_SHARES_OUTSTANDING": 1_000_000.0,
        },
    )

    result = fundamental_legacy._calculate_all_metrics(stock_id, "2025-12-31", 2025, None)

    assert result["valuation"]["EPS"] == pytest.approx(0.5)


def test_eps_prefers_reported_line_item_over_calculation(test_client):
    stock_id = _seed_stock("EPSFALLBACK_B")
    _seed_period(
        stock_id, "2025-12-31", 2025,
        {
            "NET_INCOME": 500_000.0,
            "TOTAL_COMMON_SHARES_OUTSTANDING": 1_000_000.0,
            "EPS_DILUTED": 0.42,
        },
    )

    result = fundamental_legacy._calculate_all_metrics(stock_id, "2025-12-31", 2025, None)

    assert result["valuation"]["EPS"] == pytest.approx(0.42)


def test_ev_ebit_falls_back_to_operating_income_when_ebit_missing(test_client, monkeypatch):
    """CFA: EBIT = Operating Income when the statement has no literal EBIT line."""
    monkeypatch.setattr(
        fundamental_legacy, "_fetch_yfinance_risk_data", lambda symbol: {"Current Price": 10.0}
    )

    stock_id = _seed_stock("EVEBITFALLBACK_A")
    _seed_period(
        stock_id, "2025-12-31", 2025,
        {
            "OPERATING_INCOME": 200_000.0,
            # Gives _calculate_all_metrics at least one metric to persist
            # (Book Value / Share) — without it stock_metrics stays empty
            # for the period and _compute_stock_score exits early.
            "TOTAL_EQUITY": 2_000_000.0,
            "TOTAL_COMMON_SHARES_OUTSTANDING": 1_000_000.0,
            "TOTAL_DEBT": 300_000.0,
            "CASH_EQUIVALENTS": 100_000.0,
        },
    )

    score = fundamental_legacy._compute_stock_score(stock_id, 1)

    # EV = 10 * 1,000,000 + 300,000 - 100,000 = 10,200,000
    # EV/EBIT = 10,200,000 / 200,000 = 51.0
    valuation_metrics = {row["metric"]: row for row in score["score_breakdown"]["valuation"]["metrics"]}
    assert valuation_metrics["EV/EBIT"]["value"] == pytest.approx(51.0)


def test_ev_ebit_reconstructs_from_net_income_when_operating_income_missing(test_client, monkeypatch):
    """CFA: EBIT = Net Income + Interest Expense + Income Tax Expense as the
    last-resort reconstruction when neither EBIT nor Operating Income is
    disclosed."""
    monkeypatch.setattr(
        fundamental_legacy, "_fetch_yfinance_risk_data", lambda symbol: {"Current Price": 10.0}
    )

    stock_id = _seed_stock("EVEBITFALLBACK_B")
    _seed_period(
        stock_id, "2025-12-31", 2025,
        {
            "NET_INCOME": 150_000.0,
            "INTEREST_EXPENSE": 30_000.0,
            "INCOME_TAX_EXPENSE": 20_000.0,
            "TOTAL_COMMON_SHARES_OUTSTANDING": 1_000_000.0,
            "TOTAL_DEBT": 300_000.0,
            "CASH_EQUIVALENTS": 100_000.0,
        },
    )

    score = fundamental_legacy._compute_stock_score(stock_id, 1)

    # EBIT = 150,000 + 30,000 + 20,000 = 200,000
    # EV = 10 * 1,000,000 + 300,000 - 100,000 = 10,200,000
    # EV/EBIT = 10,200,000 / 200,000 = 51.0
    valuation_metrics = {row["metric"]: row for row in score["score_breakdown"]["valuation"]["metrics"]}
    assert valuation_metrics["EV/EBIT"]["value"] == pytest.approx(51.0)

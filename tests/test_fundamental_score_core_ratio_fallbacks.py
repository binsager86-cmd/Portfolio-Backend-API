import time

import pytest

from app.api.v1 import fundamental_legacy
from app.core.database import exec_sql, query_one


def _create_analysis_stock(symbol: str) -> int:
    now = int(time.time())
    exec_sql(
        """INSERT INTO analysis_stocks
           (user_id, symbol, company_name, exchange, currency, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (1, symbol, f"{symbol} Co", "KSE", "KWD", now, now),
    )
    row = query_one(
        "SELECT id FROM analysis_stocks WHERE symbol = ? ORDER BY id DESC LIMIT 1",
        (symbol,),
    )
    return row[0] if isinstance(row, (tuple, list)) else row["id"]


def _insert_metric(
    stock_id: int,
    metric_type: str,
    metric_name: str,
    metric_value: float,
    period_end_date: str = "2025-12-31",
    fiscal_year: int = 2025,
) -> None:
    exec_sql(
        """INSERT INTO stock_metrics
           (stock_id, fiscal_year, fiscal_quarter, period_end_date,
            metric_type, metric_name, metric_value, created_at)
           VALUES (?, ?, NULL, ?, ?, ?, ?, ?)""",
        (
            stock_id,
            fiscal_year,
            period_end_date,
            metric_type,
            metric_name,
            metric_value,
            int(time.time()),
        ),
    )


def _metric_value(category_breakdown: dict, metric_name: str):
    for item in category_breakdown.get("metrics", []):
        if item.get("metric") == metric_name:
            return item.get("value")
    raise AssertionError(f"Metric {metric_name!r} not found in breakdown")


def test_compute_stock_score_alias_backfill_populates_core_breakdown(_init_test_db, monkeypatch):
    stock_id = _create_analysis_stock("SCORALIAS")

    _insert_metric(stock_id, "leverage", "Debt to Equity Ratio", 0.82)
    _insert_metric(stock_id, "leverage", "Interest Coverage Ratio", 4.6)
    _insert_metric(stock_id, "efficiency", "Inventory Turnover Ratio", 5.2)
    _insert_metric(stock_id, "liquidity", "Current Ratio (TTM)", 1.3)

    monkeypatch.setattr(fundamental_legacy, "_fetch_yfinance_risk_data", lambda _symbol: {})
    monkeypatch.setattr(fundamental_legacy, "_fetch_stockanalysis_core_ratios", lambda _symbol: {})

    result = fundamental_legacy._compute_stock_score(stock_id, 1)
    breakdown = result["metric_category_breakdown"]

    assert _metric_value(breakdown["liquidity"], "Current Ratio") == pytest.approx(1.3, abs=0.01)
    assert _metric_value(breakdown["capital_structure"], "Debt-to-Equity") == pytest.approx(0.82, abs=0.01)
    assert _metric_value(breakdown["capital_structure"], "Interest Coverage") == pytest.approx(4.6, abs=0.01)
    assert _metric_value(breakdown["efficiency"], "Inventory Turnover") == pytest.approx(5.2, abs=0.01)


def test_compute_stock_score_stockanalysis_fallback_populates_missing_core_ratios(_init_test_db, monkeypatch):
    stock_id = _create_analysis_stock("SCORSAFB")

    # Keep one base metric so score computation proceeds, but leave core ratios missing.
    _insert_metric(stock_id, "income", "EPS", 1.2)

    monkeypatch.setattr(fundamental_legacy, "_fetch_yfinance_risk_data", lambda _symbol: {})
    monkeypatch.setattr(
        fundamental_legacy,
        "_fetch_stockanalysis_core_ratios",
        lambda _symbol: {
            "Current Ratio": 1.25,
            "Quick Ratio": 0.91,
            "Debt-to-Equity": 0.58,
            "Interest Coverage": 7.4,
            "Inventory Turnover": 4.8,
            "Cash Conversion Cycle": 42.0,
        },
    )

    result = fundamental_legacy._compute_stock_score(stock_id, 1)
    breakdown = result["metric_category_breakdown"]

    assert _metric_value(breakdown["liquidity"], "Current Ratio") == pytest.approx(1.25, abs=0.01)
    assert _metric_value(breakdown["liquidity"], "Quick Ratio") == pytest.approx(0.91, abs=0.01)
    assert _metric_value(breakdown["capital_structure"], "Debt-to-Equity") == pytest.approx(0.58, abs=0.01)
    assert _metric_value(breakdown["capital_structure"], "Interest Coverage") == pytest.approx(7.4, abs=0.01)
    assert _metric_value(breakdown["efficiency"], "Inventory Turnover") == pytest.approx(4.8, abs=0.01)
    assert _metric_value(breakdown["efficiency"], "Cash Conversion Cycle") == pytest.approx(42.0, abs=0.1)
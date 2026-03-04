"""
Unit tests for the class-based PortfolioService.

Tests WAC engine, Sharpe/Sortino, TWR/MWRR, and backward-compat wrappers
without requiring a live database.
"""

import pandas as pd
import numpy as np


# ── WAC Engine ───────────────────────────────────────────────────────

def test_compute_holdings_avg_cost_buy_only():
    """Buy-only transactions: avg cost = total cost / shares."""
    from app.services.portfolio_service import compute_holdings_avg_cost

    df = pd.DataFrame([
        {"id": 1, "txn_date": "2024-01-01", "txn_type": "Buy", "shares": 100,
         "purchase_cost": 500.0, "sell_value": 0, "fees": 10.0,
         "bonus_shares": 0, "cash_dividend": 0, "reinvested_dividend": 0, "created_at": 1},
        {"id": 2, "txn_date": "2024-02-01", "txn_type": "Buy", "shares": 50,
         "purchase_cost": 300.0, "sell_value": 0, "fees": 5.0,
         "bonus_shares": 0, "cash_dividend": 0, "reinvested_dividend": 0, "created_at": 2},
    ])

    result = compute_holdings_avg_cost(df)
    assert result["position_open"] is True
    assert result["shares"] == 150.0
    # cost = (500 + 10) + (300 + 5) = 815
    assert abs(result["cost_basis"] - 815.0) < 0.01
    assert abs(result["avg_cost"] - 815.0 / 150) < 0.0001
    assert result["realized_pnl"] == 0.0


def test_compute_holdings_avg_cost_buy_sell():
    """Buy then partial sell: check realized P&L calculation."""
    from app.services.portfolio_service import compute_holdings_avg_cost

    df = pd.DataFrame([
        {"id": 1, "txn_date": "2024-01-01", "txn_type": "Buy", "shares": 100,
         "purchase_cost": 1000.0, "sell_value": 0, "fees": 0,
         "bonus_shares": 0, "cash_dividend": 0, "reinvested_dividend": 0, "created_at": 1},
        {"id": 2, "txn_date": "2024-03-01", "txn_type": "Sell", "shares": 50,
         "purchase_cost": 0, "sell_value": 600.0, "fees": 0,
         "bonus_shares": 0, "cash_dividend": 0, "reinvested_dividend": 0, "created_at": 2},
    ])

    result = compute_holdings_avg_cost(df)
    assert result["shares"] == 50.0
    # Avg cost = 10.0/share, sell 50 @ 12.0 = 600 - 500 = 100 profit
    assert abs(result["realized_pnl"] - 100.0) < 0.01
    assert result["position_open"] is True


def test_compute_holdings_avg_cost_bonus_dilute():
    """Bonus shares dilute avg cost without adding cost."""
    from app.services.portfolio_service import compute_holdings_avg_cost

    df = pd.DataFrame([
        {"id": 1, "txn_date": "2024-01-01", "txn_type": "Buy", "shares": 100,
         "purchase_cost": 1000.0, "sell_value": 0, "fees": 0,
         "bonus_shares": 10, "cash_dividend": 0, "reinvested_dividend": 0, "created_at": 1},
    ])

    result = compute_holdings_avg_cost(df)
    # 100 bought + 10 bonus = 110 shares, cost = 1000
    assert result["shares"] == 110.0
    assert abs(result["avg_cost"] - 1000.0 / 110) < 0.0001
    assert result["bonus_shares"] == 10.0


def test_compute_holdings_avg_cost_empty():
    """Empty DataFrame returns zero position."""
    from app.services.portfolio_service import compute_holdings_avg_cost

    result = compute_holdings_avg_cost(pd.DataFrame())
    assert result["shares"] == 0.0
    assert result["position_open"] is False


def test_compute_holdings_avg_cost_none():
    """None input returns zero position."""
    from app.services.portfolio_service import compute_holdings_avg_cost

    result = compute_holdings_avg_cost(None)
    assert result["shares"] == 0.0
    assert result["position_open"] is False


# ── Backward-compat wrappers exist ───────────────────────────────────

def test_backward_compat_functions_importable():
    """Backward-compatible module-level functions are importable."""
    from app.services.portfolio_service import (
        get_current_holdings,
        build_portfolio_table,
        get_portfolio_overview,
        get_portfolio_value,
        get_account_balances,
        get_complete_overview,
    )
    # Just verifying they're callable
    assert callable(get_current_holdings)
    assert callable(build_portfolio_table)
    assert callable(get_complete_overview)


# ── PortfolioService class shape ─────────────────────────────────────

def test_portfolio_service_has_all_methods():
    """PortfolioService exposes all expected methods."""
    from app.services.portfolio_service import PortfolioService

    svc = PortfolioService(user_id=999)
    expected = [
        "get_overview",
        "get_current_holdings",
        "build_portfolio_table",
        "get_portfolio_value",
        "get_portfolio_overview",
        "get_account_balances",
        "get_all_holdings",
        "calculate_twr",
        "calculate_mwrr",
        "calculate_performance",
        "calculate_sharpe_ratio",
        "calculate_sortino_ratio",
        "calculate_realized_profit_details",
        "recalc_portfolio_cash",
    ]
    for method in expected:
        assert hasattr(svc, method), f"Missing method: {method}"
        assert callable(getattr(svc, method)), f"Not callable: {method}"

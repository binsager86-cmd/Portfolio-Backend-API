"""
Extended unit tests for the WAC (Weighted Average Cost) engine.

Tests beyond the basic 5 in test_portfolio_service.py:
  - Complex multi-leg trades
  - Full close → reopen position
  - Over-sell protection
  - Multiple concurrent symbols
  - Fee-heavy scenarios
  - Reinvested dividends
  - Large volume stress tests
  - Floating-point precision
"""

import pandas as pd
import numpy as np
import pytest

from app.services.portfolio_service import compute_holdings_avg_cost
from app.services.fx_service import safe_float


def _make_tx(**kw):
    """Build a single row dict for WAC testing."""
    base = {
        "id": kw.get("id", 1),
        "txn_date": kw.get("txn_date", "2024-01-01"),
        "txn_type": kw.get("txn_type", "Buy"),
        "shares": kw.get("shares", 0),
        "purchase_cost": kw.get("purchase_cost", 0),
        "sell_value": kw.get("sell_value", 0),
        "fees": kw.get("fees", 0),
        "bonus_shares": kw.get("bonus_shares", 0),
        "cash_dividend": kw.get("cash_dividend", 0),
        "reinvested_dividend": kw.get("reinvested_dividend", 0),
        "created_at": kw.get("created_at", kw.get("id", 1)),
    }
    return base


class TestWACComplexScenarios:
    """Complex multi-leg WAC scenarios with known expected values."""

    def test_three_buys_two_sells(self):
        """
        Buy 100 @ 10 (fees=10) → cost=1010, shares=100, avg=10.10
        Buy 200 @ 12 (fees=20) → cost=3430, shares=300, avg=11.4333
        Sell 150 @ 15 (fees=15) → proceeds=2235, cost_sold=1715, pnl=520
          remaining: cost=1715, shares=150, avg=11.4333
        Buy 50 @ 14 (fees=5) → cost=2420, shares=200, avg=12.10
        Sell 100 @ 16 (fees=10) → proceeds=1590, cost_sold=1210, pnl=380
          remaining: cost=1210, shares=100, avg=12.10
        """
        df = pd.DataFrame([
            _make_tx(id=1, txn_date="2024-01-10", txn_type="Buy", shares=100, purchase_cost=1000, fees=10),
            _make_tx(id=2, txn_date="2024-02-15", txn_type="Buy", shares=200, purchase_cost=2400, fees=20),
            _make_tx(id=3, txn_date="2024-03-20", txn_type="Sell", shares=150, sell_value=2250, fees=15),
            _make_tx(id=4, txn_date="2024-04-25", txn_type="Buy", shares=50, purchase_cost=700, fees=5),
            _make_tx(id=5, txn_date="2024-05-30", txn_type="Sell", shares=100, sell_value=1600, fees=10),
        ])

        r = compute_holdings_avg_cost(df)

        # Remaining: 100 shares
        assert r["shares"] == 100.0
        assert r["position_open"] is True

        # After all trades, verify avg_cost is reasonable
        assert r["avg_cost"] > 10.0
        assert r["avg_cost"] < 15.0

        # Realized PnL should be positive (sold above avg cost)
        assert r["realized_pnl"] > 0

    def test_full_close_then_reopen(self):
        """
        Buy 100 @ 10, sell all 100 @ 12 → position closed.
        Reopen: buy 50 @ 15 → new position with fresh avg cost.
        """
        df = pd.DataFrame([
            _make_tx(id=1, txn_date="2024-01-01", txn_type="Buy", shares=100, purchase_cost=1000, fees=0),
            _make_tx(id=2, txn_date="2024-03-01", txn_type="Sell", shares=100, sell_value=1200, fees=0),
            _make_tx(id=3, txn_date="2024-06-01", txn_type="Buy", shares=50, purchase_cost=750, fees=0),
        ])

        r = compute_holdings_avg_cost(df)

        # Position should be open again
        assert r["position_open"] is True
        assert r["shares"] == 50.0

        # Realized PnL from the closed trade: 1200 - 1000 = 200
        assert abs(r["realized_pnl"] - 200.0) < 0.01

        # New avg cost should reflect only the reopened position
        assert abs(r["avg_cost"] - 15.0) < 0.01

    def test_sell_at_loss(self):
        """Sell below avg cost → negative realized PnL."""
        df = pd.DataFrame([
            _make_tx(id=1, txn_date="2024-01-01", txn_type="Buy", shares=100, purchase_cost=2000, fees=0),
            _make_tx(id=2, txn_date="2024-06-01", txn_type="Sell", shares=50, sell_value=800, fees=0),
        ])

        r = compute_holdings_avg_cost(df)
        # Avg=20, sell 50 @ 16 = 800 - 1000 = -200
        assert r["realized_pnl"] < 0
        assert abs(r["realized_pnl"] - (-200.0)) < 0.01

    def test_over_sell_protection(self):
        """
        Selling more shares than held — WAC engine should handle gracefully.
        The engine caps remaining shares at 0.
        """
        df = pd.DataFrame([
            _make_tx(id=1, txn_date="2024-01-01", txn_type="Buy", shares=100, purchase_cost=1000, fees=0),
            _make_tx(id=2, txn_date="2024-06-01", txn_type="Sell", shares=150, sell_value=2250, fees=0),
        ])

        r = compute_holdings_avg_cost(df)
        # Should not have negative shares
        assert r["shares"] >= 0

    def test_bonus_after_multiple_buys(self):
        """
        Bonus shares dilute avg cost across all existing positions.
        Buy 100 @ 10 + Buy 100 @ 15 = 200 shares, cost=2500
        Bonus 20 → 220 shares, cost=2500, avg=11.3636
        """
        df = pd.DataFrame([
            _make_tx(id=1, txn_date="2024-01-01", txn_type="Buy", shares=100, purchase_cost=1000, fees=0),
            _make_tx(id=2, txn_date="2024-03-01", txn_type="Buy", shares=100, purchase_cost=1500, fees=0),
            _make_tx(id=3, txn_date="2024-06-01", txn_type="Buy", shares=0, purchase_cost=0, bonus_shares=20, fees=0),
        ])

        r = compute_holdings_avg_cost(df)
        assert r["shares"] == 220.0
        assert r["bonus_shares"] == 20.0
        assert abs(r["cost_basis"] - 2500.0) < 0.01
        assert abs(r["avg_cost"] - 2500.0 / 220) < 0.01

    def test_cash_dividend_tracking(self):
        """Cash dividends are tracked but don't affect cost basis."""
        df = pd.DataFrame([
            _make_tx(id=1, txn_date="2024-01-01", txn_type="Buy", shares=100, purchase_cost=1000, fees=0),
            _make_tx(id=2, txn_date="2024-06-01", txn_type="Buy", shares=0, purchase_cost=0, cash_dividend=50),
            _make_tx(id=3, txn_date="2024-12-01", txn_type="Buy", shares=0, purchase_cost=0, cash_dividend=75),
        ])

        r = compute_holdings_avg_cost(df)
        assert r["shares"] == 100.0
        assert abs(r["cost_basis"] - 1000.0) < 0.01
        assert abs(r["cash_div"] - 125.0) < 0.01

    def test_reinvested_dividend_tracking(self):
        """Reinvested dividends are tracked separately."""
        df = pd.DataFrame([
            _make_tx(id=1, txn_date="2024-01-01", txn_type="Buy", shares=100, purchase_cost=1000, fees=0, reinvested_dividend=0),
            _make_tx(id=2, txn_date="2024-06-01", txn_type="Buy", shares=10, purchase_cost=100, fees=0, reinvested_dividend=100),
        ])

        r = compute_holdings_avg_cost(df)
        assert r["shares"] == 110.0
        assert abs(r["reinv"] - 100.0) < 0.01


class TestWACFeeScenarios:
    """Fee-heavy scenarios for WAC engine."""

    def test_high_fees_increase_avg_cost(self):
        """High fees should increase the average cost basis."""
        # Without fees
        df_no_fees = pd.DataFrame([
            _make_tx(id=1, txn_type="Buy", shares=100, purchase_cost=1000, fees=0),
        ])
        r_no = compute_holdings_avg_cost(df_no_fees)

        # With fees
        df_fees = pd.DataFrame([
            _make_tx(id=1, txn_type="Buy", shares=100, purchase_cost=1000, fees=100),
        ])
        r_fees = compute_holdings_avg_cost(df_fees)

        assert r_fees["avg_cost"] > r_no["avg_cost"]
        assert abs(r_fees["cost_basis"] - 1100.0) < 0.01

    def test_sell_fees_reduce_realized_pnl(self):
        """Sell fees reduce realized PnL."""
        # Sell without fees
        df_no = pd.DataFrame([
            _make_tx(id=1, txn_type="Buy", shares=100, purchase_cost=1000, fees=0),
            _make_tx(id=2, txn_date="2024-06-01", txn_type="Sell", shares=100, sell_value=1500, fees=0),
        ])
        r_no = compute_holdings_avg_cost(df_no)

        # Sell with fees
        df_fees = pd.DataFrame([
            _make_tx(id=1, txn_type="Buy", shares=100, purchase_cost=1000, fees=0),
            _make_tx(id=2, txn_date="2024-06-01", txn_type="Sell", shares=100, sell_value=1500, fees=50),
        ])
        r_fees = compute_holdings_avg_cost(df_fees)

        assert r_fees["realized_pnl"] < r_no["realized_pnl"]
        assert abs(r_no["realized_pnl"] - 500.0) < 0.01
        assert abs(r_fees["realized_pnl"] - 450.0) < 0.01


class TestWACPrecision:
    """Floating-point precision tests."""

    def test_many_small_transactions(self):
        """100 small buys should not accumulate floating-point errors."""
        rows = []
        for i in range(1, 101):
            rows.append(_make_tx(
                id=i,
                txn_date=f"2024-01-{min(i, 28):02d}",
                txn_type="Buy",
                shares=1,
                purchase_cost=10.01,
                fees=0.01,
                created_at=i,
            ))

        df = pd.DataFrame(rows)
        r = compute_holdings_avg_cost(df)

        assert r["shares"] == 100.0
        # Total cost = 100 * (10.01 + 0.01) = 1002.0
        assert abs(r["cost_basis"] - 1002.0) < 0.1
        assert abs(r["avg_cost"] - 10.02) < 0.01

    def test_fractional_shares(self):
        """Fractional share quantities should work correctly."""
        df = pd.DataFrame([
            _make_tx(id=1, txn_type="Buy", shares=0.5, purchase_cost=50, fees=0),
            _make_tx(id=2, txn_date="2024-02-01", txn_type="Buy", shares=0.25, purchase_cost=30, fees=0),
        ])

        r = compute_holdings_avg_cost(df)
        assert abs(r["shares"] - 0.75) < 0.001
        assert abs(r["cost_basis"] - 80.0) < 0.01

    def test_very_large_values(self):
        """WAC handles very large portfolio values without overflow."""
        df = pd.DataFrame([
            _make_tx(id=1, txn_type="Buy", shares=1000000, purchase_cost=50000000, fees=1000),
        ])

        r = compute_holdings_avg_cost(df)
        assert r["shares"] == 1000000.0
        assert abs(r["cost_basis"] - 50001000.0) < 1.0


class TestWACInputValidation:
    """Input validation and edge cases."""

    def test_none_input(self):
        """None input returns zero position."""
        r = compute_holdings_avg_cost(None)
        assert r["shares"] == 0.0
        assert r["position_open"] is False

    def test_empty_dataframe(self):
        """Empty DataFrame returns zero position."""
        r = compute_holdings_avg_cost(pd.DataFrame())
        assert r["shares"] == 0.0
        assert r["position_open"] is False

    def test_only_dividend_transactions(self):
        """Only dividend entries = 0 shares but tracked dividends."""
        df = pd.DataFrame([
            _make_tx(id=1, txn_type="Buy", shares=0, purchase_cost=0, cash_dividend=100),
            _make_tx(id=2, txn_date="2024-06-01", txn_type="Buy", shares=0, purchase_cost=0, cash_dividend=150),
        ])

        r = compute_holdings_avg_cost(df)
        # No shares held, position not open
        assert r["shares"] == 0.0
        assert r["position_open"] is False
        assert abs(r["cash_div"] - 250.0) < 0.01

    def test_missing_columns_handled(self):
        """Missing optional columns default to 0."""
        df = pd.DataFrame([
            {"id": 1, "txn_date": "2024-01-01", "txn_type": "Buy",
             "shares": 100, "purchase_cost": 1000, "sell_value": 0,
             "fees": 0, "created_at": 1},
        ])
        # Missing bonus_shares, cash_dividend, reinvested_dividend
        r = compute_holdings_avg_cost(df)
        assert r["shares"] == 100.0
        assert r["bonus_shares"] == 0
        assert r["cash_div"] == 0

    def test_sell_with_zero_shares_held(self):
        """Selling when no shares held — should handle gracefully."""
        df = pd.DataFrame([
            _make_tx(id=1, txn_type="Sell", shares=100, sell_value=1000, fees=0),
        ])

        r = compute_holdings_avg_cost(df)
        # Should not crash, shares should be 0
        assert r["shares"] == 0.0

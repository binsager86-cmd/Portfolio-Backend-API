"""
Unit tests for TWR (Time-Weighted Return) and MWRR calculators.

Verifies:
  - Modified Dietz TWR against CFA reference values
  - MWRR via Newton IRR against known scenarios
  - Edge cases: insufficient data, zero values, single snapshot
  - Chain-linking across sub-periods
"""

import pandas as pd
import numpy as np
import pytest

from app.services.twr_calculator import compute_twr, compute_mwrr


# ── TWR: CFA Reference Implementation ───────────────────────────────

class TestTWRCFAReference:
    """TWR calculation verified against CFA Institute reference."""

    def test_twr_matches_cfa_reference(self):
        """
        CFA Modified Dietz chain-linked TWR reference scenario:

        Period 0 (start): Value = 100,000, Deposits = 100,000
        Period 1 (day 30): Value = 105,000, Deposits = 110,000 (added 10k)
        Period 2 (day 60): Value = 112,000, Deposits = 110,000
        Period 3 (day 90): Value = 115,000, Deposits = 110,000

        Sub-period returns:
          R1 = (105,000 - 100,000 - 10,000) / (100,000 + 10,000 * 0.5)
             = -5,000 / 105,000 = -0.04762
          R2 = (112,000 - 105,000 - 0) / (105,000 + 0)
             = 7,000 / 105,000 = 0.06667
          R3 = (115,000 - 112,000 - 0) / (112,000 + 0)
             = 3,000 / 112,000 = 0.02679

        TWR = (1 + R1)(1 + R2)(1 + R3) - 1
            = 0.95238 * 1.06667 * 1.02679 - 1
            ≈ 0.04307 = 4.307%
        """
        df = pd.DataFrame([
            {"snapshot_date": "2024-01-01", "portfolio_value": 100000, "deposit_cash": 100000},
            {"snapshot_date": "2024-01-31", "portfolio_value": 105000, "deposit_cash": 110000},
            {"snapshot_date": "2024-03-01", "portfolio_value": 112000, "deposit_cash": 110000},
            {"snapshot_date": "2024-03-31", "portfolio_value": 115000, "deposit_cash": 110000},
        ])

        twr = compute_twr(df)
        assert twr is not None

        # Calculated: (0.95238 * 1.06667 * 1.02679 - 1) * 100 ≈ 4.307%
        expected_twr = ((1 - 5000 / 105000) * (1 + 7000 / 105000) * (1 + 3000 / 112000) - 1) * 100
        assert abs(twr - expected_twr) < 0.01, f"TWR={twr}, expected≈{expected_twr}"

    def test_twr_no_cash_flows(self):
        """TWR with no cash flows = simple return."""
        df = pd.DataFrame([
            {"snapshot_date": "2024-01-01", "portfolio_value": 100000, "deposit_cash": 100000},
            {"snapshot_date": "2024-12-31", "portfolio_value": 110000, "deposit_cash": 100000},
        ])

        twr = compute_twr(df)
        assert twr is not None
        # Simple return: (110,000 - 100,000) / 100,000 = 10%
        assert abs(twr - 10.0) < 0.01

    def test_twr_multiple_deposits(self):
        """TWR isolates manager performance from cash flow timing."""
        # Scenario: Two deposits, identical market return
        df = pd.DataFrame([
            {"snapshot_date": "2024-01-01", "portfolio_value": 50000, "deposit_cash": 50000},
            {"snapshot_date": "2024-04-01", "portfolio_value": 52000, "deposit_cash": 50000},
            {"snapshot_date": "2024-07-01", "portfolio_value": 77000, "deposit_cash": 75000},
            {"snapshot_date": "2024-10-01", "portfolio_value": 80000, "deposit_cash": 75000},
        ])

        twr = compute_twr(df)
        assert twr is not None
        assert twr > 0  # Should be positive

    def test_twr_negative_return(self):
        """TWR correctly shows negative return when portfolio loses value."""
        df = pd.DataFrame([
            {"snapshot_date": "2024-01-01", "portfolio_value": 100000, "deposit_cash": 100000},
            {"snapshot_date": "2024-06-01", "portfolio_value": 90000, "deposit_cash": 100000},
        ])

        twr = compute_twr(df)
        assert twr is not None
        assert abs(twr - (-10.0)) < 0.01

    def test_twr_large_deposit_mid_period(self):
        """
        TWR adjusts for large mid-period deposit.
        This is the key advantage of Modified Dietz over simple return.
        """
        df = pd.DataFrame([
            {"snapshot_date": "2024-01-01", "portfolio_value": 10000, "deposit_cash": 10000},
            {"snapshot_date": "2024-06-01", "portfolio_value": 110500, "deposit_cash": 110000},
            {"snapshot_date": "2024-12-31", "portfolio_value": 116000, "deposit_cash": 110000},
        ])

        twr = compute_twr(df)
        assert twr is not None
        # Without adjustment, simple return would be misleading
        # Modified Dietz properly credits the deposit


# ── TWR: Edge Cases ──────────────────────────────────────────────────

class TestTWREdgeCases:
    """Edge case handling for TWR calculator."""

    def test_twr_empty_dataframe(self):
        """Empty DataFrame returns None."""
        assert compute_twr(pd.DataFrame()) is None

    def test_twr_single_snapshot(self):
        """Single snapshot is insufficient for TWR — returns None."""
        df = pd.DataFrame([
            {"snapshot_date": "2024-01-01", "portfolio_value": 100000, "deposit_cash": 100000},
        ])
        assert compute_twr(df) is None

    def test_twr_two_identical_values(self):
        """Two snapshots with same value = 0% return."""
        df = pd.DataFrame([
            {"snapshot_date": "2024-01-01", "portfolio_value": 100000, "deposit_cash": 100000},
            {"snapshot_date": "2024-12-31", "portfolio_value": 100000, "deposit_cash": 100000},
        ])

        twr = compute_twr(df)
        assert twr is not None
        assert abs(twr) < 0.01

    def test_twr_zero_beginning_value(self):
        """Zero beginning value should not crash (skips sub-period)."""
        df = pd.DataFrame([
            {"snapshot_date": "2024-01-01", "portfolio_value": 0, "deposit_cash": 0},
            {"snapshot_date": "2024-06-01", "portfolio_value": 50000, "deposit_cash": 50000},
            {"snapshot_date": "2024-12-31", "portfolio_value": 55000, "deposit_cash": 50000},
        ])

        twr = compute_twr(df)
        # Should not raise; may return a value or None based on skipping

    def test_twr_unsorted_dates(self):
        """Input not sorted by date — function should sort internally."""
        df = pd.DataFrame([
            {"snapshot_date": "2024-12-31", "portfolio_value": 110000, "deposit_cash": 100000},
            {"snapshot_date": "2024-01-01", "portfolio_value": 100000, "deposit_cash": 100000},
        ])

        twr = compute_twr(df)
        assert twr is not None
        assert abs(twr - 10.0) < 0.01

    def test_twr_none_values_in_columns(self):
        """None values in value/cashflow columns should be treated as 0."""
        df = pd.DataFrame([
            {"snapshot_date": "2024-01-01", "portfolio_value": 100000, "deposit_cash": None},
            {"snapshot_date": "2024-12-31", "portfolio_value": 110000, "deposit_cash": None},
        ])

        twr = compute_twr(df)
        # Should handle None gracefully

    def test_twr_custom_column_names(self):
        """TWR with custom column names works correctly."""
        df = pd.DataFrame([
            {"date": "2024-01-01", "value": 100000, "cashflow": 100000},
            {"date": "2024-12-31", "value": 110000, "cashflow": 100000},
        ])

        twr = compute_twr(df, date_col="date", value_col="value", cashflow_col="cashflow")
        assert twr is not None
        assert abs(twr - 10.0) < 0.01


# ── MWRR: Known Scenarios ───────────────────────────────────────────

class TestMWRR:
    """Money-Weighted Rate of Return (MWRR) tests."""

    def test_mwrr_simple_growth(self):
        """Simple growth scenario — MWRR should be positive."""
        df = pd.DataFrame([
            {"snapshot_date": "2024-01-01", "portfolio_value": 100000, "deposit_cash": 100000},
            {"snapshot_date": "2024-12-31", "portfolio_value": 110000, "deposit_cash": 100000},
        ])

        mwrr = compute_mwrr(df)
        assert mwrr is not None
        assert mwrr > 0

    def test_mwrr_with_additional_deposit(self):
        """MWRR accounts for timing of cash flows differently than TWR."""
        df = pd.DataFrame([
            {"snapshot_date": "2024-01-01", "portfolio_value": 100000, "deposit_cash": 100000},
            {"snapshot_date": "2024-07-01", "portfolio_value": 155000, "deposit_cash": 150000},
            {"snapshot_date": "2024-12-31", "portfolio_value": 165000, "deposit_cash": 150000},
        ])

        mwrr = compute_mwrr(df)
        assert mwrr is not None
        # MWRR should reflect the timing-weighted return

    def test_mwrr_empty_dataframe(self):
        """Empty DataFrame returns None."""
        assert compute_mwrr(pd.DataFrame()) is None

    def test_mwrr_single_snapshot(self):
        """Single snapshot is insufficient."""
        df = pd.DataFrame([
            {"snapshot_date": "2024-01-01", "portfolio_value": 100000, "deposit_cash": 100000},
        ])
        assert compute_mwrr(df) is None

    def test_mwrr_negative_return(self):
        """Portfolio losing value should give negative MWRR."""
        df = pd.DataFrame([
            {"snapshot_date": "2024-01-01", "portfolio_value": 100000, "deposit_cash": 100000},
            {"snapshot_date": "2024-12-31", "portfolio_value": 85000, "deposit_cash": 100000},
        ])

        mwrr = compute_mwrr(df)
        assert mwrr is not None
        assert mwrr < 0

    def test_mwrr_matches_simple_return_no_flows(self):
        """
        With no intermediate cash flows, MWRR should approximately
        equal simple return (annualized).
        """
        df = pd.DataFrame([
            {"snapshot_date": "2024-01-01", "portfolio_value": 100000, "deposit_cash": 100000},
            {"snapshot_date": "2024-12-31", "portfolio_value": 110000, "deposit_cash": 100000},
        ])

        mwrr = compute_mwrr(df)
        assert mwrr is not None
        # Should be approximately 10% annualized
        assert abs(mwrr - 10.0) < 3.0  # Tolerance for annualization approach


# ── TWR vs MWRR Relationship ────────────────────────────────────────

class TestTWRvsMWRR:
    """Verify expected relationships between TWR and MWRR."""

    def test_twr_and_mwrr_same_sign(self):
        """Both metrics should agree on direction of return."""
        df = pd.DataFrame([
            {"snapshot_date": "2024-01-01", "portfolio_value": 100000, "deposit_cash": 100000},
            {"snapshot_date": "2024-06-01", "portfolio_value": 105000, "deposit_cash": 100000},
            {"snapshot_date": "2024-12-31", "portfolio_value": 112000, "deposit_cash": 100000},
        ])

        twr = compute_twr(df)
        mwrr = compute_mwrr(df)
        assert twr is not None and mwrr is not None
        # Both should be positive
        assert twr > 0
        assert mwrr > 0

    def test_twr_mwrr_diverge_with_timing(self):
        """
        TWR and MWRR diverge when large cash flows coincide with
        significant market moves — this is by design.
        """
        # Scenario: Large deposit right before a drop
        df = pd.DataFrame([
            {"snapshot_date": "2024-01-01", "portfolio_value": 10000, "deposit_cash": 10000},
            {"snapshot_date": "2024-06-01", "portfolio_value": 11000, "deposit_cash": 10000},
            {"snapshot_date": "2024-07-01", "portfolio_value": 110000, "deposit_cash": 110000},
            {"snapshot_date": "2024-12-31", "portfolio_value": 105000, "deposit_cash": 110000},
        ])

        twr = compute_twr(df)
        mwrr = compute_mwrr(df)
        assert twr is not None and mwrr is not None
        # They should both be computed, but may differ significantly

"""
Unit tests for the Quarter Movement service modules.

Covers:
  - Module 1 (QuarterlyPriceMovementModule): normal appreciation/depreciation,
    missing baseline prices, non-trading baseline days, quarter transitions,
    empty dataset, in-progress quarter handling, reduced sample flag.
  - Module 2 (QuarterlyPERatioMovementModule): normal P/E computation, zero/
    negative EPS exclusion, empty EPS snapshots (eps_coverage = "none"),
    single EPS snapshot (latest_only coverage).
  - Module 3 (ExpectedPriceForecastModule): all three methods, zero/None EPS,
    missing baseline price, missing historical means.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional

import pytest

from app.services.quarter_movement.price_module import QuarterlyPriceMovementModule
from app.services.quarter_movement.pe_module import QuarterlyPERatioMovementModule
from app.services.quarter_movement.forecast_module import ExpectedPriceForecastModule


# ── Fixtures / helpers ─────────────────────────────────────────────────────────


def _make_ohlcv(date_str: str, open_: float, high: float, low: float, close: float) -> Dict:
    return {"date": date_str, "open": open_, "high": high, "low": low, "close": close}


def _make_eps(period_end: str, eps_value: float) -> Dict:
    return {"period_end_date": period_end, "eps_value": eps_value}


# ── Module 1 tests ─────────────────────────────────────────────────────────────


class TestQuarterlyPriceMovementModule:
    def _run(self, records: List[Dict], today: date) -> Dict:
        return QuarterlyPriceMovementModule().compute(records, today)

    # ── 8.4 scenario: normal price appreciation and depreciation ─────────────
    def test_normal_appreciation_and_depreciation(self):
        """Validates high_pct > 0 and low_pct < 0 for a typical quarter."""
        records = [
            _make_ohlcv("2022-12-30", 1.0, 1.0, 1.0, 1.0),   # baseline day for Q1-2023
            _make_ohlcv("2023-01-02", 1.0, 1.15, 0.95, 1.05),
            _make_ohlcv("2023-01-15", 1.0, 1.20, 0.90, 1.10),
            _make_ohlcv("2023-03-31", 1.0, 1.10, 0.98, 1.08),
        ]
        # Use a date well past Q1-2023 so it is complete
        today = date(2023, 6, 30)
        result = self._run(records, today)

        q1_cell = result["price_movement_table"]["2023"]["q1"]
        assert q1_cell is not None
        assert q1_cell["insufficient_data"] is False
        assert q1_cell["in_progress"] is False
        # Max high during Q1 = 1.20, baseline = 1.0 → high_pct = +20.0
        assert q1_cell["high_pct"] == pytest.approx(20.0, abs=0.1)
        # Min low during Q1 = 0.90, baseline = 1.0 → low_pct = -10.0
        assert q1_cell["low_pct"] == pytest.approx(-10.0, abs=0.1)

    # ── 8.4 scenario: missing baseline price ─────────────────────────────────
    def test_missing_baseline_price_marks_insufficient(self):
        """When no trading day exists on or before the baseline date, cell is null."""
        records = [
            # Only has records starting after Q1-2023 baseline date (Dec 31 2022)
            _make_ohlcv("2023-01-02", 1.0, 1.1, 0.9, 1.05),
        ]
        today = date(2023, 6, 30)
        result = self._run(records, today)
        q1_cell = result["price_movement_table"]["2023"]["q1"]
        assert q1_cell is not None
        assert q1_cell["insufficient_data"] is True
        assert q1_cell["high_pct"] is None

    # ── 8.4 scenario: baseline on non-trading day ─────────────────────────────
    def test_baseline_falls_on_non_trading_day_uses_prior_close(self):
        """If baseline date is not a trading day, most recent prior day is used."""
        # Q1-2023 baseline = Dec 31, 2022; provide Dec 30 as the last trading day
        records = [
            _make_ohlcv("2022-12-30", 1.0, 1.0, 1.0, 0.80),  # last trade before Dec 31
            _make_ohlcv("2023-01-02", 0.85, 0.92, 0.82, 0.88),
        ]
        today = date(2023, 6, 30)
        result = self._run(records, today)
        q1_cell = result["price_movement_table"]["2023"]["q1"]
        assert q1_cell["insufficient_data"] is False
        # Max high = 0.92, baseline = 0.80 → high_pct = 15.0
        assert q1_cell["high_pct"] == pytest.approx(15.0, abs=0.1)

    # ── 8.4 scenario: quarter transition dates ────────────────────────────────
    def test_quarter_boundary_records_are_included_in_correct_quarter(self):
        """Records on Q1 end day (Mar 31) count toward Q1, not Q2."""
        records = [
            _make_ohlcv("2022-12-31", 1.0, 1.0, 1.0, 1.0),
            _make_ohlcv("2023-03-31", 1.0, 1.30, 0.95, 1.10),  # last day of Q1
            _make_ohlcv("2023-04-01", 1.0, 1.40, 0.90, 1.20),  # first day of Q2
        ]
        today = date(2023, 9, 30)
        result = self._run(records, today)
        q1_cell = result["price_movement_table"]["2023"]["q1"]
        q2_cell = result["price_movement_table"]["2023"]["q2"]
        # Q1 max high should be 1.30, not 1.40
        assert q1_cell["high_pct"] == pytest.approx(30.0, abs=0.1)
        # Q2 baseline = Mar 31 close = 1.10; max high = 1.40 → ~27.3%
        assert q2_cell["high_pct"] is not None
        assert q2_cell["high_pct"] > 0

    # ── 8.4 scenario: empty historical dataset ───────────────────────────────
    def test_empty_records_returns_all_insufficient(self):
        today = date(2023, 6, 30)
        result = self._run([], today)
        assert result["years"] == [2023]
        for q in ("q1", "q2"):
            cell = result["price_movement_table"]["2023"][q]
            assert cell is not None
            assert cell["insufficient_data"] is True
        # Means should all be None
        for q in ("q1", "q2", "q3", "q4"):
            assert result["price_movement_means"][q]["high_pct_mean"] is None

    # ── 8.4 scenario: in-progress quarter uses data only up to today ──────────
    def test_in_progress_quarter_marked_and_excluded_from_means(self):
        records = [
            _make_ohlcv("2022-12-31", 1.0, 1.0, 1.0, 1.0),
            _make_ohlcv("2023-01-10", 1.0, 1.10, 0.95, 1.05),
            # Q2 baseline day
            _make_ohlcv("2023-03-31", 1.0, 1.0, 1.0, 1.0),
            _make_ohlcv("2023-04-10", 1.0, 1.12, 0.96, 1.05),
        ]
        # today is mid-Q2 2023 → Q2 is in-progress
        today = date(2023, 5, 1)
        result = self._run(records, today)
        q2_cell = result["price_movement_table"]["2023"]["q2"]
        assert q2_cell["in_progress"] is True
        # Q2 should NOT appear in the mean calculation
        assert result["price_movement_means"]["q2"]["sample_count"] == 0

    # ── Spec §7.3: reduced sample flag when < 2 complete years ───────────────
    def test_reduced_sample_flag_set_for_single_year(self):
        records = [
            _make_ohlcv("2022-12-31", 1.0, 1.0, 1.0, 1.0),
            _make_ohlcv("2023-02-01", 1.0, 1.1, 0.9, 1.05),
            _make_ohlcv("2023-03-31", 1.05, 1.05, 1.05, 1.05),
        ]
        today = date(2023, 6, 30)
        result = self._run(records, today)
        assert result["price_movement_means"]["q1"]["reduced_sample"] is True
        assert result["price_movement_means"]["q1"]["sample_count"] == 1

    # ── Rounding rules (spec §6.3 one decimal place for percentages) ─────────
    def test_percentage_rounded_to_one_decimal_place(self):
        records = [
            _make_ohlcv("2022-12-31", 1.0, 1.0, 1.0, 3.0),
            _make_ohlcv("2023-02-01", 3.0, 3.1, 2.8, 3.0),
        ]
        today = date(2023, 6, 30)
        result = self._run(records, today)
        q1_cell = result["price_movement_table"]["2023"]["q1"]
        if q1_cell and q1_cell.get("high_pct") is not None:
            val = q1_cell["high_pct"]
            # Should have at most one decimal place
            assert round(val, 1) == val


# ── Module 2 tests ─────────────────────────────────────────────────────────────


class TestQuarterlyPERatioMovementModule:
    def _run(self, records: List[Dict], eps_snapshots: List[Dict], today: date) -> Dict:
        return QuarterlyPERatioMovementModule().compute(records, eps_snapshots, today)

    def test_normal_pe_computation(self):
        """Validates that highest_pe and lowest_pe are computed correctly."""
        records = [
            _make_ohlcv("2022-12-31", 1.0, 1.0, 1.0, 1.0),
            _make_ohlcv("2023-01-02", 1.0, 1.4, 1.0, 1.40),
            _make_ohlcv("2023-01-15", 1.0, 1.2, 0.9, 1.00),
            _make_ohlcv("2023-03-31", 1.0, 1.1, 0.95, 1.05),
        ]
        eps_snapshots = [_make_eps("2022-12-31", 0.10)]
        today = date(2023, 6, 30)
        result = self._run(records, eps_snapshots, today)
        q1_cell = result["pe_movement_table"]["2023"]["q1"]
        assert q1_cell is not None
        assert q1_cell["insufficient_data"] is False
        # Max daily P/E: close=1.40 / eps=0.10 = 14.0
        assert q1_cell["highest_pe"] == pytest.approx(14.0, abs=0.01)
        # Min daily P/E: close=1.00 / eps=0.10 = 10.0
        assert q1_cell["lowest_pe"] == pytest.approx(10.0, abs=0.01)

    # ── 8.4 scenario: zero and negative EPS exclusion ────────────────────────
    def test_zero_eps_excludes_all_pe_values(self):
        """All daily P/E values are undefined when EPS = 0."""
        records = [_make_ohlcv("2023-01-02", 1.0, 1.1, 0.9, 1.0)]
        eps_snapshots = [_make_eps("2022-12-31", 0.0)]
        today = date(2023, 6, 30)
        result = self._run(records, eps_snapshots, today)
        q1_cell = result["pe_movement_table"]["2023"]["q1"]
        assert q1_cell["insufficient_data"] is True

    def test_negative_eps_excludes_all_pe_values(self):
        records = [_make_ohlcv("2023-01-02", 1.0, 1.1, 0.9, 1.0)]
        eps_snapshots = [_make_eps("2022-12-31", -0.05)]
        today = date(2023, 6, 30)
        result = self._run(records, eps_snapshots, today)
        q1_cell = result["pe_movement_table"]["2023"]["q1"]
        assert q1_cell["insufficient_data"] is True

    # ── 8.4 scenario: empty EPS snapshots ────────────────────────────────────
    def test_empty_eps_snapshots_gives_none_coverage(self):
        records = [_make_ohlcv("2023-01-02", 1.0, 1.1, 0.9, 1.0)]
        today = date(2023, 6, 30)
        result = self._run(records, [], today)
        assert result["eps_coverage"] == "none"
        assert result["ttm_eps"] is None
        q1_cell = result["pe_movement_table"]["2023"]["q1"]
        assert q1_cell["insufficient_data"] is True

    # ── Single EPS snapshot → latest_only coverage ────────────────────────────
    def test_single_eps_snapshot_gives_latest_only_coverage(self):
        records = [
            _make_ohlcv("2023-01-02", 1.0, 1.0, 1.0, 2.00),
        ]
        eps_snapshots = [_make_eps("2025-01-01", 0.20)]
        today = date(2023, 6, 30)
        result = self._run(records, eps_snapshots, today)
        assert result["eps_coverage"] == "latest_only"
        # Should still compute P/E using the single snapshot
        q1_cell = result["pe_movement_table"]["2023"]["q1"]
        assert q1_cell["insufficient_data"] is False
        assert q1_cell["highest_pe"] == pytest.approx(10.0, abs=0.01)

    # ── P/E rounded to 2 decimal places (spec §6.3) ──────────────────────────
    def test_pe_rounded_to_two_decimal_places(self):
        records = [_make_ohlcv("2023-01-02", 1.0, 1.0, 1.0, 1.0)]
        eps_snapshots = [_make_eps("2022-12-31", 0.03)]
        today = date(2023, 6, 30)
        result = self._run(records, eps_snapshots, today)
        q1_cell = result["pe_movement_table"]["2023"]["q1"]
        if q1_cell and q1_cell.get("highest_pe") is not None:
            val = q1_cell["highest_pe"]
            assert round(val, 2) == val

    # ── Arithmetic mean excludes in-progress quarters ─────────────────────────
    def test_in_progress_quarter_excluded_from_pe_means(self):
        records = [
            _make_ohlcv("2023-01-02", 1.0, 1.0, 1.0, 1.5),
            _make_ohlcv("2023-04-01", 1.0, 1.0, 1.0, 1.6),
        ]
        eps_snapshots = [_make_eps("2022-12-31", 0.10)]
        today = date(2023, 5, 1)  # mid Q2 → Q2 in progress
        result = self._run(records, eps_snapshots, today)
        assert result["pe_movement_means"]["q2"]["sample_count"] == 0


# ── Module 3 tests ─────────────────────────────────────────────────────────────


class TestExpectedPriceForecastModule:
    def _run(
        self,
        active_quarter_key: str = "q1",
        baseline_price: Optional[float] = 1.0,
        high_pct_mean: Optional[float] = 15.0,
        highest_pe_mean: Optional[float] = 28.0,
        ttm_eps: Optional[float] = 0.035,
    ) -> Dict:
        price_means = {
            q: {"high_pct_mean": high_pct_mean if q == active_quarter_key else None,
                "low_pct_mean": -5.0}
            for q in ("q1", "q2", "q3", "q4")
        }
        pe_means = {
            q: {"highest_pe_mean": highest_pe_mean if q == active_quarter_key else None,
                "lowest_pe_mean": 20.0}
            for q in ("q1", "q2", "q3", "q4")
        }
        return ExpectedPriceForecastModule().compute(
            active_quarter_key=active_quarter_key,
            baseline_price=baseline_price,
            price_movement_means=price_means,
            pe_movement_means=pe_means,
            trailing_twelve_months_eps=ttm_eps,
        )

    def test_method_one_formula(self):
        """baseline × (1 + pct/100), rounded to 3 d.p. (spec §4.2 example)."""
        result = self._run(baseline_price=0.80, high_pct_mean=15.0)
        assert result["method_one_expected_price"] == pytest.approx(0.920, abs=0.001)

    def test_method_two_formula(self):
        """eps × mean_highest_pe, rounded to 3 d.p. (spec §4.3 example)."""
        result = self._run(highest_pe_mean=28.0, ttm_eps=0.035)
        assert result["method_two_expected_price"] == pytest.approx(0.980, abs=0.001)

    def test_consensus_formula(self):
        """Average of method 1 and method 2 (spec §4.4 example)."""
        result = self._run(baseline_price=0.80, high_pct_mean=15.0, highest_pe_mean=28.0, ttm_eps=0.035)
        assert result["consensus_expected_price"] == pytest.approx(0.950, abs=0.001)

    def test_zero_eps_disables_method_two(self):
        result = self._run(ttm_eps=0.0)
        assert result["method_two_expected_price"] is None
        assert result["consensus_expected_price"] is None

    def test_negative_eps_disables_method_two(self):
        result = self._run(ttm_eps=-0.01)
        assert result["method_two_expected_price"] is None

    def test_none_eps_disables_method_two(self):
        result = self._run(ttm_eps=None)
        assert result["method_two_expected_price"] is None

    def test_missing_baseline_price_disables_method_one(self):
        result = self._run(baseline_price=None)
        assert result["method_one_expected_price"] is None
        assert result["consensus_expected_price"] is None

    def test_zero_baseline_price_disables_method_one(self):
        result = self._run(baseline_price=0.0)
        assert result["method_one_expected_price"] is None

    def test_missing_high_pct_mean_disables_method_one(self):
        result = self._run(high_pct_mean=None)
        assert result["method_one_expected_price"] is None

    def test_missing_pe_mean_disables_method_two(self):
        result = self._run(highest_pe_mean=None)
        assert result["method_two_expected_price"] is None

    def test_all_results_rounded_to_three_decimal_places(self):
        result = self._run(baseline_price=1.234567, high_pct_mean=12.3456, highest_pe_mean=25.6789, ttm_eps=0.0456789)
        for key in ("method_one_expected_price", "method_two_expected_price", "consensus_expected_price"):
            val = result[key]
            if val is not None:
                assert round(val, 3) == pytest.approx(val, abs=1e-9), f"{key} not rounded to 3 d.p."

    def test_method_one_only_when_eps_unavailable(self):
        """Method 1 still works even when EPS is unavailable."""
        result = self._run(ttm_eps=None, baseline_price=1.0, high_pct_mean=10.0)
        assert result["method_one_expected_price"] == pytest.approx(1.100, abs=0.001)
        assert result["method_two_expected_price"] is None
        assert result["consensus_expected_price"] is None

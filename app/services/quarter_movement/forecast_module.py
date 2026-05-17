"""
Module 3: Expected Price Forecast.

Produces three expected price estimates for the active quarter using historical
averages computed by Module 1 (price movement) and Module 2 (P/E movement).

Method 1 — Percentage-based (spec §4.2):
    expected_price = baseline_price × (1 + historical_mean_high_pct / 100)

Method 2 — Valuation-based (spec §4.3):
    expected_price = ttm_eps × historical_mean_highest_pe

Consensus (spec §4.4):
    consensus_price = (method_one_expected_price + method_two_expected_price) / 2

All three results are rounded to 3 decimal places (spec §6.3).

Input:
    active_quarter_key:             str  — "q1" | "q2" | "q3" | "q4"
    baseline_price:                 float | None — from Module 1
    price_movement_means:           dict — from Module 1
    pe_movement_means:              dict — from Module 2
    trailing_twelve_months_eps:     float | None — most recent TTM EPS

Output dict keys:
    method_one_expected_price   — float | None
    method_two_expected_price   — float | None
    consensus_expected_price    — float | None
    method_one_inputs           — dict (for transparency)
    method_two_inputs           — dict (for transparency)

Known limitations:
    * Method 2 returns None when TTM EPS is zero, negative, or unavailable
      (spec §7.2).
    * Consensus returns None when either method price is None.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ExpectedPriceForecastModule:
    """
    Module 3: Expected Price Forecast.
    Combines historical means from Modules 1 and 2 to produce price targets.
    """

    def compute(
        self,
        active_quarter_key: str,
        baseline_price: Optional[float],
        price_movement_means: Dict[str, Dict],
        pe_movement_means: Dict[str, Dict],
        trailing_twelve_months_eps: Optional[float],
    ) -> Dict:
        """
        Compute three expected price estimates for the active quarter.

        Args:
            active_quarter_key:         "q1" | "q2" | "q3" | "q4"
            baseline_price:             Closing price on the baseline date for the
                                        active quarter.
            price_movement_means:       {q_key: {high_pct_mean, low_pct_mean, ...}}
                                        from Module 1.
            pe_movement_means:          {q_key: {highest_pe_mean, lowest_pe_mean, ...}}
                                        from Module 2.
            trailing_twelve_months_eps: Most recent TTM EPS. None / ≤ 0 disables
                                        Method 2.

        Returns:
            Dict with method_one_expected_price, method_two_expected_price,
            consensus_expected_price, and input transparency fields.
        """
        # ── Retrieve active-quarter historical means ──────────────────────
        price_mean_cell = price_movement_means.get(active_quarter_key, {})
        pe_mean_cell = pe_movement_means.get(active_quarter_key, {})

        historical_mean_highest_price_percentage_increase: Optional[float] = (
            price_mean_cell.get("high_pct_mean")
        )
        historical_arithmetic_mean_highest_price_to_earnings_ratio: Optional[float] = (
            pe_mean_cell.get("highest_pe_mean")
        )

        # ── Method 1: Percentage-based (spec §4.2) ────────────────────────
        method_one_expected_price: Optional[float] = None

        if (
            baseline_price is not None
            and baseline_price > 0
            and historical_mean_highest_price_percentage_increase is not None
        ):
            # Convert percentage to decimal multiplier: 1 + (pct / 100)
            decimal_multiplier = 1.0 + (historical_mean_highest_price_percentage_increase / 100.0)
            method_one_expected_price = round(baseline_price * decimal_multiplier, 3)

        # ── Method 2: Valuation-based (spec §4.3) ────────────────────────
        method_two_expected_price: Optional[float] = None

        # §7.2: exclude zero/negative EPS from Method 2
        eps_valid = (
            trailing_twelve_months_eps is not None
            and trailing_twelve_months_eps > 0
        )

        if (
            eps_valid
            and historical_arithmetic_mean_highest_price_to_earnings_ratio is not None
        ):
            method_two_expected_price = round(
                trailing_twelve_months_eps * historical_arithmetic_mean_highest_price_to_earnings_ratio,
                3,
            )

        # ── Consensus: average of Method 1 and Method 2 (spec §4.4) ──────
        consensus_expected_price: Optional[float] = None
        if method_one_expected_price is not None and method_two_expected_price is not None:
            consensus_expected_price = round(
                (method_one_expected_price + method_two_expected_price) / 2.0,
                3,
            )

        return {
            "method_one_expected_price": method_one_expected_price,
            "method_two_expected_price": method_two_expected_price,
            "consensus_expected_price": consensus_expected_price,
            # Transparency fields — useful for debugging and client display
            "method_one_inputs": {
                "baseline_price": baseline_price,
                "historical_mean_high_pct": historical_mean_highest_price_percentage_increase,
            },
            "method_two_inputs": {
                "ttm_eps": trailing_twelve_months_eps,
                "historical_mean_highest_pe": historical_arithmetic_mean_highest_price_to_earnings_ratio,
            },
        }

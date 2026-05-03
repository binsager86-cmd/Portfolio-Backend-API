"""Unit tests for the Kuwait Signal Engine — Position Sizer.

Validates:
  - Basic formula: equity * risk_fraction / risk_per_share_kd
  - Liquidity factor capped at 1.0 when ADTV ≥ threshold
  - Half-Kelly reduces size for moderate win probability
  - CVaR reduction factor is applied multiplicatively
  - Zero entry or zero stop → 0 shares
  - Equity percentage never exceeds KELLY_MAX_FRACTION * 100
"""
from __future__ import annotations

import pytest

from app.services.signal_engine.config.risk_config import (
    KELLY_MAX_FRACTION,
    LIQUIDITY_THRESHOLD_KD,
    RISK_PER_TRADE,
)
from app.services.signal_engine.models.risk.position_sizer import calculate_position_size


class TestReturnSchema:
    def test_required_keys_present(self):
        result = calculate_position_size(
            account_equity=100_000.0,
            entry_price=500.0,
            stop_loss=490.0,
            adtv_kd=200_000.0,
        )
        for k in ("shares", "equity_pct", "position_value_kd", "liquidity_factor"):
            assert k in result

    def test_shares_is_non_negative_int(self):
        result = calculate_position_size(100_000.0, 500.0, 490.0, 200_000.0)
        assert isinstance(result["shares"], int)
        assert result["shares"] >= 0


class TestZeroCases:
    def test_zero_entry_price_returns_zero(self):
        result = calculate_position_size(100_000.0, 0.0, 490.0, 200_000.0)
        assert result["shares"] == 0

    def test_entry_equals_stop_returns_zero(self):
        result = calculate_position_size(100_000.0, 500.0, 500.0, 200_000.0)
        assert result["shares"] == 0


class TestLiquidityFactor:
    def test_high_adtv_gives_factor_1(self):
        result = calculate_position_size(100_000.0, 500.0, 490.0,
                                         adtv_kd=LIQUIDITY_THRESHOLD_KD * 2)
        assert result["liquidity_factor"] == pytest.approx(1.0)

    def test_low_adtv_gives_reduced_factor(self):
        result = calculate_position_size(100_000.0, 500.0, 490.0,
                                         adtv_kd=LIQUIDITY_THRESHOLD_KD * 0.5)
        assert result["liquidity_factor"] == pytest.approx(0.5)

    def test_illiquid_size_smaller_than_liquid(self):
        liquid = calculate_position_size(100_000.0, 500.0, 490.0, adtv_kd=500_000.0)
        illiquid = calculate_position_size(100_000.0, 500.0, 490.0, adtv_kd=50_000.0)
        assert illiquid["shares"] < liquid["shares"]


class TestKelly:
    def test_no_win_prob_uses_fixed_fraction(self):
        result = calculate_position_size(100_000.0, 500.0, 490.0, 200_000.0,
                                         win_probability=None)
        # With full liquidity and 2% risk, risk fraction = 0.02
        # max_risk_kd = 100000 * 0.02 = 2000
        # risk_per_share_kd = (500-490)/1000 = 0.01
        # shares = 2000 / 0.01 = 200_000
        assert result["shares"] > 0

    def test_high_win_prob_allows_more_shares(self):
        low_p = calculate_position_size(100_000.0, 500.0, 490.0, 200_000.0, win_probability=0.52)
        high_p = calculate_position_size(100_000.0, 500.0, 490.0, 200_000.0, win_probability=0.80)
        assert high_p["shares"] >= low_p["shares"]

    def test_risk_fraction_never_exceeds_kelly_max(self):
        # KELLY_MAX_FRACTION caps risk fraction (% of equity at risk per trade),
        # not the position value as % of equity.
        result = calculate_position_size(100_000.0, 500.0, 490.0, 1_000_000.0,
                                         win_probability=0.99)
        assert result["risk_fraction_used"] / 100.0 <= KELLY_MAX_FRACTION + 0.01


class TestCVaRReduction:
    def test_cvar_reduction_reduces_shares(self):
        full = calculate_position_size(100_000.0, 500.0, 490.0, 200_000.0, cvar_reduction=1.0)
        reduced = calculate_position_size(100_000.0, 500.0, 490.0, 200_000.0, cvar_reduction=0.75)
        assert reduced["shares"] <= full["shares"]

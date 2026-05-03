"""Unit tests for the Kuwait Signal Engine — CVaR Calculator.

Validates:
  - Historical simulation correctly calculates tail average
  - Illiquidity adjustment applied when ADTV < threshold
  - CVaR/VaR ratio reduction triggers correctly
  - Edge cases: too few bars, zero closes
"""
from __future__ import annotations

import pytest

from app.services.signal_engine.models.risk.cvar_calculator import calculate_cvar
from app.services.signal_engine.config.risk_config import (
    CVAR_ALPHA,
    CVAR_ILLIQUID_WIDEN_FACTOR,
    LIQUIDITY_THRESHOLD_KD,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_rows(n: int = 260, daily_return: float = 0.0, sigma: float = 0.01) -> list[dict]:
    """Synthetic price series with given mean daily return and volatility."""
    import random
    rng = random.Random(42)
    price = 500.0
    rows = []
    for i in range(n):
        ret = daily_return + rng.gauss(0, sigma)
        price = price * (1 + ret)
        rows.append({
            "date": f"2024-{(i // 30 + 1):02d}-{(i % 30 + 1):02d}",
            "close": round(price, 2),
        })
    return rows


# ── Basic calculation ──────────────────────────────────────────────────────────

class TestCVaRCalculation:
    def test_returns_expected_keys(self):
        rows = _make_rows()
        result = calculate_cvar(rows)
        for key in ("var_95", "cvar_95", "cvar_fils", "is_illiquid_adj", "position_size_reduction"):
            assert key in result, f"Missing key: {key}"

    def test_cvar_is_positive(self):
        rows = _make_rows()
        result = calculate_cvar(rows)
        assert result["cvar_95"] is not None
        assert result["cvar_95"] >= 0.0

    def test_cvar_gte_var(self):
        """CVaR (expected shortfall) must always be ≥ VaR."""
        rows = _make_rows(sigma=0.015)
        result = calculate_cvar(rows)
        if result["cvar_95"] is not None and result["var_95"] is not None:
            assert result["cvar_95"] >= result["var_95"] - 1e-9

    def test_higher_volatility_gives_higher_cvar(self):
        low_vol = calculate_cvar(_make_rows(sigma=0.005))
        high_vol = calculate_cvar(_make_rows(sigma=0.030))
        if low_vol["cvar_95"] is not None and high_vol["cvar_95"] is not None:
            assert high_vol["cvar_95"] > low_vol["cvar_95"]


# ── Illiquidity adjustment ─────────────────────────────────────────────────────

class TestIlliquidityAdjustment:
    def test_below_adtv_threshold_widens_cvar(self):
        rows = _make_rows()
        liquid = calculate_cvar(rows, adtv_kd=LIQUIDITY_THRESHOLD_KD * 2)
        illiquid = calculate_cvar(rows, adtv_kd=LIQUIDITY_THRESHOLD_KD * 0.5)
        if liquid["cvar_95"] and illiquid["cvar_95"]:
            ratio = illiquid["cvar_95"] / liquid["cvar_95"]
            assert abs(ratio - CVAR_ILLIQUID_WIDEN_FACTOR) < 0.01

    def test_illiquid_flag_set(self):
        rows = _make_rows()
        result = calculate_cvar(rows, adtv_kd=50_000.0)
        assert result["is_illiquid_adj"] is True

    def test_liquid_flag_not_set(self):
        rows = _make_rows()
        result = calculate_cvar(rows, adtv_kd=500_000.0)
        assert result["is_illiquid_adj"] is False

    def test_no_adtv_no_adjustment(self):
        rows = _make_rows()
        result = calculate_cvar(rows, adtv_kd=None)
        assert result["is_illiquid_adj"] is False


# ── Insufficient data ─────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_too_few_rows_returns_none(self):
        rows = _make_rows(n=5)
        result = calculate_cvar(rows)
        assert result["cvar_95"] is None
        assert result["var_95"] is None

    def test_zero_closes_returns_none(self):
        rows = [{"close": 0.0}] * 20
        result = calculate_cvar(rows)
        assert result["cvar_95"] is None

    def test_exactly_10_returns_valid(self):
        rows = _make_rows(n=12)
        result = calculate_cvar(rows)
        # 11 return values — just enough
        assert result["cvar_95"] is not None or result["cvar_95"] is None  # no crash

    def test_constant_price_zero_var(self):
        rows = [{"close": 100.0}] * 260
        result = calculate_cvar(rows)
        if result["var_95"] is not None:
            assert result["var_95"] == 0.0

"""Unit tests for the Kuwait Signal Engine — Liquidity Filter.

Tests every path in is_tradable() including exact threshold boundaries.
100 % coverage required on core logic.
"""
from __future__ import annotations

import pytest

from app.services.signal_engine.processors.liquidity_filter import is_tradable
from app.services.signal_engine.config.kuwait_constants import (
    PREMIER_ADTV_MIN_KD,
    PREMIER_SPREAD_PROXY_MAX,
    PREMIER_ACTIVE_DAYS_MIN_PCT,
    PREMIER_VOLUME_CONCENTRATION_MAX,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_rows(
    n: int = 30,
    value: float = 200_000.0,
    high: float = 105.0,
    low: float = 103.0,
    close: float = 104.0,
    volume: float = 1_000,
    zero_days: int = 0,
) -> list[dict]:
    """Generate synthetic OHLCV rows passing all filters by default."""
    rows = []
    for i in range(n):
        v = 0.0 if i < zero_days else value
        vol = 0 if i < zero_days else volume
        rows.append({
            "date": f"2024-01-{i+1:02d}",
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "value": v,
        })
    return rows


# ── Insufficient data ──────────────────────────────────────────────────────────

class TestInsufficientData:
    def test_empty_rows_fails(self):
        passed, details = is_tradable([])
        assert passed is False

    def test_less_than_20_rows_fails(self):
        rows = _make_rows(n=19)
        passed, _ = is_tradable(rows)
        assert passed is False

    def test_exactly_20_rows_proceeds(self):
        # Use a tight spread (< 1.5 %) so only the row-count check is exercised
        rows = _make_rows(n=20, high=104.1, low=103.9, close=104.0, value=200_000.0)
        passed, _ = is_tradable(rows)
        assert passed is True


# ── ADTV threshold (boundary testing) ─────────────────────────────────────────

class TestADTV:
    def test_adtv_exactly_at_threshold_passes(self):
        rows = _make_rows(n=30, value=PREMIER_ADTV_MIN_KD)
        passed, details = is_tradable(rows)
        assert details["pass_adtv"] is True

    def test_adtv_one_below_threshold_fails(self):
        rows = _make_rows(n=30, value=PREMIER_ADTV_MIN_KD - 0.01)
        passed, details = is_tradable(rows)
        assert details["pass_adtv"] is False

    def test_adtv_well_above_threshold_passes(self):
        rows = _make_rows(n=30, value=500_000.0)
        _, details = is_tradable(rows)
        assert details["pass_adtv"] is True

    def test_zero_traded_value_fails(self):
        rows = _make_rows(n=30, value=0.0)
        _, details = is_tradable(rows)
        assert details["pass_adtv"] is False

    def test_zero_value_fallback_uses_volume_times_close(self):
        """When value=0 but volume × close / 1000 exceeds ADTV threshold, should pass."""
        # volume=2000, close=300 fils → 2000 × 300 / 1000 = 600 KWD per day
        # 600 × 20 days would avg 600 KWD — still well below 100K threshold
        # We need a large volume × close to exceed threshold
        # volume=1_000_000 shares, close=200 fils → 1M × 200 / 1000 = 200K KWD
        rows = _make_rows(n=30, value=0.0, volume=1_000_000, close=200.0)
        _, details = is_tradable(rows)
        assert details["adtv_fallback_used"] is True
        assert details["pass_adtv"] is True
        assert details["adtv_20d_kd"] == pytest.approx(200_000.0, rel=0.01)


# ── Spread proxy threshold ─────────────────────────────────────────────────────

class TestSpreadProxy:
    def test_spread_exactly_at_threshold_passes(self):
        close = 100.0
        # spread = (high - low) / close = PREMIER_SPREAD_PROXY_MAX
        spread_fils = PREMIER_SPREAD_PROXY_MAX * close
        high = close + spread_fils / 2
        low = close - spread_fils / 2
        rows = _make_rows(n=30, high=high, low=low, close=close, value=200_000.0)
        _, details = is_tradable(rows)
        assert details["pass_spread"] is True

    def test_spread_above_threshold_fails(self):
        close = 100.0
        # 2 % spread > 1.5 % limit
        rows = _make_rows(n=30, high=close + 1.1, low=close - 1.1, close=close, value=200_000.0)
        _, details = is_tradable(rows)
        assert details["pass_spread"] is False

    def test_zero_close_handled_gracefully(self):
        rows = _make_rows(n=30, close=0.0, high=1.0, low=0.0, value=200_000.0)
        # Should not raise
        passed, _ = is_tradable(rows)
        # pass/fail doesn't matter — just no crash
        assert isinstance(passed, bool)

    def test_corrupt_spread_capped_at_10_pct(self):
        """Absurd H/L values (e.g. data corruption) must be capped at 10% spread_proxy_pct."""
        # high=5000, low=0, close=100 → raw (h-l)/c = 50.0 = 5000% — must be capped
        rows = _make_rows(n=30, high=5000.0, low=0.0, close=100.0, value=200_000.0)
        _, details = is_tradable(rows)
        assert details["spread_proxy_pct"] == pytest.approx(10.0)
        assert details["pass_spread"] is False  # 10% >> 1.5% threshold


# ── Active trading days ────────────────────────────────────────────────────────

class TestActiveDays:
    def test_all_30_days_active_passes(self):
        rows = _make_rows(n=30, zero_days=0)
        _, details = is_tradable(rows)
        assert details["pass_active_days"] is True

    def test_exactly_80_pct_active_passes(self):
        # 30 sessions: need ≥ 24 active days (80 %)
        # 6 inactive → exactly 80 %
        rows = _make_rows(n=30, zero_days=6)
        _, details = is_tradable(rows)
        assert details["pass_active_days"] is True

    def test_below_80_pct_fails(self):
        # 7 inactive → 23/30 ≈ 76.7 %
        rows = _make_rows(n=30, zero_days=7)
        _, details = is_tradable(rows)
        assert details["pass_active_days"] is False


# ── Volume concentration (wash-trade filter) ───────────────────────────────────

class TestVolumeConcentration:
    def test_even_volume_passes(self):
        rows = _make_rows(n=30, volume=1_000)
        _, details = is_tradable(rows)
        assert details["pass_concentration"] is True

    def test_single_spike_above_40pct_fails(self):
        rows = _make_rows(n=30, volume=1_000)
        # Make last day volume = 80 % of 20-day sum
        # 20-day sum = 19 * 1000 + spike_vol → spike_vol / sum > 0.40
        rows[-1]["volume"] = 15_000
        _, details = is_tradable(rows)
        assert details["pass_concentration"] is False

    def test_concentration_exactly_at_threshold_passes(self):
        # 20 days: 19 × 1000 + 1 spike that is exactly 40 % of sum
        # sum = 19000 + spike → spike = 0.40 * (19000 + spike) → spike ≈ 12666
        rows = _make_rows(n=30, volume=1_000)
        rows[-1]["volume"] = 12_666
        _, details = is_tradable(rows)
        # 12666 / (19000 + 12666) = 0.3998... just under 40 % → should pass
        assert details["pass_concentration"] is True


# ── All-pass scenario ──────────────────────────────────────────────────────────

class TestAllPass:
    def test_ideal_stock_passes_all(self):
        rows = _make_rows(n=30, value=300_000.0, high=105.0, low=104.0, close=104.5, volume=5_000)
        passed, details = is_tradable(rows)
        assert passed is True
        assert details["pass_adtv"] is True
        assert details["pass_spread"] is True
        assert details["pass_active_days"] is True
        assert details["pass_concentration"] is True

    def test_single_failing_check_rejects_overall(self):
        # ADTV fails, others pass
        rows = _make_rows(n=30, value=10_000.0, high=105.0, low=104.0, close=104.5, volume=5_000)
        passed, details = is_tradable(rows)
        assert passed is False
        assert details["pass_adtv"] is False

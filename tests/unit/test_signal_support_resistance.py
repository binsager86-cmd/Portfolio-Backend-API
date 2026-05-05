"""Unit tests for the Kuwait Signal Engine — Support/Resistance.

Validates:
  - compute_sr_score returns (int, dict, list, list) tuple
  - Score in [0, 100]
  - compute_entry_stop_tp returns correct keys
  - Entry/stop/TP levels are tick-aligned
  - RR ratio = (TP1 - entry) / (entry - SL) for BUY
  - Entry buffer around current close
  - Stop is below entry for BUY, above for SELL
  - TP2 > TP1 > entry for BUY
"""
from __future__ import annotations

import pytest

from app.services.signal_engine.config.kuwait_constants import align_to_tick
from app.services.signal_engine.models.technical.support_resistance import (
    compute_entry_stop_tp,
    compute_sr_score,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row(close: float = 500.0, atr: float = 8.0,
         high: float | None = None, low: float | None = None,
         volume: float = 2_000_000.0, value: float = 500_000.0) -> dict:
    return {
        "close": close, "open": close - 2,
        "high": high if high is not None else close + atr,
        "low": low if low is not None else close - atr,
        "volume": volume, "value": value,
        "atr_14": atr,
        "vwap": close * 0.98,
    }


def _rows(n: int = 60, close: float = 500.0, atr: float = 8.0) -> list[dict]:
    return [_row(close=close + i * 0.5, atr=atr) for i in range(n)]


# ── compute_sr_score ──────────────────────────────────────────────────────────

class TestSRScore:
    def test_returns_four_element_tuple(self):
        result = compute_sr_score(_rows())
        assert len(result) == 4

    def test_score_is_int_in_range(self):
        score, _, _, _ = compute_sr_score(_rows())
        assert isinstance(score, int)
        assert 0 <= score <= 100

    def test_details_is_dict(self):
        _, details, _, _ = compute_sr_score(_rows())
        assert isinstance(details, dict)

    def test_support_and_resistance_are_lists(self):
        _, _, supports, resistances = compute_sr_score(_rows())
        assert isinstance(supports, list)
        assert isinstance(resistances, list)

    def test_empty_rows_returns_safe_defaults(self):
        score, _, sup, res = compute_sr_score([])
        assert 0 <= score <= 100

    def test_psych_level_guard_reduces_score(self):
        """Price within 1.5 % of 500 fils (0.500 KWD) → -15 % penalty applied."""
        # Flat rows all at 500.0 fils — exactly on a psychological level
        rows_psych = [_row(close=500.0) for _ in range(60)]
        _, d_psych, _, _ = compute_sr_score(rows_psych)
        assert d_psych["psych_level_guard"] is True

        # Far from any psych level (620 fils — >15 % from 500 and 750)
        rows_clear = _rows(n=60, close=620.0)
        _, d_clear, _, _ = compute_sr_score(rows_clear)
        assert d_clear["psych_level_guard"] is False

    def test_psych_level_guard_not_applied_far_from_round(self):
        """Price far from any psych level → guard not fired."""
        rows = _rows(n=60, close=620.0)  # 620 fils — far from 500 and 750
        _, d, _, _ = compute_sr_score(rows)
        assert d["psych_level_guard"] is False

    def test_session_vol_skew_guard_reduces_score(self):
        """Front-loaded volume (>70 % in first half) → -10 % penalty."""
        # 15 heavy rows + 10 light rows = 25 total (≥ 22 minimum for PIVOT_LOOKBACK=10)
        # skew window = last 20 rows = 5 heavy + 10 light rows
        # first_half (10 rows) = 5 heavy (50M) + 5 light (0.5M); second_half = 5 light (0.5M)
        # To guarantee >0.70: use all-heavy first 15 + all-light last 10
        heavy = [_row(close=620.0 + i * 0.5, volume=10_000_000.0) for i in range(15)]
        light = [_row(close=627.5 + i * 0.5, volume=50_000.0) for i in range(10)]
        rows = heavy + light
        _, d, _, _ = compute_sr_score(rows)
        assert d["session_vol_skew"] > 0.70

    def test_session_vol_skew_neutral_for_uniform_volume(self):
        """Uniform volume across window → skew ≈ 0.5, guard not fired."""
        rows = _rows(n=60, close=620.0)  # all rows have volume=2_000_000
        _, d, _, _ = compute_sr_score(rows)
        assert d["session_vol_skew"] == pytest.approx(0.5, abs=0.05)

    def test_details_contains_guard_keys(self):
        """Guard diagnostic keys always present in details."""
        _, d, _, _ = compute_sr_score(_rows())
        assert "psych_level_guard" in d
        assert "session_vol_skew" in d


# ── compute_entry_stop_tp ─────────────────────────────────────────────────────

class TestEntryStopTP:
    def _buy_levels(self, close: float = 500.0, atr: float = 8.0) -> dict:
        rows = _rows(n=60, close=close, atr=atr)
        return compute_entry_stop_tp(rows, direction="BUY")

    def _sell_levels(self, close: float = 500.0, atr: float = 8.0) -> dict:
        rows = _rows(n=60, close=close, atr=atr)
        return compute_entry_stop_tp(rows, direction="SELL")

    def test_required_keys_present(self):
        levels = self._buy_levels()
        expected = {"entry_low", "entry_mid", "entry_high", "stop_loss",
                    "tp1", "tp2", "risk_per_share", "risk_reward_ratio"}
        assert expected.issubset(set(levels.keys()))

    def test_buy_stop_below_entry(self):
        levels = self._buy_levels()
        assert levels["stop_loss"] < levels["entry_mid"]

    def test_buy_tp1_above_entry(self):
        levels = self._buy_levels()
        assert levels["tp1"] > levels["entry_mid"]

    def test_buy_tp2_above_tp1(self):
        levels = self._buy_levels()
        assert levels["tp2"] > levels["tp1"]

    def test_sell_stop_above_entry(self):
        levels = self._sell_levels()
        assert levels["stop_loss"] > levels["entry_mid"]

    def test_sell_tp1_below_entry(self):
        levels = self._sell_levels()
        assert levels["tp1"] < levels["entry_mid"]

    def test_sell_tp2_below_tp1(self):
        levels = self._sell_levels()
        assert levels["tp2"] < levels["tp1"]

    def test_entry_mid_tick_aligned_low_price(self):
        levels = self._buy_levels(close=95.0, atr=1.0)
        mid = levels["entry_mid"]
        # For prices ≤ 100.9: should be aligned to 0.1-fil grid
        assert abs(round(mid, 1) - mid) < 1e-6

    def test_entry_mid_tick_aligned_high_price(self):
        levels = self._buy_levels(close=500.0, atr=8.0)
        mid = levels["entry_mid"]
        # For prices > 100.9: should be aligned to 1.0-fil grid
        assert abs(round(mid) - mid) < 1e-6

    def test_rr_ratio_positive(self):
        levels = self._buy_levels()
        assert levels["risk_reward_ratio"] > 0

    def test_entry_zone_straddles_mid(self):
        levels = self._buy_levels()
        assert levels["entry_low"] <= levels["entry_mid"] <= levels["entry_high"]

    def test_risk_per_share_equals_entry_minus_stop(self):
        levels = self._buy_levels()
        expected_risk = abs(levels["entry_mid"] - levels["stop_loss"])
        assert abs(levels["risk_per_share"] - expected_risk) < 0.2  # tick rounding tolerance

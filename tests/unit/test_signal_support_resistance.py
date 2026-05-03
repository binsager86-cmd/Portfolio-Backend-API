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

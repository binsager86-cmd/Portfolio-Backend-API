"""Unit tests for the Kuwait Signal Engine — Corporate Actions.

Validates:
  - adjust_for_dividends subtracts dividend fils from prices before ex-date
  - adjust_for_splits divides prices, multiplies volume
  - is_near_ex_dividend returns True within EX_DIV_BUFFER_DAYS
  - get_corporate_action_flag priority: SPLIT > RIGHTS > BONUS > DIVIDEND > NONE
  - apply_all_adjustments runs splits first, then dividends
  - Input rows not mutated
"""
from __future__ import annotations

import pytest

from app.services.signal_engine.data.fetchers.corporate_actions import (
    CorporateActionFlag,
    EX_DIV_BUFFER_DAYS,
    adjust_for_dividends,
    adjust_for_splits,
    apply_all_adjustments,
    get_corporate_action_flag,
    is_near_ex_dividend,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ohlcv(date: str, close: float = 500.0, volume: float = 1_000_000.0) -> dict:
    return {"date": date, "open": close, "high": close + 5,
            "low": close - 5, "close": close, "volume": volume, "value": 500_000.0}


def _rows() -> list[dict]:
    return [
        _ohlcv("2024-01-01", close=500.0),
        _ohlcv("2024-01-02", close=502.0),
        _ohlcv("2024-01-03", close=498.0),
        _ohlcv("2024-01-08", close=505.0),   # after gap
    ]


# ── Dividend adjustment ────────────────────────────────────────────────────────

class TestDividendAdjustment:
    def test_prices_reduced_before_ex_date(self):
        rows = _rows()
        # Event schema: 'date' = ex_date, 'value' = fils per share
        events = [{"type": "DIVIDEND", "date": "2024-01-03", "value": 10.0}]
        adjusted = adjust_for_dividends(rows, events)
        # Rows before ex-date (2024-01-01, 2024-01-02) should have prices −10
        assert adjusted[0]["close"] == pytest.approx(490.0)
        assert adjusted[1]["close"] == pytest.approx(492.0)

    def test_prices_not_changed_on_or_after_ex_date(self):
        rows = _rows()
        events = [{"type": "DIVIDEND", "date": "2024-01-03", "value": 10.0}]
        adjusted = adjust_for_dividends(rows, events)
        assert adjusted[2]["close"] == pytest.approx(498.0)
        assert adjusted[3]["close"] == pytest.approx(505.0)

    def test_does_not_mutate_input(self):
        rows = _rows()
        original_close = rows[0]["close"]
        events = [{"type": "DIVIDEND", "date": "2024-01-03", "value": 5.0}]
        adjust_for_dividends(rows, events)
        assert rows[0]["close"] == original_close


# ── Split adjustment ───────────────────────────────────────────────────────────

class TestSplitAdjustment:
    def test_prices_divided_before_effective_date(self):
        rows = _rows()
        # Event schema: 'date' = effective_date, 'value' = split ratio
        events = [{"type": "SPLIT", "date": "2024-01-03", "value": 2.0}]
        adjusted = adjust_for_splits(rows, events)
        assert adjusted[0]["close"] == pytest.approx(250.0)
        assert adjusted[1]["close"] == pytest.approx(251.0)

    def test_volume_multiplied_before_effective_date(self):
        rows = _rows()
        events = [{"type": "SPLIT", "date": "2024-01-03", "value": 2.0}]
        adjusted = adjust_for_splits(rows, events)
        assert adjusted[0]["volume"] == pytest.approx(2_000_000.0)

    def test_post_split_prices_unchanged(self):
        rows = _rows()
        events = [{"type": "SPLIT", "date": "2024-01-03", "value": 2.0}]
        adjusted = adjust_for_splits(rows, events)
        assert adjusted[2]["close"] == pytest.approx(498.0)


# ── is_near_ex_dividend ───────────────────────────────────────────────────────

class TestNearExDividend:
    def test_same_day_as_ex_div_is_near(self):
        assert is_near_ex_dividend("2024-03-15", ["2024-03-15"]) is True

    def test_within_buffer_days_is_near(self):
        # EX_DIV_BUFFER_DAYS = 3 by default
        assert is_near_ex_dividend("2024-03-13", ["2024-03-15"]) is True
        assert is_near_ex_dividend("2024-03-18", ["2024-03-15"]) is True

    def test_beyond_buffer_is_not_near(self):
        assert is_near_ex_dividend("2024-03-10", ["2024-03-15"]) is False
        assert is_near_ex_dividend("2024-03-21", ["2024-03-15"]) is False

    def test_empty_ex_dates_returns_false(self):
        assert is_near_ex_dividend("2024-03-15", []) is False


# ── Corporate action flag ─────────────────────────────────────────────────────

class TestCorporateActionFlag:
    def test_no_events_returns_none_flag(self):
        flag = get_corporate_action_flag("2024-03-15", [])
        assert flag == CorporateActionFlag.NONE

    def test_split_takes_priority_over_dividend(self):
        events = [
            {"type": "DIVIDEND", "date": "2024-03-15"},
            {"type": "SPLIT", "date": "2024-03-15"},
        ]
        flag = get_corporate_action_flag("2024-03-15", events)
        assert flag == CorporateActionFlag.SPLIT

    def test_rights_before_bonus(self):
        events = [
            {"type": "BONUS", "date": "2024-03-15"},
            {"type": "RIGHTS", "date": "2024-03-15"},
        ]
        flag = get_corporate_action_flag("2024-03-15", events)
        assert flag == CorporateActionFlag.RIGHTS


# ── apply_all_adjustments ─────────────────────────────────────────────────────

class TestApplyAll:
    def test_combined_split_and_dividend(self):
        rows = _rows()
        events = [
            {"type": "SPLIT", "date": "2024-01-08", "value": 2.0},
            {"type": "DIVIDEND", "date": "2024-01-08", "value": 5.0},
        ]
        adjusted = apply_all_adjustments(rows, events)
        # Split first (rows before 2024-01-08 get /2), then dividend (rows before ex-date)
        # Row 2024-01-01: 500 / 2 = 250, then 250 - 5 = 245
        assert adjusted[0]["close"] == pytest.approx(245.0)

    def test_no_events_returns_unchanged(self):
        rows = _rows()
        adjusted = apply_all_adjustments(rows, [])
        assert adjusted[0]["close"] == rows[0]["close"]

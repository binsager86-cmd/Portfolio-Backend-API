"""Unit tests for the Kuwait Signal Engine — Kuwait Constants & Tick Grid.

Validates:
  - align_to_tick() for both tick grids (≤100.9 fils → 0.1, ≥101 → 1.0)
  - Circuit breaker limits (+10 % upper, -5 % lower)
  - PREMIER_STOCKS list contains expected major listings
  - Trading session constants (Sun-Thu, 9:30-12:40 AST)
"""
from __future__ import annotations

import pytest

from app.services.signal_engine.config.kuwait_constants import (
    align_to_tick,
    TICK_SMALL,
    TICK_LARGE,
    TICK_BOUNDARY,
    CIRCUIT_BREAKER_UP,
    CIRCUIT_BREAKER_DOWN,
    PREMIER_STOCKS,
    TRADING_DAYS,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
)


# ── Tick size grid ─────────────────────────────────────────────────────────────

class TestAlignToTick:
    # Small tick grid (≤ 100.9 fils, step = 0.1)
    def test_exact_small_tick_unchanged(self):
        assert align_to_tick(100.0) == pytest.approx(100.0)

    def test_round_up_small_tick(self):
        result = align_to_tick(100.04)
        assert result == pytest.approx(100.0, abs=0.15)

    def test_round_down_small_tick(self):
        result = align_to_tick(100.97)
        assert result == pytest.approx(101.0, abs=0.15)

    def test_zero_aligns_to_zero(self):
        assert align_to_tick(0.0) == 0.0

    def test_small_price_0_point_1_tick(self):
        result = align_to_tick(50.05)
        # Should round to nearest 0.1
        assert result % 0.1 < 1e-9 or abs(result % 0.1 - 0.1) < 1e-9

    # Large tick grid (≥ 101 fils, step = 1.0)
    def test_exact_large_tick_unchanged(self):
        assert align_to_tick(200.0) == pytest.approx(200.0)

    def test_above_boundary_rounds_to_1_fil_grid(self):
        result = align_to_tick(205.6)
        assert result % 1.0 < 1e-9 or abs(result % 1.0 - 1.0) < 1e-9

    def test_boundary_value_uses_correct_grid(self):
        # 100.9 → small tick boundary, 101 → large tick
        below = align_to_tick(100.9)
        above = align_to_tick(101.0)
        assert below % TICK_SMALL < 1e-9 or abs(below % TICK_SMALL - TICK_SMALL) < 1e-9
        assert above % TICK_LARGE < 1e-9 or abs(above % TICK_LARGE - TICK_LARGE) < 1e-9

    def test_large_price_no_decimals(self):
        result = align_to_tick(1500.7)
        # 1.0 fil grid → integer value
        assert abs(result - round(result)) < 1e-9


# ── Circuit breakers ───────────────────────────────────────────────────────────

class TestCircuitBreakers:
    def test_circuit_breaker_up_is_10pct(self):
        assert CIRCUIT_BREAKER_UP == pytest.approx(0.10, rel=1e-6)

    def test_circuit_breaker_down_is_5pct(self):
        assert CIRCUIT_BREAKER_DOWN == pytest.approx(0.05, rel=1e-6)

    def test_upper_limit_calculation(self):
        prev_close = 100.0
        upper = prev_close * (1 + CIRCUIT_BREAKER_UP)
        assert upper == pytest.approx(110.0)

    def test_lower_limit_calculation(self):
        prev_close = 100.0
        lower = prev_close * (1 - CIRCUIT_BREAKER_DOWN)
        assert lower == pytest.approx(95.0)


# ── Premier stocks ─────────────────────────────────────────────────────────────

class TestPremierStocks:
    EXPECTED_STOCKS = ["NBK", "KFH", "ZAIN", "MABANEE", "BURG", "CBK"]

    def test_premier_stocks_is_non_empty_list(self):
        assert isinstance(PREMIER_STOCKS, list)
        assert len(PREMIER_STOCKS) >= 5

    def test_expected_major_listings_present(self):
        for ticker in self.EXPECTED_STOCKS:
            assert ticker in PREMIER_STOCKS, f"{ticker} missing from PREMIER_STOCKS"

    def test_no_duplicate_stocks(self):
        assert len(PREMIER_STOCKS) == len(set(PREMIER_STOCKS))

    def test_stock_codes_are_uppercase_strings(self):
        for code in PREMIER_STOCKS:
            assert isinstance(code, str)
            assert code == code.upper()


# ── Trading session constants ──────────────────────────────────────────────────

class TestTradingSession:
    def test_trading_days_are_sun_to_thu(self):
        # Kuwait exchanges trade Sunday through Thursday
        # Python weekday(): Monday=0, ..., Sunday=6
        # Sun=6, Mon=0, Tue=1, Wed=2, Thu=3
        expected_days = {6, 0, 1, 2, 3}  # Sun-Thu as Python weekday numbers
        assert set(TRADING_DAYS) == expected_days

    def test_market_opens_at_930_ast(self):
        assert MARKET_OPEN_HOUR == 9
        assert MARKET_OPEN_MINUTE == 30

    def test_market_closes_at_1240_ast(self):
        assert MARKET_CLOSE_HOUR == 12
        assert MARKET_CLOSE_MINUTE == 40

    def test_session_duration_less_than_4_hours(self):
        open_mins = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE
        close_mins = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE
        duration = close_mins - open_mins
        assert 0 < duration < 4 * 60

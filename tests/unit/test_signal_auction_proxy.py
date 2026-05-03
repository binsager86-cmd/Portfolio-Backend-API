"""Unit tests for the Kuwait Signal Engine — Auction Proxy.

Validates:
  - Intensity value range [0, 2]
  - Confidence adjustments at boundary thresholds (1.0 and 1.8)
  - Edge cases: zero range, missing data
"""
from __future__ import annotations

import pytest

from app.services.signal_engine.processors.auction_proxy import (
    calculate_auction_intensity,
    auction_confidence_adjustment,
)


# ── Intensity calculation ──────────────────────────────────────────────────────

class TestAuctionIntensity:
    def test_close_at_high_gives_intensity_2(self):
        rows = [{"high": 110.0, "low": 100.0, "close": 110.0}]
        assert calculate_auction_intensity(rows) == pytest.approx(2.0)

    def test_close_at_low_gives_intensity_0(self):
        rows = [{"high": 110.0, "low": 100.0, "close": 100.0}]
        assert calculate_auction_intensity(rows) == pytest.approx(0.0)

    def test_close_at_midrange_gives_intensity_1(self):
        rows = [{"high": 110.0, "low": 100.0, "close": 105.0}]
        intensity = calculate_auction_intensity(rows)
        assert intensity == pytest.approx(1.0, abs=0.01)

    def test_empty_rows_returns_neutral(self):
        assert calculate_auction_intensity([]) == 1.0

    def test_zero_range_returns_neutral(self):
        rows = [{"high": 100.0, "low": 100.0, "close": 100.0}]
        assert calculate_auction_intensity(rows) == 1.0

    def test_intensity_non_negative(self):
        """Intensity must always be ≥ 0 (close below low is clamped to 0)."""
        row_below_low = {"high": 200.0, "low": 100.0, "close": 50.0}
        v = calculate_auction_intensity([row_below_low])
        assert v >= 0.0

    def test_uses_last_row_only(self):
        rows = [
            {"high": 110.0, "low": 100.0, "close": 100.0},  # intensity = 0
            {"high": 110.0, "low": 100.0, "close": 110.0},  # intensity = 2
        ]
        assert calculate_auction_intensity(rows) == pytest.approx(2.0)


# ── Confidence adjustments at boundary thresholds ─────────────────────────────

class TestConfidenceAdjustment:
    # auction_confidence_adjustment returns a MULTIPLIER:
    #   intensity < 1.0  → 0.80 (20 % penalty)
    #   intensity 1.0-1.8 → 1.00 (no change)
    #   intensity > 1.8  → 1.15 (+15 % boost)

    def test_intensity_below_1_returns_penalty_multiplier(self):
        adj = auction_confidence_adjustment(0.5)
        assert adj < 1.0, f"Expected multiplier < 1.0 for low intensity, got {adj}"
        assert adj == pytest.approx(0.80, abs=0.01)

    def test_intensity_exactly_1_returns_neutral_multiplier(self):
        adj = auction_confidence_adjustment(1.0)
        assert adj == pytest.approx(1.00, abs=0.01)

    def test_intensity_between_1_and_1_8_returns_neutral(self):
        for v in [1.0, 1.2, 1.5, 1.79]:
            adj = auction_confidence_adjustment(v)
            assert adj == pytest.approx(1.00, abs=0.01), f"Unexpected multiplier at intensity {v}: {adj}"

    def test_intensity_exactly_1_8_boundary(self):
        # At exactly 1.8 should still be in the normal range (multiplier = 1.0)
        adj = auction_confidence_adjustment(1.8)
        assert adj == pytest.approx(1.00, abs=0.01)

    def test_intensity_above_1_8_returns_boost_multiplier(self):
        adj = auction_confidence_adjustment(2.0)
        assert adj > 1.0, f"Expected multiplier > 1.0 above 1.8, got {adj}"
        assert adj == pytest.approx(1.15, abs=0.01)

    def test_high_institutional_boost_is_15pct(self):
        """Spec: +15 % confidence boost for high institutional activity."""
        adj = auction_confidence_adjustment(2.0)
        assert adj == pytest.approx(1.15, abs=0.01)

    def test_low_participation_penalty_is_20pct(self):
        """Spec: –20 % confidence for low institutional participation."""
        adj = auction_confidence_adjustment(0.0)
        assert adj == pytest.approx(0.80, abs=0.01)

    def test_neutral_rows_give_neutral_multiplier(self):
        """End-to-end: mid-range close → intensity ≈ 1.0 → multiplier = 1.0."""
        rows = [{"high": 110.0, "low": 100.0, "close": 105.0}]
        intensity = calculate_auction_intensity(rows)
        adj = auction_confidence_adjustment(intensity)
        assert adj == pytest.approx(1.00, abs=0.01)

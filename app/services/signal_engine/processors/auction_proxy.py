"""Auction-phase volume proxy for the Kuwait Signal Engine.

The Boursa Kuwait closes with a continuous auction from 12:30–12:40 AST.
Tick-level auction data is not available via the TickerChart daily feed, so
we approximate participation intensity using price and volume structure.
"""
from __future__ import annotations

from typing import Any

from app.services.signal_engine.config.kuwait_constants import (
    AUCTION_INTENSITY_HIGH_CONFIDENCE_BOOST,
    AUCTION_INTENSITY_HIGH_THRESHOLD,
    AUCTION_INTENSITY_LOW_CONFIDENCE_REDUCTION,
    AUCTION_INTENSITY_LOW_THRESHOLD,
    ESTIMATED_AUCTION_VOLUME_PCT,
)


def calculate_auction_intensity(rows: list[dict[str, Any]]) -> float:
    """Estimate the auction-phase participation intensity for the latest bar.

    Formula (proxy when intraday auction data is unavailable):

        closing_bias = (Close - Low) / (High - Low)      [0-1, 1 = very bullish close]
        vol_estimate = closing_bias * Daily_Volume * ESTIMATED_AUCTION_VOLUME_PCT
        avg_30min_vol = Daily_Volume * ESTIMATED_AUCTION_VOLUME_PCT  (neutral baseline)

        Intensity = vol_estimate / avg_30min_vol = closing_bias

    The ratio is then scaled relative to 0.5 (neutral mid-point) so that a
    perfectly centred close = 1.0, a top-of-range close > 1.0, and a
    bottom-of-range close < 1.0.

    Intensity < 1.0  → low institutional participation   → –20 % confidence
    Intensity 1.0-1.8 → normal range                      → no adjustment
    Intensity > 1.8  → high institutional activity        → +15 % confidence

    Args:
        rows: OHLCV rows sorted ascending by date (need ≥ 1 bar).

    Returns:
        Float auction intensity value.  Returns 1.0 (neutral) when data is
        insufficient or the high-low range is zero.
    """
    if not rows:
        return 1.0

    last = rows[-1]
    h = float(last.get("high") or 0.0)
    l = float(last.get("low") or 0.0)
    c = float(last.get("close") or 0.0)

    hl_range = h - l
    if hl_range <= 0:
        return 1.0

    closing_bias = (c - l) / hl_range          # 0 = closed at low, 1 = at high

    # Normalise: neutral mid-range close (bias = 0.5) → intensity 1.0
    # Top close (bias = 1.0) → intensity 2.0, bottom (bias = 0) → intensity 0
    intensity = closing_bias * 2.0
    return round(max(0.0, intensity), 3)


def auction_confidence_adjustment(intensity: float) -> float:
    """Return a multiplicative confidence adjustment factor based on intensity.

    Returns:
        Multiplier in [1-LOW_REDUCTION, 1+HIGH_BOOST].
    """
    if intensity < AUCTION_INTENSITY_LOW_THRESHOLD:
        return round(1.0 - AUCTION_INTENSITY_LOW_CONFIDENCE_REDUCTION, 3)
    if intensity > AUCTION_INTENSITY_HIGH_THRESHOLD:
        return round(1.0 + AUCTION_INTENSITY_HIGH_CONFIDENCE_BOOST, 3)
    return 1.0

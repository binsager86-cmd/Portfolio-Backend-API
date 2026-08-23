"""Auction analysis boundary for the Kuwait Signal Engine.

Daily OHLCV does not contain intraday timestamps or auction volume. Returning
``None`` makes that limitation explicit and prevents fabricated confidence.
"""
from __future__ import annotations

from typing import Any

from app.services.signal_engine.config.kuwait_constants import (
    AUCTION_INTENSITY_HIGH_CONFIDENCE_BOOST,
    AUCTION_INTENSITY_HIGH_THRESHOLD,
    AUCTION_INTENSITY_LOW_CONFIDENCE_REDUCTION,
    AUCTION_INTENSITY_LOW_THRESHOLD,
)


def calculate_auction_intensity(rows: list[dict[str, Any]]) -> None:
    """Return no auction metric because daily bars cannot identify auction flow."""
    return None


def auction_confidence_adjustment(intensity: float | None) -> float:
    """Return neutral confidence when auction data is unavailable."""
    if intensity is None:
        return 1.0
    if intensity < AUCTION_INTENSITY_LOW_THRESHOLD:
        return round(1.0 - AUCTION_INTENSITY_LOW_CONFIDENCE_REDUCTION, 3)
    if intensity > AUCTION_INTENSITY_HIGH_THRESHOLD:
        return round(1.0 + AUCTION_INTENSITY_HIGH_CONFIDENCE_BOOST, 3)
    return 1.0

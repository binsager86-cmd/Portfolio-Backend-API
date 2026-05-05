"""Unit tests for OrderBookImbalance processor.

Covers:
  - Strong bid imbalance detection (ratio > 0.3)
  - Liquidity wall detection (bid side and ask side)
  - Neutral imbalance when bid ≈ ask volume
  - Graceful fallback when API client is None
  - compute_imbalance_ratio description string content
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.services.signal_engine.processors.orderbook_imbalance import OrderBookImbalance


# ── Snapshot factories ─────────────────────────────────────────────────────────

def _make_snapshot(
    bid_volumes: list[int],
    ask_volumes: list[int],
    base_bid_price: float = 1.230,
    base_ask_price: float = 1.236,
) -> dict[str, Any]:
    """Build a minimal order book snapshot dict."""
    bids = [
        {"price": round(base_bid_price - i * 0.001, 4), "volume": v, "orders": 1}
        for i, v in enumerate(bid_volumes)
    ]
    asks = [
        {"price": round(base_ask_price + i * 0.001, 4), "volume": v, "orders": 1}
        for i, v in enumerate(ask_volumes)
    ]
    return {"symbol": "NBK", "bids": bids, "asks": asks}


# ── Tests: imbalance calculation ───────────────────────────────────────────────

class TestImbalanceRatio:
    def _ob(self, segment: str = "Premier") -> OrderBookImbalance:
        return OrderBookImbalance("NBK", segment, api_client=None)

    def test_imbalance_strong_bid(self):
        """Heavily bid-side book should produce imbalance_ratio > 0.3."""
        snapshot = _make_snapshot(
            bid_volumes=[40_000, 30_000, 25_000, 20_000, 18_000],
            ask_volumes=[5_000,  4_000,  3_000,  2_000,  1_000],
        )
        ob = self._ob()
        result = ob.compute_imbalance_ratio(snapshot)

        assert result["imbalance_ratio"] > 0.3, (
            f"Expected imbalance_ratio > 0.3, got {result['imbalance_ratio']}"
        )
        assert result["bid_pressure"] > 0.5
        assert result["ask_pressure"] < 0.5
        assert "strong_bid" in result["description"]

    def test_imbalance_strong_ask(self):
        """Heavily ask-side book should produce imbalance_ratio < -0.3."""
        snapshot = _make_snapshot(
            bid_volumes=[2_000, 1_500, 1_200, 1_000, 800],
            ask_volumes=[40_000, 35_000, 28_000, 20_000, 15_000],
        )
        ob = self._ob()
        result = ob.compute_imbalance_ratio(snapshot)

        assert result["imbalance_ratio"] < -0.3
        assert "ask_imbalance" in result["description"]

    def test_balanced_book_near_zero(self):
        """Equal bid/ask volumes should produce imbalance_ratio ≈ 0."""
        vols = [10_000, 9_000, 8_000, 7_000, 6_000]
        snapshot = _make_snapshot(bid_volumes=vols, ask_volumes=vols)
        ob = self._ob()
        result = ob.compute_imbalance_ratio(snapshot)

        assert abs(result["imbalance_ratio"]) < 0.05
        assert "balanced" in result["description"]

    def test_ratio_bounded_negative_one_to_one(self):
        """Imbalance ratio must always stay in [-1, +1]."""
        for bid_vols, ask_vols in [
            ([100_000, 100_000], [1, 1]),
            ([1, 1], [100_000, 100_000]),
        ]:
            snapshot = _make_snapshot(bid_volumes=bid_vols, ask_volumes=ask_vols)
            ob = self._ob()
            result = ob.compute_imbalance_ratio(snapshot)
            assert -1.0 <= result["imbalance_ratio"] <= 1.0

    def test_empty_bids_returns_neutral(self):
        """Empty bids should return the neutral fallback without raising."""
        snapshot = {"symbol": "NBK", "bids": [], "asks": [{"price": 1.0, "volume": 100}]}
        ob = self._ob()
        result = ob.compute_imbalance_ratio(snapshot)
        assert result["imbalance_ratio"] == 0.0

    def test_zero_volume_returns_neutral(self):
        """All-zero volumes should return neutral imbalance."""
        snapshot = _make_snapshot(bid_volumes=[0, 0, 0], ask_volumes=[0, 0, 0])
        ob = self._ob()
        result = ob.compute_imbalance_ratio(snapshot)
        assert result["imbalance_ratio"] == 0.0


# ── Tests: liquidity wall detection ───────────────────────────────────────────

class TestLiquidityWall:
    def _ob(self) -> OrderBookImbalance:
        return OrderBookImbalance("NBK", "Premier", api_client=None)

    def test_bid_wall_detected(self):
        """A single very large bid level should be flagged as a liquidity wall."""
        # First bid level has 5x+ average volume → wall
        snapshot = _make_snapshot(
            bid_volumes=[50_000, 3_000, 2_500, 2_000, 1_500],
            ask_volumes=[4_000,  3_800, 3_600, 3_400, 3_200],
        )
        ob = self._ob()
        result = ob.compute_imbalance_ratio(snapshot)

        assert result["liquidity_wall"] is not None, "Expected a bid liquidity wall"
        wall = result["liquidity_wall"]
        assert wall["side"] == "bid"
        assert wall["volume"] == 50_000
        assert wall["strength"] in {"strong", "moderate"}

    def test_ask_wall_detected(self):
        """A single very large ask level should be flagged as a liquidity wall."""
        snapshot = _make_snapshot(
            bid_volumes=[4_000, 3_800, 3_600, 3_400, 3_200],
            ask_volumes=[60_000, 3_000, 2_500, 2_000, 1_500],
        )
        ob = self._ob()
        result = ob.compute_imbalance_ratio(snapshot)

        assert result["liquidity_wall"] is not None, "Expected an ask liquidity wall"
        assert result["liquidity_wall"]["side"] == "ask"

    def test_no_wall_when_volumes_even(self):
        """Uniform volume distribution should not produce a wall."""
        vols = [10_000] * 5
        snapshot = _make_snapshot(bid_volumes=vols, ask_volumes=vols)
        ob = self._ob()
        result = ob.compute_imbalance_ratio(snapshot)

        assert result["liquidity_wall"] is None

    def test_wall_contains_required_keys(self):
        """Wall dict must have side, price, volume, strength."""
        snapshot = _make_snapshot(
            bid_volumes=[80_000, 2_000, 1_500, 1_000, 500],
            ask_volumes=[2_000,  1_800, 1_600, 1_400, 1_200],
        )
        ob = self._ob()
        result = ob.compute_imbalance_ratio(snapshot)

        wall = result["liquidity_wall"]
        assert wall is not None
        for key in ("side", "price", "volume", "strength"):
            assert key in wall, f"Missing key '{key}' in liquidity_wall dict"


# ── Tests: API client integration ─────────────────────────────────────────────

class TestFetchSnapshot:
    def test_fetch_returns_none_when_no_client(self):
        """Without an API client, fetch_snapshot should return None gracefully."""
        import asyncio
        ob = OrderBookImbalance("NBK", "Premier", api_client=None)
        result = asyncio.run(ob.fetch_snapshot())
        assert result is None

    def test_fetch_uses_api_client(self):
        """With a mock API client, fetch_snapshot should call get_order_book."""
        import asyncio
        mock_snapshot = {
            "symbol": "NBK",
            "bids": [{"price": 1.230, "volume": 15_000}],
            "asks": [{"price": 1.236, "volume": 8_500}],
        }
        mock_client = AsyncMock()
        mock_client.get_order_book.return_value = mock_snapshot

        ob = OrderBookImbalance("NBK", "Premier", api_client=mock_client)
        result = asyncio.run(ob.fetch_snapshot())

        assert result == mock_snapshot
        mock_client.get_order_book.assert_called_once_with(symbol="NBK", depth=20)

    def test_fetch_returns_none_on_exception(self):
        """API client errors should be swallowed and return None."""
        import asyncio
        mock_client = AsyncMock()
        mock_client.get_order_book.side_effect = ConnectionError("feed down")

        ob = OrderBookImbalance("NBK", "Premier", api_client=mock_client)
        result = asyncio.run(ob.fetch_snapshot())

        assert result is None

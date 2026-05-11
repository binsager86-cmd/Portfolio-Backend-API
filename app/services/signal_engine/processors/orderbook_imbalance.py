"""Order-book imbalance proxy utilities."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class OrderBookImbalance:
    """Compute bid/ask pressure metrics from a level-2 snapshot."""

    def __init__(self, symbol: str, segment: str, api_client: Any | None = None) -> None:
        self.symbol = str(symbol or "").upper()
        self.segment = str(segment or "PREMIER").upper()
        self.api_client = api_client

    async def fetch_snapshot(self) -> dict[str, Any] | None:
        """Fetch order book from client when available."""
        if self.api_client is None:
            return None
        try:
            return await self.api_client.get_order_book(symbol=self.symbol, depth=20)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Order book fetch failed for %s: %s", self.symbol, exc)
            return None

    def compute_imbalance_ratio(self, snapshot: dict[str, Any] | None) -> dict[str, Any]:
        """Return bounded imbalance stats in [-1, +1]."""
        if not snapshot:
            return self._neutral("no_snapshot")

        bids = list(snapshot.get("bids") or [])
        asks = list(snapshot.get("asks") or [])
        if not bids or not asks:
            return self._neutral("insufficient_levels")

        bid_total = float(sum(float(level.get("volume") or 0.0) for level in bids))
        ask_total = float(sum(float(level.get("volume") or 0.0) for level in asks))
        gross = bid_total + ask_total
        if gross <= 0:
            return self._neutral("zero_volume")

        ratio = (bid_total - ask_total) / gross
        ratio = max(-1.0, min(1.0, ratio))
        bid_pressure = bid_total / gross
        ask_pressure = ask_total / gross

        wall = self._detect_wall(bids, asks)
        if ratio > 0.3:
            desc = "strong_bid_imbalance"
        elif ratio < -0.3:
            desc = "strong_ask_imbalance"
        else:
            desc = "balanced_book"

        return {
            "imbalance_ratio": round(ratio, 4),
            "bid_pressure": round(bid_pressure, 4),
            "ask_pressure": round(ask_pressure, 4),
            "liquidity_wall": wall,
            "description": desc,
        }

    def _detect_wall(
        self,
        bids: list[dict[str, Any]],
        asks: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        bid_vols = [
            float(level.get("volume") or 0.0)
            for level in bids
            if float(level.get("volume") or 0.0) > 0
        ]
        ask_vols = [
            float(level.get("volume") or 0.0)
            for level in asks
            if float(level.get("volume") or 0.0) > 0
        ]
        if not bid_vols or not ask_vols:
            return None

        bid_max = max(bid_vols)
        ask_max = max(ask_vols)
        bid_avg = sum(bid_vols) / len(bid_vols)
        ask_avg = sum(ask_vols) / len(ask_vols)
        wall_threshold = 3.0

        if bid_avg > 0 and bid_max >= bid_avg * wall_threshold:
            level = max(bids, key=lambda x: float(x.get("volume") or 0.0))
            return {
                "side": "bid",
                "price": level.get("price"),
                "volume": int(float(level.get("volume") or 0.0)),
                "strength": "strong" if bid_max >= bid_avg * 5 else "moderate",
            }

        if ask_avg > 0 and ask_max >= ask_avg * wall_threshold:
            level = max(asks, key=lambda x: float(x.get("volume") or 0.0))
            return {
                "side": "ask",
                "price": level.get("price"),
                "volume": int(float(level.get("volume") or 0.0)),
                "strength": "strong" if ask_max >= ask_avg * 5 else "moderate",
            }

        return None

    @staticmethod
    def _neutral(reason: str) -> dict[str, Any]:
        return {
            "imbalance_ratio": 0.0,
            "bid_pressure": 0.5,
            "ask_pressure": 0.5,
            "liquidity_wall": None,
            "description": reason,
        }

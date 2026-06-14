from __future__ import annotations

from typing import Any


class OrderBookImbalance:
    """Compute bid/ask pressure from a level-2 order book snapshot."""

    def __init__(self, symbol: str, segment: str, api_client: Any | None = None) -> None:
        self.symbol = str(symbol or "").upper()
        self.segment = str(segment or "").upper()
        self.api_client = api_client

    async def fetch_snapshot(self) -> dict[str, Any] | None:
        """Fetch order book snapshot using the configured API client."""
        if self.api_client is None:
            return None
        try:
            return await self.api_client.get_order_book(symbol=self.symbol, depth=20)
        except Exception:  # noqa: BLE001
            return None

    def _detect_wall(self, levels: list[dict[str, Any]], side: str) -> dict[str, Any] | None:
        if len(levels) < 2:
            return None

        vols = [float(l.get("volume") or 0.0) for l in levels]
        if not vols or max(vols) <= 0:
            return None

        max_idx = int(max(range(len(vols)), key=lambda i: vols[i]))
        max_vol = vols[max_idx]
        baseline = sum(vols) / len(vols)
        if baseline <= 0:
            return None

        ratio = max_vol / baseline
        if ratio < 3.0:
            return None

        strength = "strong" if ratio >= 4.5 else "moderate"
        level = levels[max_idx]
        return {
            "side": side,
            "price": level.get("price"),
            "volume": int(max_vol),
            "strength": strength,
        }

    def compute_imbalance_ratio(self, snapshot: dict[str, Any] | None) -> dict[str, Any]:
        """Return normalized imbalance ratio and optional liquidity wall details."""
        bids = (snapshot or {}).get("bids") or []
        asks = (snapshot or {}).get("asks") or []

        if not bids or not asks:
            return {
                "imbalance_ratio": 0.0,
                "bid_pressure": 0.5,
                "ask_pressure": 0.5,
                "liquidity_wall": None,
                "description": "balanced_missing_side",
            }

        bid_vol = float(sum(float(b.get("volume") or 0.0) for b in bids))
        ask_vol = float(sum(float(a.get("volume") or 0.0) for a in asks))
        total = bid_vol + ask_vol

        if total <= 0:
            return {
                "imbalance_ratio": 0.0,
                "bid_pressure": 0.5,
                "ask_pressure": 0.5,
                "liquidity_wall": None,
                "description": "balanced_no_volume",
            }

        imbalance = (bid_vol - ask_vol) / total
        imbalance = max(-1.0, min(1.0, imbalance))
        bid_pressure = bid_vol / total
        ask_pressure = ask_vol / total

        bid_wall = self._detect_wall(bids, "bid")
        ask_wall = self._detect_wall(asks, "ask")
        liquidity_wall = None
        if bid_wall and ask_wall:
            liquidity_wall = bid_wall if bid_wall["volume"] >= ask_wall["volume"] else ask_wall
        elif bid_wall:
            liquidity_wall = bid_wall
        elif ask_wall:
            liquidity_wall = ask_wall

        if imbalance > 0.3:
            description = "strong_bid_imbalance"
        elif imbalance < -0.3:
            description = "strong_ask_imbalance"
        else:
            description = "balanced_orderbook"

        return {
            "imbalance_ratio": round(float(imbalance), 4),
            "bid_pressure": round(float(bid_pressure), 4),
            "ask_pressure": round(float(ask_pressure), 4),
            "liquidity_wall": liquidity_wall,
            "description": description,
        }

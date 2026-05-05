"""Order Book Imbalance processor for Kuwait Signal Engine.

Integrates with order book API to compute:
- Bid/Ask volume imbalance ratio
- Liquidity wall detection (large orders at specific price levels)
- Real-time auction intensity proxy (tighter spread + higher concentration)

Fallback: If order book unavailable, returns neutral values without blocking signal.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class OrderBookImbalance:
    """Compute imbalance metrics from order book snapshot."""
    
    def __init__(
        self,
        symbol: str,
        market_segment: str,
        api_client: Any | None = None,
    ):
        """Initialize order book analyzer.
        
        Args:
            symbol: Stock ticker (e.g. "NBK").
            market_segment: "PREMIER" or "MAIN".
            api_client: Order book API client instance (must implement get_order_book method).
        """
        self.symbol = symbol
        self.market_segment = market_segment.upper()
        self.api_client = api_client
    
    async def fetch_snapshot(self) -> dict[str, Any] | None:
        """Fetch real-time order book from API.
        
        Expected response format:
        {
            "symbol": "NBK",
            "timestamp": "2026-05-04T10:30:00Z",
            "bids": [{"price": 750.0, "volume": 12000}, ...],  # Sorted descending
            "asks": [{"price": 751.0, "volume": 8000}, ...],   # Sorted ascending
        }
        
        Returns:
            Order book snapshot dict or None if unavailable.
        """
        if self.api_client is None:
            logger.debug(f"Order book API client not configured for {self.symbol}")
            return None
        
        try:
            snapshot = await self.api_client.get_order_book(
                symbol=self.symbol,
                depth=20,  # Top 20 price levels each side
            )
            
            if not snapshot or not snapshot.get("bids") or not snapshot.get("asks"):
                logger.warning(f"Empty order book received for {self.symbol}")
                return None
            
            return snapshot
            
        except Exception as e:
            logger.warning(f"Order book fetch failed for {self.symbol}: {e}")
            return None
    
    def compute_imbalance_ratio(
        self,
        snapshot: dict[str, Any],
        lookback_levels: int = 5,
    ) -> dict[str, Any]:
        """Compute bid/ask imbalance from order book snapshot.
        
        Args:
            snapshot: Order book snapshot from fetch_snapshot().
            lookback_levels: Number of price levels to analyze (default 5 = top of book).
        
        Returns:
            {
                "imbalance_ratio": float,     # (bid_vol - ask_vol) / total_vol ∈ [-1, +1]
                "bid_pressure": float,        # bid_vol / total_vol ∈ [0, 1]
                "ask_pressure": float,        # ask_vol / total_vol ∈ [0, 1]
                "liquidity_wall": dict | None,  # {side, price, volume, strength} if detected
                "description": str
            }
        """
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])
        
        if not bids or not asks:
            return self._neutral_imbalance("empty_orderbook")
        
        # Sum volume over top N levels
        bid_vol = sum(float(b.get("volume", 0)) for b in bids[:lookback_levels])
        ask_vol = sum(float(a.get("volume", 0)) for a in asks[:lookback_levels])
        
        total_vol = bid_vol + ask_vol
        
        if total_vol < 1.0:
            return self._neutral_imbalance("zero_volume")
        
        # Imbalance ratio: +1.0 = all bids, -1.0 = all asks
        imbalance = (bid_vol - ask_vol) / total_vol
        
        # Normalize pressures to 0-1 scale
        bid_pressure = bid_vol / total_vol
        ask_pressure = ask_vol / total_vol
        
        # Detect liquidity walls (orders > 3x average level volume)
        liquidity_wall = self._detect_wall(bids, asks, bid_vol, ask_vol, lookback_levels)
        
        # Kuwait-specific interpretation
        if imbalance > 0.30:
            desc = f"strong_bid_imbalance_{imbalance:.2f}"
        elif imbalance > 0.10:
            desc = f"moderate_bid_imbalance_{imbalance:.2f}"
        elif imbalance > -0.10:
            desc = f"balanced_orderbook_{imbalance:.2f}"
        elif imbalance > -0.30:
            desc = f"moderate_ask_imbalance_{imbalance:.2f}"
        else:
            desc = f"strong_ask_imbalance_{imbalance:.2f}"
        
        return {
            "imbalance_ratio": round(imbalance, 3),
            "bid_pressure": round(bid_pressure, 3),
            "ask_pressure": round(ask_pressure, 3),
            "liquidity_wall": liquidity_wall,
            "description": desc,
        }
    
    def _neutral_imbalance(self, reason: str) -> dict[str, Any]:
        """Return neutral imbalance metrics when data unavailable."""
        return {
            "imbalance_ratio": 0.0,
            "bid_pressure": 0.5,
            "ask_pressure": 0.5,
            "liquidity_wall": None,
            "description": reason,
        }
    
    def _detect_wall(
        self,
        bids: list[dict],
        asks: list[dict],
        total_bid_vol: float,
        total_ask_vol: float,
        n_levels: int,
        threshold_multiplier: float = 3.0,
    ) -> dict[str, Any] | None:
        """Detect large liquidity walls at specific price levels.
        
        A "wall" = single order >= 3x the average volume per level.
        
        Returns:
            {
                "side": "bid" | "ask",
                "price": float,
                "volume": float,
                "strength": "strong" | "moderate"
            }
        """
        avg_bid_vol = total_bid_vol / n_levels if n_levels > 0 else 0
        avg_ask_vol = total_ask_vol / n_levels if n_levels > 0 else 0
        
        # Check bid side for walls (large buy orders = support)
        for bid in bids[:n_levels]:
            vol = float(bid.get("volume", 0))
            if vol > avg_bid_vol * threshold_multiplier and avg_bid_vol > 0:
                strength = "strong" if vol > avg_bid_vol * 5.0 else "moderate"
                return {
                    "side": "bid",
                    "price": round(float(bid.get("price", 0)), 2),
                    "volume": int(vol),
                    "strength": strength,
                }
        
        # Check ask side for walls (large sell orders = resistance)
        for ask in asks[:n_levels]:
            vol = float(ask.get("volume", 0))
            if vol > avg_ask_vol * threshold_multiplier and avg_ask_vol > 0:
                strength = "strong" if vol > avg_ask_vol * 5.0 else "moderate"
                return {
                    "side": "ask",
                    "price": round(float(ask.get("price", 0)), 2),
                    "volume": int(vol),
                    "strength": strength,
                }
        
        return None
    
    def compute_auction_intensity_proxy(
        self,
        snapshot: dict[str, Any],
        historical_avg_spread_pct: float = 0.005,
    ) -> float:
        """Compute real-time auction intensity from order book tightness.
        
        Higher intensity = tighter spread + higher volume concentration = more institutional flow.
        
        Formula:
            intensity = (1 / normalized_spread) * volume_concentration
        
        Args:
            snapshot: Order book snapshot.
            historical_avg_spread_pct: Typical spread as % of mid-price (default 0.5%).
        
        Returns:
            Auction intensity ∈ [0.5, 2.5], where:
            - < 1.0 = Low liquidity / wide spread
            - ~1.0 = Normal market conditions
            - > 1.5 = High institutional activity
        """
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])
        
        if not bids or not asks:
            return 1.0  # Neutral intensity
        
        best_bid = float(bids[0].get("price", 0))
        best_ask = float(asks[0].get("price", 0))
        
        if best_bid <= 0 or best_ask <= 0:
            return 1.0
        
        # Mid-price and spread
        mid_price = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        
        # Normalized spread (as % of mid-price)
        norm_spread = spread / mid_price if mid_price > 0 else 1.0
        
        # Volume concentration: % of total volume in top 3 levels
        top_3_bid_vol = sum(float(b.get("volume", 0)) for b in bids[:3])
        top_3_ask_vol = sum(float(a.get("volume", 0)) for a in asks[:3])
        total_bid_vol = sum(float(b.get("volume", 0)) for b in bids)
        total_ask_vol = sum(float(a.get("volume", 0)) for a in asks)
        total_vol = total_bid_vol + total_ask_vol
        
        if total_vol < 1.0:
            concentration = 0.5
        else:
            concentration = (top_3_bid_vol + top_3_ask_vol) / total_vol
        
        # Intensity formula
        # Tighter spread (lower norm_spread) → higher intensity
        # Higher concentration → higher intensity
        spread_factor = 1.0 / max(norm_spread, 0.0001)
        
        # Normalize against historical baseline
        intensity_raw = spread_factor * concentration
        
        # Scale to match existing auction_proxy range [0.5, 2.5]
        # Assume historical_avg_spread_pct = 0.005 (0.5%) as baseline
        baseline_spread_factor = 1.0 / historical_avg_spread_pct
        intensity = intensity_raw / baseline_spread_factor
        
        # Clip to valid range
        intensity = float(np.clip(intensity, 0.5, 2.5))
        
        logger.debug(
            f"Auction intensity for {self.symbol}: spread={norm_spread:.4f}, "
            f"concentration={concentration:.2f}, intensity={intensity:.2f}"
        )
        
        return intensity


def compute_orderbook_score(
    imbalance_ratio: float,
) -> tuple[int, str]:
    """Convert order book imbalance ratio to a 0-15 pt score for volume_flow_score.
    
    Args:
        imbalance_ratio: Bid/ask imbalance ∈ [-1, +1].
    
    Returns:
        (points, description)
    """
    if imbalance_ratio > 0.30:
        return 15, f"strong_bid_pressure_{imbalance_ratio:.2f}"
    if imbalance_ratio > 0.10:
        return 11, f"moderate_bid_pressure_{imbalance_ratio:.2f}"
    if imbalance_ratio > -0.10:
        return 7, f"balanced_{imbalance_ratio:.2f}"
    if imbalance_ratio > -0.30:
        return 3, f"moderate_ask_pressure_{imbalance_ratio:.2f}"
    return 0, f"strong_ask_pressure_{imbalance_ratio:.2f}"

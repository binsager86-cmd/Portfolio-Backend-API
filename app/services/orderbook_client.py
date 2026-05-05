"""Order Book API Client for Kuwait Stock Exchange.

Wrapper around TickerChart service for fetching real-time order book data.
Reuses existing TickerChart authentication (username/password, TcToken).
"""
from __future__ import annotations

import logging
from typing import Any

from app.services import tickerchart_service

logger = logging.getLogger(__name__)


class OrderBookClient:
    """Async client for order book using TickerChart service."""
    
    def __init__(
        self,
        market: str = "KSE",
        timeout: float = 5.0,
        max_connections: int = 10,
    ):
        """Initialize order book client.
        
        Args:
            market: Market abbreviation (e.g. "KSE" for Kuwait, default).
            timeout: Request timeout in seconds (not used, kept for compatibility).
            max_connections: Max concurrent connections (not used, kept for compatibility).
        """
        self.market = market
        self.timeout = timeout
        logger.info(f"Order book client initialized for market: {market}")
    
    async def connect(self):
        """Initialize client (no-op for TickerChart, auth happens per-request)."""
        logger.info("Order book client ready (using TickerChart service)")
    
    async def close(self):
        """Close client (no-op for TickerChart)."""
        logger.info("Order book client closed")
    
    async def get_order_book(
        self,
        symbol: str,
        depth: int = 20,
    ) -> dict[str, Any] | None:
        """Fetch real-time order book for a symbol.
        
        Args:
            symbol: Stock ticker (e.g. "NBK", "KFH").
            depth: Number of price levels to fetch (default 20).
        
        Returns:
            {
                "symbol": "NBK",
                "market": "KSE",
                "timestamp": "2026-05-04T10:30:00Z",
                "bids": [{"price": 750.0, "volume": 12000}, ...],
                "asks": [{"price": 751.0, "volume": 8000}, ...],
                "total_bid_volume": 29010,
                "total_ask_volume": 182254,
            }
            or None if unavailable.
        """
        try:
            # Parse symbol to extract base and market
            parsed = tickerchart_service.split_symbol(
                symbol=symbol,
                exchange=self.market,
                country=None,
            )
            
            if parsed is None:
                logger.warning(f"Could not parse symbol: {symbol}")
                # Try direct fetch with market
                base_symbol = symbol.split(".")[0].upper()
                result = await tickerchart_service.fetch_order_book(
                    base_symbol=base_symbol,
                    market_abb=self.market,
                    depth=depth,
                )
                return result
            
            base_symbol, market_abb = parsed
            
            # Fetch order book from TickerChart
            result = await tickerchart_service.fetch_order_book(
                base_symbol=base_symbol,
                market_abb=market_abb,
                depth=depth,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Order book fetch failed for {symbol}: {e}", exc_info=True)
            return None
    
    async def get_historical_spread(
        self,
        symbol: str,
        lookback_days: int = 20,
    ) -> float:
        """Fetch historical average spread for a symbol.
        
        Used as baseline for auction intensity calculation.
        Currently returns default value - can be enhanced to calculate
        from historical OHLC data.
        
        Args:
            symbol: Stock ticker.
            lookback_days: Days of historical data (default 20).
        
        Returns:
            Average spread as % of mid-price (default 0.005 = 0.5%).
        """
        try:
            # Parse symbol
            parsed = tickerchart_service.split_symbol(
                symbol=symbol,
                exchange=self.market,
                country=None,
            )
            
            if parsed is None:
                return 0.005
            
            base_symbol, market_abb = parsed
            
            # Fetch recent OHLC to estimate typical spread
            from datetime import date, timedelta
            to_date = date.today()
            from_date = to_date - timedelta(days=lookback_days)
            
            ohlcv = await tickerchart_service.fetch_ohlcv(
                base_symbol=base_symbol,
                market_abb=market_abb,
                from_d=from_date,
                to_d=to_date,
                interval="day",
            )
            
            if not ohlcv:
                return 0.005
            
            # Estimate spread from high-low range
            spreads = []
            for row in ohlcv[-lookback_days:]:
                high = row.get("high", 0)
                low = row.get("low", 0)
                close = row.get("close", 0)
                
                if high > 0 and low > 0 and close > 0:
                    spread_pct = (high - low) / close
                    spreads.append(spread_pct)
            
            if spreads:
                avg_spread = sum(spreads) / len(spreads)
                # Divide by 2 to approximate bid-ask spread from daily range
                return max(0.0001, min(0.05, avg_spread / 2))
            
            return 0.005
            
        except Exception as e:
            logger.debug(f"Historical spread calculation failed for {symbol}: {e}")
            return 0.005  # Default to 0.5% spread
    
    def __repr__(self) -> str:
        return f"<OrderBookClient market={self.market} provider=TickerChart>"

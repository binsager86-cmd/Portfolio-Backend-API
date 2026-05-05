"""Kuwait Stock Exchange market constants for the signal engine.

All prices are in Kuwait fils (1 KWD = 1000 fils).
No magic numbers anywhere else — import from this module.
"""
from __future__ import annotations

# ── Tick Size Grid ────────────────────────────────────────────────────────────
TICK_GRID_LOW_MAX_FILS: float = 100.9   # prices ≤ this use 0.1-fil grid
TICK_GRID_LOW: float = 0.1              # grid step for low-price tier
TICK_GRID_HIGH: float = 1.0             # grid step for high-price tier (≥ 101 fils)


def align_to_tick(price: float) -> float:
    """Round a price to the nearest valid Kuwait tick grid value.

    Args:
        price: Raw price in fils.

    Returns:
        Price aligned to the appropriate tick grid.
    """
    if price <= TICK_GRID_LOW_MAX_FILS:
        return round(round(price / TICK_GRID_LOW) * TICK_GRID_LOW, 1)
    return round(price)


# ── Circuit Breaker Limits ────────────────────────────────────────────────────
CIRCUIT_UPPER_PCT: float = 0.10     # +10 % from previous close
CIRCUIT_LOWER_PCT: float = -0.05    # -5 % from previous close
CIRCUIT_BUFFER_PCT: float = 0.01    # reduce confidence 30 % if within this buffer

# ── Market Hours (Arabia Standard Time = UTC+3) ───────────────────────────────
MARKET_OPEN_HOUR: int = 9
MARKET_OPEN_MINUTE: int = 30
MARKET_CLOSE_HOUR: int = 12
MARKET_CLOSE_MINUTE: int = 40
# Trading days: Sunday–Thursday (Python weekday: Mon=0 … Sun=6)
TRADING_WEEKDAYS: frozenset[int] = frozenset([6, 0, 1, 2, 3])

# ── Settlement ────────────────────────────────────────────────────────────────
SETTLEMENT_T_PLUS: int = 2

# ── Premier Market Liquidity Thresholds ──────────────────────────────────────
PREMIER_ADTV_MIN_KD: float = 100_000.0        # 20-day median traded value (KD)
PREMIER_SPREAD_PROXY_MAX: float = 0.015        # (high-low)/close ≤ 1.5 %
PREMIER_ACTIVE_DAYS_MIN_PCT: float = 0.80      # ≥ 80 % of last 30 sessions
PREMIER_VOLUME_CONCENTRATION_MAX: float = 0.40 # max single day / 20-day sum

# ── Auction Phase Proxy ───────────────────────────────────────────────────────
# 15 % of daily volume assumed to occur during the closing auction window
# (12:30–12:40 AST) — based on Kuwait market microstructure studies.
ESTIMATED_AUCTION_VOLUME_PCT: float = 0.15

# Auction intensity thresholds and adjustments
AUCTION_INTENSITY_LOW_THRESHOLD: float = 1.0
AUCTION_INTENSITY_HIGH_THRESHOLD: float = 1.8
AUCTION_INTENSITY_LOW_CONFIDENCE_REDUCTION: float = 0.20   # –20 %
AUCTION_INTENSITY_HIGH_CONFIDENCE_BOOST: float = 0.15      # +15 %

# ── Corporate Action Buffer ───────────────────────────────────────────────────
EX_DIVIDEND_BUFFER_DAYS: int = 3

# ── Premier Market Universe (top ~30 by liquidity, 2025) ─────────────────────
# ── Aliases for test imports ──────────────────────────────────────────────────
TICK_SMALL: float = TICK_GRID_LOW              # 0.1 fil
TICK_LARGE: float = TICK_GRID_HIGH             # 1.0 fil
TICK_BOUNDARY: float = TICK_GRID_LOW_MAX_FILS  # 100.9 fils boundary
CIRCUIT_BREAKER_UP: float = CIRCUIT_UPPER_PCT          # 0.10
CIRCUIT_BREAKER_DOWN: float = abs(CIRCUIT_LOWER_PCT)   # 0.05 (positive fraction)
TRADING_DAYS: frozenset[int] = TRADING_WEEKDAYS        # {0,1,2,3,6}

# ── Hurst Exponent Thresholds ─────────────────────────────────────────────────
# Premier market has clearer trends (higher liquidity); Main tolerates more noise.
HURST_THRESHOLD_PREMIER: float = 0.55
HURST_THRESHOLD_MAIN: float = 0.48

# ── Banking Sector Reference ──────────────────────────────────────────────────
# Lead-lag filter: when BKB trends strongly, non-banking mid-caps tend to follow.
BANKING_INDEX_SYMBOL: str = "BKB"

# ── Order Book Integration ────────────────────────────────────────────────────
# WebSocket base URL for live order book streaming (Boursa Kuwait).
# Replace with the real endpoint before deploying to production.
ORDERBOOK_API_BASE: str = "wss://your-api.boursa-kuwait.com/v1"

# Minimum volume (shares) to classify a price level as a "liquidity wall".
LIQUIDITY_WALL_THRESHOLD_PREMIER: int = 25_000
LIQUIDITY_WALL_THRESHOLD_MAIN: int = 50_000

# ── Sunday Session Handling ───────────────────────────────────────────────────
# Kuwait Sunday open is often thinner; widen entry zones for 30 minutes.
SUNDAY_OPEN_BUFFER_MINUTES: int = 30
SUNDAY_ENTRY_BUFFER_MULTIPLIER: float = 2.0

# ── Premier Market Universe (top ~30 by liquidity, 2025) ─────────────────────
PREMIER_STOCKS: list[str] = [
    "NBK", "KFH", "CBK", "BURG", "GBK", "ABK", "AHLI",
    "ZAIN", "OMANTEL", "VIVA",
    "AGILITY", "MABANEE", "NRE", "NMDC",
    "KIPCO", "GCC", "KPI",
    "AUM", "KCBK", "WARBA",
    "ALIMTIAZ", "COAST", "GULF",
    "HUMANSOFT", "BAYAN", "IFA",
    "ALAFCO", "KSCC", "NREC", "IHC",
]

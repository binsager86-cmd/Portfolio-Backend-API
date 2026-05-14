"""
Data layer abstraction. The engine talks to DataAdapter, not to any specific API.
Phase 1 includes a SyntheticAdapter for testing.
Phase 1 next step: implement TickerChartAdapter once we inspect your existing code.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict
import pandas as pd
import numpy as np


@dataclass
class OHLCV:
    """Single bar of price/volume data."""
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    turnover_kwd: Optional[float] = None  # price * volume in KWD


@dataclass
class StockMeta:
    ticker: str
    name_en: str
    name_ar: Optional[str]
    sector: str
    sub_sector: Optional[str]
    market_tier: str  # "premier" / "main" / "auction"
    listing_date: Optional[date]
    shares_outstanding: Optional[int]
    fils_precision: int = 3


class DataAdapter(ABC):
    """Abstract base — every data source implements this interface."""

    @abstractmethod
    def list_stocks(self) -> List[StockMeta]: ...

    @abstractmethod
    def get_ohlcv_daily(
        self, ticker: str, start: date, end: date
    ) -> pd.DataFrame:
        """
        Return DataFrame indexed by date with columns:
        open, high, low, close, volume, turnover_kwd
        """
        ...

    @abstractmethod
    def get_ohlcv_weekly(self, ticker: str, start: date, end: date) -> pd.DataFrame: ...

    @abstractmethod
    def get_corporate_actions(self, ticker: str) -> pd.DataFrame: ...

    @abstractmethod
    def get_market_index(self, index_ticker: str, start: date, end: date) -> pd.DataFrame: ...


# =============================================================================
# Synthetic adapter — generates realistic-looking Kuwait-like data for testing.
# This lets us validate the entire pipeline end-to-end before plugging into
# the real ticker chart API.
# =============================================================================
class SyntheticAdapter(DataAdapter):
    """Generates synthetic OHLCV with realistic structure: trends, consolidations,
    breakouts, distributions. Used for testing the engine before real data is wired in.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._stocks = self._make_stock_universe()

    def _make_stock_universe(self) -> List[StockMeta]:
        """Realistic-looking Kuwait stocks for testing."""
        return [
            StockMeta("NBK",      "National Bank of Kuwait",  "البنك الوطني",    "Banking",          None, "premier", date(2010,1,1), 7_000_000_000),
            StockMeta("KFH",      "Kuwait Finance House",     "بيت التمويل",     "Banking",          None, "premier", date(2010,1,1), 10_000_000_000),
            StockMeta("BOUBYAN",  "Boubyan Bank",             "بنك بوبيان",       "Banking",          None, "premier", date(2010,1,1), 3_500_000_000),
            StockMeta("ZAIN",     "Zain",                     "زين",             "Telecom",          None, "premier", date(2010,1,1), 4_300_000_000),
            StockMeta("AGILITY",  "Agility Public Warehousing","أجيليتي",         "Logistics",        None, "premier", date(2010,1,1), 2_400_000_000),
            StockMeta("MABANEE",  "Mabanee",                  "المباني",         "Real Estate",      None, "premier", date(2010,1,1), 1_200_000_000),
            StockMeta("HUMANSOFT","Humansoft Holding",        "هيومن سوفت",      "Technology",       None, "premier", date(2010,1,1),   125_000_000),
            StockMeta("GBK",      "Gulf Bank",                "بنك الخليج",       "Banking",          None, "premier", date(2010,1,1), 3_100_000_000),
            StockMeta("STC",      "Kuwait Telecom (stc)",     "stc",             "Telecom",          None, "premier", date(2010,1,1),   500_000_000),
            StockMeta("ALAFCO",   "ALAFCO Aviation Lease",    "ألافكو",          "Industrial",       None, "main",    date(2010,1,1), 1_100_000_000),
        ]

    def list_stocks(self) -> List[StockMeta]:
        return self._stocks

    def get_ohlcv_daily(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Generate synthetic OHLCV with realistic regime cycles."""
        # Stable per-ticker seed so the same ticker always gets the same history
        ticker_seed = sum(ord(c) for c in ticker) * 31
        local_rng = np.random.default_rng(ticker_seed)

        # Generate ~750 daily bars (3 years), Sun-Thu trading days
        dates = pd.bdate_range(start=start, end=end, freq='C',
                               weekmask='Sun Mon Tue Wed Thu')
        n = len(dates)
        if n == 0:
            return pd.DataFrame()

        # Build regime cycles: alternating periods of accumulation, markup, distribution, decline
        # Each regime has different return + volatility characteristics
        regime_lengths = []
        regime_types = []
        remaining = n
        while remaining > 0:
            length = int(local_rng.integers(40, 120))
            length = min(length, remaining)
            regime_lengths.append(length)
            regime_types.append(local_rng.choice(
                ['accumulation', 'markup', 'distribution', 'decline'],
                p=[0.30, 0.25, 0.20, 0.25]
            ))
            remaining -= length

        # Regime-specific (daily drift, volatility) parameters
        params = {
            'accumulation': (0.0001, 0.012),  # ~flat, low vol
            'markup':       (0.0025, 0.018),  # strong up, moderate vol
            'distribution': (0.0000, 0.020),  # flat-down, higher vol
            'decline':      (-0.0018, 0.022), # down, high vol
        }

        returns = []
        for length, regime in zip(regime_lengths, regime_types):
            drift, vol = params[regime]
            r = local_rng.normal(drift, vol, length)
            returns.append(r)
        returns = np.concatenate(returns)[:n]

        # Build close prices
        base_price = 0.5 + (ticker_seed % 100) / 30.0  # 0.5 to ~3.8 KWD
        close = base_price * np.exp(np.cumsum(returns))

        # Build OHLC around close with reasonable intraday range
        daily_range = np.abs(local_rng.normal(0, 0.012, n)) + 0.003
        high = close * (1 + daily_range / 2)
        low  = close * (1 - daily_range / 2)
        open_ = np.where(
            np.arange(n) == 0,
            close,
            np.roll(close, 1) * (1 + local_rng.normal(0, 0.004, n))
        )

        # Volume: higher during markup/distribution, lower during accumulation/dormant
        base_volume = int(local_rng.integers(50_000, 500_000))
        volume_multiplier = np.ones(n)
        idx = 0
        for length, regime in zip(regime_lengths, regime_types):
            mult = {'accumulation': 0.7, 'markup': 1.8, 'distribution': 1.4, 'decline': 1.2}[regime]
            volume_multiplier[idx:idx+length] = mult * (1 + local_rng.normal(0, 0.3, length))
            idx += length
        volume_multiplier = np.maximum(volume_multiplier, 0.2)
        volume = (base_volume * volume_multiplier).astype(int)

        # Turnover
        turnover = close * volume

        df = pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'turnover_kwd': turnover,
        }, index=dates)
        df.index.name = 'date'
        return df

    def get_ohlcv_weekly(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        daily = self.get_ohlcv_daily(ticker, start, end)
        if daily.empty:
            return daily
        weekly = daily.resample('W-THU').agg({
            'open': 'first',
            'high': 'max',
            'low':  'min',
            'close':'last',
            'volume': 'sum',
            'turnover_kwd': 'sum',
        }).dropna()
        return weekly

    def get_corporate_actions(self, ticker: str) -> pd.DataFrame:
        return pd.DataFrame(columns=['date', 'action_type', 'ratio', 'amount'])

    def get_market_index(self, index_ticker: str, start: date, end: date) -> pd.DataFrame:
        # Generate an index series the same way but smoother (composite behavior)
        dates = pd.bdate_range(start=start, end=end, freq='C',
                               weekmask='Sun Mon Tue Wed Thu')
        n = len(dates)
        if n == 0:
            return pd.DataFrame()
        local_rng = np.random.default_rng(7777)
        returns = local_rng.normal(0.0003, 0.008, n)
        close = 7000 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            'open': close * 0.999, 'high': close * 1.005,
            'low': close * 0.995, 'close': close,
            'volume': 0, 'turnover_kwd': 0,
        }, index=dates)
        df.index.name = 'date'
        return df


# =============================================================================
# Placeholder for the real adapter — to be implemented during API discovery
# =============================================================================
class TickerChartAdapter(DataAdapter):
    """
    PHASE 1 NEXT STEP: implement this against the actual ticker chart API
    used in your existing React Native app under the technical analysis page.

    Once filled in, no other code in the engine needs to change.
    """

    def __init__(self, api_base_url: str, api_key: Optional[str] = None):
        self.api_base_url = api_base_url
        self.api_key = api_key
        raise NotImplementedError(
            "TickerChartAdapter pending API discovery. Use SyntheticAdapter for now."
        )

    def list_stocks(self) -> List[StockMeta]:
        raise NotImplementedError

    def get_ohlcv_daily(self, ticker, start, end):
        raise NotImplementedError

    def get_ohlcv_weekly(self, ticker, start, end):
        raise NotImplementedError

    def get_corporate_actions(self, ticker):
        raise NotImplementedError

    def get_market_index(self, index_ticker, start, end):
        raise NotImplementedError

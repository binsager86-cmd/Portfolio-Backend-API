"""
Eagle Eye Data Adapter.

Wraps the existing TickerChart service so the Eagle Eye engine never
touches tickerchart_service.py directly. Implements the DataAdapter
abstract interface from the reference implementation.

The engine calls this adapter; this adapter calls tickerchart_service.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Abstract interface — the engine only knows about DataAdapter
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TickerChart adapter — wraps the existing tickerchart_service
# ---------------------------------------------------------------------------

class TickerChartAdapter(DataAdapter):
    """
    Wraps app.services.tickerchart_service so the Eagle Eye engine never
    imports it directly. All TickerChart-specific logic stays here.
    """

    # Stocks in the KSE universe are looked up from the analysis_stocks
    # table (exchange = 'KW' or currency = 'KWD').  When the table is
    # empty or missing (common in fresh dev environments), we fall back
    # to the hardcoded KUWAIT_STOCKS reference list so the Eagle Eye
    # pipeline works out of the box without a pre-populated DB.
    def list_stocks(self) -> List[StockMeta]:
        from app.core.database import query_all  # type: ignore[import-untyped]

        try:
            rows = query_all(
                """
                SELECT DISTINCT symbol, company_name, exchange, currency
                FROM analysis_stocks
                WHERE (exchange IN ('KW', 'KSE') OR currency = 'KWD')
                ORDER BY symbol
                """,
                (),
            )
        except Exception:
            rows = []

        result: List[StockMeta] = []
        for r in (rows or []):
            sym = str(r.get("symbol") or "").upper()
            ticker = sym.replace(".KW", "").strip()
            if not ticker:
                continue
            result.append(StockMeta(
                ticker=ticker,
                name_en=str(r.get("company_name") or ticker),
                name_ar=None,
                sector=str(r.get("sector") or "Kuwait"),
                sub_sector=None,
                market_tier="premier",
                listing_date=None,
                shares_outstanding=None,
            ))

        # Fall back to the static KUWAIT_STOCKS list when the DB table is
        # empty or unavailable (dev environments, fresh deployments).
        if not result:
            from app.data.stock_lists import KUWAIT_STOCKS

            for s in KUWAIT_STOCKS:
                sym = str(s.get("symbol") or "").upper().replace(".KW", "").strip()
                if not sym:
                    continue
                result.append(StockMeta(
                    ticker=sym,
                    name_en=str(s.get("name") or sym),
                    name_ar=None,
                    sector="Kuwait",
                    sub_sector=None,
                    market_tier="premier",
                    listing_date=None,
                    shares_outstanding=None,
                ))

        return result

    def get_ohlcv_daily(
        self, ticker: str, start: date, end: date
    ) -> pd.DataFrame:
        """
        Call the TickerChart service to get OHLCV for ticker.KW, then
        convert to a DataFrame indexed by date with columns:
        open, high, low, close, volume, turnover_kwd
        Phantom rows (open=high=low=0) are dropped.
        Duplicated dates are deduplicated (keep last).
        """
        import asyncio
        import threading
        from app.services import tickerchart_service as tc  # type: ignore[import-untyped]

        def _run_in_new_loop(coro):
            """Run a coroutine in a fresh event loop on a background thread.

            Safe to call from both plain sync code and from within a running
            event loop (e.g. FastAPI startup events, uvicorn reload handlers)
            where asyncio.run() would raise RuntimeError.
            """
            result_box: list = []
            exc_box: list = []

            def target():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result_box.append(loop.run_until_complete(coro))
                except Exception as _exc:  # noqa: BLE001
                    exc_box.append(_exc)
                finally:
                    loop.close()

            t = threading.Thread(target=target, daemon=True)
            t.start()
            t.join(timeout=30)
            if t.is_alive():
                raise TimeoutError("Async fetch timed out after 30 s")
            if exc_box:
                raise exc_box[0]
            return result_box[0] if result_box else None

        try:
            rows = _run_in_new_loop(tc.fetch_ohlcv(ticker, "KSE", from_d=start, to_d=end))
        except Exception as exc:
            logger.warning("TickerChart fetch failed for %s: %s", ticker, exc)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "turnover_kwd"])

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "turnover_kwd"])

        df = pd.DataFrame(rows)
        df.rename(columns={"date": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Compute turnover_kwd if not provided
        if "turnover_kwd" not in df.columns:
            # Use 'value' column from TickerChart if available, else close * volume
            if "value" in df.columns:
                df["turnover_kwd"] = pd.to_numeric(df["value"], errors="coerce").fillna(
                    df["close"] * df["volume"]
                )
            else:
                df["turnover_kwd"] = df["close"] * df["volume"]

        # Drop phantom rows
        phantom = (df["open"] == 0) & (df["high"] == 0) & (df["low"] == 0)
        df = df[~phantom]

        # Deduplicate by date (keep last)
        df = df[~df.index.duplicated(keep="last")]

        return df[["open", "high", "low", "close", "volume", "turnover_kwd"]]

    def get_ohlcv_weekly(
        self, ticker: str, start: date, end: date
    ) -> pd.DataFrame:
        """
        Resample daily OHLCV to weekly bars ending on Thursday
        (Kuwait market week ends Thursday).
        """
        daily = self.get_ohlcv_daily(ticker, start, end)
        if daily.empty:
            return daily
        weekly = daily.resample("W-THU").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "turnover_kwd": "sum",
        }).dropna(subset=["close"])
        return weekly

    def get_corporate_actions(self, ticker: str) -> pd.DataFrame:
        """Placeholder — returns empty DataFrame with expected schema."""
        return pd.DataFrame(columns=["date", "action_type", "ratio", "amount"])

    def get_market_index(
        self, index_ticker: str, start: date, end: date
    ) -> pd.DataFrame:
        """
        Attempt to fetch the Premier Market Index. If unavailable, returns
        a synthetic flat series so downstream callers don't break.
        """
        import asyncio
        import threading
        from app.services import tickerchart_service as tc  # type: ignore[import-untyped]

        def _run_in_new_loop(coro):
            result_box: list = []
            exc_box: list = []

            def target():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result_box.append(loop.run_until_complete(coro))
                except Exception as _exc:
                    exc_box.append(_exc)
                finally:
                    loop.close()

            t = threading.Thread(target=target, daemon=True)
            t.start()
            t.join(timeout=30)
            if t.is_alive():
                raise TimeoutError("Async fetch timed out after 30 s")
            if exc_box:
                raise exc_box[0]
            return result_box[0] if result_box else None

        try:
            rows = _run_in_new_loop(tc.fetch_ohlcv(index_ticker, "KSE", from_d=start, to_d=end))
        except Exception:
            rows = []

        if rows:
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            if "turnover_kwd" not in df.columns:
                df["turnover_kwd"] = df.get("value", df["close"] * df.get("volume", 0))
            return df[["open", "high", "low", "close", "volume", "turnover_kwd"]]

        # Synthetic flat placeholder
        dates = pd.bdate_range(start=start, end=end, freq="C",
                               weekmask="Sun Mon Tue Wed Thu")
        n = len(dates)
        if n == 0:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "turnover_kwd"])
        close = np.full(n, 7000.0)
        return pd.DataFrame({
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.zeros(n, dtype=int),
            "turnover_kwd": np.zeros(n),
        }, index=dates)


# ---------------------------------------------------------------------------
# Synthetic adapter — for testing without a live TickerChart connection
# ---------------------------------------------------------------------------

class SyntheticAdapter(DataAdapter):
    """
    Generates realistic-looking Kuwait-like OHLCV for testing the engine
    end-to-end before plugging into the real TickerChart API.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._stocks = self._make_stock_universe()

    def _make_stock_universe(self) -> List[StockMeta]:
        return [
            StockMeta("NBK",      "National Bank of Kuwait",   "البنك الوطني",   "Banking",   None, "premier", date(2010,1,1), 7_000_000_000),
            StockMeta("KFH",      "Kuwait Finance House",      "بيت التمويل",    "Banking",   None, "premier", date(2010,1,1), 10_000_000_000),
            StockMeta("BOUBYAN",  "Boubyan Bank",              "بنك بوبيان",      "Banking",   None, "premier", date(2010,1,1), 3_500_000_000),
            StockMeta("ZAIN",     "Zain",                      "زين",            "Telecom",   None, "premier", date(2010,1,1), 4_300_000_000),
            StockMeta("AGILITY",  "Agility Public Warehousing","أجيليتي",         "Logistics", None, "premier", date(2010,1,1), 2_400_000_000),
            StockMeta("MABANEE",  "Mabanee",                   "المباني",        "Real Estate", None, "premier", date(2010,1,1), 1_200_000_000),
            StockMeta("HUMANSOFT","Humansoft Holding",         "هيومن سوفت",     "Technology",None, "premier", date(2010,1,1),   125_000_000),
            StockMeta("GBK",      "Gulf Bank",                 "بنك الخليج",      "Banking",   None, "premier", date(2010,1,1), 3_100_000_000),
            StockMeta("STC",      "Kuwait Telecom (stc)",      "stc",            "Telecom",   None, "premier", date(2010,1,1),   500_000_000),
            StockMeta("ALAFCO",   "ALAFCO Aviation Lease",     "ألافكو",         "Industrial",None, "main",    date(2010,1,1), 1_100_000_000),
        ]

    def list_stocks(self) -> List[StockMeta]:
        return self._stocks

    def get_ohlcv_daily(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        ticker_seed = sum(ord(c) for c in ticker) * 31
        local_rng = np.random.default_rng(ticker_seed)

        dates = pd.bdate_range(start=start, end=end, freq="C",
                               weekmask="Sun Mon Tue Wed Thu")
        n = len(dates)
        if n == 0:
            return pd.DataFrame()

        regime_lengths, regime_types = [], []
        remaining = n
        while remaining > 0:
            length = int(local_rng.integers(40, 120))
            length = min(length, remaining)
            regime_lengths.append(length)
            regime_types.append(local_rng.choice(
                ["accumulation", "markup", "distribution", "decline"],
                p=[0.30, 0.25, 0.20, 0.25]
            ))
            remaining -= length

        params = {
            "accumulation": (0.0001, 0.012),
            "markup":       (0.0025, 0.018),
            "distribution": (0.0000, 0.020),
            "decline":      (-0.0018, 0.022),
        }

        returns = []
        for length, regime in zip(regime_lengths, regime_types):
            drift, vol = params[regime]
            r = local_rng.normal(drift, vol, length)
            returns.append(r)
        returns = np.concatenate(returns)[:n]

        base_price = 0.5 + (ticker_seed % 100) / 30.0
        close = base_price * np.exp(np.cumsum(returns))

        daily_range = np.abs(local_rng.normal(0, 0.012, n)) + 0.003
        high = close * (1 + daily_range / 2)
        low  = close * (1 - daily_range / 2)
        open_ = np.where(
            np.arange(n) == 0,
            close,
            np.roll(close, 1) * (1 + local_rng.normal(0, 0.004, n))
        )

        base_volume = int(local_rng.integers(50_000, 500_000))
        volume_multiplier = np.ones(n)
        idx = 0
        for length, regime in zip(regime_lengths, regime_types):
            mult = {"accumulation": 0.7, "markup": 1.8, "distribution": 1.4, "decline": 1.2}[regime]
            volume_multiplier[idx:idx+length] = mult * (1 + local_rng.normal(0, 0.3, length))
            idx += length
        volume_multiplier = np.maximum(volume_multiplier, 0.2)
        volume = (base_volume * volume_multiplier).astype(int)

        df = pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "turnover_kwd": close * volume,
        }, index=dates)
        df.index.name = "date"
        return df

    def get_ohlcv_weekly(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        daily = self.get_ohlcv_daily(ticker, start, end)
        if daily.empty:
            return daily
        return daily.resample("W-THU").agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
            "volume": "sum", "turnover_kwd": "sum",
        }).dropna()

    def get_corporate_actions(self, ticker: str) -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "action_type", "ratio", "amount"])

    def get_market_index(self, index_ticker: str, start: date, end: date) -> pd.DataFrame:
        dates = pd.bdate_range(start=start, end=end, freq="C",
                               weekmask="Sun Mon Tue Wed Thu")
        n = len(dates)
        if n == 0:
            return pd.DataFrame()
        local_rng = np.random.default_rng(7777)
        returns = local_rng.normal(0.0003, 0.008, n)
        close = 7000 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            "open": close * 0.999, "high": close * 1.005,
            "low": close * 0.995, "close": close,
            "volume": np.zeros(n, dtype=int), "turnover_kwd": np.zeros(n),
        }, index=dates)
        df.index.name = "date"
        return df

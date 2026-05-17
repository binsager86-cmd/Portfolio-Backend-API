"""
ml/market_context.py — Phase 1: Market context features.

Computes per-row market context features (Section 1.4 — Market Context):
  - index_return_Nd    : KSE All-Share index return over N days
  - sector_return_Nd   : sector index return over N days
  - market_regime      : current regime tag (RISK_ON / NEUTRAL / RISK_OFF)
  - stock_beta_60d     : rolling 60-day beta vs KSE index
  - index_vol_20d      : 20-day realized volatility of the KSE index

All computations are point-in-time: only data at or before the row date
is used (expanding/trailing windows only — never centered).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# KSE All-Share Index ticker (as used in the TickerChart adapter)
KSE_INDEX_TICKER = "KSE_ALL"

# Lookback windows (trading days) for return features
RETURN_WINDOWS: Tuple[int, ...] = (1, 3, 5, 10, 20, 60)

# Rolling beta window
BETA_WINDOW = 60


def _load_index_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    """Load index OHLCV from the ee_ohlcv_cache; returns None on failure."""
    try:
        from app.services.eagle_eye.store import load_ohlcv
        df = load_ohlcv(ticker)
        return df if (df is not None and not df.empty) else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Market context: index load failed (%s): %s", ticker, exc)
        return None


def _rolling_returns(close: pd.Series, windows: Tuple[int, ...]) -> pd.DataFrame:
    """
    Compute trailing log-returns for each window length.
    Uses .shift(1) so the return at time T is computed from close[T-N] to
    close[T-1] — strictly no look-ahead.
    """
    cols: Dict[str, pd.Series] = {}
    log_close = np.log(close.clip(lower=1e-10))
    for w in windows:
        cols[f"return_{w}d"] = log_close.diff(w).shift(1)
    return pd.DataFrame(cols, index=close.index)


class MarketContextBuilder:
    """
    Enriches a feature DataFrame with KSE index and sector context columns.

    Usage
    -----
        ctx = MarketContextBuilder()
        df_enriched = ctx.enrich(df, date_col="event_date")
    """

    def __init__(
        self,
        index_ticker: str = KSE_INDEX_TICKER,
        return_windows: Tuple[int, ...] = RETURN_WINDOWS,
        beta_window: int = BETA_WINDOW,
    ) -> None:
        self.index_ticker = index_ticker
        self.return_windows = return_windows
        self.beta_window = beta_window
        self._index_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(self, df: pd.DataFrame, date_col: str = "event_date") -> pd.DataFrame:
        """
        Add market context feature columns to ``df`` in-place (returns copy).

        Parameters
        ----------
        df       : DataFrame with at least a date column
        date_col : name of the date column (values convertible to Timestamp)
        """
        if date_col not in df.columns:
            logger.warning("MarketContextBuilder: date column '%s' not found", date_col)
            return df

        index_df = self._load_index()
        if index_df is None or index_df.empty:
            logger.info("Market context: KSE index not available — skipping enrichment")
            return df

        df = df.copy()
        dates = pd.to_datetime(df[date_col], errors="coerce")

        # Build lookup series for index features
        idx_rets = _rolling_returns(index_df["close"], self.return_windows)
        idx_vol = (
            np.log(index_df["close"].clip(lower=1e-10))
            .diff()
            .shift(1)
            .rolling(20)
            .std()
            * np.sqrt(252)
        )

        # Map each row's date to the index features on that date (or nearest prior)
        for w in self.return_windows:
            col = f"index_return_{w}d"
            df[col] = self._lookup_series(dates, idx_rets[f"return_{w}d"])

        df["index_vol_20d"] = self._lookup_series(dates, idx_vol)

        # Market regime from existing Eagle Eye system
        df["market_regime"] = self._lookup_regime(dates, index_df)

        # Sector-level beta (requires per-stock OHLCV — added by caller if available)
        # We provide the helper function; the caller passes stock close series.

        return df

    def compute_rolling_beta(
        self,
        stock_close: pd.Series,
        date_index: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Compute trailing ``beta_window``-day beta of a stock vs the KSE index.

        Returns a Series aligned to ``date_index``.  Uses shift(1) so beta at
        time T is computed from data ending at T-1.

        Parameters
        ----------
        stock_close : daily close prices of the stock (DatetimeIndex)
        date_index  : dates for which to compute beta
        """
        index_df = self._load_index()
        if index_df is None or stock_close.empty:
            return pd.Series(np.nan, index=date_index)

        # Align on common dates
        s = np.log(stock_close.clip(lower=1e-10)).diff().shift(1)
        idx_ret = np.log(index_df["close"].clip(lower=1e-10)).diff().shift(1)
        common = s.index.intersection(idx_ret.index)
        s = s.loc[common]
        idx_ret = idx_ret.loc[common]

        # Rolling beta = cov(stock, index) / var(index)
        cov = s.rolling(self.beta_window).cov(idx_ret)
        var = idx_ret.rolling(self.beta_window).var()
        beta = cov / var.replace(0, np.nan)

        # Map back to date_index
        result = beta.reindex(date_index, method="ffill")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> Optional[pd.DataFrame]:
        if self._index_df is None:
            self._index_df = _load_index_ohlcv(self.index_ticker)
        return self._index_df

    def _lookup_series(
        self, dates: pd.Series, series: pd.Series
    ) -> pd.Series:
        """
        For each date in ``dates``, find the value in ``series`` at that date
        or the nearest PRIOR date (forward-fill in the past → no future data).
        """
        if series.empty:
            return pd.Series(np.nan, index=dates.index)

        series_sorted = series.sort_index()
        result = pd.Series(np.nan, index=dates.index)
        for i, t in enumerate(dates):
            if pd.isna(t):
                continue
            mask = series_sorted.index <= t
            if mask.any():
                result.iloc[i] = float(series_sorted[mask].iloc[-1])
        return result

    def _lookup_regime(
        self, dates: pd.Series, index_df: pd.DataFrame
    ) -> pd.Series:
        """Derive a simple regime tag from KSE index trailing volatility."""
        close = index_df["close"]
        log_ret = np.log(close.clip(lower=1e-10)).diff().shift(1)
        vol_20 = log_ret.rolling(20).std() * np.sqrt(252)

        # Percentile buckets on the trailing vol
        vol_60pct = vol_20.expanding().quantile(0.60)
        vol_40pct = vol_20.expanding().quantile(0.40)

        regime_raw = pd.Series("NEUTRAL", index=vol_20.index)
        regime_raw[vol_20 < vol_40pct] = "RISK_ON"
        regime_raw[vol_20 > vol_60pct] = "RISK_OFF"

        return self._lookup_series(dates, regime_raw)

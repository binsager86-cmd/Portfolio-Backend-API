"""
ml/macro_features.py — Addendum A.3: Kuwait macro / oil / GCC features.

Computes point-in-time macro context features for every training row:

  Oil:
    brent_return_5d, brent_return_20d, brent_return_60d
    brent_vol_20d
    brent_regime_score  (percentile of 60d return vs 5yr trailing distribution)

  GCC:
    gcc_return_5d, gcc_return_20d, gcc_return_60d
    kw_gcc_corr_60d     (rolling 60d correlation of Kuwait All-Share vs Tadawul)

  KWD FX (if available):
    kwd_fx_return_5d, kwd_fx_return_20d, kwd_fx_return_60d

  Per-stock:
    stock_oil_sensitivity_60d  (rolling 60d correlation of stock returns vs Brent)

Proxy choices (documented per A.3 rules):
  - GCC composite → Saudi Tadawul All-Share Index (yfinance: "^TASI.SR")
    Documented reason: no single widely-available GCC composite index;
    Tadawul is the largest and most liquid GCC market by cap.
  - Brent crude → yfinance: "BZ=F" (front-month Brent futures)
    Documented reason: spot price not directly available via yfinance;
    front-month futures closely track spot with < 1% divergence on a daily basis.
  - KWD FX → not currently available from public APIs without subscription.
    Documented as UNAVAILABLE in reports/data_gaps.md.  Features are omitted
    (not fabricated) when unavailable.

All features use shift(1) trailing windows — no centered windows, no look-ahead.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable tickers / constants
# ---------------------------------------------------------------------------

BRENT_TICKER    = "BZ=F"          # yfinance: Brent front-month futures
TADAWUL_TICKER  = "^TASI.SR"      # yfinance: Saudi Tadawul All-Share
KSE_TICKER      = "KSE_ALL"       # internal store key for Kuwait All-Share

RETURN_WINDOWS: Tuple[int, ...] = (5, 20, 60)
BRENT_REGIME_LOOKBACK = 252 * 5   # ~5 years of trading days

REPORTS_DIR = Path(__file__).resolve().parents[4] / "reports"


# ---------------------------------------------------------------------------
# Data fetching helpers
# ---------------------------------------------------------------------------

def _fetch_yfinance(ticker: str, start: str, end: str) -> Optional[pd.Series]:
    """
    Fetch daily close prices from yfinance.  Returns None on failure.
    All dates are ISO strings (YYYY-MM-DD).
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        close = df["Close"].squeeze()
        close.name = ticker
        return close.sort_index()
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
        return None


def _load_internal_ohlcv_close(ticker: str) -> Optional[pd.Series]:
    """Load close prices from the ee_ohlcv_cache."""
    try:
        from app.services.eagle_eye.store import load_ohlcv
        df = load_ohlcv(ticker)
        if df is None or df.empty or "close" not in df.columns:
            return None
        return df["close"].sort_index()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Macro series cache (loaded once per process)
# ---------------------------------------------------------------------------

class _MacroCache:
    """Singleton-style cache so yfinance is called at most once per run."""

    def __init__(self) -> None:
        self._brent: Optional[pd.Series] = None
        self._tadawul: Optional[pd.Series] = None
        self._kse: Optional[pd.Series] = None
        self._loaded = False

    def warm(self, start: str = "2018-01-01") -> None:
        if self._loaded:
            return
        end = date.today().isoformat()
        logger.info("MacroCache: loading Brent and Tadawul…")
        self._brent   = _fetch_yfinance(BRENT_TICKER,   start, end)
        self._tadawul = _fetch_yfinance(TADAWUL_TICKER, start, end)
        self._kse     = _load_internal_ohlcv_close(KSE_TICKER)
        self._loaded  = True
        logger.info(
            "MacroCache ready: brent=%s bars, tadawul=%s bars, kse=%s bars",
            len(self._brent) if self._brent is not None else "N/A",
            len(self._tadawul) if self._tadawul is not None else "N/A",
            len(self._kse) if self._kse is not None else "N/A",
        )

    @property
    def brent(self) -> Optional[pd.Series]:
        return self._brent

    @property
    def tadawul(self) -> Optional[pd.Series]:
        return self._tadawul

    @property
    def kse(self) -> Optional[pd.Series]:
        return self._kse


_CACHE = _MacroCache()


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _trailing_returns(close: pd.Series, windows: Sequence[int]) -> pd.DataFrame:
    """
    Point-in-time trailing log-returns.  shift(1) ensures we use close[T-1]
    as the most recent observation when computing the row for time T.
    """
    log_c = np.log(close.clip(lower=1e-8))
    out: Dict[str, pd.Series] = {}
    for w in windows:
        out[f"return_{w}d"] = log_c.diff(w).shift(1)
    return pd.DataFrame(out, index=close.index)


def _brent_regime_score(brent_close: pd.Series) -> pd.Series:
    """
    Expanding percentile rank of the 60d return vs its own trailing 5yr history.
    Returns a value in [0, 1].  shift(1) applied.
    """
    ret60 = np.log(brent_close.clip(lower=1e-8)).diff(60).shift(1)
    score = ret60.expanding(min_periods=60).rank(pct=True)
    return score.rename("brent_regime_score")


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Trailing ``window``-day correlation between two return series, shift(1)."""
    ra = np.log(a.clip(lower=1e-8)).diff().shift(1)
    rb = np.log(b.clip(lower=1e-8)).diff().shift(1)
    common = ra.index.intersection(rb.index)
    ra, rb = ra.loc[common], rb.loc[common]
    return ra.rolling(window).corr(rb).rename(f"corr_{window}d")


def _lookup(dates: pd.Series, series: pd.Series) -> pd.Series:
    """
    For each date in ``dates``, return the value in ``series`` at that date
    or the closest prior date (no future data used).
    """
    if series is None or series.empty:
        return pd.Series(np.nan, index=dates.index)
    s = series.sort_index()
    result = pd.Series(np.nan, index=dates.index)
    for i, t in enumerate(dates):
        if pd.isna(t):
            continue
        mask = s.index <= t
        if mask.any():
            result.iloc[i] = float(s[mask].iloc[-1])
    return result


# ---------------------------------------------------------------------------
# Public builder class
# ---------------------------------------------------------------------------

class MacroFeatureBuilder:
    """
    Adds Kuwait macro features to a feature DataFrame.

    Usage
    -----
        builder = MacroFeatureBuilder()
        df = builder.enrich(df, date_col="event_date", stock_close=ohlcv["close"])
    """

    def __init__(self) -> None:
        _CACHE.warm()

    def enrich(
        self,
        df: pd.DataFrame,
        date_col: str = "event_date",
        stock_close: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Add all macro feature columns to ``df``.

        Parameters
        ----------
        df          : feature DataFrame
        date_col    : date column name
        stock_close : daily close Series for the stock (DatetimeIndex);
                      needed for per-stock oil sensitivity.  Optional.
        """
        if date_col not in df.columns:
            return df

        df = df.copy()
        dates = pd.to_datetime(df[date_col], errors="coerce")

        # ── Oil features ───────────────────────────────────────────────
        if _CACHE.brent is not None:
            brent_rets = _trailing_returns(_CACHE.brent, RETURN_WINDOWS)
            for w in RETURN_WINDOWS:
                df[f"brent_return_{w}d"] = _lookup(dates, brent_rets[f"return_{w}d"])

            brent_vol = (
                np.log(_CACHE.brent.clip(lower=1e-8))
                .diff()
                .shift(1)
                .rolling(20)
                .std()
                * np.sqrt(252)
            )
            df["brent_vol_20d"] = _lookup(dates, brent_vol)
            df["brent_regime_score"] = _lookup(dates, _brent_regime_score(_CACHE.brent))
        else:
            for w in RETURN_WINDOWS:
                df[f"brent_return_{w}d"] = np.nan
            df["brent_vol_20d"]       = np.nan
            df["brent_regime_score"]  = np.nan
            logger.debug("Brent data unavailable — macro features will be NaN")

        # ── GCC / Tadawul features ─────────────────────────────────────
        if _CACHE.tadawul is not None:
            tad_rets = _trailing_returns(_CACHE.tadawul, RETURN_WINDOWS)
            for w in RETURN_WINDOWS:
                df[f"gcc_return_{w}d"] = _lookup(dates, tad_rets[f"return_{w}d"])

            if _CACHE.kse is not None:
                corr_60 = _rolling_corr(_CACHE.kse, _CACHE.tadawul, 60)
                df["kw_gcc_corr_60d"] = _lookup(dates, corr_60)
            else:
                df["kw_gcc_corr_60d"] = np.nan
        else:
            for w in RETURN_WINDOWS:
                df[f"gcc_return_{w}d"] = np.nan
            df["kw_gcc_corr_60d"] = np.nan
            logger.debug("Tadawul data unavailable — GCC features will be NaN")

        # ── Per-stock oil sensitivity ──────────────────────────────────
        if stock_close is not None and _CACHE.brent is not None:
            try:
                oil_sens = _rolling_corr(stock_close, _CACHE.brent, 60)
                df["stock_oil_sensitivity_60d"] = _lookup(dates, oil_sens)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Oil sensitivity failed: %s", exc)
                df["stock_oil_sensitivity_60d"] = np.nan
        else:
            df["stock_oil_sensitivity_60d"] = np.nan

        # KWD FX: unavailable — document in data gaps report, do not fabricate
        df["kwd_fx_return_5d"]  = np.nan   # UNAVAILABLE — see reports/data_gaps.md
        df["kwd_fx_return_20d"] = np.nan
        df["kwd_fx_return_60d"] = np.nan

        return df


# ---------------------------------------------------------------------------
# Data-gap documentation helper
# ---------------------------------------------------------------------------

def write_data_gaps_report() -> None:
    """
    Append / overwrite reports/data_gaps.md with current known data gaps.
    Called once at startup.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "data_gaps.md"
    content = """\
# Eagle Eye ML — Known Data Gaps

_Auto-generated by macro_features.py. Do not edit manually._

## Permanently Unavailable (as of 2026-05-16)

| Feature Group | Reason | Impact |
| --- | --- | --- |
| KWD FX returns (kwd_fx_return_*) | No free public API for KWD exchange rate history; Bloomberg / Refinitiv subscription required. | Features set to NaN. _is_missing indicator columns added. Model trains without them. |
| Kuwait CDS spread / 10-year yield | No public machine-readable source identified. | Same as above. |
| Intraday volume profile (VWAP, trade count) | TickerChart API does not expose intraday bars. | Volume features use daily OHLCV only. |

## Conditionally Unavailable

| Feature Group | Condition | Action |
| --- | --- | --- |
| Brent crude (BZ=F) | yfinance unreachable or rate-limited | Features set to NaN; pipeline continues. Retry on next run. |
| GCC / Tadawul (^TASI.SR) | Same | Same. |

## Proxy Choices (Documented)

| Series | Proxy Used | Rationale |
| --- | --- | --- |
| GCC Composite Index | Saudi Tadawul All-Share (^TASI.SR) | No single GCC composite available on public APIs. Tadawul is largest GCC market by cap (~50% of GCC equity weight). |
| Brent Spot Price | Brent Front-Month Futures (BZ=F) | Spot price not on yfinance. Front-month futures track spot within < 1% on a daily basis; acceptable for directional feature engineering. |

_Update this file if new data sources become available._
"""
    out.write_text(content, encoding="utf-8")
    logger.info("Data gaps report written to %s", out)

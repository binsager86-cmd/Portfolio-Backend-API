"""
Unit tests for Eagle Eye indicators.

Tests all 10 core indicator functions using synthetic OHLCV data.
No network calls are made — all data generated in-memory.
"""
from __future__ import annotations

import sys
import os

# Ensure backend root is on the path so app.* imports resolve
_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

import numpy as np
import pandas as pd
import pytest

from app.services.eagle_eye.indicators import (
    compute_all_indicators,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate realistic-looking OHLCV with a drift + noise process."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.015, n)
    close = 1.0 * np.exp(np.cumsum(returns))
    daily_range = np.abs(rng.normal(0, 0.012, n)) + 0.005
    high = close * (1 + daily_range / 2)
    low = close * (1 - daily_range / 2)
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.004, n))
    volume = rng.integers(100_000, 1_000_000, n)

    dates = pd.bdate_range(end="2024-12-31", periods=n, freq="C",
                           weekmask="Sun Mon Tue Wed Thu")
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume.astype(float),
        "turnover_kwd": close * volume,
    }, index=dates)
    df.index.name = "date"
    return df


def _make_constant(n: int = 300, price: float = 1.0) -> pd.DataFrame:
    """Create OHLCV where close == open == high == low == price (no movement)."""
    dates = pd.bdate_range(end="2024-12-31", periods=n, freq="C",
                           weekmask="Sun Mon Tue Wed Thu")
    df = pd.DataFrame({
        "open": price, "high": price, "low": price, "close": price,
        "volume": 100_000.0,
        "turnover_kwd": price * 100_000,
    }, index=dates)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Import indicator functions directly for isolated tests
# (all functions take a DataFrame with open/high/low/close/volume columns)
# ---------------------------------------------------------------------------

from app.services.eagle_eye.indicators import (
    adx,
    atr,
    bollinger_bands,
    cmf,
    macd,
    mfi,
    obv,
    rsi,
    ema as _ema_fn,
)


# ---------------------------------------------------------------------------
# Test 1: EMA on constant input
# ---------------------------------------------------------------------------

def test_ema_constant_input():
    """EMA of a constant series should equal the constant."""
    prices = pd.Series(np.full(100, 5.0))
    df = pd.DataFrame({"close": prices})
    result = _ema_fn(df, 20)
    # After warm-up period, all values should equal 5.0
    assert result.dropna().shape[0] > 0, "EMA returned all NaN on constant input"
    tail = result.dropna().iloc[20:]
    np.testing.assert_allclose(
        tail.values, 5.0, rtol=1e-3,
        err_msg="EMA should equal constant input for flat price series"
    )


# ---------------------------------------------------------------------------
# Test 2: RSI must stay within 0-100
# ---------------------------------------------------------------------------

def test_rsi_range():
    df = _make_ohlcv(300)
    rsi_series = rsi(df, period=14)
    valid = rsi_series.dropna()
    assert len(valid) > 0, "RSI returned all NaN"
    assert (valid >= 0).all() and (valid <= 100).all(), \
        f"RSI out of range [0, 100]: min={valid.min():.2f}, max={valid.max():.2f}"


# ---------------------------------------------------------------------------
# Test 3: MACD components
# ---------------------------------------------------------------------------

def test_macd_components():
    df = _make_ohlcv(300)
    macd_line, signal_line, histogram = macd(df)
    assert not macd_line.dropna().empty, "MACD line is all NaN"
    assert not signal_line.dropna().empty, "Signal line is all NaN"
    assert not histogram.dropna().empty, "MACD histogram is all NaN"
    # Histogram should equal macd_line - signal_line (where both are valid)
    both_valid = macd_line.notna() & signal_line.notna()
    diff = (macd_line - signal_line)[both_valid]
    hist_v = histogram[both_valid]
    np.testing.assert_allclose(
        diff.values, hist_v.values, rtol=1e-6,
        err_msg="MACD histogram != MACD line - signal line"
    )


# ---------------------------------------------------------------------------
# Test 4: ATR must be non-negative
# ---------------------------------------------------------------------------

def test_atr_positive():
    df = _make_ohlcv(300)
    atr_series = atr(df, period=14)
    valid = atr_series.dropna()
    assert len(valid) > 0, "ATR returned all NaN"
    assert (valid >= 0).all(), "ATR must be non-negative"


# ---------------------------------------------------------------------------
# Test 5: Bollinger Band ordering
# ---------------------------------------------------------------------------

def test_bollinger_bands_ordering():
    df = _make_ohlcv(300)
    bb = bollinger_bands(df, period=20, stddev=2.0)
    upper, middle, lower = bb["upper"], bb["middle"], bb["lower"]
    valid = upper.notna() & middle.notna() & lower.notna()
    assert valid.sum() > 0, "Bollinger Bands all NaN"
    assert (upper[valid] >= middle[valid]).all(), "Upper band must >= middle band"
    assert (middle[valid] >= lower[valid]).all(), "Middle band must >= lower band"


# ---------------------------------------------------------------------------
# Test 6: OBV responds to price direction
# ---------------------------------------------------------------------------

def test_obv_responds_to_direction():
    """OBV should increase on up-days and decrease on down-days."""
    n = 50
    dates = pd.bdate_range(end="2024-12-31", periods=n)
    # Strictly increasing prices
    close = np.linspace(1.0, 2.0, n)
    df = pd.DataFrame({
        "open": close * 0.99, "high": close * 1.01,
        "low": close * 0.98, "close": close,
        "volume": np.full(n, 10_000.0),
        "turnover_kwd": close * 10_000,
    }, index=dates)
    obv_series = obv(df)
    assert obv_series.iloc[-1] > obv_series.iloc[0], \
        "OBV should increase on a pure uptrend"


# ---------------------------------------------------------------------------
# Test 7: CMF must stay within [-1, +1]
# ---------------------------------------------------------------------------

def test_cmf_range():
    df = _make_ohlcv(300)
    cmf_series = cmf(df, period=20)
    valid = cmf_series.dropna()
    assert len(valid) > 0, "CMF returned all NaN"
    assert (valid >= -1.0).all() and (valid <= 1.0).all(), \
        f"CMF out of range [-1, 1]: min={valid.min():.4f}, max={valid.max():.4f}"


# ---------------------------------------------------------------------------
# Test 8: MFI must stay within 0-100
# ---------------------------------------------------------------------------

def test_mfi_range():
    df = _make_ohlcv(300)
    mfi_series = mfi(df, period=14)
    valid = mfi_series.dropna()
    assert len(valid) > 0, "MFI returned all NaN"
    assert (valid >= 0).all() and (valid <= 100).all(), \
        f"MFI out of range [0, 100]: min={valid.min():.2f}, max={valid.max():.2f}"


# ---------------------------------------------------------------------------
# Test 9: ADX must be non-negative
# ---------------------------------------------------------------------------

def test_adx_range():
    df = _make_ohlcv(300)
    adx_series, plus_di, minus_di = adx(df, period=14)
    valid = adx_series.dropna()
    assert len(valid) > 0, "ADX returned all NaN"
    assert (valid >= 0).all(), "ADX must be non-negative"


# ---------------------------------------------------------------------------
# Test 10: compute_all_indicators — no crash on normal data
# ---------------------------------------------------------------------------

def test_compute_all_indicators_no_crashes():
    df = _make_ohlcv(300)
    result = compute_all_indicators(df)
    assert result is not None, "compute_all_indicators returned None"
    assert isinstance(result, pd.DataFrame), "compute_all_indicators must return a DataFrame"
    assert len(result) > 0, "compute_all_indicators returned empty DataFrame"
    # Should include at least these key columns
    required_cols = ["rsi", "macd_histogram", "adx", "atr", "cmf", "obv"]
    for col in required_cols:
        assert col in result.columns, f"Missing expected column: {col}"
    # Should not be entirely NaN for the last 50 rows
    last_50 = result.tail(50)
    non_nan_cols = last_50.notna().any(axis=0)
    assert non_nan_cols.sum() > len(required_cols), \
        "Too many all-NaN columns in compute_all_indicators output"

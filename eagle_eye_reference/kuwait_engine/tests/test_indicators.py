"""Sanity tests for indicator engine.
Validates indicators against simple known cases to ensure math is correct.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from indicators.engine import (
    _ema, _wilder_ema, rsi, macd, atr, bollinger_bands,
    obv, cmf, mfi, adx, compute_all_indicators,
)


def _make_test_df(n=300, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    returns = rng.normal(0.0005, 0.015, n)
    close = 100 * np.exp(np.cumsum(returns))
    daily_range = np.abs(rng.normal(0, 0.012, n)) + 0.005
    df = pd.DataFrame({
        'open':   close * (1 + rng.normal(0, 0.003, n)),
        'high':   close * (1 + daily_range/2),
        'low':    close * (1 - daily_range/2),
        'close':  close,
        'volume': rng.integers(100_000, 1_000_000, n),
    }, index=dates)
    return df


def test_ema_constant_input():
    """EMA of constant series should equal that constant."""
    s = pd.Series([5.0] * 100)
    result = _ema(s, 10)
    assert abs(result.dropna().iloc[-1] - 5.0) < 1e-9
    print("✓ EMA constant input")


def test_rsi_range():
    """RSI must be in [0, 100]."""
    df = _make_test_df()
    r = rsi(df)
    valid = r.dropna()
    assert valid.min() >= 0 and valid.max() <= 100
    print(f"✓ RSI in [0,100], min={valid.min():.2f}, max={valid.max():.2f}")


def test_macd_components():
    df = _make_test_df()
    m, s, h = macd(df)
    diff = (m - s) - h
    valid = diff.dropna()
    assert valid.abs().max() < 1e-9
    print("✓ MACD: histogram = macd - signal")


def test_atr_positive():
    df = _make_test_df()
    a = atr(df)
    valid = a.dropna()
    assert (valid > 0).all()
    print(f"✓ ATR always positive, mean={valid.mean():.4f}")


def test_bollinger_bands_ordering():
    df = _make_test_df()
    bb = bollinger_bands(df)
    valid = pd.DataFrame({'u': bb['upper'], 'm': bb['middle'], 'l': bb['lower']}).dropna()
    assert (valid['u'] >= valid['m']).all() and (valid['m'] >= valid['l']).all()
    print("✓ Bollinger Bands: upper >= middle >= lower")


def test_obv_responds_to_direction():
    """OBV should rise on up days, fall on down days."""
    df = pd.DataFrame({
        'open': [100, 100, 101, 101, 100],
        'high': [101, 102, 102, 102, 101],
        'low':  [99,  100, 100, 100, 99],
        'close':[100.5, 101.5, 100.5, 101.5, 100.5],
        'volume':[1000, 2000, 3000, 4000, 5000],
    }, index=pd.date_range('2020-01-01', periods=5))
    o = obv(df).values
    # day 1: no prior, treat as 0 contribution
    # day 2: 101.5 > 100.5 → +2000
    # day 3: 100.5 < 101.5 → -3000
    # day 4: 101.5 > 100.5 → +4000
    # day 5: 100.5 < 101.5 → -5000
    expected = np.array([0, 2000, -1000, 3000, -2000])
    assert np.allclose(o, expected), f"Expected {expected}, got {o}"
    print("✓ OBV direction logic correct")


def test_cmf_range():
    df = _make_test_df()
    c = cmf(df).dropna()
    assert c.min() >= -1.0 - 1e-9 and c.max() <= 1.0 + 1e-9
    print(f"✓ CMF in [-1,1], range=[{c.min():.3f}, {c.max():.3f}]")


def test_mfi_range():
    df = _make_test_df()
    m = mfi(df).dropna()
    assert m.min() >= 0 and m.max() <= 100
    print(f"✓ MFI in [0,100], range=[{m.min():.2f}, {m.max():.2f}]")


def test_adx_range():
    df = _make_test_df()
    a, plus_di, minus_di = adx(df)
    a_valid = a.dropna()
    assert a_valid.min() >= 0 and a_valid.max() <= 100
    print(f"✓ ADX in [0,100], mean={a_valid.mean():.2f}")


def test_compute_all_indicators_no_crashes():
    df = _make_test_df(n=500)
    out = compute_all_indicators(df)
    n_cols = len(out.columns)
    n_no_nan = out.dropna(how='all').shape[0]
    print(f"✓ compute_all_indicators produced {n_cols} columns, {n_no_nan} usable rows")


def run_all():
    print("Running indicator engine sanity tests…\n")
    test_ema_constant_input()
    test_rsi_range()
    test_macd_components()
    test_atr_positive()
    test_bollinger_bands_ordering()
    test_obv_responds_to_direction()
    test_cmf_range()
    test_mfi_range()
    test_adx_range()
    test_compute_all_indicators_no_crashes()
    print("\nAll sanity tests passed.")


if __name__ == "__main__":
    run_all()

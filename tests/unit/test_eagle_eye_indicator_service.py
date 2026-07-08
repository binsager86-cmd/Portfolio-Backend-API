from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.eagle_eye.indicator_service import _cmf, _expanding_percentile, _norm_flow_slope, _norm_lr_slope


def test_norm_lr_slope_linear_plus_12pct_over_40():
    series = pd.Series(np.linspace(100.0, 112.0, 40), dtype=float)
    out = _norm_lr_slope(series, 40)
    value = float(out.iloc[-1])
    assert abs(value - 0.12) <= 0.005


def test_norm_lr_slope_flat_is_near_zero():
    series = pd.Series([123.45] * 60, dtype=float)
    out = _norm_lr_slope(series, 40)
    value = float(out.iloc[-1])
    assert abs(value) <= 1e-6


def test_norm_flow_slope_alternating_obv_is_near_zero():
    lookback = 40
    volume = pd.Series([1000.0] * 80, dtype=float)
    flow = []
    cur = 0.0
    for i in range(80):
        cur += 1000.0 if (i % 2 == 0) else -1000.0
        flow.append(cur)
    obv = pd.Series(flow, dtype=float)
    out = _norm_flow_slope(obv, volume, lookback)
    value = float(out.iloc[-1])
    assert abs(value) <= 0.05


def test_norm_flow_slope_all_up_days_is_near_positive_one():
    lookback = 40
    volume = pd.Series([1000.0] * 80, dtype=float)
    obv = pd.Series(np.cumsum([1000.0] * 80), dtype=float)
    out = _norm_flow_slope(obv, volume, lookback)
    value = float(out.iloc[-1])
    assert 0.95 <= value <= 1.05


def test_expanding_percentile_requires_60_prior_and_no_forced_one():
    atr_pct = pd.Series([0.03] * 60 + [0.02, 0.05, 0.04, 0.06] + [0.03] * 16, dtype=float)
    out = _expanding_percentile(atr_pct, min_history=60)

    assert out.iloc[59] is np.nan or np.isnan(out.iloc[59])
    assert not np.isnan(out.iloc[60])
    assert float(out.iloc[60]) < 1.0


def test_cmf_definition_three_bar_balanced_flow_is_zero():
    # mfm sequence is [+1, -1, 0], so MFV sum is zero and CMF should be zero.
    df = pd.DataFrame(
        {
            "high": [10.0, 10.0, 10.0],
            "low": [0.0, 0.0, 0.0],
            "close": [10.0, 0.0, 5.0],
            "volume": [100.0, 100.0, 100.0],
        }
    )
    out = _cmf(df, period=3)
    assert abs(float(out.iloc[-1])) <= 1e-9


def test_cmf_definition_three_bar_positive_one_third():
    # mfm sequence is [+1, 0, 0], MFV sum is 100 over total volume 300 => 1/3.
    df = pd.DataFrame(
        {
            "high": [10.0, 10.0, 10.0],
            "low": [0.0, 0.0, 0.0],
            "close": [10.0, 5.0, 5.0],
            "volume": [100.0, 100.0, 100.0],
        }
    )
    out = _cmf(df, period=3)
    assert abs(float(out.iloc[-1]) - (1.0 / 3.0)) <= 1e-9

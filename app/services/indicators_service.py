"""Technical indicator computation using TA-Lib.

Uses the same C library (TA-Lib) that TickerChart Live's desktop app bundles
(RTSoft.SDK.MS.Static.dll), so values match the desktop charts bit-for-bit
on identical input.

Input:  list of OHLCV dicts as returned by tickerchart_service.fetch_ohlcv
Output: same list, each row augmented with indicator values (or None for
        warmup periods where the indicator is undefined).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import talib


def _to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    o = np.array([float(r["open"]) for r in rows], dtype=float)
    h = np.array([float(r["high"]) for r in rows], dtype=float)
    l = np.array([float(r["low"]) for r in rows], dtype=float)
    c = np.array([float(r["close"]) for r in rows], dtype=float)
    v = np.array([float(r["volume"]) for r in rows], dtype=float)
    return o, h, l, c, v


def _none_if_nan(x: float) -> Optional[float]:
    return None if not np.isfinite(x) else float(x)


def attach_indicators(rows: list[dict]) -> list[dict]:
    """Return a new list where each row has indicator fields attached.

    Indicators (TA-Lib defaults / standard settings):
      - sma_20, sma_50, sma_200
      - ema_20, ema_50
      - rsi_14
      - macd, macd_signal, macd_hist  (12, 26, 9)
      - bb_upper, bb_middle, bb_lower (20, 2)
      - atr_14
      - adx_14
      - stoch_k, stoch_d  (14, 3, 3)
      - cmf_20  (Chaikin Money Flow — TA-Lib has no CMF; computed manually)
      - ad_line  (Chaikin A/D line)
      - obv
      - vwap_session  (rolling cumulative VWAP)
    """
    if not rows:
        return rows

    o, h, l, c, v = _to_arrays(rows)
    n = len(c)

    sma_20 = talib.SMA(c, timeperiod=20)
    sma_50 = talib.SMA(c, timeperiod=50)
    sma_200 = talib.SMA(c, timeperiod=200)
    ema_20 = talib.EMA(c, timeperiod=20)
    ema_50 = talib.EMA(c, timeperiod=50)
    rsi_14 = talib.RSI(c, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    atr_14 = talib.ATR(h, l, c, timeperiod=14)
    adx_14 = talib.ADX(h, l, c, timeperiod=14)
    stoch_k, stoch_d = talib.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    ad_line = talib.AD(h, l, c, v)
    obv = talib.OBV(c, v)

    # Chaikin Money Flow (CMF, period 20). TA-Lib has ADOSC (oscillator) but
    # not CMF directly. Standard formula:
    #   MFM = ((C-L) - (H-C)) / (H-L)         (0 if H==L)
    #   MFV = MFM * V
    #   CMF = sum(MFV, period) / sum(V, period)
    hl_range = h - l
    safe_range = np.where(hl_range == 0, 1.0, hl_range)
    mfm = np.where(hl_range == 0, 0.0, ((c - l) - (h - c)) / safe_range)
    mfv = mfm * v
    cmf_20 = np.full(n, np.nan)
    period = 20
    if n >= period:
        cumsum_mfv = np.cumsum(mfv)
        cumsum_v = np.cumsum(v)
        for i in range(period - 1, n):
            window_mfv = cumsum_mfv[i] - (cumsum_mfv[i - period] if i >= period else 0.0)
            window_v = cumsum_v[i] - (cumsum_v[i - period] if i >= period else 0.0)
            cmf_20[i] = window_mfv / window_v if window_v > 0 else 0.0

    # Cumulative VWAP from the beginning of the returned series.
    typical = (h + l + c) / 3.0
    cum_pv = np.cumsum(typical * v)
    cum_v = np.cumsum(v)
    vwap_cum = np.where(cum_v > 0, cum_pv / np.where(cum_v == 0, 1.0, cum_v), np.nan)

    out: list[dict] = []
    for i, row in enumerate(rows):
        out.append({
            **row,
            "sma_20": _none_if_nan(sma_20[i]),
            "sma_50": _none_if_nan(sma_50[i]),
            "sma_200": _none_if_nan(sma_200[i]),
            "ema_20": _none_if_nan(ema_20[i]),
            "ema_50": _none_if_nan(ema_50[i]),
            "rsi_14": _none_if_nan(rsi_14[i]),
            "macd": _none_if_nan(macd[i]),
            "macd_signal": _none_if_nan(macd_signal[i]),
            "macd_hist": _none_if_nan(macd_hist[i]),
            "bb_upper": _none_if_nan(bb_upper[i]),
            "bb_middle": _none_if_nan(bb_middle[i]),
            "bb_lower": _none_if_nan(bb_lower[i]),
            "atr_14": _none_if_nan(atr_14[i]),
            "adx_14": _none_if_nan(adx_14[i]),
            "stoch_k": _none_if_nan(stoch_k[i]),
            "stoch_d": _none_if_nan(stoch_d[i]),
            "cmf_20": _none_if_nan(cmf_20[i]),
            "ad_line": _none_if_nan(ad_line[i]),
            "obv": _none_if_nan(obv[i]),
            "vwap": _none_if_nan(vwap_cum[i]),
        })
    return out

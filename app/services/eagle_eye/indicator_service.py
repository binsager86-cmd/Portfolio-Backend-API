from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.core.database import exec_sql, query_one
from app.services.eagle_eye.market_data_service import CONCEPT_VERSION, load_symbol_ohlcv


@dataclass
class IndicatorResult:
    symbol: str
    trade_date: int
    payload: dict[str, Any]


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def _norm_lr_slope(series: pd.Series, lookback: int) -> pd.Series:
    def _fit(values: np.ndarray) -> float:
        y = np.asarray(values, dtype=float)
        if len(y) < lookback or np.any(~np.isfinite(y)):
            return np.nan
        x = np.arange(lookback, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        fit_start = float(intercept)
        fit_end = float(intercept + slope * (lookback - 1))
        mean_abs = float(np.mean(np.abs(y))) if len(y) else 1.0
        denom = max(abs(fit_start), 1e-9 * max(mean_abs, 1.0))
        return float((fit_end - fit_start) / denom)

    return series.rolling(lookback).apply(lambda v: _fit(np.asarray(v, dtype=float)), raw=True)


def _norm_flow_slope(flow_series: pd.Series, scale_series: pd.Series, lookback: int) -> pd.Series:
    def _fit(flow_values: np.ndarray, scale_values: np.ndarray) -> float:
        flow = np.asarray(flow_values, dtype=float)
        scale = np.asarray(scale_values, dtype=float)
        if len(flow) < lookback or len(scale) < lookback:
            return np.nan
        if np.any(~np.isfinite(flow)) or np.any(~np.isfinite(scale)):
            return np.nan

        mean_scale = float(np.mean(scale))
        denom = max(mean_scale * lookback, 1e-9)
        normalized_flow = flow / denom
        x = np.arange(lookback, dtype=float)
        slope = float(np.polyfit(x, normalized_flow, 1)[0])
        return float(slope * (lookback - 1))

    return pd.Series(
        [
            _fit(
                flow_series.iloc[max(0, i - lookback + 1) : i + 1].to_numpy(dtype=float),
                scale_series.iloc[max(0, i - lookback + 1) : i + 1].to_numpy(dtype=float),
            )
            if i + 1 >= lookback
            else np.nan
            for i in range(len(flow_series))
        ],
        index=flow_series.index,
        dtype=float,
    )


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 19) -> tuple[pd.Series, pd.Series, pd.Series]:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx, plus_di, minus_di


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast = _ema(close, 12)
    slow = _ema(close, 26)
    line = fast - slow
    signal = _ema(line, 9)
    hist = line - signal
    return line, signal, hist


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume.fillna(0.0)).cumsum()


def _cmf(df: pd.DataFrame, period: int = 10) -> pd.Series:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"].fillna(0.0)
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfv = mfm.fillna(0.0) * volume
    return mfv.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)


def _stoch_slow(df: pd.DataFrame, k_period: int = 14, smooth_k: int = 3, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    low_k = df["low"].rolling(k_period).min()
    high_k = df["high"].rolling(k_period).max()
    fast_k = 100 * (df["close"] - low_k) / (high_k - low_k).replace(0, np.nan)
    slow_k = fast_k.rolling(smooth_k).mean()
    slow_d = slow_k.rolling(d_period).mean()
    return slow_k, slow_d


def _cci(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(period).mean()
    mad = (tp - sma).abs().rolling(period).mean().replace(0, np.nan)
    return (tp - sma) / (0.015 * mad)


def _expanding_percentile(series: pd.Series, min_history: int = 60) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    for i in range(len(values)):
        current = values[i]
        if not np.isfinite(current):
            continue
        prior = values[:i]
        prior = prior[np.isfinite(prior)]
        if len(prior) < min_history:
            continue
        out[i] = float(np.sum(prior <= current) / len(prior))
    return pd.Series(out, index=series.index, dtype=float)


def _pivot_high(series: pd.Series, window: int = 5) -> pd.Series:
    # Causal pivot approximation: avoids future leakage by using only past bars.
    roll_max = series.rolling(window).max()
    return series.where(series == roll_max)


def compute_symbol_indicators(symbol: str) -> list[IndicatorResult]:
    rows = load_symbol_ohlcv(symbol)
    if not rows:
        return []

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_numeric(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"]).sort_values("trade_date")
    for col in ["open", "high", "low", "close", "volume", "value_kwd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["close"]
    volume = df["volume"].fillna(0.0)

    ema10 = _ema(close, 10)
    ema30 = _ema(close, 30)
    sma200 = _sma(close, 200)
    ema10_slope = _norm_lr_slope(ema10, 20)
    ema30_slope = _norm_lr_slope(ema30, 20)
    sma200_slope = _norm_lr_slope(sma200, 20)

    rsi14 = _rsi(close, 14)
    adx19, plus_di, minus_di = _adx(df, 19)
    macd_line, macd_signal, macd_hist = _macd(close)
    stoch_k, stoch_d = _stoch_slow(df, 14, 3, 3)
    cci14 = _cci(df, 14)
    atr14 = _atr(df, 14)
    atr_pct = atr14 / close.replace(0, np.nan)

    obv = _obv(close, volume)
    cmf10 = _cmf(df, 10)
    signed = np.sign(close.diff().fillna(0.0))
    anv = (signed * close.fillna(0.0) * volume).cumsum()
    lnv = signed * close.fillna(0.0) * volume

    vol_sma20 = volume.rolling(20).mean()
    rel_volume = volume / vol_sma20.replace(0, np.nan)

    range_high_60 = close.rolling(60).max().shift(1)
    range_low_60 = close.rolling(60).min().shift(1)
    range_high_120 = close.rolling(120).max().shift(1)
    range_low_120 = close.rolling(120).min().shift(1)
    range_width_pct = (range_high_60 - range_low_60) / range_low_60.replace(0, np.nan)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    atr_pct_pctile_252 = _expanding_percentile(atr_pct, 60)

    price_slope = _norm_lr_slope(close, 40)
    obv_slope = _norm_flow_slope(obv, volume, 40)
    anv_slope = _norm_flow_slope(anv, df["value_kwd"].fillna(0.0), 40)

    accumulation_div = (price_slope.abs() < 0.02) & ((obv_slope > 0.10) | (anv_slope > 0.10))

    ph = _pivot_high(close, 5)
    oh = _pivot_high(obv, 5)
    rh = _pivot_high(rsi14, 5)
    price_hh = ph.ffill() > ph.shift(20).ffill()
    obv_lh = oh.ffill() < oh.shift(20).ffill()
    rsi_lh = rh.ffill() < rh.shift(20).ffill()
    distribution_div = price_hh & (obv_lh | rsi_lh)

    high_252 = close.rolling(252).max()
    low_252 = close.rolling(252).min()
    dist_52w_high = (high_252 - close) / high_252.replace(0, np.nan)
    dist_52w_low = (close - low_252) / low_252.replace(0, np.nan)

    days_since_52w_high = pd.Series(index=df.index, dtype=float)
    last_high_ix = -1
    for i in range(len(df)):
        win_start = max(0, i - 251)
        win = close.iloc[win_start : i + 1]
        if len(win) > 0 and close.iloc[i] >= win.max():
            last_high_ix = i
        days_since_52w_high.iloc[i] = np.nan if last_high_ix < 0 else float(i - last_high_ix)

    results: list[IndicatorResult] = []
    for i, row in df.iterrows():
        payload = {
            "open": float(row.get("open") or 0.0),
            "high": float(row.get("high") or 0.0),
            "low": float(row.get("low") or 0.0),
            "close": float(row.get("close") or 0.0),
            "volume": float(row.get("volume") or 0.0),
            "value_kwd": float(row.get("value_kwd") or 0.0),
            "sma200": _f(sma200.iloc[i]),
            "ema30": _f(ema30.iloc[i]),
            "ema10": _f(ema10.iloc[i]),
            "ema10_slope": _f(ema10_slope.iloc[i]),
            "ema30_slope": _f(ema30_slope.iloc[i]),
            "sma200_slope": _f(sma200_slope.iloc[i]),
            "rsi_14": _f(rsi14.iloc[i]),
            "adx_19": _f(adx19.iloc[i]),
            "plus_di": _f(plus_di.iloc[i]),
            "minus_di": _f(minus_di.iloc[i]),
            "macd_line": _f(macd_line.iloc[i]),
            "macd_signal": _f(macd_signal.iloc[i]),
            "macd_hist": _f(macd_hist.iloc[i]),
            "slow_stoch_k": _f(stoch_k.iloc[i]),
            "slow_stoch_d": _f(stoch_d.iloc[i]),
            "cci_14": _f(cci14.iloc[i]),
            "atr_14": _f(atr14.iloc[i]),
            "atr_pct": _f(atr_pct.iloc[i]),
            "obv": _f(obv.iloc[i]),
            "cmf_10": _f(cmf10.iloc[i]),
            "accumulated_net_value": _f(anv.iloc[i]),
            "liquidity_net_value": _f(lnv.iloc[i]),
            "vol_sma20": _f(vol_sma20.iloc[i]),
            "rel_volume": _f(rel_volume.iloc[i]),
            "range_high_60": _f(range_high_60.iloc[i]),
            "range_low_60": _f(range_low_60.iloc[i]),
            "range_high_120": _f(range_high_120.iloc[i]),
            "range_low_120": _f(range_low_120.iloc[i]),
            "range_width_pct": _f(range_width_pct.iloc[i]),
            "bb_width": _f(bb_width.iloc[i]),
            "atr_pct_percentile_252": _f(atr_pct_pctile_252.iloc[i]),
            "price_slope_40": _f(price_slope.iloc[i]),
            "obv_slope_40": _f(obv_slope.iloc[i]),
            "anv_slope_40": _f(anv_slope.iloc[i]),
            "accumulation_divergence": bool(accumulation_div.iloc[i]) if pd.notna(accumulation_div.iloc[i]) else False,
            "distribution_divergence": bool(distribution_div.iloc[i]) if pd.notna(distribution_div.iloc[i]) else False,
            "dist_52w_high": _f(dist_52w_high.iloc[i]),
            "dist_52w_low": _f(dist_52w_low.iloc[i]),
            "days_since_52w_high": _f(days_since_52w_high.iloc[i]),
        }
        results.append(
            IndicatorResult(
                symbol=symbol,
                trade_date=int(row["trade_date"]),
                payload=payload,
            )
        )

    return results


def store_indicator_results(results: list[IndicatorResult]) -> None:
    for row in results:
        exec_sql(
            """
            INSERT INTO ee_indicators (symbol, trade_date, payload_json, concept_version)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol, trade_date) DO UPDATE SET
                payload_json = excluded.payload_json,
                concept_version = excluded.concept_version
            """,
            (
                row.symbol,
                row.trade_date,
                json.dumps(row.payload, ensure_ascii=True, separators=(",", ":")),
                CONCEPT_VERSION,
            ),
        )


def compute_and_store_symbol(symbol: str) -> int:
    results = compute_symbol_indicators(symbol)
    store_indicator_results(results)
    return len(results)


def load_latest_indicator(symbol: str, trade_date: int | None = None) -> dict[str, Any] | None:
    if trade_date is None:
        row = query_one(
            "SELECT trade_date, payload_json FROM ee_indicators WHERE symbol = ? ORDER BY trade_date DESC LIMIT 1",
            (symbol,),
        )
    else:
        row = query_one(
            "SELECT trade_date, payload_json FROM ee_indicators WHERE symbol = ? AND trade_date = ?",
            (symbol, trade_date),
        )
    if not row:
        return None
    payload = json.loads(str(row.get("payload_json") or "{}"))
    payload["trade_date"] = int(row.get("trade_date") or 0)
    return payload


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if np.isnan(out) or np.isinf(out):
        return None
    return out

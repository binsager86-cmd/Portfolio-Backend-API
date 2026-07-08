from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.eagle_eye import indicator_service as ee_indicator_service


OUT_DIR = Path(__file__).resolve().parent
START_DATE = "2021-01-01"
MASTER_SEED = 20260704
NOISE_SIGMA = 0.004  # 0.4%
PREFIX_SESSIONS = 220


@dataclass
class SeriesBundle:
    close: np.ndarray
    volume: np.ndarray
    range_frac: np.ndarray
    accumulation_windows: list[tuple[int, int]] = field(default_factory=list)
    segments: dict[str, tuple[int, int]] = field(default_factory=dict)


def _neutral_prefix(
    rng: np.random.Generator,
    n: int,
    start_price: float,
    base_vol: float,
    range_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shocks = rng.normal(0.0, 0.006, size=n)
    log_path = np.cumsum(shocks)
    # Remove end drift so total move remains near-flat over the prefix.
    log_path = log_path - np.linspace(0.0, float(log_path[-1]), n)
    close = np.maximum(start_price * np.exp(log_path), 0.1)
    volume = base_vol * rng.uniform(0.85, 1.15, size=n)
    range_frac = np.full(n, range_level)
    return close, volume, range_frac


def _prepend_prefix(
    rng: np.random.Generator,
    close: np.ndarray,
    volume: np.ndarray,
    range_frac: np.ndarray,
    accumulation_windows: list[tuple[int, int]],
    segments: dict[str, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], dict[str, tuple[int, int]]]:
    pref_close, pref_vol, pref_range = _neutral_prefix(
        rng,
        PREFIX_SESSIONS,
        float(close[0]) * 0.98,
        float(np.mean(volume)),
        float(np.median(range_frac)),
    )
    scale = float(pref_close[-1]) / max(float(close[0]), 1e-9)
    scaled_close = close * scale
    out_close = np.concatenate([pref_close, scaled_close])
    out_volume = np.concatenate([pref_vol, volume])
    out_range = np.concatenate([pref_range, range_frac])
    shifted_windows = [(s + PREFIX_SESSIONS, e + PREFIX_SESSIONS) for s, e in accumulation_windows]
    shifted_segments = {
        "prefix": (0, PREFIX_SESSIONS),
        **{name: (start + PREFIX_SESSIONS, end + PREFIX_SESSIONS) for name, (start, end) in segments.items()},
    }
    return out_close, out_volume, out_range, shifted_windows, shifted_segments


def _segment_entry(df: pd.DataFrame, start: int, end_exclusive: int) -> dict[str, int]:
    end_inclusive = max(start, end_exclusive - 1)
    d_start = pd.to_datetime(df.iloc[start]["date"], format="%d/%m/%Y", dayfirst=True)
    d_end = pd.to_datetime(df.iloc[end_inclusive]["date"], format="%d/%m/%Y", dayfirst=True)
    ts_start = int(pd.Timestamp(d_start, tz="UTC").timestamp())
    ts_end = int(pd.Timestamp(d_end, tz="UTC").timestamp())
    return {
        "bar_start": int(start),
        "bar_end": int(end_inclusive),
        "trade_date_start": ts_start,
        "trade_date_end": ts_end,
    }


def _assert_warmup_alignment(df: pd.DataFrame, symbol: str, first_pattern_idx: int, is_joiner: bool = False) -> None:
    close = df["close"].to_numpy(dtype=float)
    close_s = pd.Series(close, dtype=float)
    ema10 = ee_indicator_service._ema(close_s, 10).to_numpy(dtype=float)
    ema30 = ee_indicator_service._ema(close_s, 30).to_numpy(dtype=float)
    sma200 = ee_indicator_service._sma(close_s, 200).to_numpy(dtype=float)
    range_low_120 = close_s.rolling(120).min().shift(1).to_numpy(dtype=float)
    range_high_120 = close_s.rolling(120).max().shift(1).to_numpy(dtype=float)

    ready = np.where(
        (~np.isnan(sma200))
        & (~np.isnan(range_low_120))
        & (~np.isnan(range_high_120))
        & (range_low_120 > 0)
        & (range_high_120 > 0)
    )[0]
    assert len(ready) > 0, f"{symbol}: warmup_ready_date not found"
    warmup_idx = int(ready[0])

    if not is_joiner:
        assert warmup_idx < first_pattern_idx, (
            f"{symbol}: warmup_ready ({warmup_idx}) must precede first pattern segment ({first_pattern_idx})"
        )
        return

    # JOINER requirement: join conditions hold on >=5 bars inside warmup window.
    end = min(len(close), warmup_idx + 41)
    hits = 0
    for i in range(warmup_idx, end):
        if close[i] > sma200[i] and ema10[i] > ema30[i] and close[i] >= (range_low_120[i] * 1.15):
            hits += 1
    assert hits >= 5, f"{symbol}: expected >=5 join-eligible bars in [warmup, warmup+40], got {hits}"


def _business_dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(START_DATE, periods=n)


def _apply_close_noise(close: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, NOISE_SIGMA, size=len(close))
    out = close * (1.0 + noise)
    return np.maximum(out, 0.1)


def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    direction = np.sign(d)
    return np.cumsum(direction * volume)


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(series, dtype=float).ewm(span=span, adjust=False).mean().to_numpy()


def _rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    s = pd.Series(series, dtype=float)
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.bfill().fillna(50.0).to_numpy()


def _norm_lr_fractional_change(series: np.ndarray) -> float:
    lookback = len(series)
    y = np.asarray(series, dtype=float)
    x = np.arange(lookback, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fit_start = float(intercept)
    fit_end = float(intercept + slope * (lookback - 1))
    denom = max(abs(fit_start), 1e-9 * max(abs(y.mean()), 1.0))
    return float((fit_end - fit_start) / denom)


def _norm_flow_window(flow_cumsum: np.ndarray, scale_window: np.ndarray) -> float:
    lookback = len(flow_cumsum)
    mean_scale = float(np.mean(scale_window))
    denom = max(mean_scale * lookback, 1e-9)
    normalized = flow_cumsum / denom
    x = np.arange(lookback, dtype=float)
    slope = float(np.polyfit(x, normalized, 1)[0])
    return float(slope * (lookback - 1))


def _in_any_window(index: int, windows: list[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in windows)


def _cmf_from_df(df: pd.DataFrame, period: int = 10) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    rng = (high - low).replace(0.0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfm = mfm.fillna(0.0)
    mfv = mfm * vol
    den = vol.rolling(period).sum().replace(0.0, np.nan)
    return (mfv.rolling(period).sum() / den).fillna(0.0)


def _assert_accumulation_cmf_window(df: pd.DataFrame, window: tuple[int, int], symbol: str) -> None:
    start, end = window
    cmf_10 = _cmf_from_df(df, period=10).to_numpy()
    hit = False
    for i in range(start + 9, end):
        left = max(start, i - 9)
        right = i + 1
        hits = int(np.sum(cmf_10[left:right] > 0.05))
        if hits >= 5:
            hit = True
            break
    assert hit, (
        f"{symbol}: accumulation CMF self-check failed for window {window}; "
        "expected cmf_10 > 0.05 on >=5 of a trailing-10 slice"
    )


def _assert_breakout_crossing(df: pd.DataFrame, symbol: str, base_window: tuple[int, int], breakout_window: tuple[int, int]) -> None:
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)
    rv = vol / pd.Series(vol).rolling(20).mean().to_numpy()

    base_start, base_end = base_window
    brk_start, brk_end = breakout_window
    frozen_candidate = float(np.nanmax(high[base_start:base_end]))
    breakout_close = close[brk_start:brk_end]
    breakout_rv = rv[brk_start:brk_end]

    assert len(breakout_close) > 0, f"{symbol}: breakout segment empty"
    assert np.all(breakout_close > (frozen_candidate * 1.005)), (
        f"{symbol}: breakout close failed frozen-base test against {frozen_candidate:.3f}"
    )
    assert int(np.sum(breakout_rv >= 2.5)) >= 2, (
        f"{symbol}: breakout rel_volume >= 2.5 on fewer than 2 bars"
    )


def _ohlcv_from_close(
    symbol: str,
    close: np.ndarray,
    volume: np.ndarray,
    range_frac: np.ndarray,
    accumulation_windows: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    dates = _business_dates(len(close))
    close = close.copy()
    seen: set[float] = set()
    for i in range(len(close)):
        v = round(float(close[i]), 3)
        if v in seen:
            close[i] = close[i] + ((i + 1) * 1e-4)
            v = round(float(close[i]), 3)
        seen.add(v)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    open_px = (prev_close + close) / 2.0
    windows = accumulation_windows or []
    high = np.zeros(len(close), dtype=float)
    low = np.zeros(len(close), dtype=float)

    for i in range(len(close)):
        o = float(open_px[i])
        c = float(close[i])
        r = float(range_frac[i])
        if _in_any_window(i, windows):
            # Accumulation realism rule:
            # up-day close in upper 40% of range; down-day close at mid-range or higher.
            up_day = c >= float(prev_close[i])
            target_pos = 0.62 if up_day else 0.52
            span = max(abs(c), abs(o), 1.0) * max(r, 0.004) * 2.0
            low_i = c - (target_pos * span)
            high_i = c + ((1.0 - target_pos) * span)
            if o > high_i:
                high_i = o + (0.05 * span)
            if o < low_i:
                low_i = o - (0.05 * span)
        else:
            high_i = max(o, c) * (1.0 + r)
            low_i = min(o, c) * max(0.5, (1.0 - r))

        high[i] = max(high_i, o, c)
        low[i] = min(low_i, o, c)
        if low[i] <= 0:
            low[i] = max(0.01, min(o, c) * 0.95)
    value = close * volume

    df = pd.DataFrame(
        {
            "date": dates.strftime("%d/%m/%Y"),
            "open": np.round(open_px, 3),
            "high": np.round(high, 3),
            "low": np.round(low, 3),
            "close": np.round(close, 3),
            "volume": np.round(volume, 0),
            "value": np.round(value, 3),
        }
    )
    assert len(df) >= 320, f"{symbol}: expected at least 320 sessions, got {len(df)}"
    assert not df["close"].duplicated().any(), f"{symbol}: close contains repeated exact values"
    return df


def _build_tijara(rng: np.random.Generator) -> SeriesBundle:
    pre = np.linspace(86.0, 92.0, 40) + 1.4 * np.sin(np.linspace(0, 6.2, 40))
    base_early = 90.0 + 8.5 * np.sin(np.linspace(0, 8.0, 100))
    base_late = 98.0 + 1.2 * np.sin(np.linspace(0, 4.0 * np.pi, 40))
    base = np.concatenate([base_early, base_late])
    breakout = np.array([101.2, 106.4])
    markup = np.linspace(107.0, 185.0, 90)
    markup[24:27] -= np.array([4.0, 5.5, 2.0])
    markup[56:59] -= np.array([4.5, 6.0, 3.0])
    drift = np.linspace(184.0, 188.0, 60)
    tail = 188.0 + 1.6 * np.sin(np.linspace(0, 8.0, 30))
    close = np.concatenate([pre, base, breakout, markup, drift, tail])
    close = _apply_close_noise(close, rng)

    base_vol = 120_000.0
    volume = np.full(len(close), base_vol)
    range_frac = np.full(len(close), 0.012)

    base_start = len(pre)
    base_end = base_start + len(base)
    for i in range(base_end - 40, base_end):
        up_day = close[i] >= close[i - 1]
        volume[i] = base_vol * rng.uniform(1.35, 1.65) if up_day else base_vol * rng.uniform(0.65, 0.90)

    # Force two-of-five elevated bars right before breakout to trigger BREAKOUT_WATCH.
    volume[base_end - 2] = base_vol * 1.9
    volume[base_end - 1] = base_vol * 1.9

    brk_i = base_end
    volume[brk_i] = base_vol * 5.5
    volume[brk_i + 1] = base_vol * 5.0
    range_frac[brk_i] = 0.018
    range_frac[brk_i + 1] = 0.019

    for k in [brk_i + 2, brk_i + 5, brk_i + 9]:
        if k < len(volume):
            volume[k] = base_vol * 3.4
            range_frac[k] = 0.010

    volume[base_end + 24] = base_vol * 1.8
    volume[base_end + 56] = base_vol * 1.8

    close, volume, range_frac, windows, segments = _prepend_prefix(
        rng,
        close,
        volume,
        range_frac,
        [(base_end - 40, base_end)],
        {
            "pre": (0, len(pre)),
            "base": (len(pre), len(pre) + len(base)),
            "breakout": (len(pre) + len(base), len(pre) + len(base) + len(breakout)),
            "markup": (len(pre) + len(base) + len(breakout), len(pre) + len(base) + len(breakout) + len(markup)),
            "drift": (len(pre) + len(base) + len(breakout) + len(markup), len(pre) + len(base) + len(breakout) + len(markup) + len(drift)),
            "tail": (len(pre) + len(base) + len(breakout) + len(markup) + len(drift), len(close)),
        },
    )
    return SeriesBundle(close=close, volume=volume, range_frac=range_frac, accumulation_windows=windows, segments=segments)


def _build_bpcc(rng: np.random.Generator) -> SeriesBundle:
    decline = np.linspace(700.0, 560.0, 120)
    base = 585.0 + 23.0 * np.sin(np.linspace(0, 10.0, 80))
    breakout = np.array([628.0, 636.0, 644.0])
    markup = np.linspace(650.0, 760.0, 70)
    tail = 758.0 + 6.0 * np.sin(np.linspace(0, 7.0, 60))
    close = np.concatenate([decline, base, breakout, markup, tail])
    close = _apply_close_noise(close, rng)

    base_vol = 95_000.0
    volume = np.full(len(close), base_vol)
    range_frac = np.full(len(close), 0.012)

    base_start = len(decline)
    base_end = base_start + len(base)
    for i in range(base_start, base_end):
        up_day = close[i] >= close[i - 1]
        volume[i] = base_vol * rng.uniform(1.35, 1.70) if up_day else base_vol * rng.uniform(0.65, 0.90)

    volume[base_end - 2] = base_vol * 1.85
    volume[base_end - 1] = base_vol * 1.85

    brk_i = base_end
    volume[brk_i] = base_vol * 4.2
    volume[brk_i + 1] = base_vol * 6.0
    volume[brk_i + 2] = base_vol * 5.5
    range_frac[brk_i] = 0.017
    range_frac[brk_i + 1] = 0.018
    range_frac[brk_i + 2] = 0.019

    for k in [brk_i + 4, brk_i + 8, brk_i + 12, brk_i + 16]:
        if k < len(volume):
            volume[k] = base_vol * 3.3
            range_frac[k] = 0.010

    close, volume, range_frac, windows, segments = _prepend_prefix(
        rng,
        close,
        volume,
        range_frac,
        [(base_start, base_end)],
        {
            "decline": (0, len(decline)),
            "base": (len(decline), len(decline) + len(base)),
            "breakout": (len(decline) + len(base), len(decline) + len(base) + len(breakout)),
            "markup": (len(decline) + len(base) + len(breakout), len(decline) + len(base) + len(breakout) + len(markup)),
            "tail": (len(decline) + len(base) + len(breakout) + len(markup), len(close)),
        },
    )
    return SeriesBundle(close=close, volume=volume, range_frac=range_frac, accumulation_windows=windows, segments=segments)


def _build_zain(rng: np.random.Generator) -> SeriesBundle:
    rise = np.linspace(470.0, 510.0, 30)
    base = 515.0 + 1.5 * np.sin(np.linspace(0, 10.0, 110))
    breakout = np.array([531.0, 537.0])
    markup = np.linspace(541.0, 609.0, 100)
    for j in [20, 46, 74]:
        markup[j : j + 3] -= np.array([5.5, 7.0, 2.5])
    tail = 610.0 + 4.0 * np.sin(np.linspace(0, 9.0, 90))
    close = np.concatenate([rise, base, breakout, markup, tail])
    close = _apply_close_noise(close, rng)

    base_start = len(rise)
    base_end = base_start + len(base)
    for _ in range(3):
        for i in range(base_start + 39, base_end):
            anchor = close[i - 39]
            upper = anchor * 1.014
            lower = anchor * 0.986
            close[i] = min(max(close[i], lower), upper)

    base_vol = 110_000.0
    volume = np.full(len(close), base_vol)
    range_frac = np.full(len(close), 0.011)

    volume[base_end - 2] = base_vol * 1.9
    volume[base_end - 1] = base_vol * 1.9

    brk_i = len(rise) + len(base)
    volume[brk_i] = base_vol * 4.0
    volume[brk_i + 1] = base_vol * 3.8
    range_frac[brk_i] = 0.016
    range_frac[brk_i + 1] = 0.017

    for k in [brk_i + 4, brk_i + 8, brk_i + 12]:
        if k < len(volume):
            volume[k] = base_vol * 3.2
            range_frac[k] = 0.009
            close[k] = close[k] + 3.0

    for j in [len(rise) + len(base) + 20, len(rise) + len(base) + 46, len(rise) + len(base) + 74]:
        volume[j] = base_vol * 1.8

    close, volume, range_frac, windows, segments = _prepend_prefix(
        rng,
        close,
        volume,
        range_frac,
        [(base_start, base_end)],
        {
            "rise": (0, len(rise)),
            "base": (len(rise), len(rise) + len(base)),
            "breakout": (len(rise) + len(base), len(rise) + len(base) + len(breakout)),
            "markup": (len(rise) + len(base) + len(breakout), len(rise) + len(base) + len(breakout) + len(markup)),
            "tail": (len(rise) + len(base) + len(breakout) + len(markup), len(close)),
        },
    )
    return SeriesBundle(close=close, volume=volume, range_frac=range_frac, accumulation_windows=windows, segments=segments)


def _build_sanam(rng: np.random.Generator) -> SeriesBundle:
    base_early = 205.0 + 8.5 * np.sin(np.linspace(0, 8.0, 60))
    base_late = 210.0 + 1.4 * np.sin(np.linspace(0, 4.0 * np.pi, 40))
    base = np.concatenate([base_early, base_late])
    breakout = np.array([222.0, 232.0])
    markup = np.linspace(228.0, 333.0, 120)
    tail = np.linspace(334.0, 338.0, 120)
    close = np.concatenate([base, breakout, markup, tail])
    close = _apply_close_noise(close, rng)

    base_vol = 100_000.0
    volume = np.full(len(close), base_vol)
    range_frac = np.full(len(close), 0.015)
    range_frac[50:100] = 0.0105

    base_start = 0
    base_end = len(base)
    for i in range(base_end - 40, base_end):
        up_day = close[i] >= close[max(0, i - 1)]
        volume[i] = base_vol * rng.uniform(1.35, 1.70) if up_day else base_vol * rng.uniform(0.65, 0.90)

    for i in range(base_end - 40, base_end):
        up_day = close[i] >= close[max(0, i - 1)]
        volume[i] = base_vol * rng.uniform(1.35, 1.70) if up_day else base_vol * rng.uniform(0.65, 0.90)

    brk_i = len(base)
    volume[brk_i] = base_vol * 5.0
    volume[brk_i + 1] = base_vol * 4.8
    range_frac[brk_i] = 0.018
    range_frac[brk_i + 1] = 0.019

    close, volume, range_frac, windows, segments = _prepend_prefix(
        rng,
        close,
        volume,
        range_frac,
        [(base_start, base_end)],
        {
            "base": (0, len(base)),
            "breakout": (len(base), len(base) + len(breakout)),
            "markup": (len(base) + len(breakout), len(base) + len(breakout) + len(markup)),
            "tail": (len(base) + len(breakout) + len(markup), len(close)),
        },
    )
    return SeriesBundle(close=close, volume=volume, range_frac=range_frac, accumulation_windows=windows, segments=segments)


def _build_mabanee(rng: np.random.Generator) -> SeriesBundle:
    base = 882.0 + 13.0 * np.sin(np.linspace(0, 8.0 * np.pi, 160))
    breakout = np.array([912.0, 926.0, 940.0])
    markup = np.linspace(940.0, 1160.0, 90)
    top = 1150.0 + 12.0 * np.sin(np.linspace(0, 3.5 * np.pi, 24))
    decline = np.linspace(1138.0, 920.0, 90)
    tail = 925.0 + 6.0 * np.sin(np.linspace(0, 3.0 * np.pi, 40))
    close = np.concatenate([base, breakout, markup, top, decline, tail])
    close = _apply_close_noise(close, rng)

    base_vol = 95_000.0
    volume = np.full(len(close), base_vol)
    range_frac = np.full(len(close), 0.012)

    base_end = len(base)
    for i in range(base_end - 40, base_end):
        up_day = close[i] >= close[i - 1]
        volume[i] = base_vol * rng.uniform(1.35, 1.65) if up_day else base_vol * rng.uniform(0.70, 0.92)

    brk_i = base_end
    volume[brk_i] = base_vol * 4.8
    volume[brk_i + 1] = base_vol * 4.5
    volume[brk_i + 2] = base_vol * 5.2
    range_frac[brk_i : brk_i + 3] = np.array([0.020, 0.021, 0.018])

    top_start = len(base) + len(breakout) + len(markup)
    for i in range(top_start, top_start + len(top)):
        swing_idx = i - top_start
        fade = 1.0 - (0.15 * (swing_idx // 8))
        up_day = close[i] >= close[i - 1]
        if up_day:
            volume[i] = base_vol * max(0.65, fade)
        else:
            volume[i] = base_vol * max(0.85, fade) * 1.6

    climax_i = top_start + (len(top) // 2)
    volume[climax_i] = base_vol * 6.5
    range_frac[climax_i] = 0.036

    decline_start = len(base) + len(breakout) + len(markup) + len(top)
    for i in range(decline_start, len(close)):
        volume[i] = base_vol * rng.uniform(1.05, 1.35)

    close, volume, range_frac, windows, segments = _prepend_prefix(
        rng,
        close,
        volume,
        range_frac,
        [(0, base_end)],
        {
            "base": (0, len(base)),
            "breakout": (len(base), len(base) + len(breakout)),
            "markup": (len(base) + len(breakout), len(base) + len(breakout) + len(markup)),
            "top": (len(base) + len(breakout) + len(markup), len(base) + len(breakout) + len(markup) + len(top)),
            "decline": (len(base) + len(breakout) + len(markup) + len(top), len(base) + len(breakout) + len(markup) + len(top) + len(decline)),
            "tail": (len(base) + len(breakout) + len(markup) + len(top) + len(decline), len(close)),
        },
    )
    return SeriesBundle(close=close, volume=volume, range_frac=range_frac, accumulation_windows=windows, segments=segments)


def _build_joiner(rng: np.random.Generator) -> SeriesBundle:
    # JOINER prefix must be a clean uptrend so warmup-ready bars satisfy trend-join conditions.
    daily_ret = rng.normal(0.0020, 0.0007, size=PREFIX_SESSIONS)
    prefix = 760.0 * np.exp(np.cumsum(daily_ret))

    early_join = np.linspace(prefix[-1] * 1.10, prefix[-1] * 1.28, 40)
    late_markup = np.linspace(early_join[-1] * 1.01, early_join[-1] * 1.08, 40)
    topping_mid = late_markup[-1]
    topping = topping_mid + (0.012 * topping_mid) * np.sin(np.linspace(0, 3.0 * np.pi, 20))
    decline = np.linspace(topping[-1] * 0.985, topping[-1] * 0.70, 90)
    tail = np.linspace(decline[-1] * 0.99, decline[-1] * 0.94, 30)
    close = np.concatenate([prefix, early_join, late_markup, topping, decline, tail])
    close = _apply_close_noise(close, rng)

    base_vol = 105_000.0
    volume = np.full(len(close), base_vol)
    range_frac = np.full(len(close), 0.011)
    join_start = PREFIX_SESSIONS
    decline_start = PREFIX_SESSIONS + len(early_join) + len(late_markup) + len(topping)
    for i in range(join_start, decline_start):
        volume[i] = base_vol * rng.uniform(1.15, 1.45)
    volume[join_start] = base_vol * 3.6
    volume[join_start + 1] = base_vol * 3.8
    for i in range(decline_start, len(close)):
        volume[i] = base_vol * rng.uniform(1.55, 2.10)
        range_frac[i] = 0.020
    segments = {
        "prefix": (0, PREFIX_SESSIONS),
        "breakout": (PREFIX_SESSIONS, PREFIX_SESSIONS + len(early_join)),
        "markup": (PREFIX_SESSIONS + len(early_join), PREFIX_SESSIONS + len(early_join) + len(late_markup)),
        "top": (
            PREFIX_SESSIONS + len(early_join) + len(late_markup),
            PREFIX_SESSIONS + len(early_join) + len(late_markup) + len(topping),
        ),
        "decline": (
            PREFIX_SESSIONS + len(early_join) + len(late_markup) + len(topping),
            PREFIX_SESSIONS + len(early_join) + len(late_markup) + len(topping) + len(decline),
        ),
        "tail": (
            PREFIX_SESSIONS + len(early_join) + len(late_markup) + len(topping) + len(decline),
            len(close),
        ),
    }
    return SeriesBundle(close=close, volume=volume, range_frac=range_frac, segments=segments)


def _self_check_tijara(df: pd.DataFrame) -> None:
    close = df["close"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)
    rv = vol / pd.Series(vol).rolling(20).mean().to_numpy()
    obv = _obv(close, vol)

    pre = PREFIX_SESSIONS + 40
    base_len = 140
    base_last_40 = slice(pre + base_len - 40, pre + base_len)

    frac = _norm_lr_fractional_change(close[base_last_40])
    flow = _norm_flow_window(obv[base_last_40], vol[base_last_40])
    assert abs(frac) < 0.02, f"TIJARA: expected flat price in late base, got {frac:.4f}"
    assert flow > 0.10, f"TIJARA: expected positive OBV flow > 0.10, got {flow:.4f}"

    brk_i = pre + base_len
    assert rv[brk_i] >= 3.5, f"TIJARA: breakout rel_volume < 3.5 ({rv[brk_i]:.3f})"
    _assert_breakout_crossing(df, "TIJARA", (pre, pre + base_len), (brk_i, brk_i + 2))


def _self_check_bpcc(df: pd.DataFrame) -> None:
    close = df["close"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)
    obv = _obv(close, vol)

    decline = close[PREFIX_SESSIONS : PREFIX_SESSIONS + 120]
    assert decline[-1] < decline[0] * 0.84, "BPCC: decline segment is not steep enough"

    base_window = slice(PREFIX_SESSIONS + 120 + 40, PREFIX_SESSIONS + 120 + 80)
    flow = _norm_flow_window(obv[base_window], vol[base_window])
    assert flow > 0.10, f"BPCC: base OBV flow should be >0.10, got {flow:.4f}"

    rv = vol / pd.Series(vol).rolling(20).mean().to_numpy()
    brk_i = PREFIX_SESSIONS + 200
    assert rv[brk_i] >= 3.0, f"BPCC: breakout rel_volume < 3.0 ({rv[brk_i]:.3f})"
    _assert_breakout_crossing(df, "BPCC", (PREFIX_SESSIONS + 120, PREFIX_SESSIONS + 200), (brk_i, brk_i + 3))


def _self_check_zain(df: pd.DataFrame) -> None:
    close = df["close"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)

    base = close[PREFIX_SESSIONS + 30 : PREFIX_SESSIONS + 30 + 110]
    max_drift = 0.0
    for i in range(40, len(base) + 1):
        win = base[i - 40 : i]
        drift = abs((win[-1] / win[0]) - 1.0)
        max_drift = max(max_drift, drift)
    assert max_drift <= 0.015, f"ZAIN: 40-session base drift too high ({max_drift:.4f})"

    rv = vol / pd.Series(vol).rolling(20).mean().to_numpy()
    brk_i = PREFIX_SESSIONS + 140
    assert rv[brk_i] >= 3.0, f"ZAIN: breakout rel_volume < 3.0 ({rv[brk_i]:.3f})"
    _assert_breakout_crossing(df, "ZAIN", (PREFIX_SESSIONS + 30, PREFIX_SESSIONS + 140), (brk_i, brk_i + 2))


def _self_check_sanam(df: pd.DataFrame) -> None:
    close = df["close"].to_numpy(dtype=float)
    rsi = _rsi(close, 14)

    streak = 0
    best = 0
    for x in rsi:
        if x > 70:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    assert best >= 15, f"SANAM: RSI>70 streak too short ({best})"
    _assert_breakout_crossing(df, "SANAM", (PREFIX_SESSIONS, PREFIX_SESSIONS + 100), (PREFIX_SESSIONS + 100, PREFIX_SESSIONS + 102))


def _self_check_mabanee(df: pd.DataFrame) -> None:
    close = df["close"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)
    obv = _obv(close, vol)

    climax_i = int(np.argmax(vol))
    top_start = max(0, climax_i - 40)
    top = close[top_start:climax_i]
    assert top.max() > close[PREFIX_SESSIONS + 59], "MABANEE: top segment missing higher highs"

    top_obv = obv[top_start:climax_i]
    assert top_obv[-1] < top_obv[10], "MABANEE: OBV should fade in top segment"

    rv = vol / pd.Series(vol).rolling(20).mean().to_numpy()
    assert rv[climax_i] >= 4.0, f"MABANEE: climax rel_volume < 4.0 ({rv[climax_i]:.3f})"

    sma200 = pd.Series(close).rolling(200).mean().to_numpy()
    post = close[climax_i + 1 :]
    post_sma = sma200[climax_i + 1 :]
    below = np.where(post < post_sma)[0]
    assert len(below) > 0, "MABANEE: decline never crosses below SMA200"
    _assert_breakout_crossing(df, "MABANEE", (PREFIX_SESSIONS, PREFIX_SESSIONS + 160), (PREFIX_SESSIONS + 160, PREFIX_SESSIONS + 163))


def _self_check_joiner(df: pd.DataFrame) -> None:
    close = df["close"].to_numpy(dtype=float)
    close_s = pd.Series(close, dtype=float)
    ema10 = ee_indicator_service._ema(close_s, 10).to_numpy(dtype=float)
    ema30 = ee_indicator_service._ema(close_s, 30).to_numpy(dtype=float)
    sma200 = ee_indicator_service._sma(close_s, 200).to_numpy(dtype=float)
    range_low_120 = close_s.rolling(120).min().shift(1).to_numpy(dtype=float)
    range_high_120 = close_s.rolling(120).max().shift(1).to_numpy(dtype=float)

    warmup_ready = np.where(
        (~np.isnan(sma200))
        & (~np.isnan(range_low_120))
        & (~np.isnan(range_high_120))
        & (range_low_120 > 0)
        & (range_high_120 > 0)
    )[0]
    assert len(warmup_ready) > 0, "JOINER: warmup_ready_date not found"
    warmup_idx = int(warmup_ready[0])

    join_window_end = min(len(close), warmup_idx + 41)
    joined_hits = 0
    for i in range(warmup_idx, join_window_end):
        if close[i] > sma200[i] and ema10[i] > ema30[i] and close[i] >= (range_low_120[i] * 1.15):
            joined_hits += 1
    assert joined_hits >= 5, (
        "JOINER: expected >=5 trend-join eligible bars inside first 40 sessions after warmup_ready_date"
    )

    upper = float(np.nanmax(close[warmup_idx + 20 : warmup_idx + 140]))
    assert upper > 1150.0, "JOINER: expected late-cycle strong markup"
    assert np.nanmean(close[-80:] < sma200[-80:]) > 0.7, "JOINER: expected late decline below SMA200"
    _assert_breakout_crossing(df, "JOINER", (0, PREFIX_SESSIONS), (PREFIX_SESSIONS, PREFIX_SESSIONS + 40))


def _write(symbol: str, bundle: SeriesBundle) -> None:
    df = _ohlcv_from_close(
        symbol,
        bundle.close,
        bundle.volume,
        bundle.range_frac,
        accumulation_windows=bundle.accumulation_windows,
    )
    target = OUT_DIR / f"synthetic_{symbol.lower()}.csv"
    df.to_csv(target, index=False)


def main() -> None:
    rng = np.random.default_rng(MASTER_SEED)

    builders = {
        "TIJARA": _build_tijara,
        "BPCC": _build_bpcc,
        "ZAIN": _build_zain,
        "SANAM": _build_sanam,
        "MABANEE": _build_mabanee,
        "JOINER": _build_joiner,
    }
    checkers = {
        "TIJARA": _self_check_tijara,
        "BPCC": _self_check_bpcc,
        "ZAIN": _self_check_zain,
        "SANAM": _self_check_sanam,
        "MABANEE": _self_check_mabanee,
        "JOINER": _self_check_joiner,
    }
    first_pattern_index = {
        "TIJARA": PREFIX_SESSIONS,
        "BPCC": PREFIX_SESSIONS,
        "ZAIN": PREFIX_SESSIONS,
        "SANAM": PREFIX_SESSIONS,
        "MABANEE": PREFIX_SESSIONS,
        "JOINER": PREFIX_SESSIONS,
    }
    all_segments: dict[str, dict[str, dict[str, int]]] = {}

    for symbol in ["TIJARA", "BPCC", "ZAIN", "SANAM", "MABANEE", "JOINER"]:
        local_rng = np.random.default_rng(rng.integers(0, 1_000_000_000))
        bundle = builders[symbol](local_rng)
        df = _ohlcv_from_close(
            symbol,
            bundle.close,
            bundle.volume,
            bundle.range_frac,
            accumulation_windows=bundle.accumulation_windows,
        )
        _assert_warmup_alignment(
            df,
            symbol,
            first_pattern_idx=first_pattern_index[symbol],
            is_joiner=(symbol == "JOINER"),
        )
        for window in bundle.accumulation_windows:
            _assert_accumulation_cmf_window(df, window, symbol)
        checkers[symbol](df)
        target = OUT_DIR / f"synthetic_{symbol.lower()}.csv"
        df.to_csv(target, index=False)
        all_segments[symbol] = {
            name: _segment_entry(df, bounds[0], bounds[1])
            for name, bounds in sorted(bundle.segments.items())
            if bounds[0] < bounds[1]
        }
        print(f"generated {target.name}: {len(df)} rows")

    (OUT_DIR / "segments.json").write_text(
        json.dumps(all_segments, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("generated segments.json")


if __name__ == "__main__":
    main()

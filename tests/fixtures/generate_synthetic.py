from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
START_DATE = "2021-01-01"
MASTER_SEED = 20260704
NOISE_SIGMA = 0.004  # 0.4%


@dataclass
class SeriesBundle:
    close: np.ndarray
    volume: np.ndarray
    range_frac: np.ndarray
    accumulation_windows: list[tuple[int, int]] = field(default_factory=list)


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
    breakout = np.array([99.4, 105.4])
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

    return SeriesBundle(
        close=close,
        volume=volume,
        range_frac=range_frac,
        accumulation_windows=[(base_end - 40, base_end)],
    )


def _build_bpcc(rng: np.random.Generator) -> SeriesBundle:
    decline = np.linspace(700.0, 560.0, 120)
    base = 585.0 + 23.0 * np.sin(np.linspace(0, 10.0, 80))
    breakout = np.array([607.0, 616.0])
    markup = np.linspace(620.0, 697.0, 70)
    tail = 695.0 + 5.0 * np.sin(np.linspace(0, 7.0, 60))
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
    volume[brk_i] = base_vol * 4.5
    volume[brk_i + 1] = base_vol * 4.2
    range_frac[brk_i] = 0.017
    range_frac[brk_i + 1] = 0.018

    for k in [brk_i + 4, brk_i + 8, brk_i + 12, brk_i + 16]:
        if k < len(volume):
            volume[k] = base_vol * 3.3
            range_frac[k] = 0.010

    return SeriesBundle(
        close=close,
        volume=volume,
        range_frac=range_frac,
        accumulation_windows=[(base_start, base_end)],
    )


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

    return SeriesBundle(
        close=close,
        volume=volume,
        range_frac=range_frac,
        accumulation_windows=[(base_start, base_end)],
    )


def _build_sanam(rng: np.random.Generator) -> SeriesBundle:
    base_early = 205.0 + 8.5 * np.sin(np.linspace(0, 8.0, 60))
    base_late = 210.0 + 1.4 * np.sin(np.linspace(0, 4.0 * np.pi, 40))
    base = np.concatenate([base_early, base_late])
    breakout = np.array([214.0, 223.0])
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
    volume[brk_i] = base_vol * 3.2
    volume[brk_i + 1] = base_vol * 3.0
    range_frac[brk_i] = 0.018
    range_frac[brk_i + 1] = 0.019

    return SeriesBundle(
        close=close,
        volume=volume,
        range_frac=range_frac,
        accumulation_windows=[(base_start, base_end)],
    )


def _build_mabanee(rng: np.random.Generator) -> SeriesBundle:
    base = 882.0 + 13.0 * np.sin(np.linspace(0, 8.0 * np.pi, 160))
    breakout = np.array([901.0, 915.0, 932.0])
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
    volume[brk_i + 2] = base_vol * 3.8
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

    return SeriesBundle(
        close=close,
        volume=volume,
        range_frac=range_frac,
        accumulation_windows=[(0, base_end)],
    )


def _build_joiner(rng: np.random.Generator) -> SeriesBundle:
    warmup = np.linspace(760.0, 980.0, 220)
    early_join = np.linspace(986.0, 1120.0, 40)
    late_markup = np.linspace(1125.0, 1195.0, 40)
    topping = 1188.0 + 8.0 * np.sin(np.linspace(0, 3.0 * np.pi, 20))
    decline = np.linspace(1170.0, 820.0, 90)
    tail = np.linspace(810.0, 770.0, 30)
    close = np.concatenate([warmup, early_join, late_markup, topping, decline, tail])
    close = _apply_close_noise(close, rng)

    base_vol = 105_000.0
    volume = np.full(len(close), base_vol)
    range_frac = np.full(len(close), 0.011)
    join_start = len(warmup)
    decline_start = len(warmup) + len(early_join) + len(late_markup) + len(topping)
    for i in range(join_start, decline_start):
        volume[i] = base_vol * rng.uniform(1.15, 1.45)
    for i in range(decline_start, len(close)):
        volume[i] = base_vol * rng.uniform(1.55, 2.10)
        range_frac[i] = 0.020
    return SeriesBundle(close=close, volume=volume, range_frac=range_frac)


def _self_check_tijara(df: pd.DataFrame) -> None:
    close = df["close"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)
    rv = vol / pd.Series(vol).rolling(20).mean().to_numpy()
    obv = _obv(close, vol)

    pre = 40
    base_len = 140
    base_last_40 = slice(pre + base_len - 40, pre + base_len)

    frac = _norm_lr_fractional_change(close[base_last_40])
    flow = _norm_flow_window(obv[base_last_40], vol[base_last_40])
    assert abs(frac) < 0.02, f"TIJARA: expected flat price in late base, got {frac:.4f}"
    assert flow > 0.10, f"TIJARA: expected positive OBV flow > 0.10, got {flow:.4f}"

    brk_i = pre + base_len
    assert rv[brk_i] >= 3.5, f"TIJARA: breakout rel_volume < 3.5 ({rv[brk_i]:.3f})"


def _self_check_bpcc(df: pd.DataFrame) -> None:
    close = df["close"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)
    obv = _obv(close, vol)

    decline = close[:120]
    assert decline[-1] < decline[0] * 0.84, "BPCC: decline segment is not steep enough"

    base_window = slice(120 + 40, 120 + 80)
    flow = _norm_flow_window(obv[base_window], vol[base_window])
    assert flow > 0.10, f"BPCC: base OBV flow should be >0.10, got {flow:.4f}"

    rv = vol / pd.Series(vol).rolling(20).mean().to_numpy()
    assert rv[200] >= 3.0, f"BPCC: breakout rel_volume < 3.0 ({rv[200]:.3f})"


def _self_check_zain(df: pd.DataFrame) -> None:
    close = df["close"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)

    base = close[30 : 30 + 110]
    max_drift = 0.0
    for i in range(40, len(base) + 1):
        win = base[i - 40 : i]
        drift = abs((win[-1] / win[0]) - 1.0)
        max_drift = max(max_drift, drift)
    assert max_drift <= 0.015, f"ZAIN: 40-session base drift too high ({max_drift:.4f})"

    rv = vol / pd.Series(vol).rolling(20).mean().to_numpy()
    assert rv[140] >= 3.0, f"ZAIN: breakout rel_volume < 3.0 ({rv[140]:.3f})"


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


def _self_check_mabanee(df: pd.DataFrame) -> None:
    close = df["close"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)
    obv = _obv(close, vol)

    climax_i = int(np.argmax(vol))
    top_start = max(0, climax_i - 40)
    top = close[top_start:climax_i]
    assert top.max() > close[59], "MABANEE: top segment missing higher highs"

    top_obv = obv[top_start:climax_i]
    assert top_obv[-1] < top_obv[10], "MABANEE: OBV should fade in top segment"

    rv = vol / pd.Series(vol).rolling(20).mean().to_numpy()
    assert rv[climax_i] >= 4.0, f"MABANEE: climax rel_volume < 4.0 ({rv[climax_i]:.3f})"

    sma200 = pd.Series(close).rolling(200).mean().to_numpy()
    post = close[climax_i + 1 :]
    post_sma = sma200[climax_i + 1 :]
    below = np.where(post < post_sma)[0]
    assert len(below) > 0, "MABANEE: decline never crosses below SMA200"


def _self_check_joiner(df: pd.DataFrame) -> None:
    close = df["close"].to_numpy(dtype=float)
    ema10 = _ema(close, 10)
    ema30 = _ema(close, 30)
    sma200 = pd.Series(close).rolling(200).mean().to_numpy()
    range_low_120 = pd.Series(close).rolling(120).min().to_numpy()

    warmup_ready = np.where((~np.isnan(sma200)) & (~np.isnan(range_low_120)) & (range_low_120 > 0))[0]
    assert len(warmup_ready) > 0, "JOINER: warmup_ready_date not found"
    warmup_idx = int(warmup_ready[0])

    join_window_end = min(len(close), warmup_idx + 41)
    joined_eligible = False
    for i in range(warmup_idx, join_window_end):
        if close[i] > sma200[i] and ema10[i] > ema30[i] and close[i] >= (range_low_120[i] * 1.15):
            joined_eligible = True
            break
    assert joined_eligible, "JOINER: no trend-join eligibility inside first 40 sessions after warmup_ready_date"

    upper = float(np.nanmax(close[warmup_idx + 20 : warmup_idx + 140]))
    assert upper > 1150.0, "JOINER: expected late-cycle strong markup"
    assert np.nanmean(close[-80:] < sma200[-80:]) > 0.7, "JOINER: expected late decline below SMA200"


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
        for window in bundle.accumulation_windows:
            _assert_accumulation_cmf_window(df, window, symbol)
        checkers[symbol](df)
        target = OUT_DIR / f"synthetic_{symbol.lower()}.csv"
        df.to_csv(target, index=False)
        print(f"generated {target.name}: {len(df)} rows")


if __name__ == "__main__":
    main()

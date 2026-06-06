"""
Indicator Engine — every technical indicator the analysis layer needs.
Pure numpy/pandas implementations. No TA-Lib dependency, no system libs needed.
Every indicator is unit-testable. Validated math, no hand-wavy approximations.
"""
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from app.services.eagle_eye.config import CONFIG


# =============================================================================
# Helper utilities
# =============================================================================

def _wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (used by RSI, ADX, ATR). Equivalent to EMA with alpha=1/period."""
    return series.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _rolling_pctile(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of last value within the rolling window (0-100)."""
    def _pctile_of_last(x):
        last = x[-1]
        return 100.0 * (x <= last).sum() / len(x)
    return series.rolling(window).apply(_pctile_of_last, raw=True)


# =============================================================================
# TREND
# =============================================================================

def ema(df: pd.DataFrame, period: int) -> pd.Series:
    return _ema(df['close'], period)


def ema_ribbon_aligned(df: pd.DataFrame, periods=(8, 21, 50, 100, 200)) -> pd.Series:
    """1 if EMA8 > EMA21 > EMA50 > EMA100 > EMA200 (bullish stack),
       -1 if inverse (bearish stack), 0 if mixed."""
    emas = {p: _ema(df['close'], p) for p in periods}
    sorted_p = sorted(periods)
    bullish = pd.Series(True, index=df.index)
    bearish = pd.Series(True, index=df.index)
    for i in range(len(sorted_p) - 1):
        bullish &= emas[sorted_p[i]] > emas[sorted_p[i+1]]
        bearish &= emas[sorted_p[i]] < emas[sorted_p[i+1]]
    return pd.Series(np.where(bullish, 1, np.where(bearish, -1, 0)), index=df.index)


def macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    macd_line = _ema(df['close'], fast) - _ema(df['close'], slow)
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (adx, +DI, -DI)."""
    high, low, close = df['high'], df['low'], df['close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_ = _wilder_ema(tr, period)
    plus_di = 100 * _wilder_ema(pd.Series(plus_dm, index=df.index), period) / atr_
    minus_di = 100 * _wilder_ema(pd.Series(minus_dm, index=df.index), period) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = _wilder_ema(dx, period)
    return adx_, plus_di, minus_di


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """Returns +1 for uptrend, -1 for downtrend."""
    atr_ = atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    upper = hl2 + multiplier * atr_
    lower = hl2 - multiplier * atr_
    trend = pd.Series(index=df.index, dtype=float)
    trend.iloc[0] = 1
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper.iloc[i-1]:
            trend.iloc[i] = 1
        elif df['close'].iloc[i] < lower.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    return trend


def parabolic_sar(df: pd.DataFrame, af_start: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Classic Wilder's Parabolic SAR. Returns SAR value series."""
    high, low = df['high'].values, df['low'].values
    sar = np.zeros(len(df))
    bull = True
    af = af_start
    ep = high[0]
    sar[0] = low[0]
    for i in range(1, len(df)):
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        if bull:
            if low[i] < sar[i]:
                bull = False; sar[i] = ep; ep = low[i]; af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]; af = min(af + af_start, af_max)
        else:
            if high[i] > sar[i]:
                bull = True; sar[i] = ep; ep = high[i]; af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]; af = min(af + af_start, af_max)
    return pd.Series(sar, index=df.index)


def hull_ma(df: pd.DataFrame, period: int = 16) -> pd.Series:
    half = int(period / 2)
    sqrt_p = int(np.sqrt(period))
    wma_half = df['close'].rolling(half).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)
    wma_full = df['close'].rolling(period).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)
    diff = 2 * wma_half - wma_full
    return diff.rolling(sqrt_p).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)


def linear_regression_slope(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Slope of best-fit line over rolling window (price units / day)."""
    def slope(y):
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]
    return df['close'].rolling(period).apply(slope, raw=True)


def ichimoku(df: pd.DataFrame) -> Dict[str, pd.Series]:
    high, low, close = df['high'], df['low'], df['close']
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = close.shift(-26)
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    position = np.where(close > cloud_top, 1, np.where(close < cloud_bot, -1, 0))
    return {
        'tenkan': tenkan, 'kijun': kijun, 'senkou_a': senkou_a, 'senkou_b': senkou_b,
        'chikou': chikou,
        'cloud_position': pd.Series(position, index=df.index),
        'tk_cross': pd.Series(np.where(tenkan > kijun, 1, -1), index=df.index),
    }


# =============================================================================
# MOMENTUM
# =============================================================================

def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _wilder_ema(gain, period)
    avg_loss = _wilder_ema(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rsi_divergence(df: pd.DataFrame, lookback: int = 28) -> pd.Series:
    """Returns +1 (bullish divergence), -1 (bearish), 0 (none)."""
    r = rsi(df)
    out = pd.Series(0, index=df.index)
    for i in range(lookback, len(df)):
        window_price = df['close'].iloc[i-lookback:i+1]
        window_rsi = r.iloc[i-lookback:i+1]
        # bearish divergence: latest price near top, RSI making lower high
        if window_price.iloc[-1] >= window_price.iloc[-10:].max() * 0.99:
            if window_rsi.iloc[-1] < window_rsi.iloc[-20:-10].max():
                out.iloc[i] = -1
        # bullish divergence: latest price near bottom, RSI making higher low
        if window_price.iloc[-1] <= window_price.iloc[-10:].min() * 1.01:
            if window_rsi.iloc[-1] > window_rsi.iloc[-20:-10].min():
                out.iloc[i] = 1
    return out


def stochastic(
    df: pd.DataFrame,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 1,
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic oscillator.

    When smooth_k > 1, returns Slow Stochastic values:
      - stoch_k = SMA(smooth_k) of raw %K
      - stoch_d = SMA(d) of stoch_k
    """
    lowest = df['low'].rolling(k).min()
    highest = df['high'].rolling(k).max()
    raw_k = 100 * (df['close'] - lowest) / (highest - lowest).replace(0, np.nan)
    stoch_k = raw_k.rolling(smooth_k).mean() if smooth_k > 1 else raw_k
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def stoch_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    r = rsi(df, period)
    low = r.rolling(period).min()
    high = r.rolling(period).max()
    return 100 * (r - low) / (high - low).replace(0, np.nan)


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high'].rolling(period).max()
    low = df['low'].rolling(period).min()
    return -100 * (high - df['close']) / (high - low).replace(0, np.nan)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
    return 100 * (df['close'] / df['close'].shift(period) - 1)


def tsi(df: pd.DataFrame, long: int = 25, short: int = 13) -> pd.Series:
    pc = df['close'].diff()
    double_smooth = _ema(_ema(pc, long), short)
    double_smooth_abs = _ema(_ema(pc.abs(), long), short)
    return 100 * double_smooth / double_smooth_abs.replace(0, np.nan)


def awesome_oscillator(df: pd.DataFrame) -> pd.Series:
    median = (df['high'] + df['low']) / 2
    return median.rolling(5).mean() - median.rolling(34).mean()


def connors_rsi(df: pd.DataFrame) -> pd.Series:
    """Three-component Connors RSI."""
    rsi_3 = rsi(df, 3)
    # Streak
    change = df['close'].diff()
    streak = pd.Series(0, index=df.index, dtype=float)
    for i in range(1, len(df)):
        if change.iloc[i] > 0:
            streak.iloc[i] = streak.iloc[i-1] + 1 if streak.iloc[i-1] >= 0 else 1
        elif change.iloc[i] < 0:
            streak.iloc[i] = streak.iloc[i-1] - 1 if streak.iloc[i-1] <= 0 else -1
        else:
            streak.iloc[i] = 0
    streak_rsi = rsi(pd.DataFrame({'close': streak}), 2)
    # Percent rank of 1-day ROC over last 100
    pct_change = df['close'].pct_change()
    pct_rank = pct_change.rolling(100).apply(
        lambda x: 100 * (x < x.iloc[-1]).sum() / len(x), raw=False
    )
    return (rsi_3 + streak_rsi + pct_rank) / 3


# =============================================================================
# VOLATILITY
# =============================================================================

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return _wilder_ema(tr, period)


def atr_percentile(df: pd.DataFrame, period: int = 14, window: int = 252) -> pd.Series:
    a = atr(df, period)
    return _rolling_pctile(a, window)


def bollinger_bands(df: pd.DataFrame, period: int = 20, stddev: float = 2.0):
    mid = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    upper = mid + stddev * std
    lower = mid - stddev * std
    pct_b = (df['close'] - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    return {'upper': upper, 'middle': mid, 'lower': lower, 'pct_b': pct_b, 'bandwidth': bandwidth}


def bb_squeeze(df: pd.DataFrame, period: int = 20, lookback: int = 252) -> pd.Series:
    bb = bollinger_bands(df, period)
    bw_pct = _rolling_pctile(bb['bandwidth'], lookback)
    return (bw_pct < 20).astype(int)


def keltner_channels(df: pd.DataFrame, period: int = 20, mult: float = 2.0):
    mid = _ema(df['close'], period)
    a = atr(df, period)
    return {'upper': mid + mult * a, 'middle': mid, 'lower': mid - mult * a}


def donchian(df: pd.DataFrame, period: int = 20):
    upper = df['high'].rolling(period).max()
    lower = df['low'].rolling(period).min()
    middle = (upper + lower) / 2
    return {'upper': upper, 'middle': middle, 'lower': lower}


def historical_volatility(df: pd.DataFrame, period: int = 30) -> pd.Series:
    log_returns = np.log(df['close'] / df['close'].shift(1))
    return log_returns.rolling(period).std() * np.sqrt(252) * 100


# =============================================================================
# VOLUME / FLOW
# =============================================================================

def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df['close'].diff().fillna(0))
    return (direction * df['volume']).cumsum()


def obv_slope(df: pd.DataFrame, period: int = 20) -> pd.Series:
    o = obv(df)
    def s(y):
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]
    return o.rolling(period).apply(s, raw=True)


def ad_line(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution Line."""
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)
    mfm = mfm.fillna(0)
    mfv = mfm * df['volume']
    return mfv.cumsum()


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)
    mfm = mfm.fillna(0)
    mfv = mfm * df['volume']
    return mfv.rolling(period).sum() / df['volume'].rolling(period).sum().replace(0, np.nan)


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos = pd.Series(np.where(tp > tp.shift(), mf, 0), index=df.index)
    neg = pd.Series(np.where(tp < tp.shift(), mf, 0), index=df.index)
    mfr = pos.rolling(period).sum() / neg.rolling(period).sum().replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP proxy for daily bars (typical price)."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    return tp


def vwap_distance_sigma(df: pd.DataFrame) -> pd.Series:
    """Distance from VWAP in standard deviations."""
    v = vwap(df)
    diff = df['close'] - v
    std = diff.rolling(20).std()
    return diff / std.replace(0, np.nan)


def relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df['volume'] / df['volume'].rolling(period).mean().replace(0, np.nan)


def force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
    fi = (df['close'] - df['close'].shift()) * df['volume']
    return _ema(fi, period)


def ease_of_movement(df: pd.DataFrame, period: int = 14) -> pd.Series:
    distance = ((df['high'] + df['low'])/2) - ((df['high'].shift() + df['low'].shift())/2)
    box_ratio = (df['volume'] / 100_000_000) / (df['high'] - df['low']).replace(0, np.nan)
    return (distance / box_ratio).rolling(period).mean()


def klinger(df: pd.DataFrame) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    trend = pd.Series(np.where(tp > tp.shift(), 1, -1), index=df.index)
    dm = df['high'] - df['low']
    cm = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if trend.iloc[i] == trend.iloc[i-1]:
            cm.iloc[i] = cm.iloc[i-1] + dm.iloc[i]
        else:
            cm.iloc[i] = dm.iloc[i-1] + dm.iloc[i]
    vf = df['volume'] * trend * (2 * (dm / cm.replace(0, np.nan)) - 1) * 100
    return _ema(vf, 34) - _ema(vf, 55)


# =============================================================================
# STRUCTURE / SUPPORT-RESISTANCE
# =============================================================================

def swing_points(df: pd.DataFrame, window: int = 5):
    """Detect swing highs/lows using fractal logic (window bars on each side)."""
    highs = df['high']
    lows = df['low']
    swing_high = pd.Series(False, index=df.index)
    swing_low = pd.Series(False, index=df.index)
    for i in range(window, len(df) - window):
        if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
            swing_high.iloc[i] = True
        if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
            swing_low.iloc[i] = True
    return swing_high, swing_low


def volume_profile(df: pd.DataFrame, lookback: int = 90, buckets: int = 50) -> Dict[str, float]:
    """Compute Volume Profile (POC, VAH, VAL) over last N bars."""
    recent = df.tail(lookback)
    if len(recent) < 10:
        return {'poc': np.nan, 'vah': np.nan, 'val': np.nan}
    price_min = recent['low'].min()
    price_max = recent['high'].max()
    edges = np.linspace(price_min, price_max, buckets + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vol_per_bucket = np.zeros(buckets)
    for _, row in recent.iterrows():
        low_b = max(0, np.searchsorted(edges, row['low']) - 1)
        high_b = min(buckets - 1, np.searchsorted(edges, row['high']) - 1)
        if high_b <= low_b:
            vol_per_bucket[low_b] += row['volume']
        else:
            spread = high_b - low_b + 1
            per = row['volume'] / spread
            vol_per_bucket[low_b:high_b+1] += per
    poc_idx = int(np.argmax(vol_per_bucket))
    poc = float(centers[poc_idx])
    # Value area (70% of volume around POC)
    total = vol_per_bucket.sum()
    target = total * 0.70
    accumulated = vol_per_bucket[poc_idx]
    low_i, high_i = poc_idx, poc_idx
    while accumulated < target and (low_i > 0 or high_i < buckets - 1):
        left_v = vol_per_bucket[low_i - 1] if low_i > 0 else -1
        right_v = vol_per_bucket[high_i + 1] if high_i < buckets - 1 else -1
        if right_v >= left_v and high_i < buckets - 1:
            high_i += 1; accumulated += vol_per_bucket[high_i]
        elif low_i > 0:
            low_i -= 1; accumulated += vol_per_bucket[low_i]
        else:
            break
    return {
        'poc': poc,
        'vah': float(centers[high_i]),
        'val': float(centers[low_i]),
        'distribution': dict(zip(centers.round(4), vol_per_bucket)),
    }


def fibonacci_levels(df: pd.DataFrame, lookback: int = 252) -> Dict[str, float]:
    """Fib retracements/extensions from most significant swing."""
    recent = df.tail(lookback)
    if recent.empty:
        return {}
    hi = recent['high'].max()
    lo = recent['low'].min()
    hi_date = recent['high'].idxmax()
    lo_date = recent['low'].idxmin()
    is_uptrend = hi_date > lo_date
    diff = hi - lo
    if is_uptrend:
        return {
            'fib_0':     hi,
            'fib_23.6':  hi - 0.236 * diff,
            'fib_38.2':  hi - 0.382 * diff,
            'fib_50':    hi - 0.500 * diff,
            'fib_61.8':  hi - 0.618 * diff,
            'fib_78.6':  hi - 0.786 * diff,
            'fib_100':   lo,
            'fib_127.2': hi + 0.272 * diff,
            'fib_161.8': hi + 0.618 * diff,
            'fib_261.8': hi + 1.618 * diff,
        }
    else:
        return {
            'fib_0':     lo,
            'fib_23.6':  lo + 0.236 * diff,
            'fib_38.2':  lo + 0.382 * diff,
            'fib_50':    lo + 0.500 * diff,
            'fib_61.8':  lo + 0.618 * diff,
            'fib_78.6':  lo + 0.786 * diff,
            'fib_100':   hi,
            'fib_127.2': lo - 0.272 * diff,
            'fib_161.8': lo - 0.618 * diff,
            'fib_261.8': lo - 1.618 * diff,
        }


def pivot_points(df: pd.DataFrame) -> Dict[str, float]:
    """Classic floor pivots based on previous day's HLC."""
    if len(df) < 2:
        return {}
    prev = df.iloc[-2]
    p = (prev['high'] + prev['low'] + prev['close']) / 3
    return {
        'pivot': p,
        'r1': 2*p - prev['low'], 'r2': p + (prev['high'] - prev['low']),
        'r3': prev['high'] + 2*(p - prev['low']),
        's1': 2*p - prev['high'], 's2': p - (prev['high'] - prev['low']),
        's3': prev['low'] - 2*(prev['high'] - p),
    }


# =============================================================================
# STATISTICAL
# =============================================================================

def zscore_vs_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    ma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    return (df['close'] - ma) / std.replace(0, np.nan)


def hurst_exponent(series: pd.Series, lags=range(2, 20)) -> float:
    """Hurst exponent: >0.5 trending, <0.5 mean-reverting, ~0.5 random."""
    if len(series) < max(lags) * 3 or series.isna().all():
        return np.nan
    s = series.dropna().values
    if len(s) < max(lags) * 3:
        return np.nan
    tau = []
    for lag in lags:
        diff = s[lag:] - s[:-lag]
        tau.append(np.std(diff))
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(poly[0])


# =============================================================================
# INSTITUTIONAL ACCUMULATION SCORE (Kuwait-adapted, EOD-based)
# =============================================================================

def accumulation_score(df: pd.DataFrame) -> pd.Series:
    """Composite 0-100 institutional accumulation score."""
    if len(df) < 60:
        return pd.Series(np.nan, index=df.index)

    # Component 1: OBV slope (normalized)
    obv_s = obv_slope(df, 60)
    obv_norm = (obv_s.rank(pct=True) * 100).fillna(50)

    # Component 2: CMF
    cmf_v = cmf(df, CONFIG.CMF_PERIOD)
    cmf_norm = ((cmf_v + 0.3) / 0.6 * 100).clip(0, 100).fillna(50)

    # Component 3: A/D Line slope
    ad = ad_line(df)
    ad_slope = ad.rolling(60).apply(lambda y: np.polyfit(np.arange(len(y)), y, 1)[0], raw=True)
    ad_norm = (ad_slope.rank(pct=True) * 100).fillna(50)

    # Component 4: % of last 30 days closing in upper third of range
    upper_third = ((df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)) > 0.66
    upper_third_pct = upper_third.rolling(30).mean() * 100

    # Component 5: Up-volume to down-volume ratio (30d)
    up_vol = df['volume'].where(df['close'] > df['close'].shift(), 0)
    down_vol = df['volume'].where(df['close'] < df['close'].shift(), 0)
    ud_ratio = up_vol.rolling(30).sum() / down_vol.rolling(30).sum().replace(0, np.nan)
    ud_norm = ((ud_ratio - 0.5) / 2.0 * 100).clip(0, 100).fillna(50)

    # Component 6: Narrowing range + rising volume (compression signature)
    range_pct = (df['high'] - df['low']) / df['close']
    range_compression = (range_pct.rolling(20).mean() < range_pct.rolling(60).mean()).astype(int)
    vol_rising = (df['volume'].rolling(20).mean() > df['volume'].rolling(60).mean()).astype(int)
    compression_score = (range_compression & vol_rising) * 100

    # Weighted composite
    composite = (
        0.25 * obv_norm +
        0.20 * cmf_norm +
        0.15 * ad_norm +
        0.15 * upper_third_pct +
        0.15 * ud_norm +
        0.10 * compression_score
    )
    return composite.clip(0, 100)


def wyckoff_phase(df: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """Simplified Wyckoff phase classifier."""
    if len(df) < lookback:
        return pd.Series('UNKNOWN', index=df.index)

    out = pd.Series('UNKNOWN', index=df.index, dtype=object)
    acc = accumulation_score(df)
    a = atr(df, 14)
    a_pct = atr_percentile(df, 14, 252)

    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i+1]
        cur_close = df['close'].iloc[i]
        recent_high = window['high'].max()
        recent_low = window['low'].min()
        range_pos = (cur_close - recent_low) / (recent_high - recent_low + 1e-9)

        atr_val = a_pct.iloc[i] if pd.notna(a_pct.iloc[i]) else 50
        acc_val = acc.iloc[i] if pd.notna(acc.iloc[i]) else 50

        if atr_val < 30 and acc_val > 55 and range_pos < 0.4:
            out.iloc[i] = 'B_BUILDING_CAUSE'
        elif atr_val < 25 and range_pos < 0.3:
            out.iloc[i] = 'A_STOPPING_ACTION'
        elif acc_val > 70 and range_pos > 0.5 and a_pct.iloc[i] > 40:
            out.iloc[i] = 'D_MARKUP'
        elif acc_val > 60 and range_pos > 0.3 and range_pos < 0.7:
            out.iloc[i] = 'C_TEST_SPRING'
        elif range_pos > 0.8 and atr_val > 60:
            out.iloc[i] = 'E_MARKUP_EXPANSION'
        else:
            out.iloc[i] = 'UNCLASSIFIED'
    return out


# =============================================================================
# THE PUBLIC INTERFACE — compute every indicator in one pass
# =============================================================================

def compute_all_indicators(
    df: pd.DataFrame,
    market_close: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Run every indicator and return a DataFrame indexed by date with all values
    as columns. This is the canonical 'indicator snapshot' consumed by the
    forensics engine, ML pipeline, and live rating engine.
    """
    if len(df) < 50:
        raise ValueError(f"Need at least 50 bars to compute indicators, got {len(df)}")

    out = pd.DataFrame(index=df.index)

    # Trend
    for p in CONFIG.EMA_PERIODS:
        out[f'ema_{p}'] = _ema(df['close'], p)
    # Canonical moving-average fields expected by downstream diagnostics/features.
    out['ema_20'] = _ema(df['close'], 20)
    out['ema_10'] = _ema(df['close'], 10)
    out['ema_30'] = _ema(df['close'], 30)
    out['sma_200'] = df['close'].rolling(200, min_periods=1).mean()
    out['ema_ribbon_aligned'] = ema_ribbon_aligned(df, CONFIG.EMA_PERIODS)
    m, s, h = macd(
        df,
        fast=int(CONFIG.MACD_FAST),
        slow=int(CONFIG.MACD_SLOW),
        signal=CONFIG.MACD_SIGNAL,
    )
    out['macd_line'] = m; out['macd_signal'] = s; out['macd_histogram'] = h
    a_, pd_, md_ = adx(df, CONFIG.ADX_PERIOD)
    out['adx'] = a_; out['plus_di'] = pd_; out['minus_di'] = md_
    out['plus_di_minus_di_diff'] = out['plus_di'] - out['minus_di']
    out['di_spread'] = out['plus_di_minus_di_diff']
    out['supertrend'] = supertrend(df, CONFIG.SUPERTREND_PERIOD, CONFIG.SUPERTREND_MULTIPLIER)
    out['psar'] = parabolic_sar(df)
    out['hull_ma'] = hull_ma(df)
    out['linreg_slope'] = linear_regression_slope(df, 20)
    ich = ichimoku(df)
    out['ichimoku_cloud_pos'] = ich['cloud_position']
    out['ichimoku_tk_cross'] = ich['tk_cross']

    # Momentum
    out['rsi'] = rsi(df, CONFIG.RSI_PERIOD)
    out['rsi_divergence'] = rsi_divergence(df)
    sk, sd = stochastic(
        df,
        k=CONFIG.STOCH_K,
        d=CONFIG.STOCH_D,
        smooth_k=CONFIG.STOCH_SMOOTH_K,
    )
    out['stoch_k'] = sk; out['stoch_d'] = sd
    out['stochastic_k'] = sk
    out['stochastic_d'] = sd
    out['stoch_rsi'] = stoch_rsi(df)
    out['williams_r'] = williams_r(df)
    out['cci'] = cci(df, CONFIG.CCI_PERIOD)
    out['roc'] = roc(df)
    out['tsi'] = tsi(df)
    out['ao'] = awesome_oscillator(df)
    out['connors_rsi'] = connors_rsi(df)
    momentum_rsi_bullish = ((out['rsi'] >= 40) & (out['rsi'] <= 70)).astype(float)
    momentum_macd_bullish = (out['macd_histogram'] > 0).astype(float)
    momentum_adx_bullish = (out['adx'] > 20).astype(float)
    out['momentum_confluence'] = (momentum_rsi_bullish + momentum_macd_bullish + momentum_adx_bullish) / 3.0

    # Volatility
    out['atr'] = atr(df, CONFIG.ATR_PERIOD)
    out['atr_percentile_252'] = atr_percentile(df, CONFIG.ATR_PERIOD, 252)
    bb = bollinger_bands(df, CONFIG.BB_PERIOD, CONFIG.BB_STDDEV)
    out['bb_upper'] = bb['upper']; out['bb_middle'] = bb['middle']; out['bb_lower'] = bb['lower']
    out['bb_pct_b'] = bb['pct_b']; out['bb_bandwidth'] = bb['bandwidth']
    out['bb_squeeze'] = bb_squeeze(df)
    kc = keltner_channels(df, CONFIG.KELTNER_PERIOD)
    out['kc_upper'] = kc['upper']; out['kc_lower'] = kc['lower']
    dc = donchian(df, CONFIG.DONCHIAN_PERIOD)
    out['dc_upper'] = dc['upper']; out['dc_lower'] = dc['lower']
    out['hist_vol_30d'] = historical_volatility(df, 30)

    # Volume / Flow
    out['obv'] = obv(df)
    out['obv_value'] = out['obv']
    out['obv_change_5d'] = out['obv'] - out['obv'].shift(5)
    out['obv_change_10d'] = out['obv'] - out['obv'].shift(10)
    out['obv_change_20d'] = out['obv'] - out['obv'].shift(20)
    out['obv_direction_10d'] = np.sign(out['obv_change_10d'])
    out['obv_slope_20'] = obv_slope(df, 20)
    out['obv_slope_60'] = obv_slope(df, 60)
    out['ad_line'] = ad_line(df)
    out['cmf'] = cmf(df, CONFIG.CMF_PERIOD)
    out['cmf_change_5d'] = out['cmf'] - out['cmf'].shift(5)
    out['cmf_change_10d'] = out['cmf'] - out['cmf'].shift(10)
    out['mfi'] = mfi(df, CONFIG.MFI_PERIOD)
    out['vwap'] = vwap(df)
    out['vwap_session'] = out['vwap']
    out['vwap_distance_sigma'] = vwap_distance_sigma(df)
    out['rel_volume'] = relative_volume(df, CONFIG.VOLUME_AVG_PERIOD)
    out['volume_ratio_20d'] = df['volume'] / df['volume'].rolling(20).mean().replace(0, np.nan)
    out['obv_trajectory_slope'] = out['obv_slope_20']
    open_px = df['open'] if 'open' in df.columns else df['close']
    green_vol = df['volume'] * (df['close'] > open_px).astype(float)
    red_vol = df['volume'] * (df['close'] < open_px).astype(float)
    out['green_red_volume_ratio_5d'] = green_vol.rolling(5).sum() / red_vol.rolling(5).sum().replace(0, np.nan)
    out['green_red_volume_ratio_10d'] = green_vol.rolling(10).sum() / red_vol.rolling(10).sum().replace(0, np.nan)
    flow_cmf_positive = (out['cmf'] > 0).astype(float)
    flow_obv_rising = (out['obv_change_10d'] > 0).astype(float)
    flow_mfi_healthy = (out['mfi'] > 40).astype(float)
    out['volume_flow_confluence'] = (flow_cmf_positive + flow_obv_rising + flow_mfi_healthy) / 3.0
    out['force_index'] = force_index(df)
    out['eom'] = ease_of_movement(df)
    out['klinger'] = klinger(df)

    # Structure
    sw_h, sw_l = swing_points(df)
    out['swing_high'] = sw_h.astype(int)
    out['swing_low'] = sw_l.astype(int)

    # Statistical
    out['zscore_20'] = zscore_vs_ma(df, 20)
    out['zscore_50'] = zscore_vs_ma(df, 50)
    out['zscore_200'] = zscore_vs_ma(df, 200)
    out['zscore_vs_ma_20'] = out['zscore_20']

    # Institutional
    out['accumulation_score'] = accumulation_score(df)
    out['wyckoff_phase'] = wyckoff_phase(df)

    # Price context for downstream (kept for joins and DNA chart examples)
    open_series = df['open'] if 'open' in df.columns else df['close']
    open_series = pd.to_numeric(open_series, errors='coerce')
    out['open'] = open_series.where(open_series.notna() & (open_series != 0), df['close'])
    out['close'] = df['close']
    out['volume'] = df['volume']
    out['high'] = df['high']
    out['low'] = df['low']

    # Net liquidity: KWD turnover (preferred) or fallback to shares × close
    # 'turnover_kwd' is the actual exchange-reported KWD value traded per day.
    # 'volume' (shares count) is kept separately above for share-count signals.
    _turnover_series = (
        df['turnover_kwd'] if 'turnover_kwd' in df.columns
        else df['volume'] * df['close']
    )
    out['dollar_volume'] = _turnover_series
    out['avg_20d_turnover_kwd'] = pd.to_numeric(_turnover_series, errors='coerce').rolling(20).mean()

    # Prevent highly fragmented DataFrame blocks after many column inserts.
    out = out.copy()

    # ── Pattern features (chart structure and activity) ────────────────────
    out['higher_lows_20d'] = (df['low'] > df['low'].shift(1)).rolling(20).sum()
    out['higher_highs_20d'] = (df['high'] > df['high'].shift(1)).rolling(20).sum()

    _daily_range_pct = ((df['high'] - df['low']) / df['low'].replace(0, np.nan)) * 100.0
    _range_recent_5 = _daily_range_pct.rolling(5).mean()
    _range_older_10 = _daily_range_pct.shift(10).rolling(10).mean()
    out['range_contraction_ratio'] = _range_recent_5 / _range_older_10.replace(0, np.nan)

    out['volume_trend_20d'] = ((df['volume'] / df['volume'].shift(20).replace(0, np.nan)) - 1.0) * 100.0

    # ── Entry quality features ────────────────────────────────────────────────
    # These two features measure HOW EARLY and HOW CLEAN the entry is.
    # Both are available at prediction time (no forward-looking data).
    # The rating engine uses price_extension_from_20d_low_pct to penalise
    # entries that are already extended well above the setup base.

    # price_extension_from_20d_low_pct:
    #   How far has price risen above the 20-day rolling low?
    #   0% = price IS the 20d low (ideal entry at the base).
    #   >5% = starting to extend; >10% = moderately extended; >20% = chase zone.
    low_20d = df['low'].rolling(20).min()
    out['price_extension_from_20d_low_pct'] = (
        (df['close'] / low_20d.replace(0, np.nan) - 1.0) * 100.0
    )

    # Medium/long extension context for anti-chasing safeguards.
    low_60d = df['low'].rolling(60).min()
    out['price_extension_from_60d_low_pct'] = (
        (df['close'] / low_60d.replace(0, np.nan) - 1.0) * 100.0
    )

    low_120d = df['low'].rolling(120).min()
    out['price_extension_from_120d_low_pct'] = (
        (df['close'] / low_120d.replace(0, np.nan) - 1.0) * 100.0
    )

    high_60d = df['high'].rolling(60).max()
    _range_60d = high_60d - low_60d
    out['position_in_60d_range_pct'] = (
        (df['close'] - low_60d) / _range_60d.replace(0, np.nan)
    ) * 100.0
    trend_ema_bullish = (out['ema_ribbon_aligned'] == 1).astype(float)
    trend_above_ema50 = (df['close'] > out['ema_50']).astype(float)
    out['trend_confluence'] = (trend_ema_bullish + trend_above_ema50) / 2.0

    if len(df) >= 252:
        high_252d = df['high'].rolling(252).max()
        out['distance_from_52w_high_pct'] = (
            (1.0 - (df['close'] / high_252d.replace(0, np.nan))) * 100.0
        )
    else:
        high_all = df['high'].expanding().max()
        out['distance_from_52w_high_pct'] = (
            (1.0 - (df['close'] / high_all.replace(0, np.nan))) * 100.0
        )

    # accumulation_compression_days:
    #   Consecutive days (ending at each bar) where the 5-bar coefficient of
    #   variation (std/mean) of close prices is < 2.5% — indicating a tight,
    #   low-volatility accumulation range.
    #   Longer compression = smart-money quiet build-up = higher signal quality.
    _cv5 = (
        df['close'].rolling(5).std()
        / df['close'].rolling(5).mean().replace(0, np.nan)
    )
    _is_compressed = (_cv5 < 0.025).fillna(False)
    # Streak: resets to 0 whenever compression breaks, counts up while it holds.
    _groups = (~_is_compressed).cumsum()
    out['accumulation_compression_days'] = (
        _is_compressed.groupby(_groups).cumsum().astype(float)
    )

    # ── Capitulation/Reversal features (v10) ────────────────────────────────
    high_20d = df['high'].rolling(20).max()
    out['selloff_from_20d_high_pct'] = (
        (high_20d - df['close']) / high_20d.replace(0, np.nan)
    ) * 100.0
    high_60d = df['high'].rolling(60).max()
    out['selloff_from_60d_high_pct'] = (
        (high_60d - df['close']) / high_60d.replace(0, np.nan)
    ) * 100.0

    up_vol = df['volume'].where(df['close'] > df['close'].shift(1), 0.0)
    down_vol = df['volume'].where(df['close'] < df['close'].shift(1), 0.0)
    out['up_down_volume_ratio_5d'] = (
        up_vol.rolling(5).sum() / down_vol.rolling(5).sum().replace(0, np.nan)
    )
    out['up_down_volume_ratio_10d'] = (
        up_vol.rolling(10).sum() / down_vol.rolling(10).sum().replace(0, np.nan)
    )

    vol_sma_5 = df['volume'].rolling(5).mean()
    vol_sma_20 = df['volume'].rolling(20).mean().replace(0, np.nan)
    out['volume_acceleration'] = vol_sma_5 / vol_sma_20

    low_5d = df['low'].rolling(5).min()
    out['bounce_from_5d_low_pct'] = (
        (df['close'] / low_5d.replace(0, np.nan) - 1.0) * 100.0
    )
    low_10d = df['low'].rolling(10).min()
    out['bounce_from_10d_low_pct'] = (
        (df['close'] / low_10d.replace(0, np.nan) - 1.0) * 100.0
    )

    higher_low_flag = (df['low'] > df['low'].shift(1)).fillna(False).astype(int)
    hl_groups = (higher_low_flag == 0).cumsum()
    out['consecutive_higher_lows_5d'] = higher_low_flag.groupby(hl_groups).cumsum().astype(float)
    out['consecutive_higher_lows'] = out['consecutive_higher_lows_5d']

    out['rsi_velocity_3d'] = out['rsi'] - out['rsi'].shift(3)
    out['rsi_velocity_5d'] = out['rsi'] - out['rsi'].shift(5)
    out['macd_hist_velocity_3d'] = out['macd_histogram'] - out['macd_histogram'].shift(3)
    out['macd_hist_velocity_5d'] = out['macd_histogram'] - out['macd_histogram'].shift(5)

    selloff_component = (out['selloff_from_20d_high_pct'] / 25.0 * 100.0).clip(0, 100).fillna(0)
    volume_component = ((out['volume_ratio_20d'] - 1.0) * 100.0).clip(0, 100).fillna(0)
    bounce_component = (out['bounce_from_5d_low_pct'] / 10.0 * 100.0).clip(0, 100).fillna(0)
    rsi_component = ((35.0 - out['rsi']) / 35.0 * 100.0).clip(0, 100).fillna(0)
    hl_component = (out['consecutive_higher_lows_5d'] / 5.0 * 100.0).clip(0, 100).fillna(0)

    out['capitulation_reversal_score'] = (
        0.30 * selloff_component
        + 0.25 * volume_component
        + 0.20 * bounce_component
        + 0.15 * rsi_component
        + 0.10 * hl_component
    ).clip(0, 100)

    out['institutional_confluence'] = (out['accumulation_score'] > 30).astype(float)
    out['overall_confluence'] = (
        0.30 * out['momentum_confluence']
        + 0.30 * out['volume_flow_confluence']
        + 0.20 * out['trend_confluence']
        + 0.20 * out['institutional_confluence']
    ).clip(0.0, 1.0)
    out['flow_momentum_divergence'] = out['volume_flow_confluence'] - out['momentum_confluence']

    # Backward-compatible aliases used by older callers.
    if 'ema_21' in out.columns:
        out['ema_fast'] = out['ema_21']
    if 'ema_200' in out.columns:
        out['ema_slow'] = out['ema_200']

    # Consolidate blocks before the large curated feature assignment block.
    out = out.copy()

    # ---------------------------------------------------------------------
    # Phase 1 curated indicator set (liquidity-first, non-redundant)
    # ---------------------------------------------------------------------
    out['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
    out['sma_200'] = df['close'].rolling(200, min_periods=1).mean()
    out['stock_close_vs_50sma'] = (df['close'] / out['sma_50'].replace(0, np.nan)) - 1.0
    out['stock_close_vs_200sma'] = (df['close'] / out['sma_200'].replace(0, np.nan)) - 1.0
    out['stock_50sma_slope_20d'] = out['sma_50'].rolling(20).apply(
        lambda y: np.polyfit(np.arange(len(y)), y, 1)[0], raw=True
    )

    # Market context: use provided premier-market proxy if available; otherwise
    # fall back to local price series and mark source for observability.
    if market_close is not None and isinstance(market_close, pd.Series) and len(market_close) > 0:
        market_aligned = pd.to_numeric(market_close.reindex(df.index), errors='coerce').ffill()
        out['market_proxy_source'] = 'premier_composite'
    else:
        market_aligned = pd.to_numeric(df['close'], errors='coerce')
        out['market_proxy_source'] = 'self_fallback'

    market_sma_50 = market_aligned.rolling(50, min_periods=1).mean()
    market_sma_200 = market_aligned.rolling(200, min_periods=1).mean()
    out['market_close_vs_200sma'] = (market_aligned / market_sma_200.replace(0, np.nan)) - 1.0
    out['market_50sma_slope_20d'] = market_sma_50.rolling(20).apply(
        lambda y: np.polyfit(np.arange(len(y)), y, 1)[0], raw=True
    )

    out['return_3m'] = (df['close'] / df['close'].shift(63)) - 1.0
    out['return_6m'] = (df['close'] / df['close'].shift(126)) - 1.0
    market_return_3m = (market_aligned / market_aligned.shift(63)) - 1.0
    out['relative_strength_3m'] = out['return_3m'] - market_return_3m

    out['rsi_14'] = out['rsi']
    out['cci_20'] = out['cci']
    out['cci_change_5d'] = out['cci_20'] - out['cci_20'].shift(5)
    out['macd_histogram_slope_5d'] = out['macd_histogram'].rolling(5).apply(
        lambda y: np.polyfit(np.arange(len(y)), y, 1)[0], raw=True
    )

    out['absolute_daily_traded_value'] = pd.to_numeric(df['close'] * df['volume'], errors='coerce')
    out['avg_traded_value_20d'] = out['absolute_daily_traded_value'].rolling(20).mean()
    out['traded_value_ratio_20d'] = (
        out['absolute_daily_traded_value'] / out['avg_traded_value_20d'].replace(0, np.nan)
    )

    active_threshold = max(2000.0, float(CONFIG.MIN_DAILY_TURNOVER_KWD) * 0.1)
    out['active_trading_days_ratio_60d'] = (
        (out['absolute_daily_traded_value'] > active_threshold).astype(float).rolling(60).mean()
    )

    out['cmf_10'] = cmf(df, 10)
    # Compatibility alias: many rule/scoring paths still consume cmf_20* keys.
    if CONFIG.CMF_PERIOD == 10:
        out['cmf_20'] = out['cmf_10']
    else:
        out['cmf_20'] = cmf(df, 20)
    out['obv_slope_20d'] = out['obv_slope_20']

    prev_close = df['close'].shift(1)
    up_mask = (df['close'] > prev_close).astype(float)
    down_mask = (df['close'] < prev_close).astype(float)
    out['up_day_volume_20d'] = (df['volume'] * up_mask).rolling(20).sum()
    out['down_day_volume_20d'] = (df['volume'] * down_mask).rolling(20).sum()
    out['up_down_volume_ratio_20d'] = (
        out['up_day_volume_20d'] / out['down_day_volume_20d'].replace(0, np.nan)
    )

    # Recompute CLV in a dedicated field required by the rules engine.
    out['close_location_value'] = (
        ((df['close'] - df['low']) - (df['high'] - df['close']))
        / (df['high'] - df['low']).replace(0, np.nan)
    ).clip(-1.0, 1.0)

    out['high_volume_weak_close_flag'] = (
        (out['traded_value_ratio_20d'] > 1.5)
        & (out['close_location_value'] < 0.0)
    ).astype(int)

    out['atr_14'] = out['atr']
    out['atr_percent'] = out['atr_14'] / df['close'].replace(0, np.nan)
    out['bb_width_20'] = out['bb_bandwidth']
    out['bb_width_percentile_252d'] = (
        out['bb_width_20'].rolling(252).apply(
            lambda x: (x <= x.iloc[-1]).sum() / len(x), raw=False
        )
    )
    out['donchian_breakout_50d'] = (df['close'] > df['high'].shift(1).rolling(50).max()).astype(int)

    sr_lookback = 120
    nearest_res = df['high'].shift(1).rolling(sr_lookback).max()
    nearest_sup = df['low'].shift(1).rolling(sr_lookback).min()
    out['distance_to_major_resistance'] = (
        (nearest_res - df['close']) / df['close'].replace(0, np.nan)
    ).clip(lower=0.0)
    out['distance_to_major_support'] = (
        (df['close'] - nearest_sup) / df['close'].replace(0, np.nan)
    ).clip(lower=0.0)
    out['failed_breakout_flag'] = (
        (out['donchian_breakout_50d'].shift(1).rolling(3).max().fillna(0) >= 1)
        & (df['close'] <= df['high'].shift(1).rolling(50).max())
    ).astype(int)

    def _days_since_last_breakout(x: pd.Series) -> float:
        arr = x.to_numpy()
        idx = np.where(arr > 0)[0]
        if len(idx) == 0:
            return 60.0
        return float(min(60, len(arr) - 1 - idx[-1]))

    out['days_since_breakout'] = out['donchian_breakout_50d'].rolling(60).apply(
        _days_since_last_breakout, raw=False
    ).fillna(60.0)

    out['price_extension_from_50sma'] = out['stock_close_vs_50sma']
    out['atr_stop_distance'] = 1.5 * out['atr_14']
    rr_raw = (
        out['distance_to_major_resistance']
        / (out['atr_stop_distance'] / df['close'].replace(0, np.nan)).replace(0, np.nan)
    )
    rr_base = rr_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Fresh-high handling: when there is no overhead resistance, do not force
    # strong breakouts to 0 R:R. Only assign strong upside when trend direction
    # confirms an advancing setup.
    no_overhead = out['distance_to_major_resistance'] <= 1e-9
    advancing = (
        (out['stock_close_vs_50sma'] > 0)
        & (out['plus_di'] > out['minus_di'])
        & (out['stock_50sma_slope_20d'] > 0)
    )

    rr_base = rr_base.where(~(no_overhead & advancing), 3.0)
    rr_base = rr_base.where(~(no_overhead & ~advancing), 1.0)

    out['risk_reward_ratio'] = rr_base.clip(0.0, 10.0)
    out['_risk_reward_ratio'] = out['risk_reward_ratio']

    # Consolidate again before data-quality and flow-inflection tail features.
    out = out.copy()

    # Kuwait data-quality module + limit-day awareness.
    daily_ret = (df['close'] / df['close'].shift(1) - 1.0) * 100.0
    out['limit_day_flag'] = (daily_ret.abs() >= 9.5).astype(int)

    # Exclude potential limit-day distortion from volatility-context fields.
    non_limit_close = df['close'].where(out['limit_day_flag'] == 0)
    bb_mid_nl = non_limit_close.rolling(20).mean()
    bb_std_nl = non_limit_close.rolling(20).std()
    bb_upper_nl = bb_mid_nl + 2.0 * bb_std_nl
    bb_lower_nl = bb_mid_nl - 2.0 * bb_std_nl
    out['bb_width_20'] = (
        (bb_upper_nl - bb_lower_nl) / bb_mid_nl.replace(0, np.nan)
    )

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    tr_filtered = tr.where(out['limit_day_flag'] == 0)
    out['atr_14'] = tr_filtered.rolling(14).mean()
    out['atr_percent'] = out['atr_14'] / df['close'].replace(0, np.nan)

    zero_volume_days_60 = (pd.to_numeric(df['volume'], errors='coerce').fillna(0.0) <= 0.0).rolling(60).sum()
    out['near_zero_volume_flag'] = (
        pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)
        < 0.1 * pd.to_numeric(df['volume'], errors='coerce').rolling(20).median().fillna(0.0)
    ).astype(int)

    gap_suspects_60 = (daily_ret.abs() >= 15.0).rolling(60).sum()
    recency_days = np.zeros(len(out), dtype=float)
    try:
        if len(df.index) > 0:
            last_bar = pd.Timestamp(df.index[-1]).to_pydatetime().date()
            recency_days[:] = float((datetime.utcnow().date() - last_bar).days)
    except Exception:
        recency_days[:] = 30.0
    recency_penalty = np.clip(recency_days / 30.0, 0.0, 1.0)

    activity_component = out['active_trading_days_ratio_60d'].fillna(0.0)
    zero_vol_component = (1.0 - (zero_volume_days_60 / 60.0).clip(0.0, 1.0)).fillna(0.0)
    corp_action_component = (1.0 - (gap_suspects_60 / 8.0).clip(0.0, 1.0)).fillna(0.0)
    recency_component = (1.0 - recency_penalty)

    out['data_quality_score'] = (
        100.0
        * (
            0.45 * activity_component
            + 0.25 * zero_vol_component
            + 0.20 * corp_action_component
            + 0.10 * recency_component
        )
    ).clip(0.0, 100.0)

    # Keep legacy names in sync where downstream still expects them.
    out['atr'] = out['atr_14']
    out['bb_bandwidth'] = out['bb_width_20']
    out['cmf'] = out['cmf_20']

    # Flow inflection features (detect the turn, not just absolute level).
    if 'cmf_20' in out.columns:
        out['cmf_20_change_5d'] = out['cmf_20'] - out['cmf_20'].shift(5)
        out['cmf_20_change_10d'] = out['cmf_20'] - out['cmf_20'].shift(10)

    if 'obv_slope_20d' in out.columns:
        out['obv_slope_change_10d'] = out['obv_slope_20d'] - out['obv_slope_20d'].shift(10)

    if 'rsi_14' in out.columns:
        out['rsi_14_change_5d'] = out['rsi_14'] - out['rsi_14'].shift(5)

    if 'close' in out.columns:
        up_close = (out['close'] > out['close'].shift(1)).astype(int)
        up_groups = (up_close == 0).cumsum()
        out['consecutive_up_closes'] = up_close.groupby(up_groups).cumsum()

        low_60 = out['close'].rolling(60).min()
        out['pct_above_60d_low'] = (
            (out['close'] / low_60.replace(0, np.nan) - 1.0) * 100.0
        )

    return out

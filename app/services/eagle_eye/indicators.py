"""
Indicator Engine — every technical indicator the analysis layer needs.
Pure numpy/pandas implementations. No TA-Lib dependency, no system libs needed.
Every indicator is unit-testable. Validated math, no hand-wavy approximations.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
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


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    lowest = df['low'].rolling(k).min()
    highest = df['high'].rolling(k).max()
    stoch_k = 100 * (df['close'] - lowest) / (highest - lowest).replace(0, np.nan)
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
    """Rolling 20-day VWAP."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    pv = tp * df['volume']
    return pv.rolling(20).sum() / df['volume'].rolling(20).sum().replace(0, np.nan)


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
    cmf_v = cmf(df, 20)
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

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
    out['ema_ribbon_aligned'] = ema_ribbon_aligned(df, CONFIG.EMA_PERIODS)
    m, s, h = macd(df)
    out['macd_line'] = m; out['macd_signal'] = s; out['macd_histogram'] = h
    a_, pd_, md_ = adx(df, CONFIG.ADX_PERIOD)
    out['adx'] = a_; out['plus_di'] = pd_; out['minus_di'] = md_
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
    sk, sd = stochastic(df, CONFIG.STOCH_K, CONFIG.STOCH_D)
    out['stoch_k'] = sk; out['stoch_d'] = sd
    out['stoch_rsi'] = stoch_rsi(df)
    out['williams_r'] = williams_r(df)
    out['cci'] = cci(df, CONFIG.CCI_PERIOD)
    out['roc'] = roc(df)
    out['tsi'] = tsi(df)
    out['ao'] = awesome_oscillator(df)
    out['connors_rsi'] = connors_rsi(df)

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
    out['obv_slope_20'] = obv_slope(df, 20)
    out['obv_slope_60'] = obv_slope(df, 60)
    out['ad_line'] = ad_line(df)
    out['cmf'] = cmf(df, CONFIG.CMF_PERIOD)
    out['mfi'] = mfi(df, CONFIG.MFI_PERIOD)
    out['vwap'] = vwap(df)
    out['vwap_distance_sigma'] = vwap_distance_sigma(df)
    out['rel_volume'] = relative_volume(df, CONFIG.VOLUME_AVG_PERIOD)
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

    # Institutional
    out['accumulation_score'] = accumulation_score(df)
    out['wyckoff_phase'] = wyckoff_phase(df)

    # Price context for downstream (kept for joins)
    out['close'] = df['close']
    out['volume'] = df['volume']
    out['high'] = df['high']
    out['low'] = df['low']

    return out

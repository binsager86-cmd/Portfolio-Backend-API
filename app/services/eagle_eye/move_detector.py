"""
Historical Move Detector — the forensic eye.
For every stock, scans its full history and identifies significant moves at
multiple thresholds, the acceleration day, and fakeouts (control group).
"""
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import List, Optional
import pandas as pd
import numpy as np
from app.services.eagle_eye.config import CONFIG


@dataclass
class MoveEvent:
    """A single detected historical move event."""
    ticker: str
    event_id: str
    direction: str                  # 'UP' or 'DOWN'
    threshold_pct: float
    start_date: date
    acceleration_date: date
    peak_date: date
    start_price: float
    acceleration_price: float
    peak_price: float
    gain_pct: float
    duration_days: int
    days_consolidating_before: int
    pre_move_volatility_pct: float
    is_fakeout: bool = False
    failed_at_pct: Optional[float] = None


def detect_moves(
    ticker: str,
    ohlcv: pd.DataFrame,
    thresholds_pct=CONFIG.MOVE_THRESHOLDS_PCT,
    min_duration_days: int = CONFIG.MIN_MOVE_DURATION_DAYS,
    max_lookahead_days: int = CONFIG.MAX_MOVE_LOOKAHEAD_DAYS,
) -> List[MoveEvent]:
    """Detect every significant move in this stock's history at every threshold."""
    if len(ohlcv) < 100:
        return []

    events: List[MoveEvent] = []
    closes = ohlcv['close'].values
    highs = ohlcv['high'].values
    lows = ohlcv['low'].values
    volumes = ohlcv['volume'].values
    dates = ohlcv.index.date

    avg_vol_20 = pd.Series(volumes).rolling(20).mean().values
    high_20 = pd.Series(highs).rolling(20).max().values

    used_starts = set()

    for threshold in sorted(thresholds_pct):
        for i in range(20, len(ohlcv) - min_duration_days):

            if i >= 10 and lows[i] > np.min(lows[i-10:i]):
                continue

            if (i, threshold) in used_starts:
                continue

            start_price = lows[i]
            end_window = min(i + max_lookahead_days, len(ohlcv))

            window_highs = highs[i+1:end_window]
            if len(window_highs) == 0:
                continue

            peak_idx_in_window = int(np.argmax(window_highs))
            peak_price = float(window_highs[peak_idx_in_window])
            peak_idx = i + 1 + peak_idx_in_window
            gain_pct = (peak_price - start_price) / start_price * 100

            if gain_pct < threshold:
                continue

            duration = peak_idx - i
            if duration < min_duration_days:
                continue

            accel_idx = None
            for j in range(i + 1, peak_idx + 1):
                if (not np.isnan(avg_vol_20[j]) and volumes[j] > 1.5 * avg_vol_20[j] and
                    not np.isnan(high_20[j-1]) and closes[j] > high_20[j-1]):
                    accel_idx = j
                    break
            if accel_idx is None:
                for j in range(i + 1, min(i + 15, peak_idx + 1)):
                    if (closes[j] / closes[j-1] - 1 > 0.02 and
                        not np.isnan(avg_vol_20[j]) and volumes[j] > avg_vol_20[j]):
                        accel_idx = j
                        break
            if accel_idx is None:
                continue

            is_fakeout = False
            failed_at = None
            check_end = min(accel_idx + 5, len(ohlcv))
            min_after_accel = np.min(lows[accel_idx:check_end])
            if min_after_accel < closes[accel_idx] * (1 - threshold/200):
                is_fakeout = True
                failed_at = float((min_after_accel - start_price) / start_price * 100)

            consolidation_count = 0
            for k in range(i - 1, max(0, i - 90), -1):
                if abs(closes[k] - start_price) / start_price < 0.05:
                    consolidation_count += 1
                else:
                    break

            if i >= 30:
                pre_returns = np.diff(np.log(closes[i-30:i]))
                pre_vol = float(np.std(pre_returns) * np.sqrt(252) * 100)
            else:
                pre_vol = np.nan

            event = MoveEvent(
                ticker=ticker,
                event_id=f"{ticker}_{dates[i].isoformat()}_up_{int(threshold)}",
                direction='UP',
                threshold_pct=threshold,
                start_date=dates[i],
                acceleration_date=dates[accel_idx],
                peak_date=dates[peak_idx],
                start_price=float(start_price),
                acceleration_price=float(closes[accel_idx]),
                peak_price=float(peak_price),
                gain_pct=float(gain_pct),
                duration_days=int(duration),
                days_consolidating_before=int(consolidation_count),
                pre_move_volatility_pct=pre_vol,
                is_fakeout=is_fakeout,
                failed_at_pct=failed_at,
            )
            events.append(event)
            for t in thresholds_pct:
                if t <= threshold:
                    used_starts.add((i, t))

    return events


def detect_fakeouts(
    ticker: str,
    ohlcv: pd.DataFrame,
    breakout_threshold_pct: float = 5.0,
) -> List[MoveEvent]:
    """Find breakouts that LOOKED real but failed quickly (control group)."""
    fakeouts = []
    if len(ohlcv) < 50:
        return fakeouts

    closes = ohlcv['close'].values
    highs = ohlcv['high'].values
    lows = ohlcv['low'].values
    volumes = ohlcv['volume'].values
    dates = ohlcv.index.date

    avg_vol = pd.Series(volumes).rolling(20).mean().values
    high_20 = pd.Series(highs).rolling(20).max().values

    for i in range(25, len(ohlcv) - 10):
        if (not np.isnan(avg_vol[i]) and volumes[i] > 1.8 * avg_vol[i] and
            not np.isnan(high_20[i-1]) and closes[i] > high_20[i-1] * 1.005):
            window = ohlcv.iloc[i+1:i+6]
            if len(window) < 3:
                continue
            min_after = window['low'].min()
            failed_pct = (min_after - closes[i]) / closes[i] * 100
            if failed_pct < -breakout_threshold_pct:
                fakeouts.append(MoveEvent(
                    ticker=ticker,
                    event_id=f"{ticker}_{dates[i].isoformat()}_fakeout",
                    direction='UP',
                    threshold_pct=0.0,
                    start_date=dates[i],
                    acceleration_date=dates[i],
                    peak_date=dates[i],
                    start_price=float(closes[i]),
                    acceleration_price=float(closes[i]),
                    peak_price=float(closes[i]),
                    gain_pct=0.0,
                    duration_days=0,
                    days_consolidating_before=0,
                    pre_move_volatility_pct=np.nan,
                    is_fakeout=True,
                    failed_at_pct=failed_pct,
                ))
    return fakeouts


def summarize_events(events: List[MoveEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in events])

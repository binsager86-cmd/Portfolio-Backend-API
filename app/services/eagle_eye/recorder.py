"""
Forensic Event Recorder.
For every detected move, capture the indicator state at multiple lookbacks
(t-90, t-60, t-30, t-14, t-7, t-3, t-1, t-0) so the engine can learn
"what did the indicators look like in the days leading up to this move?"
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from app.services.eagle_eye.config import CONFIG
from app.services.eagle_eye.move_detector import MoveEvent


@dataclass
class ForensicSnapshot:
    """The complete forensic record of one historical move."""
    event: MoveEvent
    indicator_snapshots: Dict[int, Dict[str, Any]]  # {lookback_days: {indicator: value}}
    signal_sequence: List[Dict[str, Any]]           # what fired and when (in order)


# Indicators whose state at a snapshot is most diagnostic
DIAGNOSTIC_INDICATORS = [
    'close', 'volume',
    'ema_8', 'ema_21', 'ema_50', 'ema_ribbon_aligned',
    'macd_line', 'macd_histogram',
    'adx', 'plus_di', 'minus_di',
    'supertrend',
    'rsi', 'rsi_divergence',
    'stoch_k', 'cci', 'roc', 'tsi',
    'atr', 'atr_percentile_252',
    'bb_pct_b', 'bb_bandwidth', 'bb_squeeze',
    'obv', 'obv_slope_20', 'obv_slope_60',
    'cmf', 'mfi',
    'vwap_distance_sigma', 'rel_volume',
    'accumulation_score', 'wyckoff_phase',
    'zscore_20', 'zscore_50',
    'ichimoku_cloud_pos',
]


# Signal definitions — atomic conditions that either fire or don't
SIGNAL_DEFS = {
    "obv_60d_slope_strongly_positive":
        lambda r: pd.notna(r.get('obv_slope_60')) and r['obv_slope_60'] > 0,

    "accumulation_above_65":
        lambda r: pd.notna(r.get('accumulation_score')) and r['accumulation_score'] > 65,

    "accumulation_above_75":
        lambda r: pd.notna(r.get('accumulation_score')) and r['accumulation_score'] > 75,

    "bb_squeeze_active":
        lambda r: pd.notna(r.get('bb_squeeze')) and r['bb_squeeze'] == 1,

    "macd_histogram_turned_positive":
        lambda r: pd.notna(r.get('macd_histogram')) and r['macd_histogram'] > 0,

    "adx_crossed_20":
        lambda r: pd.notna(r.get('adx')) and r['adx'] > 20,

    "adx_strong_trend":
        lambda r: pd.notna(r.get('adx')) and r['adx'] > 25,

    "plus_di_dominates":
        lambda r: (pd.notna(r.get('plus_di')) and pd.notna(r.get('minus_di'))
                   and r['plus_di'] > r['minus_di']),

    "rsi_in_bullish_zone":
        lambda r: pd.notna(r.get('rsi')) and 50 < r['rsi'] < 70,

    "rsi_bullish_divergence":
        lambda r: pd.notna(r.get('rsi_divergence')) and r['rsi_divergence'] == 1,

    "cmf_above_010":
        lambda r: pd.notna(r.get('cmf')) and r['cmf'] > 0.10,

    "mfi_in_bullish_zone":
        lambda r: pd.notna(r.get('mfi')) and 50 < r['mfi'] < 80,

    "ema_ribbon_bullish":
        lambda r: pd.notna(r.get('ema_ribbon_aligned')) and r['ema_ribbon_aligned'] == 1,

    "price_above_vwap":
        lambda r: pd.notna(r.get('vwap_distance_sigma')) and r['vwap_distance_sigma'] > 0,

    "volume_breakout_15x":
        lambda r: pd.notna(r.get('rel_volume')) and r['rel_volume'] > 1.5,

    "volume_breakout_2x":
        lambda r: pd.notna(r.get('rel_volume')) and r['rel_volume'] > 2.0,

    "wyckoff_in_accumulation":
        lambda r: r.get('wyckoff_phase') in ('A_STOPPING_ACTION', 'B_BUILDING_CAUSE', 'C_TEST_SPRING'),

    "wyckoff_in_markup":
        lambda r: r.get('wyckoff_phase') in ('D_MARKUP', 'E_MARKUP_EXPANSION'),

    "above_ichimoku_cloud":
        lambda r: pd.notna(r.get('ichimoku_cloud_pos')) and r['ichimoku_cloud_pos'] == 1,

    "supertrend_bullish":
        lambda r: pd.notna(r.get('supertrend')) and r['supertrend'] == 1,
}


def record_forensic_snapshot(
    event: MoveEvent,
    indicators_df: pd.DataFrame,
    lookbacks=CONFIG.PRE_MOVE_LOOKBACK_DAYS,
) -> ForensicSnapshot:
    """Build the full forensic record for a single move event."""

    accel_date = event.acceleration_date
    if accel_date not in indicators_df.index.date:
        accel_pos = indicators_df.index.searchsorted(pd.Timestamp(accel_date))
        if accel_pos >= len(indicators_df):
            accel_pos = len(indicators_df) - 1
    else:
        accel_pos = int(np.where(indicators_df.index.date == accel_date)[0][0])

    snapshots = {}
    for lb in lookbacks:
        pos = accel_pos - lb
        if pos < 0:
            continue
        row = indicators_df.iloc[pos]
        snap = {k: row.get(k) for k in DIAGNOSTIC_INDICATORS if k in indicators_df.columns}
        snap_clean = {}
        for k, v in snap.items():
            if isinstance(v, (np.integer, np.int64, np.int32)):
                snap_clean[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                snap_clean[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, (np.bool_,)):
                snap_clean[k] = bool(v)
            else:
                snap_clean[k] = v
        snapshots[lb] = snap_clean

    sequence = []
    for signal_name, signal_fn in SIGNAL_DEFS.items():
        first_fire_pos = None
        for offset in range(0, min(91, accel_pos + 1)):
            pos = accel_pos - offset
            if pos < 0:
                break
            try:
                if signal_fn(indicators_df.iloc[pos]):
                    first_fire_pos = pos
                else:
                    if first_fire_pos is not None:
                        break
            except Exception:
                continue
        if first_fire_pos is not None:
            days_before = accel_pos - first_fire_pos
            sequence.append({
                "signal": signal_name,
                "days_before_acceleration": days_before,
                "fired_on": indicators_df.index[first_fire_pos].date().isoformat(),
            })
    sequence.sort(key=lambda x: -x['days_before_acceleration'])

    return ForensicSnapshot(
        event=event,
        indicator_snapshots=snapshots,
        signal_sequence=sequence,
    )


def record_all_events(
    events: List[MoveEvent],
    indicators_df: pd.DataFrame,
) -> List[ForensicSnapshot]:
    snapshots = []
    for ev in events:
        try:
            snap = record_forensic_snapshot(ev, indicators_df)
            snapshots.append(snap)
        except Exception as e:
            print(f"  warn: failed to snapshot {ev.event_id}: {e}")
    return snapshots

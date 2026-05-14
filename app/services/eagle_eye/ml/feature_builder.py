from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.services.eagle_eye.adapter import StockMeta, TickerChartAdapter
from app.services.eagle_eye.config import STAGES
from app.services.eagle_eye.indicators import compute_all_indicators
from app.services.eagle_eye.move_detector import detect_fakeouts, detect_moves
from app.services.eagle_eye.recorder import SIGNAL_DEFS
from app.services.eagle_eye.stage_classifier import classify_stage
from app.services.eagle_eye.store import list_tickers_with_ohlcv, load_ohlcv

LOGGER = logging.getLogger(__name__)

REGIMES = ("RISK_ON", "NEUTRAL", "RISK_OFF")
CORE_VELOCITY_COLUMNS = (
    "rsi",
    "macd_histogram",
    "obv",
    "accumulation_score",
    "cmf",
    "adx",
    "rel_volume",
)
TRAJECTORY_COLUMNS = (
    "obv",
    "accumulation_score",
    "bb_bandwidth",
)
TRAJECTORY_OFFSETS = (30, 14, 7, 3, 1, 0)
CONTEXT_LOOKBACKS = (1, 3, 7, 14, 30, 60, 90)
CONTEXT_COLUMNS = (
    "rsi",
    "macd_histogram",
    "obv",
    "accumulation_score",
    "cmf",
    "adx",
    "rel_volume",
    "bb_bandwidth",
)

# Indicators computed with right-side context are unsafe for causal training.
LEAKY_INDICATOR_COLUMNS = {
    "swing_high",
    "swing_low",
}

# Gregorian date windows for Ramadan (approximate, sufficient for a binary seasonality flag).
RAMADAN_WINDOWS = (
    (date(2020, 4, 23), date(2020, 5, 23)),
    (date(2021, 4, 13), date(2021, 5, 12)),
    (date(2022, 4, 2), date(2022, 5, 1)),
    (date(2023, 3, 22), date(2023, 4, 21)),
    (date(2024, 3, 10), date(2024, 4, 9)),
    (date(2025, 2, 28), date(2025, 3, 30)),
    (date(2026, 2, 17), date(2026, 3, 19)),
    (date(2027, 2, 7), date(2027, 3, 8)),
    (date(2028, 1, 27), date(2028, 2, 25)),
    (date(2029, 1, 15), date(2029, 2, 13)),
    (date(2030, 1, 5), date(2030, 2, 3)),
)

NON_FEATURE_COLUMNS = {
    "ticker",
    "event_id",
    "event_date",
    "acceleration_date",
    "start_date",
    "peak_date",
    "sector_raw",
    "market_tier_raw",
    "day_of_week_raw",
    "month_raw",
    "regime_at_event",
    "current_stage",
    "stage_before",
    "is_fakeout",
    "threshold_pct",
    "duration_days",
    "acceleration_price",
    "peak_price",
    "gain_pct",
    "failed_at_pct",
    "tp1_hit_day",
    "tp2_hit_day",
    "stop_hit_day",
    "max_excursion_pct",
    "days_in_current_stage",
    "earliest_signal_lead_days",
    "signal_acceleration",
    "n_signals_fired_in_last_30d",
    "n_signals_fired_in_last_7d",
    *(f"current_stage_{stage.lower()}" for stage in STAGES),
    *(f"stage_before_{stage.lower()}" for stage in STAGES),
}

NON_FEATURE_PREFIXES = (
    "current_stage_",
    "stage_before_",
)


@dataclass
class FeatureBuildResult:
    frame: pd.DataFrame
    rejected_counts: Dict[str, int]
    total_before: int
    total_after: int


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(v) or math.isinf(v):
        return float("nan")
    return v


def _log1p_or_nan(value: Any) -> float:
    v = _safe_float(value)
    if math.isnan(v) or v < 0:
        return float("nan")
    return float(math.log1p(v))


def _is_ramadan_period(dt: date) -> int:
    for start, end in RAMADAN_WINDOWS:
        if start <= dt <= end:
            return 1
    return 0


def _is_earnings_window(dt: date) -> int:
    return 1 if dt.month in {1, 2, 4, 5, 7, 8, 10, 11} else 0


def _encode_wyckoff(phase: Any) -> float:
    mapping = {
        "A_STOPPING_ACTION": 1.0,
        "B_BUILDING_CAUSE": 2.0,
        "C_TEST_SPRING": 3.0,
        "D_MARKUP": 4.0,
        "E_MARKUP_EXPANSION": 5.0,
    }
    return mapping.get(str(phase or "").upper(), 0.0)


def _normalize_sector(name_en: str, stock_meta_sector: Optional[str], ticker: str) -> str:
    sector = (stock_meta_sector or "").strip().lower()
    if sector and sector != "kuwait":
        return sector.replace(" ", "_")

    n = (name_en or ticker or "").lower()
    if "bank" in n:
        return "banking"
    if "real estate" in n or "resort" in n or "hotel" in n:
        return "real_estate"
    if "insurance" in n or "takaful" in n or "reinsurance" in n:
        return "insurance"
    if "telecom" in n or "telecommunications" in n or "mobile" in n:
        return "telecom"
    if "technology" in n or "digital" in n or "systems" in n:
        return "technology"
    if "petroleum" in n or "energy" in n or "fuel" in n or "power" in n:
        return "energy"
    if "airways" in n or "aviation" in n or "logistics" in n or "transport" in n or "ship" in n:
        return "transport"
    if "cement" in n or "industr" in n or "engineering" in n or "electrical" in n:
        return "industrial"
    if "investment" in n or "financial" in n or "capital" in n or "leasing" in n:
        return "investment"
    if "food" in n or "consumer" in n or "clinic" in n or "cinema" in n or "retail" in n:
        return "consumer"
    return "holding_misc"


def _build_stock_meta_map() -> Dict[str, StockMeta]:
    adapter = TickerChartAdapter()
    return {s.ticker.upper(): s for s in adapter.list_stocks()}


def _build_regime_frame(
    start: date,
    end: date,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    log = logger or LOGGER
    adapter = TickerChartAdapter()

    try:
        pmi = adapter.get_market_index("PMI", start, end)
    except Exception as exc:  # pragma: no cover - fallback branch
        log.warning("Regime PMI fetch failed: %s", exc)
        pmi = pd.DataFrame()

    try:
        brent = adapter.get_market_index("BRENT", start, end)
    except Exception as exc:  # pragma: no cover - fallback branch
        log.warning("Regime Brent fetch failed: %s", exc)
        brent = pd.DataFrame()

    index = pd.date_range(start=start, end=end, freq="D")
    regime = pd.DataFrame(index=index)

    if not pmi.empty and "close" in pmi.columns:
        pmi_close = pmi["close"].copy()
        pmi_close.index = pd.to_datetime(pmi_close.index).normalize()
        pmi_close = pmi_close[~pmi_close.index.duplicated(keep="last")]
        regime["pmi_close"] = pmi_close.reindex(index).ffill()
    else:
        regime["pmi_close"] = 0.0

    if not brent.empty and "close" in brent.columns:
        brent_close = brent["close"].copy()
        brent_close.index = pd.to_datetime(brent_close.index).normalize()
        brent_close = brent_close[~brent_close.index.duplicated(keep="last")]
        regime["brent_close"] = brent_close.reindex(index).ffill()
    else:
        regime["brent_close"] = 0.0

    regime["pmi_50w_trend"] = regime["pmi_close"].pct_change(250)
    regime["brent_30d_trend"] = regime["brent_close"].pct_change(30)

    def _state(row: pd.Series) -> str:
        p = _safe_float(row.get("pmi_50w_trend"))
        b = _safe_float(row.get("brent_30d_trend"))
        if p > 0 and b > 0:
            return "RISK_ON"
        if p < 0 and b < 0:
            return "RISK_OFF"
        return "NEUTRAL"

    regime["regime_at_event"] = regime.apply(_state, axis=1)
    regime = regime[["pmi_50w_trend", "brent_30d_trend", "regime_at_event"]]
    regime = regime.ffill().fillna({"pmi_50w_trend": 0.0, "brent_30d_trend": 0.0, "regime_at_event": "NEUTRAL"})
    return regime


def _lookup_regime(regime_frame: pd.DataFrame, dt: pd.Timestamp) -> Tuple[str, float, float]:
    if regime_frame.empty:
        return "NEUTRAL", 0.0, 0.0
    key = pd.Timestamp(dt).normalize()
    found = regime_frame.loc[:key].tail(1)
    if found.empty:
        return "NEUTRAL", 0.0, 0.0
    row = found.iloc[0]
    return (
        str(row.get("regime_at_event") or "NEUTRAL"),
        _safe_float(row.get("pmi_50w_trend") or 0.0),
        _safe_float(row.get("brent_30d_trend") or 0.0),
    )


def _value_at_offset(df: pd.DataFrame, pos: int, col: str, offset: int) -> float:
    i = pos - offset
    if i < 0 or i >= len(df):
        return float("nan")
    return _safe_float(df.iloc[i].get(col))


def _velocity(df: pd.DataFrame, pos: int, col: str, lookback: int = 3) -> float:
    now_v = _value_at_offset(df, pos, col, 0)
    past_v = _value_at_offset(df, pos, col, lookback)
    if math.isnan(now_v) or math.isnan(past_v) or lookback == 0:
        return float("nan")
    return (now_v - past_v) / float(lookback)


def _trajectory_slope(df: pd.DataFrame, pos: int, col: str, offsets: Sequence[int]) -> float:
    values: List[float] = []
    xs: List[float] = []
    for off in offsets:
        v = _value_at_offset(df, pos, col, off)
        if not math.isnan(v):
            xs.append(float(-off))
            values.append(v)
    if len(values) < 3:
        return float("nan")
    return float(np.polyfit(xs, values, deg=1)[0])


def _days_since_flag(df: pd.DataFrame, pos: int, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    for i in range(pos, -1, -1):
        try:
            val = int(df.iloc[i][col])
        except Exception:
            val = 0
        if val == 1:
            return float(pos - i)
    return float("nan")


def _compute_trade_outcome(ohlcv: pd.DataFrame, accel_pos: int, entry: float) -> Dict[str, Any]:
    if math.isnan(entry) or entry <= 0:
        return {
            "tp1_hit_day": None,
            "tp2_hit_day": None,
            "stop_hit_day": None,
            "max_excursion_pct": float("nan"),
        }

    future = ohlcv.iloc[accel_pos + 1: accel_pos + 21]
    if future.empty:
        return {
            "tp1_hit_day": None,
            "tp2_hit_day": None,
            "stop_hit_day": None,
            "max_excursion_pct": float("nan"),
        }

    tp1_target = entry * 1.05
    tp2_target = entry * 1.10
    stop_target = entry * 0.95

    tp1_day: Optional[int] = None
    tp2_day: Optional[int] = None
    stop_day: Optional[int] = None

    for day_num, (_, row) in enumerate(future.iterrows(), start=1):
        day_high = _safe_float(row.get("high"))
        day_low = _safe_float(row.get("low"))

        stop_hit = (not math.isnan(day_low)) and day_low <= stop_target
        tp1_hit = (not math.isnan(day_high)) and day_high >= tp1_target
        tp2_hit = (not math.isnan(day_high)) and day_high >= tp2_target

        # Ambiguous intraday touch (both hit): choose conservative stop-first ordering.
        if stop_hit and (tp1_hit or tp2_hit):
            if tp1_day is None and tp2_day is None:
                stop_day = day_num
                break

        if tp1_hit and tp1_day is None:
            tp1_day = day_num
        if tp2_hit and tp2_day is None:
            tp2_day = day_num

        if stop_hit and stop_day is None and tp2_day is None:
            stop_day = day_num
            break

    if stop_day is None:
        scope = future
    else:
        scope = future.iloc[: max(stop_day - 1, 1)]

    scope_high = _safe_float(scope["high"].max()) if not scope.empty else float("nan")
    if math.isnan(scope_high):
        scope_high = entry
    max_exc = (scope_high / entry - 1.0) * 100.0

    return {
        "tp1_hit_day": tp1_day,
        "tp2_hit_day": tp2_day,
        "stop_hit_day": stop_day,
        "max_excursion_pct": float(max_exc),
    }


def _signal_slug(name: str) -> str:
    chars: List[str] = []
    for ch in str(name).lower():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    out = "".join(chars)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _extract_signal_features_asof(indicators: pd.DataFrame, pos: int) -> Dict[str, float]:
    """
    Build signal features at timestamp T=pos using only bars up to and including T.
    """
    features: Dict[str, float] = {}

    last_7_start = max(0, pos - 6)
    last_30_start = max(0, pos - 29)
    last_60_start = max(0, pos - 59)
    last_90_start = max(0, pos - 89)

    n_7 = 0
    n_30 = 0
    n_60 = 0

    for signal_name, signal_fn in SIGNAL_DEFS.items():
        fired_positions: List[int] = []
        for i in range(last_90_start, pos + 1):
            try:
                if signal_fn(indicators.iloc[i]):
                    fired_positions.append(i)
            except Exception:
                continue

        slug = _signal_slug(signal_name)
        if fired_positions:
            first_fire = min(fired_positions)
            days_since_first = float(pos - first_fire)
        else:
            days_since_first = float("nan")

        in_7 = any(i >= last_7_start for i in fired_positions)
        in_30 = any(i >= last_30_start for i in fired_positions)
        in_60 = any(i >= last_60_start for i in fired_positions)

        if in_7:
            n_7 += 1
        if in_30:
            n_30 += 1
        if in_60:
            n_60 += 1

        features[f"days_since_signal_{slug}_first_fired_as_of_t"] = days_since_first
        features[f"signal_{slug}_active_last_7d_as_of_t"] = 1.0 if in_7 else 0.0
        features[f"signal_{slug}_active_last_30d_as_of_t"] = 1.0 if in_30 else 0.0

    features["n_distinct_signals_active_in_last_7d_as_of_t"] = float(n_7)
    features["n_distinct_signals_active_in_last_30d_as_of_t"] = float(n_30)
    features["n_distinct_signals_active_in_last_60d_as_of_t"] = float(n_60)

    if n_60 > 0:
        features["signal_density_acceleration_30d_vs_60d_as_of_t"] = float(n_30 / (n_60 / 2.0))
    else:
        features["signal_density_acceleration_30d_vs_60d_as_of_t"] = 0.0

    return features


def _dedupe_move_events_for_ml(events: Sequence[Any]) -> List[Any]:
    """
    Keep one representative event per (acceleration_date, fakeout) anchor.

    The detector can emit many overlapping starts that share the same
    acceleration day and produce near-identical feature rows. Keeping only one
    prevents row-duplication leakage across CV folds and OOT checks.
    """
    if not events:
        return []

    best_by_key: Dict[Tuple[date, int], Any] = {}
    for ev in events:
        accel = getattr(ev, "acceleration_date", None)
        if accel is None:
            continue

        key = (accel, int(bool(getattr(ev, "is_fakeout", False))))
        current = best_by_key.get(key)
        if current is None:
            best_by_key[key] = ev
            continue

        cur_thr = _safe_float(getattr(current, "threshold_pct", 0.0))
        new_thr = _safe_float(getattr(ev, "threshold_pct", 0.0))
        cur_gain = _safe_float(getattr(current, "gain_pct", 0.0))
        new_gain = _safe_float(getattr(ev, "gain_pct", 0.0))

        # Prefer stronger threshold, then larger realized gain.
        if (new_thr, new_gain) > (cur_thr, cur_gain):
            best_by_key[key] = ev

    deduped = sorted(best_by_key.values(), key=lambda e: (getattr(e, "acceleration_date", date.min), getattr(e, "event_id", "")))
    return deduped


def _sample_non_event_positions(
    ohlcv: pd.DataFrame,
    accel_positions: Sequence[int],
    n_samples: int,
    seed: int = 42,
) -> List[int]:
    """
    Sample calm non-event anchors for negative training rows.
    """
    if ohlcv is None or ohlcv.empty or n_samples <= 0:
        return []

    returns_abs = ohlcv["close"].pct_change().abs().fillna(0.0)
    intraday = ((ohlcv["high"] - ohlcv["low"]) / ohlcv["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    blocked: set[int] = set()
    for pos in accel_positions:
        for j in range(pos - 5, pos + 6):
            if 0 <= j < len(ohlcv):
                blocked.add(j)

    candidates: List[int] = []
    start = max(90, 1)
    end = max(start, len(ohlcv) - 21)
    for pos in range(start, end):
        if pos in blocked:
            continue
        if returns_abs.iloc[pos] > 0.015:
            continue
        if intraday.iloc[pos] > 0.03:
            continue
        candidates.append(pos)

    if not candidates:
        return []

    if len(candidates) <= n_samples:
        return sorted(candidates)

    rng = np.random.default_rng(seed)
    picked = rng.choice(candidates, size=n_samples, replace=False)
    return sorted(int(x) for x in picked)


def build_event_feature_rows_for_ticker(
    ticker: str,
    ohlcv: pd.DataFrame,
    stock_meta: Optional[StockMeta],
    regime_frame: pd.DataFrame,
    include_fakeouts: bool = True,
) -> List[Dict[str, Any]]:
    if ohlcv is None or ohlcv.empty or len(ohlcv) < 120:
        return []

    indicators = compute_all_indicators(ohlcv)
    if indicators.empty:
        return []

    stage_series = indicators.apply(lambda row: classify_stage(row.to_dict()), axis=1)

    moves = detect_moves(ticker, ohlcv)
    if include_fakeouts:
        moves.extend(detect_fakeouts(ticker, ohlcv))

    # moves = _dedupe_move_events_for_ml(moves)  # Relaxed deduplication

    accel_positions: List[int] = []
    for event in moves:
        accel_raw = pd.Timestamp(event.acceleration_date)
        accel_pos = indicators.index.get_indexer([accel_raw], method="nearest")[0]
        if 0 <= accel_pos < len(indicators):
            accel_positions.append(accel_pos)

    negative_positions = _sample_non_event_positions(
        ohlcv=ohlcv,
        accel_positions=accel_positions,
        n_samples=len(moves) * 5,
        seed=42,
    )

    rows: List[Dict[str, Any]] = []

    name_en = stock_meta.name_en if stock_meta else ticker
    sector = _normalize_sector(name_en, stock_meta.sector if stock_meta else None, ticker)
    market_tier = (stock_meta.market_tier if stock_meta and stock_meta.market_tier else "premier").lower()

    anchor_specs: List[Tuple[Optional[Any], int, bool]] = []
    for event in moves:
        accel_raw = pd.Timestamp(event.acceleration_date)
        accel_pos = indicators.index.get_indexer([accel_raw], method="nearest")[0]
        if accel_pos <= 0 or accel_pos >= len(indicators):
            continue
        anchor_specs.append((event, accel_pos - 1, False))

    for pos in negative_positions:
        if pos <= 0 or pos >= len(indicators):
            continue
        anchor_specs.append((None, int(pos), True))

    anchor_specs.sort(key=lambda x: x[1])

    for event, pred_pos, is_control in anchor_specs:
        pred_ts = pd.Timestamp(indicators.index[pred_pos]).normalize()

        if is_control:
            accel_pos = min(pred_pos + 1, len(indicators) - 1)
            accel_ts = pd.Timestamp(indicators.index[accel_pos]).normalize()
        else:
            accel_raw = pd.Timestamp(event.acceleration_date)
            accel_pos = indicators.index.get_indexer([accel_raw], method="nearest")[0]
            if accel_pos <= pred_pos or accel_pos >= len(indicators):
                continue
            accel_ts = pd.Timestamp(indicators.index[accel_pos]).normalize()

        stage_now = stage_series.iloc[pred_pos]

        days_in_stage = 0
        stage_before = "UNKNOWN"
        for i in range(pred_pos, -1, -1):
            if stage_series.iloc[i] == stage_now:
                days_in_stage += 1
            else:
                stage_before = stage_series.iloc[i]
                break

        if is_control:
            close_t = _safe_float(indicators.iloc[pred_pos].get("close"))
            entry = close_t
            outcome = {
                "tp1_hit_day": None,
                "tp2_hit_day": None,
                "stop_hit_day": None,
                "max_excursion_pct": 0.0,
            }
            event_id = f"{ticker.upper()}_{pred_ts.date().isoformat()}_non_event"
            start_date = None
            peak_date = None
            is_fakeout = 0
            threshold_pct = 0.0
            duration_days = 0.0
            peak_price = float("nan")
            gain_pct = 0.0
            failed_at_pct = float("nan")
        else:
            close_t = _safe_float(indicators.iloc[pred_pos].get("close"))
            entry = _safe_float(event.acceleration_price)
            outcome = _compute_trade_outcome(ohlcv, accel_pos, entry)
            event_id = event.event_id
            start_date = event.start_date.isoformat() if event.start_date else None
            peak_date = event.peak_date.isoformat() if event.peak_date else None
            is_fakeout = int(bool(event.is_fakeout))
            threshold_pct = _safe_float(event.threshold_pct)
            duration_days = _safe_float(event.duration_days)
            peak_price = _safe_float(event.peak_price)
            gain_pct = _safe_float(event.gain_pct)
            failed_at_pct = _safe_float(event.failed_at_pct)

        signal_features = _extract_signal_features_asof(indicators, pred_pos)
        regime_name, pmi_trend, brent_trend = _lookup_regime(regime_frame, pred_ts)

        cap_price = close_t if not math.isnan(close_t) else entry

        row: Dict[str, Any] = {
            "ticker": ticker.upper(),
            "event_id": event_id,
            "event_date": pred_ts.date().isoformat(),
            "acceleration_date": accel_ts.date().isoformat(),
            "start_date": start_date,
            "peak_date": peak_date,
            "is_fakeout": is_fakeout,
            "threshold_pct": threshold_pct,
            "duration_days": duration_days,
            "acceleration_price": entry,
            "peak_price": peak_price,
            "gain_pct": gain_pct,
            "failed_at_pct": failed_at_pct,
            "sector": sector,
            "market_tier": market_tier,
            "log_market_cap": _log1p_or_nan(
                (stock_meta.shares_outstanding if stock_meta else None) * cap_price
                if stock_meta and stock_meta.shares_outstanding and not math.isnan(cap_price)
                else None
            ),
            "avg_daily_turnover_log": _log1p_or_nan(
                ohlcv["turnover_kwd"].iloc[max(0, pred_pos - 60):pred_pos].mean() if "turnover_kwd" in ohlcv.columns else None
            ),
            "current_stage": stage_now,
            "stage_before": stage_before,
            "days_in_current_stage": float(days_in_stage),
            "regime_at_event": regime_name,
            "pmi_50w_trend": pmi_trend,
            "brent_30d_trend": brent_trend,
            "is_ramadan_period": float(_is_ramadan_period(pred_ts.date())),
            "is_earnings_window": float(_is_earnings_window(pred_ts.date())),
            "day_of_week": float(pred_ts.weekday()),
            "month": float(pred_ts.month),
            "tp1_hit_day": outcome["tp1_hit_day"],
            "tp2_hit_day": outcome["tp2_hit_day"],
            "stop_hit_day": outcome["stop_hit_day"],
            "max_excursion_pct": outcome["max_excursion_pct"],
        }
        row.update(signal_features)

        for col in indicators.columns:
            if col in LEAKY_INDICATOR_COLUMNS:
                continue
            val = indicators.iloc[pred_pos].get(col)
            if col == "wyckoff_phase":
                row[f"t0_{col}_code"] = _encode_wyckoff(val)
            else:
                row[f"t0_{col}"] = _safe_float(val)

        for col in CORE_VELOCITY_COLUMNS:
            row[f"{col}_velocity"] = _velocity(indicators, pred_pos, col, 3)

        row["obv_trajectory_slope"] = _trajectory_slope(indicators, pred_pos, "obv", TRAJECTORY_OFFSETS)
        row["accumulation_trajectory_slope"] = _trajectory_slope(indicators, pred_pos, "accumulation_score", TRAJECTORY_OFFSETS)
        row["bb_bandwidth_trajectory"] = _trajectory_slope(indicators, pred_pos, "bb_bandwidth", TRAJECTORY_OFFSETS)

        for lookback in CONTEXT_LOOKBACKS:
            for col in CONTEXT_COLUMNS:
                row[f"t{lookback}_{col}"] = _value_at_offset(indicators, pred_pos, col, lookback)

        for stage_name in STAGES:
            stage_key = stage_name.lower()
            row[f"current_stage_{stage_key}"] = 1.0 if stage_now == stage_name else 0.0
            row[f"stage_before_{stage_key}"] = 1.0 if stage_before == stage_name else 0.0

        for regime in REGIMES:
            row[f"regime_{regime.lower()}"] = 1.0 if regime_name == regime else 0.0

        rows.append(row)

    return rows


def load_forensic_events_from_db(
    db_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Best-effort loader for pre-materialized forensic cache tables.

    Expected fields are flexible; the fallback generator is used when no usable
    event table is found. This keeps the trainer compatible with both older and
    future schemas.
    """
    log = logger or LOGGER
    settings = get_settings()
    path = db_path or settings.database_abs_path

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]

    preferred = [
        "ee_events_cache",
        "ee_forensic_events",
        "forensic_events",
        "eagle_eye_events",
    ]
    candidates = [t for t in preferred if t in tables]
    if not candidates:
        candidates = [t for t in tables if "event" in t.lower() and t.startswith("ee_")]

    selected: Optional[str] = None
    cols: List[str] = []
    for table in candidates:
        c = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]  # nosec B608
        cset = {x.lower() for x in c}
        if "ticker" in cset and ("acceleration_date" in cset or "event_date" in cset):
            selected = table
            cols = c
            break

    if not selected:
        conn.close()
        return []

    log.info("Using forensic event cache table: %s", selected)
    records: List[Dict[str, Any]] = []
    rows = conn.execute(f"SELECT * FROM {selected}").fetchall()  # nosec B608
    conn.close()

    for r in rows:
        item = dict(r)
        for key in (
            "indicator_snapshots",
            "indicator_snapshots_json",
            "snapshots_json",
            "signal_sequence",
            "signal_sequence_json",
        ):
            if key in item and isinstance(item[key], str):
                try:
                    item[key] = json.loads(item[key])
                except Exception:
                    pass
        records.append(item)

    return records


def build_events_from_ohlcv_cache(
    tickers: Optional[Sequence[str]] = None,
    include_fakeouts: bool = True,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    log = logger or LOGGER
    meta_map = _build_stock_meta_map()

    if tickers is None:
        tickers = list_tickers_with_ohlcv()

    end = date.today()
    start = date(end.year - 4, end.month, min(end.day, 28))
    regime = _build_regime_frame(start, end, logger=log)

    all_rows: List[Dict[str, Any]] = []
    for ticker in tickers:
        try:
            ohlcv = load_ohlcv(ticker)
            rows = build_event_feature_rows_for_ticker(
                ticker=ticker,
                ohlcv=ohlcv,
                stock_meta=meta_map.get(ticker.upper()),
                regime_frame=regime,
                include_fakeouts=include_fakeouts,
            )
            all_rows.extend(rows)
        except Exception as exc:
            log.warning("Event build failed for %s: %s", ticker, exc)

    return all_rows


def build_feature_matrix(
    events: Sequence[Mapping[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> FeatureBuildResult:
    log = logger or LOGGER
    if not events:
        return FeatureBuildResult(frame=pd.DataFrame(), rejected_counts={}, total_before=0, total_after=0)

    df = pd.DataFrame(events).copy()
    total_before = len(df)

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    if "is_fakeout" not in df.columns:
        df["is_fakeout"] = 0

    # before_anchor_dedupe = len(df)
    # df = df.sort_values(["ticker", "event_date", "is_fakeout", "threshold_pct", "gain_pct"], ascending=[True, True, True, False, False])
    # df = df.drop_duplicates(subset=["ticker", "event_date", "is_fakeout"], keep="first")
    # dropped = before_anchor_dedupe - len(df)
    # if dropped > 0:
    #     log.info("Dropped %d duplicate anchor rows (same ticker/event_date/fakeout)", dropped)

    df = df.sort_values(["ticker", "event_date"]).reset_index(drop=True)

    df["sector_raw"] = df.get("sector", "unknown").fillna("unknown").astype(str)
    df["market_tier_raw"] = df.get("market_tier", "unknown").fillna("unknown").astype(str)
    df["day_of_week_raw"] = df.get("day_of_week")
    df["month_raw"] = df.get("month")

    df = pd.get_dummies(
        df,
        columns=["sector", "market_tier", "day_of_week", "month"],
        prefix=["sector", "tier", "dow", "month"],
        dummy_na=False,
    )

    feature_cols = get_feature_columns(df)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_ratio = df[feature_cols].isna().mean(axis=1)
    reject_mask = missing_ratio > 0.30
    rejected_counts = df.loc[reject_mask].groupby("ticker").size().astype(int).to_dict()

    if rejected_counts:
        for tk, n in sorted(rejected_counts.items()):
            log.info("Rejected %d events for %s due to >30%% NaN feature ratio", n, tk)

    df = df.loc[~reject_mask].copy()

    if df.empty:
        return FeatureBuildResult(frame=df, rejected_counts=rejected_counts, total_before=total_before, total_after=0)

    # Category-aware imputation: forward-fill within ticker, then ticker median, then global median.
    df = df.sort_values(["ticker", "event_date"]).reset_index(drop=True)
    df[feature_cols] = df.groupby("ticker", dropna=False)[feature_cols].ffill()

    ticker_medians = df.groupby("ticker", dropna=False)[feature_cols].transform("median")
    df[feature_cols] = df[feature_cols].fillna(ticker_medians)

    global_median = df[feature_cols].median(numeric_only=True)
    df[feature_cols] = df[feature_cols].fillna(global_median)

    one_hot_cols = [c for c in feature_cols if c.startswith(("sector_", "tier_", "dow_", "month_", "regime_", "current_stage_", "stage_before_"))]
    if one_hot_cols:
        df[one_hot_cols] = df[one_hot_cols].fillna(0.0)

    df[feature_cols] = df[feature_cols].fillna(0.0)

    total_after = len(df)
    return FeatureBuildResult(
        frame=df,
        rejected_counts=rejected_counts,
        total_before=total_before,
        total_after=total_after,
    )


def get_feature_columns(frame: pd.DataFrame) -> List[str]:
    return [
        c
        for c in frame.columns
        if c not in NON_FEATURE_COLUMNS
        and not c.startswith("y_")
        and not c.startswith(NON_FEATURE_PREFIXES)
    ]

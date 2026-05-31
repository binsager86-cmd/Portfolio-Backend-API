from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.services.eagle_eye.adapter import StockMeta, TickerChartAdapter
from app.services.eagle_eye.config import STAGES
from app.services.eagle_eye.indicators import compute_all_indicators
from app.services.eagle_eye.ml.labelers import compute_training_target, label_phases
from app.services.eagle_eye.move_detector import detect_fakeouts, detect_moves
from app.services.eagle_eye.recorder import SIGNAL_DEFS
from app.services.eagle_eye.stage_classifier import classify_stage
from app.services.eagle_eye.store import list_tickers_with_ohlcv, load_ohlcv

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Multi-horizon forward windows — variable profit capture per stock personality
# The ML learns WHICH horizon best fits each setup, not a fixed 20-day gate.
# ---------------------------------------------------------------------------
MULTI_HORIZON_DAYS = (20, 40, 60, 90, 180)

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
    "date",
    "bar_index",
    "target_score",
    "label",
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
    # Multi-horizon outcomes — forward-looking, must never become features
    "max_gain_pct_20d",
    "max_gain_pct_40d",
    "max_gain_pct_60d",
    "max_gain_pct_90d",
    "max_gain_pct_180d",
    "days_to_peak",
    "max_drawdown_before_peak_pct",
    "risk_adjusted_gain",
    "y_opportunity_score",
    *(f"current_stage_{stage.lower()}" for stage in STAGES),
    *(f"stage_before_{stage.lower()}" for stage in STAGES),
}

NON_FEATURE_PREFIXES = (
    "current_stage_",
    "stage_before_",
)

CURATED_FEATURES: Dict[str, str] = {
    # Momentum
    "rsi": "RSI value 0-100",
    "rsi_velocity_5d": "RSI change over 5 bars",
    "macd_histogram": "MACD histogram value",
    "macd_hist_velocity_5d": "MACD histogram change over 5 bars",
    "adx": "ADX trend strength",
    "stochastic_k": "Stochastic %K",
    "momentum_confluence": "Momentum confluence score",
    # Flow / volume
    "cmf": "Chaikin Money Flow",
    "cmf_change_5d": "CMF change over 5 bars",
    "obv_change_10d": "OBV change over 10 bars",
    "obv_change_20d": "OBV change over 20 bars",
    "volume_ratio_20d": "Volume vs 20-day average",
    "volume_acceleration": "5-day volume SMA divided by 20-day SMA",
    "green_red_volume_ratio_10d": "Up-volume/down-volume ratio over 10 bars",
    "volume_flow_confluence": "Flow confluence score",
    # Trend
    "ema_ribbon_aligned": "EMA ribbon alignment",
    "di_spread": "+DI minus -DI",
    "trend_confluence": "Trend confluence score",
    # Position
    "price_extension_from_20d_low_pct": "Distance above 20-day low",
    "position_in_60d_range_pct": "Position in 60-day range",
    "selloff_from_60d_high_pct": "Distance below 60-day high",
    "bounce_from_10d_low_pct": "Recovery from 10-day low",
    "consecutive_higher_lows": "Consecutive higher lows",
    # Institutional
    "accumulation_score": "Accumulation composite",
    "institutional_confluence": "Institutional confluence score",
    # Composite
    "overall_confluence": "Weighted confluence across categories",
    "flow_momentum_divergence": "Flow confluence minus momentum confluence",
    "capitulation_reversal_score": "Capitulation reversal composite",
    # Context
    "stage_encoded": "Encoded lifecycle stage",
}

CURATED_FEATURE_ORDER: Tuple[str, ...] = tuple(CURATED_FEATURES.keys())

STAGE_ENCODING: Dict[str, float] = {
    "DORMANT": 0.0,
    "STEALTH_ACCUMULATION": 2.0,
    "EARLY_BREAKOUT": 3.0,
    "MARKUP_TRENDING": 4.0,
    "ACCELERATION_CLIMAX": 5.0,
    "DISTRIBUTION_TOPPING": 6.0,
    "MARKDOWN_DECLINE": 7.0,
    "CAPITULATION_EXHAUSTION": 1.0,
}

CURATED_FALLBACKS: Dict[str, Tuple[str, ...]] = {
    "stochastic_k": ("stoch_k",),
    "di_spread": ("plus_di_minus_di_diff",),
    "volume_ratio_20d": ("rel_volume",),
    "consecutive_higher_lows": ("consecutive_higher_lows_5d",),
}


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


def build_curated_feature_row(indicators_dict: Mapping[str, Any]) -> Dict[str, float]:
    """
    Extract only the curated v14 feature set from an indicator snapshot.
    """
    snapshot = dict(indicators_dict or {})
    stage_name = str(snapshot.get("stage") or classify_stage(snapshot))
    stage_code = STAGE_ENCODING.get(stage_name, 0.0)

    row: Dict[str, float] = {}
    for feature_name in CURATED_FEATURE_ORDER:
        if feature_name == "stage_encoded":
            row[feature_name] = float(stage_code)
            continue

        val = snapshot.get(feature_name)
        if val is None:
            for alias in CURATED_FALLBACKS.get(feature_name, ()):  # pragma: no branch
                val = snapshot.get(alias)
                if val is not None:
                    break

        fval = _safe_float(val)
        row[feature_name] = float("nan") if math.isnan(fval) else float(fval)

    return row


def build_training_dataset(
    ticker: str,
    df: pd.DataFrame,
    indicators: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build one row per bar using curated features and direct 0-100 target score.
    """
    if df is None or df.empty or indicators is None or indicators.empty:
        return pd.DataFrame(), []

    n = min(len(df), len(indicators))
    if n < 120:
        return pd.DataFrame(), []

    rows: List[Dict[str, Any]] = []
    records = indicators.to_dict("records")
    idx = indicators.index

    for i in range(60, n - 40):
        feature_row = build_curated_feature_row(records[i])
        feature_row["target_score"] = compute_training_target(df, i, forward_days=40)
        feature_row["ticker"] = ticker.upper()
        feature_row["bar_index"] = int(i)
        bar_ts = pd.Timestamp(idx[i]).normalize()
        feature_row["date"] = bar_ts.date().isoformat()
        feature_row["event_date"] = feature_row["date"]
        feature_row["event_id"] = f"{ticker.upper()}_{feature_row['date']}_{i}"
        rows.append(feature_row)

    result = pd.DataFrame(rows)
    metadata_cols = [c for c in result.columns if c not in CURATED_FEATURE_ORDER and c != "target_score"]
    return result, metadata_cols


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


def _trajectory_slope_np(col_arr: np.ndarray, pos: int, offsets: Sequence[int]) -> float:
    """Like _trajectory_slope but operates on a pre-extracted numpy column — zero iloc overhead."""
    xs: List[float] = []
    values: List[float] = []
    n = len(col_arr)
    for off in offsets:
        i = pos - off
        if 0 <= i < n:
            v = float(col_arr[i])
            if not (math.isnan(v) or math.isinf(v)):
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


def _compute_multi_horizon_outcome(
    ohlcv: pd.DataFrame,
    accel_pos: int,
    entry: float,
) -> Dict[str, Any]:
    """
    Compute trade outcomes across all configured time horizons.

    This replaces the legacy fixed 20-day window.  The model learns the TRUE
    maximum profit each setup can deliver — whether that takes 3 weeks or 6
    months — and also learns how much pain (drawdown) occurred before the peak
    so it can score entries on risk-adjusted quality, not raw return alone.

    Returns
    -------
    Legacy fields (tp1_hit_day, tp2_hit_day, stop_hit_day, max_excursion_pct)
    kept for labeler/labeler_v2 backward compatibility, plus:
      max_gain_pct_20d … max_gain_pct_180d  — best high in each window
      days_to_peak                           — bars until global max (180d)
      max_drawdown_before_peak_pct           — deepest low vs entry before peak
      risk_adjusted_gain                     — peak gain / pre-peak drawdown
    """
    _empty: Dict[str, Any] = {
        "tp1_hit_day": None,
        "tp2_hit_day": None,
        "stop_hit_day": None,
        "max_excursion_pct": float("nan"),
        "max_gain_pct_20d":  float("nan"),
        "max_gain_pct_40d":  float("nan"),
        "max_gain_pct_60d":  float("nan"),
        "max_gain_pct_90d":  float("nan"),
        "max_gain_pct_180d": float("nan"),
        "days_to_peak": float("nan"),
        "max_drawdown_before_peak_pct": float("nan"),
        "risk_adjusted_gain": float("nan"),
    }

    if math.isnan(entry) or entry <= 0:
        return _empty

    # Broadest forward slice we ever need — 180 trading days
    future_full = ohlcv.iloc[accel_pos + 1: accel_pos + 181]
    if future_full.empty:
        return _empty

    highs = future_full["high"].to_numpy(dtype=float)
    lows  = future_full["low"].to_numpy(dtype=float)

    # ── Legacy TP1 / TP2 / Stop scan (first 20 days only) ─────────────────
    # Hard-coded percentage levels kept for backward compatibility with
    # labelers and existing model bundles that reference binary TP columns.
    tp1_target  = entry * 1.05
    tp2_target  = entry * 1.10
    stop_target = entry * 0.95

    tp1_day:  Optional[int] = None
    tp2_day:  Optional[int] = None
    stop_day: Optional[int] = None

    for day_num, (_, row) in enumerate(future_full.iterrows(), start=1):
        if day_num > 20:
            break
        day_high = _safe_float(row.get("high"))
        day_low  = _safe_float(row.get("low"))

        stop_hit = not math.isnan(day_low)  and day_low  <= stop_target
        tp1_hit  = not math.isnan(day_high) and day_high >= tp1_target
        tp2_hit  = not math.isnan(day_high) and day_high >= tp2_target

        # Ambiguous intraday: conservative stop-first ordering
        if stop_hit and (tp1_hit or tp2_hit) and tp1_day is None and tp2_day is None:
            stop_day = day_num
            break
        if tp1_hit and tp1_day is None:
            tp1_day = day_num
        if tp2_hit and tp2_day is None:
            tp2_day = day_num
        if stop_hit and stop_day is None and tp2_day is None:
            stop_day = day_num
            break

    # Legacy max_excursion_pct — 20-day stop-adjusted high
    scope_slice = future_full.iloc[:20] if stop_day is None else future_full.iloc[:max(stop_day - 1, 1)]
    scope_high  = _safe_float(scope_slice["high"].max()) if not scope_slice.empty else float("nan")
    max_exc     = (scope_high / entry - 1.0) * 100.0 if not math.isnan(scope_high) else float("nan")

    # ── Multi-horizon max gain ─────────────────────────────────────────────
    # For each horizon we find the single best high within that window.
    # Uses high prices (not close) — captures the real achievable exit level.
    horizon_gains: Dict[int, float] = {}
    for horizon in MULTI_HORIZON_DAYS:
        window_highs = highs[:horizon]
        valid = window_highs[~np.isnan(window_highs)]
        if valid.size > 0:
            horizon_gains[horizon] = float((valid.max() / entry - 1.0) * 100.0)
        else:
            horizon_gains[horizon] = float("nan")

    # ── Days to peak and global max (180d scope) ───────────────────────────
    valid_mask = ~np.isnan(highs)
    if valid_mask.any():
        peak_idx       = int(np.argmax(np.where(valid_mask, highs, -np.inf)))
        peak_price     = float(highs[peak_idx])
        days_to_peak   = float(peak_idx + 1)           # 1-based trading day count
        global_max_gain = (peak_price / entry - 1.0) * 100.0
    else:
        days_to_peak    = float("nan")
        global_max_gain = float("nan")

    # ── Max drawdown BEFORE the peak — measures entry quality ─────────────
    # A setup that runs straight up with no drawdown is far safer than one
    # that drops 8% before eventually recovering.  The model learns to prefer
    # entries where the risk-before-reward is minimal.
    max_drawdown_before_peak = 0.0
    if not math.isnan(days_to_peak):
        pre_peak_lows = lows[:int(days_to_peak)]
        valid_lows    = pre_peak_lows[~np.isnan(pre_peak_lows)]
        if valid_lows.size > 0:
            max_drawdown_before_peak = float(
                max(0.0, (entry - valid_lows.min()) / entry * 100.0)
            )

    # ── Risk-adjusted gain ─────────────────────────────────────────────────
    # How much reward per unit of pre-peak pain.
    # Infinite reward (no drawdown) is capped at gain × 10 to stay numeric.
    if not math.isnan(global_max_gain) and global_max_gain > 0:
        if max_drawdown_before_peak > 0:
            risk_adjusted_gain = global_max_gain / max_drawdown_before_peak
        else:
            risk_adjusted_gain = global_max_gain * 10.0   # pristine: no drawdown at all
    else:
        risk_adjusted_gain = float("nan")

    return {
        # Legacy
        "tp1_hit_day":       tp1_day,
        "tp2_hit_day":       tp2_day,
        "stop_hit_day":      stop_day,
        "max_excursion_pct": float(max_exc) if not math.isnan(max_exc) else float("nan"),
        # Multi-horizon gains
        "max_gain_pct_20d":  horizon_gains.get(20,  float("nan")),
        "max_gain_pct_40d":  horizon_gains.get(40,  float("nan")),
        "max_gain_pct_60d":  horizon_gains.get(60,  float("nan")),
        "max_gain_pct_90d":  horizon_gains.get(90,  float("nan")),
        "max_gain_pct_180d": horizon_gains.get(180, float("nan")),
        # Quality metrics
        "days_to_peak":                  days_to_peak,
        "max_drawdown_before_peak_pct":  max_drawdown_before_peak,
        "risk_adjusted_gain":            risk_adjusted_gain,
    }


# Backward-compatible alias so any call sites referencing the old name still work.
_compute_trade_outcome = _compute_multi_horizon_outcome


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


def _precompute_signal_matrix(
    indicators: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """Evaluate every signal function once per row across the full series.

    Returns (matrix, records) where:
      matrix  — {signal_name: bool_array[len(indicators)]}
      records — plain-dict list of all indicator rows, reused for fast
                per-anchor lookups (avoids repeated pandas iloc calls).
    Computed ONCE per stock; every per-anchor extraction is then O(1).
    """
    records = indicators.to_dict("records")  # list of plain dicts — faster than iloc
    n = len(records)
    matrix: Dict[str, np.ndarray] = {}
    for signal_name, signal_fn in SIGNAL_DEFS.items():
        fires = np.zeros(n, dtype=bool)
        for i, row in enumerate(records):
            try:
                fires[i] = bool(signal_fn(row))
            except Exception:
                pass
        matrix[signal_name] = fires
    return matrix, records


def _extract_signal_features_asof(
    indicators: pd.DataFrame,
    pos: int,
    signal_matrix: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Build signal features at timestamp T=pos using only bars up to and including T.

    Pass ``signal_matrix`` (from ``_precompute_signal_matrix``) to avoid
    recomputing signal firings from scratch for every anchor — reduces training
    time from O(anchors × 90 × signals) to O(signals) per anchor.
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
        slug = _signal_slug(signal_name)

        if signal_matrix is not None:
            fires = signal_matrix[signal_name]
            # Slice the pre-built boolean array — O(1) numpy ops
            window_90 = fires[last_90_start: pos + 1]
            window_7  = fires[last_7_start:  pos + 1]
            window_30 = fires[last_30_start: pos + 1]
            window_60 = fires[last_60_start: pos + 1]

            if window_90.any():
                first_local = int(np.argmax(window_90))  # index of first True
                days_since_first = float(pos - (last_90_start + first_local))
            else:
                days_since_first = float("nan")

            in_7  = bool(window_7.any())
            in_30 = bool(window_30.any())
            in_60 = bool(window_60.any())
        else:
            # Fallback path (inference without pre-built matrix)
            fired_positions: List[int] = []
            for i in range(last_90_start, pos + 1):
                try:
                    if signal_fn(indicators.iloc[i]):
                        fired_positions.append(i)
                except Exception:
                    continue

            if fired_positions:
                days_since_first = float(pos - min(fired_positions))
            else:
                days_since_first = float("nan")

            in_7  = any(i >= last_7_start  for i in fired_positions)
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


def _compute_signal_pattern_features_asof(
    pred_pos: int,
    signal_matrix: Mapping[str, np.ndarray],
    indicator_records: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    """
    Pattern features that capture signal sequencing, clustering, and
    price/volume behavior while signals are firing.

    Used by both training row construction and live inference so feature
    semantics remain identical.
    """
    _defaults: Dict[str, float] = {
        "signals_active_now": 0.0,
        "signals_active_now_pct": 0.0,
        "signals_fired_5d": 0.0,
        "signals_fired_10d": 0.0,
        "signals_fired_20d": 0.0,
        "signals_fired_40d": 0.0,
        "signals_fired_5d_pct": 0.0,
        "signals_fired_20d_pct": 0.0,
        "most_recent_signal_days_ago": 999.0,
        "oldest_active_signal_days_ago": 999.0,
        "mean_signal_recency": 999.0,
        "num_signals_in_lookback": 0.0,
        "signal_cluster_span_days": 0.0,
        "signal_clustering_density": 0.0,
        "signal_acceleration_5d": 0.0,
        "signal_acceleration_20d": 0.0,
        "price_change_during_signals_pct": float("nan"),
        "price_cv_during_signals": float("nan"),
        "volume_trend_20d": float("nan"),
        "higher_lows_20d": 0.0,
        "higher_highs_20d": 0.0,
        "range_contraction_ratio": float("nan"),
    }

    if pred_pos < 0 or pred_pos >= len(indicator_records):
        return _defaults.copy()

    _active_now = 0
    _active_5d = 0
    _active_10d = 0
    _active_20d = 0
    _active_40d = 0

    for _sig_arr in signal_matrix.values():
        if pred_pos < len(_sig_arr) and bool(_sig_arr[pred_pos]):
            _active_now += 1

        for _lookback in (5, 10, 20, 40):
            _start = max(0, pred_pos - _lookback + 1)
            _end = pred_pos + 1
            if _start >= _end or _end > len(_sig_arr):
                continue
            if bool(_sig_arr[_start:_end].any()):
                if _lookback == 5:
                    _active_5d += 1
                elif _lookback == 10:
                    _active_10d += 1
                elif _lookback == 20:
                    _active_20d += 1
                else:
                    _active_40d += 1

    _total_signals = len(signal_matrix)

    _recency_list: List[int] = []
    for _sig_arr in signal_matrix.values():
        _found = False
        for _back in range(min(90, pred_pos + 1)):
            _idx = pred_pos - _back
            if 0 <= _idx < len(_sig_arr) and bool(_sig_arr[_idx]):
                _recency_list.append(_back)
                _found = True
                break
        if not _found:
            _recency_list.append(999)

    _valid_recencies = [r for r in _recency_list if r < 999]

    out: Dict[str, float] = {
        "signals_active_now": float(_active_now),
        "signals_active_now_pct": (_active_now / _total_signals * 100.0) if _total_signals > 0 else 0.0,
        "signals_fired_5d": float(_active_5d),
        "signals_fired_10d": float(_active_10d),
        "signals_fired_20d": float(_active_20d),
        "signals_fired_40d": float(_active_40d),
        "signals_fired_5d_pct": (_active_5d / _total_signals * 100.0) if _total_signals > 0 else 0.0,
        "signals_fired_20d_pct": (_active_20d / _total_signals * 100.0) if _total_signals > 0 else 0.0,
        "most_recent_signal_days_ago": float(min(_valid_recencies)) if _valid_recencies else 999.0,
        "oldest_active_signal_days_ago": float(max(_valid_recencies)) if _valid_recencies else 999.0,
        "mean_signal_recency": (sum(_valid_recencies) / len(_valid_recencies)) if _valid_recencies else 999.0,
        "num_signals_in_lookback": float(len(_valid_recencies)),
    }

    if len(_valid_recencies) >= 2:
        _sorted_rec = sorted(_valid_recencies)
        _span = _sorted_rec[-1] - _sorted_rec[0]
        out["signal_cluster_span_days"] = float(_span)
        out["signal_clustering_density"] = len(_valid_recencies) / max(1.0, float(_span))
    else:
        out["signal_cluster_span_days"] = 0.0
        out["signal_clustering_density"] = 0.0

    out["signal_acceleration_5d"] = float(_active_5d - _active_10d)
    out["signal_acceleration_20d"] = float(_active_20d - _active_40d)

    _price_now = _safe_float(indicator_records[pred_pos].get("close"))
    _span_bars = int(out["signal_cluster_span_days"])
    if _span_bars > 0 and not math.isnan(_price_now) and _price_now > 0:
        _span_idx = max(0, pred_pos - _span_bars)
        _price_at_start = _safe_float(indicator_records[_span_idx].get("close"))
        if not math.isnan(_price_at_start) and _price_at_start > 0:
            out["price_change_during_signals_pct"] = (_price_now / _price_at_start - 1.0) * 100.0
        else:
            out["price_change_during_signals_pct"] = float("nan")
    else:
        out["price_change_during_signals_pct"] = float("nan")

    if _span_bars > 2:
        _start_idx = max(0, pred_pos - _span_bars)
        _close_slice: List[float] = []
        for _k in range(_start_idx, min(pred_pos + 1, len(indicator_records))):
            _c = _safe_float(indicator_records[_k].get("close"))
            if not math.isnan(_c):
                _close_slice.append(_c)
        if len(_close_slice) >= 3:
            _arr = np.asarray(_close_slice, dtype=float)
            _mean = float(_arr.mean())
            out["price_cv_during_signals"] = float((_arr.std() / _mean) * 100.0) if _mean > 0 else float("nan")
        else:
            out["price_cv_during_signals"] = float("nan")
    else:
        out["price_cv_during_signals"] = float("nan")

    _vol_now = _safe_float(indicator_records[pred_pos].get("volume"))
    if pred_pos >= 20:
        _vol_20ago = _safe_float(indicator_records[pred_pos - 20].get("volume"))
        if not math.isnan(_vol_now) and not math.isnan(_vol_20ago) and _vol_20ago > 0:
            out["volume_trend_20d"] = (_vol_now / _vol_20ago - 1.0) * 100.0
        else:
            out["volume_trend_20d"] = float("nan")
    else:
        out["volume_trend_20d"] = float("nan")

    _higher_lows = 0
    for _k in range(max(1, pred_pos - 19), pred_pos + 1):
        _low_now = _safe_float(indicator_records[_k].get("low"))
        _low_prev = _safe_float(indicator_records[_k - 1].get("low"))
        if not math.isnan(_low_now) and not math.isnan(_low_prev) and _low_now > _low_prev:
            _higher_lows += 1
    out["higher_lows_20d"] = float(_higher_lows)

    _higher_highs = 0
    for _k in range(max(1, pred_pos - 19), pred_pos + 1):
        _high_now = _safe_float(indicator_records[_k].get("high"))
        _high_prev = _safe_float(indicator_records[_k - 1].get("high"))
        if not math.isnan(_high_now) and not math.isnan(_high_prev) and _high_now > _high_prev:
            _higher_highs += 1
    out["higher_highs_20d"] = float(_higher_highs)

    if pred_pos >= 20:
        _ranges_recent: List[float] = []
        _ranges_older: List[float] = []
        for _k in range(pred_pos - 4, pred_pos + 1):
            if 0 <= _k < len(indicator_records):
                _h = _safe_float(indicator_records[_k].get("high"))
                _l = _safe_float(indicator_records[_k].get("low"))
                if not math.isnan(_h) and not math.isnan(_l) and _l > 0:
                    _ranges_recent.append((_h - _l) / _l * 100.0)
        for _k in range(pred_pos - 19, pred_pos - 9):
            if 0 <= _k < len(indicator_records):
                _h = _safe_float(indicator_records[_k].get("high"))
                _l = _safe_float(indicator_records[_k].get("low"))
                if not math.isnan(_h) and not math.isnan(_l) and _l > 0:
                    _ranges_older.append((_h - _l) / _l * 100.0)

        if _ranges_recent and _ranges_older:
            _mean_recent = sum(_ranges_recent) / len(_ranges_recent)
            _mean_older = sum(_ranges_older) / len(_ranges_older)
            out["range_contraction_ratio"] = (_mean_recent / _mean_older) if _mean_older > 0 else float("nan")
        else:
            out["range_contraction_ratio"] = float("nan")
    else:
        out["range_contraction_ratio"] = float("nan")

    merged = _defaults.copy()
    merged.update(out)
    return merged


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

    # ── Precompute run-length arrays — O(N_rows) forward pass replaces the
    # O(pred_pos) backward scan previously done inside the per-anchor loop.
    _sv = stage_series.to_numpy(dtype=object)
    _n_sv = len(_sv)
    _days_arr = np.ones(_n_sv, dtype=np.int32)
    _sbefore_arr = np.full(_n_sv, "UNKNOWN", dtype=object)
    for _p in range(1, _n_sv):
        if _sv[_p] == _sv[_p - 1]:
            _days_arr[_p] = _days_arr[_p - 1] + 1
            _sbefore_arr[_p] = _sbefore_arr[_p - 1]
        else:
            _days_arr[_p] = 1
            _sbefore_arr[_p] = _sv[_p - 1]

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

    # Precompute signal firings once for the full series — avoids 90×21 iloc
    # calls per anchor (the main training bottleneck for large stocks).
    signal_matrix, indicator_records = _precompute_signal_matrix(indicators)

    # ── Per-column numpy arrays for zero-overhead trajectory / velocity ────
    _n_ind = len(indicators)
    _nan_col = np.full(_n_ind, float("nan"))
    _obv_np = indicators["obv"].to_numpy(dtype=float) if "obv" in indicators.columns else _nan_col
    _acc_np = indicators["accumulation_score"].to_numpy(dtype=float) if "accumulation_score" in indicators.columns else _nan_col
    _bb_np  = indicators["bb_bandwidth"].to_numpy(dtype=float) if "bb_bandwidth" in indicators.columns else _nan_col
    _ohlcv_low_np   = ohlcv["low"].to_numpy(dtype=float)
    _ohlcv_close_np = ohlcv["close"].to_numpy(dtype=float)
    _ohlcv_turn_np  = ohlcv["turnover_kwd"].to_numpy(dtype=float) if "turnover_kwd" in ohlcv.columns else None

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

        stage_now = str(_sv[pred_pos])
        days_in_stage = int(_days_arr[pred_pos])
        stage_before = str(_sbefore_arr[pred_pos])

        _row_pred = indicator_records[pred_pos]
        if is_control:
            close_t = _safe_float(_row_pred.get("close"))
            entry = close_t
            outcome = {
                "tp1_hit_day": None,
                "tp2_hit_day": None,
                "stop_hit_day": None,
                "max_excursion_pct": 0.0,
                "max_gain_pct_20d":  0.0,
                "max_gain_pct_40d":  0.0,
                "max_gain_pct_60d":  0.0,
                "max_gain_pct_90d":  0.0,
                "max_gain_pct_180d": 0.0,
                "days_to_peak":                 float("nan"),
                "max_drawdown_before_peak_pct": 0.0,
                "risk_adjusted_gain":           float("nan"),
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
            close_t = _safe_float(_row_pred.get("close"))
            entry = _safe_float(event.acceleration_price)
            outcome = _compute_multi_horizon_outcome(ohlcv, accel_pos, entry)
            event_id = event.event_id
            start_date = event.start_date.isoformat() if event.start_date else None
            peak_date = event.peak_date.isoformat() if event.peak_date else None
            is_fakeout = int(bool(event.is_fakeout))
            threshold_pct = _safe_float(event.threshold_pct)
            duration_days = _safe_float(event.duration_days)
            peak_price = _safe_float(event.peak_price)
            gain_pct = _safe_float(event.gain_pct)
            failed_at_pct = _safe_float(event.failed_at_pct)

        signal_features = _extract_signal_features_asof(indicators, pred_pos, signal_matrix)
        signal_pattern_features = _compute_signal_pattern_features_asof(
            pred_pos=pred_pos,
            signal_matrix=signal_matrix,
            indicator_records=indicator_records,
        )
        regime_name, pmi_trend, brent_trend = _lookup_regime(regime_frame, pred_ts)

        # ── Entry quality features (available at prediction time) ──────────
        # price_extension_from_20d_low_pct: how far has price risen from the
        # recent consolidation base?  Low = still near the anchor = good entry.
        # High = already extended = risky late entry.
        close_at_pred = close_t  # already computed above — no duplicate iloc
        low_20d = float(_ohlcv_low_np[max(0, pred_pos - 19): pred_pos + 1].min()) \
            if pred_pos >= 0 else float("nan")
        if not math.isnan(close_at_pred) and not math.isnan(low_20d) and low_20d > 0:
            price_extension_20d = (close_at_pred / low_20d - 1.0) * 100.0
        else:
            price_extension_20d = float("nan")

        # accumulation_compression_days: how many consecutive days before the
        # setup was the stock in a tight low-volatility range?
        # Long compression = smart money quietly building = early, high-quality signal.
        compression_days = float("nan")
        if pred_pos >= 10:
            _pre_close = _ohlcv_close_np[max(0, pred_pos - 60): pred_pos + 1]
            if len(_pre_close) >= 5:
                comp = 0
                for _i in range(len(_pre_close) - 1, 4, -1):
                    _w = _pre_close[max(0, _i - 4): _i + 1]
                    w_mean = float(_w.mean())
                    w_std  = float(_w.std())
                    if w_mean > 0 and (w_std / w_mean) < 0.025:
                        comp += 1
                    else:
                        break
                compression_days = float(comp)

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
                float(_ohlcv_turn_np[max(0, pred_pos - 60): pred_pos].mean()) if _ohlcv_turn_np is not None else None
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
            # ── Legacy outcome fields ──────────────────────────────────────
            "tp1_hit_day":       outcome["tp1_hit_day"],
            "tp2_hit_day":       outcome["tp2_hit_day"],
            "stop_hit_day":      outcome["stop_hit_day"],
            "max_excursion_pct": outcome["max_excursion_pct"],
            # ── Multi-horizon gains (forward-looking, non-feature columns) ─
            "max_gain_pct_20d":  outcome["max_gain_pct_20d"],
            "max_gain_pct_40d":  outcome["max_gain_pct_40d"],
            "max_gain_pct_60d":  outcome["max_gain_pct_60d"],
            "max_gain_pct_90d":  outcome["max_gain_pct_90d"],
            "max_gain_pct_180d": outcome["max_gain_pct_180d"],
            "days_to_peak":                 outcome["days_to_peak"],
            "max_drawdown_before_peak_pct": outcome["max_drawdown_before_peak_pct"],
            "risk_adjusted_gain":           outcome["risk_adjusted_gain"],
        }
        row.update(signal_features)
        row.update(signal_pattern_features)

        # Entry quality features — these ARE features (available at pred time)
        row["price_extension_from_20d_low_pct"] = price_extension_20d
        row["price_extension_from_60d_low_pct"] = _safe_float(_row_pred.get("price_extension_from_60d_low_pct"))
        row["price_extension_from_120d_low_pct"] = _safe_float(_row_pred.get("price_extension_from_120d_low_pct"))
        row["position_in_60d_range_pct"] = _safe_float(_row_pred.get("position_in_60d_range_pct"))
        row["distance_from_52w_high_pct"] = _safe_float(_row_pred.get("distance_from_52w_high_pct"))
        row["accumulation_compression_days"]     = compression_days

        for col in indicators.columns:
            if col in LEAKY_INDICATOR_COLUMNS:
                continue
            val = _row_pred.get(col)  # dict lookup — no pandas iloc overhead
            if col == "wyckoff_phase":
                row[f"t0_{col}_code"] = _encode_wyckoff(val)
            else:
                row[f"t0_{col}"] = _safe_float(val)

        _i3 = pred_pos - 3
        _rec3 = indicator_records[_i3] if _i3 >= 0 else None
        for col in CORE_VELOCITY_COLUMNS:
            now_v  = _safe_float(_row_pred.get(col))
            past_v = _safe_float(_rec3.get(col)) if _rec3 is not None else float("nan")
            row[f"{col}_velocity"] = (now_v - past_v) / 3.0 if not (math.isnan(now_v) or math.isnan(past_v)) else float("nan")

        row["obv_trajectory_slope"] = _trajectory_slope_np(_obv_np, pred_pos, TRAJECTORY_OFFSETS)
        row["accumulation_trajectory_slope"] = _trajectory_slope_np(_acc_np, pred_pos, TRAJECTORY_OFFSETS)
        row["bb_bandwidth_trajectory"] = _trajectory_slope_np(_bb_np, pred_pos, TRAJECTORY_OFFSETS)

        for lookback in CONTEXT_LOOKBACKS:
            _i_lb = pred_pos - lookback
            _rec_lb = indicator_records[_i_lb] if _i_lb >= 0 else None
            for col in CONTEXT_COLUMNS:
                row[f"t{lookback}_{col}"] = _safe_float(_rec_lb.get(col)) if _rec_lb is not None else float("nan")

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
    use_pg = settings.use_postgres

    from app.core.database import query_all, query_one

    preferred = [
        "ee_events_cache",
        "ee_forensic_events",
        "forensic_events",
        "eagle_eye_events",
    ]

    # Discover which table exists
    table_check_sql_pg = (
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name=?"
    )
    table_check_sql_sq = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"

    tables_available: List[str] = []
    for t in preferred:
        row = query_one(table_check_sql_pg if use_pg else table_check_sql_sq, (t,))
        if row:
            tables_available.append(t)

    if not tables_available and not use_pg:
        # Fallback: find any event-like table in SQLite
        all_rows = query_all(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables_available = [
            r[0] for r in all_rows
            if "event" in str(r[0]).lower() and str(r[0]).startswith("ee_")
        ]

    selected: Optional[str] = None
    cols: List[str] = []
    for table in tables_available:
        if use_pg:
            col_rows = query_all(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=? ORDER BY ordinal_position",
                (table,),
            )
            c = [r[0] for r in col_rows]
        else:
            col_rows = query_all(f"PRAGMA table_info({table})")  # nosec B608
            c = [r[1] for r in col_rows]
        cset = {x.lower() for x in c}
        if "ticker" in cset and ("acceleration_date" in cset or "event_date" in cset):
            selected = table
            cols = c
            break

    if not selected:
        return []

    log.info("Using forensic event cache table: %s", selected)
    rows = query_all(f"SELECT * FROM {selected}")  # nosec B608

    records: List[Dict[str, Any]] = []
    for r in rows:
        item = dict(r.items())
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
    n_tickers = len(tickers)
    for idx, ticker in enumerate(tickers, 1):
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
            log.info("[%d/%d] %s \u2192 %d rows (running total %d)", idx, n_tickers, ticker, len(rows), len(all_rows))
        except Exception as exc:
            log.warning("Event build failed for %s: %s", ticker, exc)

    return all_rows


def _compute_signal_pattern_features(
    row: Dict[str, float],
    pos: int,
    signal_matrix: Mapping[str, np.ndarray],
    n: int,
) -> None:
    """Add signal-count/recency/cluster features used by the classifier."""
    del n  # retained for compatibility with the public function signature

    total_signals = len(signal_matrix)
    counts = {5: 0, 10: 0, 20: 0, 40: 0}
    active_now = 0

    recencies: List[int] = []
    for arr in signal_matrix.values():
        if pos < len(arr) and bool(arr[pos]):
            active_now += 1

        for lookback in (5, 10, 20, 40):
            start = max(0, pos - lookback + 1)
            end = min(pos + 1, len(arr))
            if start < end and bool(arr[start:end].any()):
                counts[lookback] += 1

        found = False
        for back in range(min(90, pos + 1)):
            idx = pos - back
            if 0 <= idx < len(arr) and bool(arr[idx]):
                recencies.append(back)
                found = True
                break
        if not found:
            recencies.append(999)

    valid = [r for r in recencies if r < 999]

    row["signals_active_now"] = float(active_now)
    row["signals_active_now_pct"] = (active_now / total_signals * 100.0) if total_signals > 0 else 0.0
    row["signals_fired_5d"] = float(counts[5])
    row["signals_fired_10d"] = float(counts[10])
    row["signals_fired_20d"] = float(counts[20])
    row["signals_fired_40d"] = float(counts[40])

    row["most_recent_signal_days_ago"] = float(min(valid)) if valid else 999.0
    row["oldest_active_signal_days_ago"] = float(max(valid)) if valid else 999.0
    row["mean_signal_recency"] = (sum(valid) / len(valid)) if valid else 999.0
    row["num_signals_in_lookback"] = float(len(valid))

    if len(valid) >= 2:
        sv = sorted(valid)
        span = sv[-1] - sv[0]
        row["signal_cluster_span_days"] = float(span)
        row["signal_clustering_density"] = len(valid) / max(1.0, float(span))
    else:
        row["signal_cluster_span_days"] = 0.0
        row["signal_clustering_density"] = 0.0

    row["signal_acceleration_5d"] = float(counts[5] - counts[10])
    row["signal_acceleration_20d"] = float(counts[20] - counts[40])


def _compute_price_structure_features(
    row: Dict[str, float],
    pos: int,
    records: Sequence[Mapping[str, Any]],
    n: int,
) -> None:
    """Add local structure metrics: higher highs/lows and range contraction."""
    higher_lows = 0
    for k in range(max(1, pos - 19), pos + 1):
        if 0 <= k < n:
            low_now = _safe_float(records[k].get("low"))
            low_prev = _safe_float(records[k - 1].get("low"))
            if low_now is not None and low_prev is not None and low_now > low_prev:
                higher_lows += 1
    row["higher_lows_20d"] = float(higher_lows)

    higher_highs = 0
    for k in range(max(1, pos - 19), pos + 1):
        if 0 <= k < n:
            high_now = _safe_float(records[k].get("high"))
            high_prev = _safe_float(records[k - 1].get("high"))
            if high_now is not None and high_prev is not None and high_now > high_prev:
                higher_highs += 1
    row["higher_highs_20d"] = float(higher_highs)

    if pos >= 20:
        ranges_recent: List[float] = []
        ranges_older: List[float] = []
        for k in range(pos - 4, pos + 1):
            if 0 <= k < n:
                h = _safe_float(records[k].get("high"))
                l = _safe_float(records[k].get("low"))
                if h is not None and l is not None and l > 0:
                    ranges_recent.append((h - l) / l * 100.0)
        for k in range(pos - 19, pos - 9):
            if 0 <= k < n:
                h = _safe_float(records[k].get("high"))
                l = _safe_float(records[k].get("low"))
                if h is not None and l is not None and l > 0:
                    ranges_older.append((h - l) / l * 100.0)

        if ranges_recent and ranges_older:
            mr = sum(ranges_recent) / len(ranges_recent)
            mo = sum(ranges_older) / len(ranges_older)
            row["range_contraction_ratio"] = mr / mo if mo > 0 else float("nan")
        else:
            row["range_contraction_ratio"] = float("nan")
    else:
        row["range_contraction_ratio"] = float("nan")


def _add_market_context(row: Dict[str, float], market_df: Optional[pd.DataFrame], bar_date: pd.Timestamp) -> None:
    """Attach coarse market regime context to each training row."""
    if market_df is None or market_df.empty:
        row["market_rsi"] = float("nan")
        row["market_return_5d"] = float("nan")
        row["market_return_20d"] = float("nan")
        return

    key = pd.Timestamp(bar_date).normalize()
    if key not in market_df.index:
        prior = market_df.loc[:key]
        if prior.empty:
            row["market_rsi"] = float("nan")
            row["market_return_5d"] = float("nan")
            row["market_return_20d"] = float("nan")
            return
        key = prior.index[-1]

    idx_pos = int(market_df.index.get_loc(key))
    row["market_rsi"] = _safe_float(market_df.iloc[idx_pos].get("rsi")) or float("nan")

    c_now = _safe_float(market_df.iloc[idx_pos].get("close"))
    if c_now is None or c_now <= 0:
        row["market_return_5d"] = float("nan")
        row["market_return_20d"] = float("nan")
        return

    if idx_pos >= 5:
        c_5ago = _safe_float(market_df.iloc[idx_pos - 5].get("close"))
        row["market_return_5d"] = ((c_now / c_5ago - 1.0) * 100.0) if c_5ago and c_5ago > 0 else float("nan")
    else:
        row["market_return_5d"] = float("nan")

    if idx_pos >= 20:
        c_20ago = _safe_float(market_df.iloc[idx_pos - 20].get("close"))
        row["market_return_20d"] = ((c_now / c_20ago - 1.0) * 100.0) if c_20ago and c_20ago > 0 else float("nan")
    else:
        row["market_return_20d"] = float("nan")


def build_labeled_training_data(
    ticker: str,
    df: pd.DataFrame,
    indicators: pd.DataFrame,
    labels: pd.Series,
    signal_matrix: Mapping[str, np.ndarray],
    indicator_records: Sequence[Mapping[str, Any]],
    market_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build training rows from phase-labeled bars instead of event anchors.
    """
    del ticker, df

    n = len(indicators)
    if n < 30 or len(indicator_records) != n:
        return pd.DataFrame()

    aligned_labels = labels.reindex(indicators.index).fillna(1).astype(int)
    label_arr = aligned_labels.to_numpy(dtype=int)

    non_neutral_positions = [i for i, v in enumerate(label_arr) if v != 1]
    neutral_positions = [i for i, v in enumerate(label_arr) if v == 1]

    n_signal = len(non_neutral_positions)
    if n_signal == 0:
        return pd.DataFrame()

    max_neutral = n_signal * 1
    if len(neutral_positions) > max_neutral:
        rng = np.random.default_rng(42)
        neutral_positions = sorted(int(v) for v in rng.choice(neutral_positions, size=max_neutral, replace=False))

    all_positions = sorted(non_neutral_positions + neutral_positions)

    lookback_columns = [
        "rsi", "adx", "macd_histogram", "cmf", "plus_di", "minus_di",
        "plus_di_minus_di_diff", "volume_ratio_20d", "obv_trajectory_slope",
        "price_extension_from_20d_low_pct",
        "accumulation_compression_days",
    ]
    lookback_offsets = [1, 2, 3, 5]

    rows: List[Dict[str, Any]] = []
    for pos in all_positions:
        if pos < 20:
            continue

        rec = indicator_records[pos]
        bar_ts = pd.Timestamp(indicators.index[pos]).normalize()
        row: Dict[str, Any] = {
            "bar_index": int(pos),
            "date": bar_ts.date().isoformat(),
            "event_date": bar_ts.date().isoformat(),
        }

        for col in indicators.columns:
            val = rec.get(col)
            row[f"t0_{col}"] = _safe_float(val) if _safe_float(val) is not None else float("nan")

        for offset in lookback_offsets:
            lb_pos = pos - offset
            if 0 <= lb_pos < n:
                lb_rec = indicator_records[lb_pos]
                for col in lookback_columns:
                    val = _safe_float(lb_rec.get(col))
                    row[f"t{offset}_{col}"] = val if val is not None else float("nan")
            else:
                for col in lookback_columns:
                    row[f"t{offset}_{col}"] = float("nan")

        for col in lookback_columns:
            t0_val = row.get(f"t0_{col}", float("nan"))
            t5_val = row.get(f"t5_{col}", float("nan"))
            t3_val = row.get(f"t3_{col}", float("nan"))
            t1_val = row.get(f"t1_{col}", float("nan"))

            row[f"delta5_{col}"] = (t0_val - t5_val) if np.isfinite(t0_val) and np.isfinite(t5_val) else float("nan")
            row[f"delta3_{col}"] = (t0_val - t3_val) if np.isfinite(t0_val) and np.isfinite(t3_val) else float("nan")
            row[f"delta1_{col}"] = (t0_val - t1_val) if np.isfinite(t0_val) and np.isfinite(t1_val) else float("nan")

        # Canonical aliases used by diagnostics and model reviews.
        row["delta3_obv_slope"] = row.get("delta3_obv_trajectory_slope", float("nan"))
        row["delta5_obv_slope"] = row.get("delta5_obv_trajectory_slope", float("nan"))

        _compute_signal_pattern_features(row, pos, signal_matrix, n)
        _compute_price_structure_features(row, pos, indicator_records, n)
        _add_market_context(row, market_df, bar_ts)

        row["label"] = int(label_arr[pos])
        rows.append(row)

    return pd.DataFrame(rows)


def build_labeled_rows_from_ohlcv_cache(
    tickers: Optional[Sequence[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Build supervised rows from all cached OHLCV bars using curated v14 features.
    """
    log = logger or LOGGER

    if tickers is None:
        tickers = list_tickers_with_ohlcv()

    all_rows: List[Dict[str, Any]] = []
    n_tickers = len(tickers)
    for idx, ticker in enumerate(tickers, start=1):
        try:
            ohlcv = load_ohlcv(ticker)
            if ohlcv is None or ohlcv.empty or len(ohlcv) < 120:
                continue

            indicators = compute_all_indicators(ohlcv)
            if indicators is None or indicators.empty:
                continue

            frame, _ = build_training_dataset(ticker=ticker, df=ohlcv, indicators=indicators)
            if frame.empty:
                continue

            rows = frame.to_dict(orient="records")
            all_rows.extend(rows)
            log.info(
                "[%d/%d] %s -> %d curated rows (running total %d)",
                idx,
                n_tickers,
                ticker,
                len(rows),
                len(all_rows),
            )
        except Exception as exc:
            log.warning("Labeled row build failed for %s: %s", ticker, exc)

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

    for col in CURATED_FEATURE_ORDER:
        if col not in df.columns:
            df[col] = float("nan")
    if "target_score" not in df.columns:
        df["target_score"] = float("nan")

    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    elif "date" in df.columns:
        df["event_date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["event_date"] = pd.NaT

    if "ticker" not in df.columns:
        df["ticker"] = "UNKNOWN"

    feature_cols = list(CURATED_FEATURE_ORDER)
    for col in feature_cols + ["target_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_ratio = df[feature_cols].isna().mean(axis=1)
    reject_mask = missing_ratio > 0.30
    rejected_counts = df.loc[reject_mask].groupby("ticker").size().astype(int).to_dict()

    if rejected_counts:
        for tk, n in sorted(rejected_counts.items()):
            log.info("Rejected %d rows for %s due to >30%% NaN curated-feature ratio", n, tk)

    df = df.loc[~reject_mask].copy()
    df = df.loc[df["target_score"].notna()].copy()

    if df.empty:
        return FeatureBuildResult(frame=df, rejected_counts=rejected_counts, total_before=total_before, total_after=0)

    df = df.sort_values(["ticker", "event_date", "bar_index"], na_position="last").reset_index(drop=True)
    df[feature_cols] = df.groupby("ticker", dropna=False)[feature_cols].ffill()

    ticker_medians = df.groupby("ticker", dropna=False)[feature_cols].transform("median")
    df[feature_cols] = df[feature_cols].fillna(ticker_medians)

    global_median = df[feature_cols].median(numeric_only=True)
    df[feature_cols] = df[feature_cols].fillna(global_median)
    df[feature_cols] = df[feature_cols].fillna(0.0)
    df["target_score"] = df["target_score"].clip(0.0, 100.0)

    total_after = len(df)
    return FeatureBuildResult(
        frame=df,
        rejected_counts=rejected_counts,
        total_before=total_before,
        total_after=total_after,
    )


def get_feature_columns(frame: pd.DataFrame) -> List[str]:
    curated = [c for c in CURATED_FEATURE_ORDER if c in frame.columns]
    if curated:
        return curated
    return [
        c
        for c in frame.columns
        if c not in NON_FEATURE_COLUMNS
        and not c.startswith("y_")
        and not c.startswith(NON_FEATURE_PREFIXES)
    ]


# ---------------------------------------------------------------------------
# Phase 3 — single-row inference helper
# ---------------------------------------------------------------------------

def build_inference_row(
    ticker: str,
    ohlcv: pd.DataFrame,
    T: Optional[date] = None,
    regime_frame: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build one curated v14 feature dict for inference at date T.

    The returned row contains only the curated feature schema used at train
    time, preserving strict train/inference parity.
    """
    del ticker, regime_frame

    if ohlcv is None or ohlcv.empty or len(ohlcv) < 60:
        return None

    indicators = compute_all_indicators(ohlcv)
    if indicators is None or indicators.empty:
        return None

    if T is None:
        pred_pos = len(indicators) - 1
    else:
        t_ts = pd.Timestamp(T).normalize()
        pred_pos = int(indicators.index.searchsorted(t_ts, side="right")) - 1
        if pred_pos < 0:
            return None

    if pred_pos < 0:
        return None

    latest = indicators.iloc[pred_pos].to_dict()
    latest["stage"] = classify_stage(latest)
    return build_curated_feature_row(latest)

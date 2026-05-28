from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


# ---------------------------------------------------------------------------
# Legacy fixed-percentage TP constants (kept for backward-compatible columns)
# ---------------------------------------------------------------------------
TP1_PCT = 5.0
TP2_PCT = 10.0

# ---------------------------------------------------------------------------
# Variable horizons — the ML learns WHICH window captures each stock's move.
# Never hard-code the hold period; let the data decide.
# ---------------------------------------------------------------------------
GAIN_HORIZONS = (20, 40, 60, 90, 180)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(v) or math.isinf(v):
        return float("nan")
    return v


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    return v


def detect_buy_sell_points(
    df: pd.DataFrame,
    min_gain_pct: float = 10.0,
    min_drop_pct: float = 5.0,
    smoothing_window: int = 5,
    extrema_order: int = 10,
) -> pd.Series:
    """
    Label each bar as BUY (1), SELL (-1), or HOLD (0) from realized turning points.

    BUY bars are local troughs followed by >= ``min_gain_pct`` rise.
    SELL bars are local peaks followed by >= ``min_drop_pct`` decline.
    """
    n = len(df)
    labels = pd.Series(0, index=df.index, dtype=int)

    if n < smoothing_window + extrema_order * 2 + 20:
        return labels
    if not {"close", "high", "low"}.issubset(set(df.columns)):
        return labels

    smoothed = (
        pd.to_numeric(df["close"], errors="coerce")
        .rolling(smoothing_window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    if np.isnan(smoothed).all():
        return labels

    trough_indices = argrelextrema(smoothed, np.less_equal, order=extrema_order)[0]
    peak_indices = argrelextrema(smoothed, np.greater_equal, order=extrema_order)[0]

    high_arr = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low_arr = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)

    for idx in trough_indices:
        if idx >= n - 20:
            continue
        trough_price = low_arr[idx]
        if not np.isfinite(trough_price) or trough_price <= 0:
            continue

        forward_window = min(60, n - idx - 1)
        if forward_window < 10:
            continue

        future_highs = high_arr[idx + 1 : idx + 1 + forward_window]
        finite = future_highs[np.isfinite(future_highs)]
        if finite.size == 0:
            continue

        gain_pct = (finite.max() / trough_price - 1.0) * 100.0
        if gain_pct >= min_gain_pct:
            labels.iloc[idx] = 1

    for idx in peak_indices:
        if idx >= n - 10:
            continue
        peak_price = high_arr[idx]
        if not np.isfinite(peak_price) or peak_price <= 0:
            continue

        forward_window = min(40, n - idx - 1)
        if forward_window < 5:
            continue

        future_lows = low_arr[idx + 1 : idx + 1 + forward_window]
        finite = future_lows[np.isfinite(future_lows)]
        if finite.size == 0:
            continue

        drop_pct = (1.0 - finite.min() / peak_price) * 100.0
        if drop_pct >= min_drop_pct:
            labels.iloc[idx] = -1

    # Resolve tight BUY/SELL conflicts in a ±3-bar neighborhood.
    for i in range(1, n - 1):
        if labels.iloc[i] == 0:
            continue

        window_start = max(0, i - 3)
        window_end = min(n, i + 4)
        window = labels.iloc[window_start:window_end]
        if not ((window == 1).any() and (window == -1).any()):
            continue

        buy_idx = window_start + np.where(window.values == 1)[0]
        sell_idx = window_start + np.where(window.values == -1)[0]

        buy_gain = 0.0
        sell_drop = 0.0

        for bi in buy_idx:
            if bi < n - 20:
                fw = min(60, n - bi - 1)
                slice_high = high_arr[bi + 1 : bi + 1 + fw]
                finite = slice_high[np.isfinite(slice_high)]
                if finite.size > 0 and low_arr[bi] > 0 and np.isfinite(low_arr[bi]):
                    buy_gain = max(buy_gain, (finite.max() / low_arr[bi] - 1.0) * 100.0)

        for si in sell_idx:
            if si < n - 10:
                fw = min(40, n - si - 1)
                slice_low = low_arr[si + 1 : si + 1 + fw]
                finite = slice_low[np.isfinite(slice_low)]
                if finite.size > 0 and high_arr[si] > 0 and np.isfinite(high_arr[si]):
                    sell_drop = max(sell_drop, (1.0 - finite.min() / high_arr[si]) * 100.0)

        if buy_gain < sell_drop:
            for bi in buy_idx:
                labels.iloc[bi] = 0
        else:
            for si in sell_idx:
                labels.iloc[si] = 0

    return labels


def label_quality_stats(labels: pd.Series) -> Dict[str, Any]:
    """Return coarse label-distribution diagnostics for sanity checks."""
    n = int(len(labels))
    n_buy = int((labels == 1).sum())
    n_sell = int((labels == -1).sum())
    n_hold = int((labels == 0).sum())
    return {
        "total_bars": n,
        "buy_labels": n_buy,
        "sell_labels": n_sell,
        "hold_labels": n_hold,
        "buy_pct": round(n_buy / n * 100, 2) if n > 0 else 0.0,
        "sell_pct": round(n_sell / n * 100, 2) if n > 0 else 0.0,
    }


def compute_opportunity_score(row: dict) -> float:
    """
    Training label = best forward gain produced by this setup (0-100).

    The model learns a direct mapping from indicator state -> realized gain,
    without hand-coded opportunity formulas.
    """
    gains: List[float] = []
    for h in (20, 40, 60, 90, 180):
        g = row.get(f"max_gain_pct_{h}d")
        if g is not None:
            try:
                gf = float(g)
                if not math.isnan(gf):
                    gains.append(gf)
            except (TypeError, ValueError):
                pass

    if not gains:
        return 0.0

    best_gain = max(gains)
    best_gain = max(0.0, best_gain)
    best_gain = min(100.0, best_gain)
    return round(best_gain, 2)


# ---------------------------------------------------------------------------
# Core labeller
# ---------------------------------------------------------------------------

def label_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Build all target labels for one forensic event row.

    Two label families are produced:

    1. Legacy binary / ordinal labels (y_tp1_Xd, y_tp2_20d, etc.)
       Kept for backward compatibility with existing model bundles.

    2. Multi-horizon continuous labels (y_max_gain_Xd)
       The primary training signal for the new ML objective:
       maximize captured profit, not just hit a fixed 5% target in 20 days.

       Additionally:
         y_days_to_peak            -- bars to global max gain (best exit timing)
         y_max_drawdown_before_peak -- max pain before the gain (risk quality)
         y_risk_adjusted_gain      -- gain / drawdown (reward-per-risk unit)
         y_entry_quality_score     -- how early the signal fired (0-100)
                                      100 = caught the whole move from the bottom
                                        0 = entered at the peak, missed everything
    """
    # -- 1. Legacy TP / stop fields -----------------------------------------
    tp1_day  = _as_int(event.get("tp1_hit_day"))
    tp2_day  = _as_int(event.get("tp2_hit_day"))
    stop_day = _as_int(event.get("stop_hit_day"))

    max_exc = _as_float(event.get("max_excursion_pct"))
    if math.isnan(max_exc):
        peak  = _as_float(event.get("peak_price"))
        entry = _as_float(event.get("acceleration_price"))
        if not math.isnan(peak) and not math.isnan(entry) and entry > 0:
            max_exc = (peak / entry - 1.0) * 100.0

    duration_days = _as_int(event.get("duration_days"))
    if tp1_day is None and not math.isnan(max_exc) and max_exc >= TP1_PCT and duration_days is not None:
        tp1_day = duration_days
    if tp2_day is None and not math.isnan(max_exc) and max_exc >= TP2_PCT and duration_days is not None:
        tp2_day = duration_days

    y_tp1_5d  = int(tp1_day is not None and tp1_day <= 5)
    y_tp1_10d = int(tp1_day is not None and tp1_day <= 10)
    y_tp1_20d = int(tp1_day is not None and tp1_day <= 20)
    y_tp2_20d = int(tp2_day is not None and tp2_day <= 20)

    if tp1_day is not None and tp1_day <= 5:
        category = "TP1_FAST"
    elif tp1_day is not None and tp1_day <= 20:
        category = "TP1_SLOW"
    elif stop_day is not None and (tp1_day is None or stop_day <= tp1_day):
        category = "STOPPED_OUT"
    else:
        category = "TIMED_OUT"

    # -- 2. Multi-horizon max-gain labels -----------------------------------
    # Each label = best high within that many trading days from entry, as %.
    # NaN means data is not yet available for that horizon.
    gain_labels: Dict[str, float] = {}
    for horizon in GAIN_HORIZONS:
        raw = _as_float(event.get(f"max_gain_pct_{horizon}d"))
        # Graceful fallback for 20d: use legacy max_excursion_pct
        if math.isnan(raw) and horizon == 20 and not math.isnan(max_exc):
            raw = max_exc
        gain_labels[f"y_max_gain_{horizon}d"] = raw

    # -- 3. Quality labels --------------------------------------------------
    days_to_peak    = _as_float(event.get("days_to_peak"))
    max_dd_pre_peak = _as_float(event.get("max_drawdown_before_peak_pct"))
    risk_adj_gain   = _as_float(event.get("risk_adjusted_gain"))

    # Entry quality score (0-100).
    # What fraction of the full 180-day move was captured in the first 20 days?
    #   100 -> signal fired at the bottom, 20d gain ≈ 180d gain
    #     0 -> signal fired near the top, almost nothing left to capture
    max_gain_180 = gain_labels.get("y_max_gain_180d", float("nan"))
    max_gain_20  = gain_labels.get("y_max_gain_20d",  float("nan"))
    if (
        not math.isnan(max_gain_180) and max_gain_180 > 0.5
        and not math.isnan(max_gain_20)
    ):
        entry_quality_score = float(np.clip((max_gain_20 / max_gain_180) * 100.0, 0.0, 100.0))
    else:
        entry_quality_score = float("nan")

    opportunity_score = compute_opportunity_score(event)

    return {
        # Legacy binary labels
        "y_tp1_5d":            y_tp1_5d,
        "y_tp1_10d":           y_tp1_10d,
        "y_tp1_20d":           y_tp1_20d,
        "y_tp2_20d":           y_tp2_20d,
        "y_max_excursion_pct": max_exc,
        "y_days_to_tp1":       float(tp1_day) if tp1_day is not None else float("nan"),
        "y_outcome_category":  category,
        # Multi-horizon continuous labels
        **gain_labels,
        # Quality labels
        "y_days_to_peak":             days_to_peak,
        "y_max_drawdown_before_peak": max_dd_pre_peak,
        "y_risk_adjusted_gain":       risk_adj_gain,
        "y_entry_quality_score":      entry_quality_score,
        "y_opportunity_score":        opportunity_score,
    }


def build_labels(events: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised label builder.  Returns a DataFrame aligned to *events* index
    with every label column from label_event().
    """
    if events is None or events.empty:
        return pd.DataFrame(
            columns=[
                "y_tp1_5d",
                "y_tp1_10d",
                "y_tp1_20d",
                "y_tp2_20d",
                "y_max_excursion_pct",
                "y_days_to_tp1",
                "y_outcome_category",
                *(f"y_max_gain_{h}d" for h in GAIN_HORIZONS),
                "y_days_to_peak",
                "y_max_drawdown_before_peak",
                "y_risk_adjusted_gain",
                "y_entry_quality_score",
                "y_opportunity_score",
            ]
        )

    rows = [label_event(rec) for rec in events.to_dict(orient="records")]
    out  = pd.DataFrame(rows, index=events.index)

    # Stable dtype enforcement
    for col in ("y_tp1_5d", "y_tp1_10d", "y_tp1_20d", "y_tp2_20d"):
        out[col] = out[col].astype(int)

    numeric_cols: List[str] = [
        "y_max_excursion_pct",
        "y_days_to_tp1",
        *(f"y_max_gain_{h}d" for h in GAIN_HORIZONS),
        "y_days_to_peak",
        "y_max_drawdown_before_peak",
        "y_risk_adjusted_gain",
        "y_entry_quality_score",
        "y_opportunity_score",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["y_outcome_category"] = out["y_outcome_category"].astype(str)

    return out

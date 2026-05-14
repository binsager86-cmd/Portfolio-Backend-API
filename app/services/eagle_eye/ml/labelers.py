from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd


TP1_PCT = 5.0
TP2_PCT = 10.0


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


def label_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Build target labels for one forensic event.

    Labels are driven by event fields populated by feature_builder:
    - tp1_hit_day, tp2_hit_day, stop_hit_day, max_excursion_pct
    plus fallback heuristics from peak/entry when needed.
    """
    tp1_day = _as_int(event.get("tp1_hit_day"))
    tp2_day = _as_int(event.get("tp2_hit_day"))
    stop_day = _as_int(event.get("stop_hit_day"))

    max_exc = _as_float(event.get("max_excursion_pct"))
    if math.isnan(max_exc):
        peak = _as_float(event.get("peak_price"))
        entry = _as_float(event.get("acceleration_price"))
        if not math.isnan(peak) and not math.isnan(entry) and entry > 0:
            max_exc = (peak / entry - 1.0) * 100.0

    duration_days = _as_int(event.get("duration_days"))
    if tp1_day is None and not math.isnan(max_exc) and max_exc >= TP1_PCT and duration_days is not None:
        tp1_day = duration_days
    if tp2_day is None and not math.isnan(max_exc) and max_exc >= TP2_PCT and duration_days is not None:
        tp2_day = duration_days

    y_tp1_5d = int(tp1_day is not None and tp1_day <= 5)
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

    return {
        "y_tp1_5d": y_tp1_5d,
        "y_tp1_10d": y_tp1_10d,
        "y_tp1_20d": y_tp1_20d,
        "y_tp2_20d": y_tp2_20d,
        "y_max_excursion_pct": max_exc,
        "y_days_to_tp1": float(tp1_day) if tp1_day is not None else float("nan"),
        "y_outcome_category": category,
    }


def build_labels(events: pd.DataFrame) -> pd.DataFrame:
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
            ]
        )

    rows = [label_event(rec) for rec in events.to_dict(orient="records")]
    out = pd.DataFrame(rows, index=events.index)

    # Ensure stable dtypes.
    for col in ("y_tp1_5d", "y_tp1_10d", "y_tp1_20d", "y_tp2_20d"):
        out[col] = out[col].astype(int)

    out["y_max_excursion_pct"] = pd.to_numeric(out["y_max_excursion_pct"], errors="coerce")
    out["y_days_to_tp1"] = pd.to_numeric(out["y_days_to_tp1"], errors="coerce")
    out["y_outcome_category"] = out["y_outcome_category"].astype(str)

    return out

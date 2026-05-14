from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .calibrator import reliability_diagram_data


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-8, 1 - 1e-8)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(log_loss(y, p))


def calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    rel = reliability_diagram_data(np.asarray(y_true), np.asarray(y_prob), n_bins=n_bins)
    if not rel:
        return {"max_error": float("nan"), "mean_error": float("nan")}
    errors = np.asarray([r["abs_error"] for r in rel], dtype=float)
    return {
        "max_error": float(np.nanmax(errors)),
        "mean_error": float(np.nanmean(errors)),
    }


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    auc = _safe_auc(y_true, y_prob)
    ll = _safe_log_loss(y_true, y_prob)
    cal = calibration_error(y_true, y_prob, n_bins=n_bins)
    return {
        "auc": auc,
        "log_loss": ll,
        "calibration_max_error": cal["max_error"],
        "calibration_mean_error": cal["mean_error"],
    }


def top_feature_importance(
    model: Any,
    feature_list: Sequence[str],
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    if model is None:
        return []
    gain = model.feature_importance(importance_type="gain")
    rows = []
    for i, feature in enumerate(feature_list):
        score = float(gain[i]) if i < len(gain) else 0.0
        rows.append({"feature": feature, "gain": score})
    rows.sort(key=lambda x: x["gain"], reverse=True)
    return rows[:top_n]


def failure_cases(
    event_frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_cases: int = 10,
) -> List[Dict[str, Any]]:
    if event_frame is None or event_frame.empty:
        return []

    df = event_frame.copy().reset_index(drop=True)
    df["actual"] = np.asarray(y_true, dtype=float)
    df["predicted"] = np.asarray(y_prob, dtype=float)
    df["abs_error"] = (df["actual"] - df["predicted"]).abs()

    worst = df.sort_values("abs_error", ascending=False).head(n_cases)
    out: List[Dict[str, Any]] = []
    for _, row in worst.iterrows():
        out.append(
            {
                "ticker": str(row.get("ticker")),
                "event_id": str(row.get("event_id")),
                "event_date": str(row.get("event_date")),
                "actual": float(row.get("actual")),
                "predicted": float(row.get("predicted")),
                "abs_error": float(row.get("abs_error")),
                "outcome_category": str(row.get("y_outcome_category", "")),
            }
        )
    return out


def build_model_report(
    *,
    tier: str,
    identifier: str,
    event_frame: pd.DataFrame,
    fold_metrics: List[Dict[str, Any]],
    mean_metrics: Dict[str, float],
    std_auc: float,
    calibration_summary: Dict[str, Any],
    feature_importances: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if event_frame.empty:
        date_range = {"start": None, "end": None}
        class_balance = float("nan")
    else:
        dt = pd.to_datetime(event_frame["event_date"], errors="coerce")
        date_range = {
            "start": dt.min().date().isoformat() if dt.notna().any() else None,
            "end": dt.max().date().isoformat() if dt.notna().any() else None,
        }
        class_balance = float(pd.to_numeric(event_frame["y_tp1_20d"], errors="coerce").mean())

    return {
        "tier": tier,
        "identifier": identifier,
        "training_set": {
            "n_events": int(len(event_frame)),
            "date_range": date_range,
            "class_balance": class_balance,
        },
        "walk_forward_cv": {
            "folds": fold_metrics,
            "mean_auc": mean_metrics.get("auc"),
            "std_auc": std_auc,
            "mean_log_loss": mean_metrics.get("log_loss"),
            "mean_calibration_max_error": mean_metrics.get("calibration_max_error"),
            "mean_calibration_mean_error": mean_metrics.get("calibration_mean_error"),
        },
        "calibration": calibration_summary,
        "top_features": feature_importances,
        "failure_cases": failures,
    }

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

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


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = pd.Series(np.asarray(y_true, dtype=float))
    p = pd.Series(np.asarray(y_pred, dtype=float))
    if y.nunique(dropna=True) < 2 or p.nunique(dropna=True) < 2:
        return float("nan")
    return float(y.corr(p, method="spearman"))


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return {
            "spearman": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
        }
    yv = y[mask]
    pv = p[mask]
    rmse = float(np.sqrt(mean_squared_error(yv, pv)))
    return {
        "spearman": _safe_spearman(yv, pv),
        "mae": float(mean_absolute_error(yv, pv)),
        "rmse": rmse,
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute BUY/SELL/HOLD classifier diagnostics.

    Label convention:
      0 = SELL, 1 = HOLD, 2 = BUY
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    proba = np.asarray(y_proba, dtype=float)

    if proba.ndim != 2 or proba.shape[1] < 3 or yt.size == 0:
        return {
            "auc_buy": float("nan"),
            "auc_sell": float("nan"),
            "mean_auc": float("nan"),
            "precision_buy": 0.0,
            "recall_buy": 0.0,
            "f1_buy": 0.0,
            "n_buy_true": int((yt == 2).sum()),
            "n_sell_true": int((yt == 0).sum()),
            "n_buy_pred": int((yp == 2).sum()),
        }

    buy_true = (yt == 2).astype(int)
    sell_true = (yt == 0).astype(int)
    buy_proba = proba[:, 2]
    sell_proba = proba[:, 0]

    auc_buy = roc_auc_score(buy_true, buy_proba) if np.unique(buy_true).size > 1 else float("nan")
    auc_sell = roc_auc_score(sell_true, sell_proba) if np.unique(sell_true).size > 1 else float("nan")

    buy_pred = (yp == 2).astype(int)
    precision_buy = precision_score(buy_true, buy_pred, zero_division=0)
    recall_buy = recall_score(buy_true, buy_pred, zero_division=0)
    f1_buy = f1_score(buy_true, buy_pred, zero_division=0)

    finite_aucs = [v for v in (auc_buy, auc_sell) if np.isfinite(v)]
    mean_auc = float(np.mean(finite_aucs)) if finite_aucs else float("nan")

    return {
        "auc_buy": round(float(auc_buy), 4) if np.isfinite(auc_buy) else float("nan"),
        "auc_sell": round(float(auc_sell), 4) if np.isfinite(auc_sell) else float("nan"),
        "mean_auc": round(float(mean_auc), 4) if np.isfinite(mean_auc) else float("nan"),
        "precision_buy": round(float(precision_buy), 4),
        "recall_buy": round(float(recall_buy), 4),
        "f1_buy": round(float(f1_buy), 4),
        "n_buy_true": int(buy_true.sum()),
        "n_sell_true": int(sell_true.sum()),
        "n_buy_pred": int(buy_pred.sum()),
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
    task: str = "binary",
    target_col: str = "y_tp1_20d",
) -> Dict[str, Any]:
    if event_frame.empty:
        date_range = {"start": None, "end": None}
        target_mean = float("nan")
        target_min = float("nan")
        target_max = float("nan")
    else:
        dt = pd.to_datetime(event_frame["event_date"], errors="coerce")
        date_range = {
            "start": dt.min().date().isoformat() if dt.notna().any() else None,
            "end": dt.max().date().isoformat() if dt.notna().any() else None,
        }
        target_series = pd.to_numeric(event_frame.get(target_col), errors="coerce")
        target_mean = float(target_series.mean()) if target_series is not None else float("nan")
        target_min = float(target_series.min()) if target_series is not None else float("nan")
        target_max = float(target_series.max()) if target_series is not None else float("nan")

    walk_forward_payload: Dict[str, Any] = {
        "folds": fold_metrics,
    }
    if task == "regression":
        walk_forward_payload.update(
            {
                "mean_spearman": mean_metrics.get("spearman"),
                "std_spearman": std_auc,
                "mean_mae": mean_metrics.get("mae"),
                "mean_rmse": mean_metrics.get("rmse"),
            }
        )
    elif task == "classification":
        walk_forward_payload.update(
            {
                "mean_auc": mean_metrics.get("mean_auc"),
                "mean_auc_buy": mean_metrics.get("auc_buy"),
                "mean_auc_sell": mean_metrics.get("auc_sell"),
                "mean_precision_buy": mean_metrics.get("precision_buy"),
                "mean_recall_buy": mean_metrics.get("recall_buy"),
                "mean_f1_buy": mean_metrics.get("f1_buy"),
                "std_auc": std_auc,
            }
        )
    else:
        walk_forward_payload.update(
            {
                "mean_auc": mean_metrics.get("auc"),
                "std_auc": std_auc,
                "mean_log_loss": mean_metrics.get("log_loss"),
                "mean_calibration_max_error": mean_metrics.get("calibration_max_error"),
                "mean_calibration_mean_error": mean_metrics.get("calibration_mean_error"),
            }
        )

    training_set: Dict[str, Any] = {
        "n_events": int(len(event_frame)),
        "date_range": date_range,
        "target_col": target_col,
        "target_mean": target_mean,
        "target_min": target_min,
        "target_max": target_max,
    }
    if task != "regression":
        training_set["class_balance"] = target_mean

    return {
        "tier": tier,
        "identifier": identifier,
        "task": task,
        "training_set": training_set,
        "walk_forward_cv": walk_forward_payload,
        "calibration": calibration_summary,
        "top_features": feature_importances,
        "failure_cases": failures,
    }

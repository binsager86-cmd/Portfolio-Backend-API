from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibrationResult:
    calibrator: Optional[IsotonicRegression]
    summary: Dict[str, Any]
    reliability: List[Dict[str, Any]]


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> List[Dict[str, Any]]:
    if len(y_true) == 0:
        return []

    data = pd.DataFrame({"y": y_true.astype(float), "p": y_prob.astype(float)})
    data = data.dropna()
    if data.empty:
        return []

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    data["bin"] = pd.cut(data["p"], bins=bins, include_lowest=True)

    out: List[Dict[str, Any]] = []
    for bucket, group in data.groupby("bin", observed=False):
        if group.empty:
            continue
        p_mean = float(group["p"].mean())
        y_rate = float(group["y"].mean())
        out.append(
            {
                "bin": str(bucket),
                "count": int(len(group)),
                "predicted_mean": p_mean,
                "actual_rate": y_rate,
                "abs_error": abs(p_mean - y_rate),
            }
        )
    return out


def fit_isotonic_calibrator(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    n_bins: int = 10,
) -> CalibrationResult:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(raw_scores, dtype=float)

    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]

    if len(y) == 0 or len(np.unique(y)) < 2:
        reliability = reliability_diagram_data(y, s, n_bins=n_bins)
        errors = [r["abs_error"] for r in reliability] if reliability else [0.0]
        summary = {
            "n_samples": int(len(y)),
            "max_error": float(max(errors)),
            "mean_error": float(np.mean(errors)),
            "warning": False,
            "fitted": False,
        }
        return CalibrationResult(calibrator=None, summary=summary, reliability=reliability)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(s, y)
    calibrated = calibrator.predict(s)

    reliability = reliability_diagram_data(y, calibrated, n_bins=n_bins)
    errors = [r["abs_error"] for r in reliability] if reliability else [0.0]
    max_err = float(max(errors))
    mean_err = float(np.mean(errors))

    summary = {
        "n_samples": int(len(y)),
        "max_error": max_err,
        "mean_error": mean_err,
        "warning": bool(max_err > 0.15),
        "fitted": True,
    }
    return CalibrationResult(calibrator=calibrator, summary=summary, reliability=reliability)


def apply_calibrator(
    calibrator: Optional[IsotonicRegression],
    raw_scores: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=float)
    if calibrator is None:
        return np.clip(scores, 0.0, 1.0)
    return np.clip(calibrator.predict(scores), 0.0, 1.0)

"""
evaluation_v2.py — Phase 2B Stage 1: Proper evaluation infrastructure.

Provides:
  - evaluate_predictions()       : comprehensive metric dict
  - compute_reliability_diagram(): calibration diagram data
  - stratified_walk_forward_cv() : chronological CV with class-balance guards
  - honest_train_test_split()    : smart chronological split with positive-rate check
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Dict[str, Any]:
    """Return a comprehensive metric dict for a binary classifier.

    Parameters
    ----------
    y_true : array-like of 0/1
    y_pred_proba : array-like of predicted positive probabilities

    Returns
    -------
    dict with keys:
        auc_roc, auc_pr, brier_score, log_loss,
        max_calibration_error, mean_calibration_error,
        precision_at_top_10pct, precision_at_top_20pct,
        n_positives, n_negatives, positive_rate,
        prevalence_baseline_auc_pr
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)

    n_total = len(y_true)
    n_positives = int(y_true.sum())
    n_negatives = n_total - n_positives
    positive_rate = float(n_positives / n_total) if n_total > 0 else 0.0
    prevalence_baseline_auc_pr = positive_rate  # random classifier PR-AUC

    # Core metrics
    try:
        auc_roc = float(roc_auc_score(y_true, y_pred_proba))
    except ValueError:
        auc_roc = float("nan")

    try:
        auc_pr = float(average_precision_score(y_true, y_pred_proba))
    except ValueError:
        auc_pr = float("nan")

    brier = float(brier_score_loss(y_true, y_pred_proba))

    try:
        ll = float(log_loss(y_true, y_pred_proba))
    except ValueError:
        ll = float("nan")

    # Calibration errors from reliability diagram
    diagram = compute_reliability_diagram(y_true, y_pred_proba, n_bins=10)
    cal_errors = [
        abs(pred - actual)
        for pred, actual, count in zip(
            diagram["predicted_means"],
            diagram["actual_rates"],
            diagram["bin_counts"],
        )
        if count > 0
    ]
    max_cal_error = float(max(cal_errors)) if cal_errors else float("nan")
    mean_cal_error = float(np.mean(cal_errors)) if cal_errors else float("nan")

    # Precision at top K%
    precision_top10 = _precision_at_top_k(y_true, y_pred_proba, k=0.10)
    precision_top20 = _precision_at_top_k(y_true, y_pred_proba, k=0.20)

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "brier_score": brier,
        "log_loss": ll,
        "max_calibration_error": max_cal_error,
        "mean_calibration_error": mean_cal_error,
        "precision_at_top_10pct": precision_top10,
        "precision_at_top_20pct": precision_top20,
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "positive_rate": positive_rate,
        "prevalence_baseline_auc_pr": prevalence_baseline_auc_pr,
    }


def compute_reliability_diagram(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, List[float]]:
    """Compute calibration reliability diagram data.

    Parameters
    ----------
    y_true : array-like of 0/1
    y_pred_proba : array-like of predicted positive probabilities
    n_bins : number of equal-width bins in [0, 1]

    Returns
    -------
    dict with:
        bin_centers      : midpoint of each bin
        predicted_means  : mean predicted probability per bin
        actual_rates     : fraction of positives per bin
        bin_counts       : number of samples per bin
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers: List[float] = []
    predicted_means: List[float] = []
    actual_rates: List[float] = []
    bin_counts: List[int] = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        center = float((lo + hi) / 2)
        # include right edge only for last bin
        if i < n_bins - 1:
            mask = (y_pred_proba >= lo) & (y_pred_proba < hi)
        else:
            mask = (y_pred_proba >= lo) & (y_pred_proba <= hi)

        count = int(mask.sum())
        bin_centers.append(center)
        bin_counts.append(count)

        if count == 0:
            predicted_means.append(float("nan"))
            actual_rates.append(float("nan"))
        else:
            predicted_means.append(float(y_pred_proba[mask].mean()))
            actual_rates.append(float(y_true[mask].mean()))

    return {
        "bin_centers": bin_centers,
        "predicted_means": predicted_means,
        "actual_rates": actual_rates,
        "bin_counts": bin_counts,
    }


def stratified_walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    n_folds: int = 5,
) -> List[Dict[str, Any]]:
    """Chronological walk-forward CV with per-fold class-balance guards.

    Folds are created by sorting chronologically and using the most-recent
    ~15% of remaining data as the test window.  Each test fold must have
    at least 10 positives AND 10 negatives; if not, the window is expanded
    backward until both conditions are met (or the fold is skipped if
    impossible).

    Parameters
    ----------
    X : feature DataFrame (index aligned with y and dates)
    y : binary target Series
    dates : date Series (same index as X/y)
    n_folds : number of CV folds (default 5)

    Returns
    -------
    List of dicts, each containing:
        fold          : int (1-based)
        train_indices : list of integer positions
        test_indices  : list of integer positions
        train_size    : int
        test_size     : int
        test_n_pos    : int
        test_n_neg    : int
        test_date_min : str
        test_date_max : str
    """
    # Align all inputs by sorting chronologically
    sort_order = np.argsort(dates.values, kind="stable")
    y_sorted = np.asarray(y.values, dtype=float)[sort_order]
    dates_sorted = pd.Series(dates.values[sort_order]).reset_index(drop=True)
    n_total = len(y_sorted)

    MIN_POS_IN_TEST = 10
    MIN_NEG_IN_TEST = 10

    folds: List[Dict[str, Any]] = []
    # We walk backward — the k-th fold uses data near the end of the series
    # Each fold's test region is a non-overlapping slice of ~15%
    fold_size = max(1, n_total // (n_folds + 1))  # rough slice target

    # Compute non-overlapping test end indices from right side
    test_end_positions = []
    for k in range(n_folds):
        # fold 0 covers the rightmost fold_size rows, fold 1 the next, etc.
        end = n_total - k * fold_size
        start = end - fold_size
        if start < 1:
            break
        test_end_positions.append((start, end))

    # Reverse so fold 1 is earliest
    test_end_positions = list(reversed(test_end_positions))

    for fold_num, (test_start, test_end) in enumerate(test_end_positions, start=1):
        # Try to satisfy class balance; expand test window backward if needed
        actual_test_start = test_start
        while actual_test_start > 0:
            test_idx = list(range(actual_test_start, test_end))
            y_test = y_sorted[test_idx]
            n_pos = int(y_test.sum())
            n_neg = int(len(y_test) - n_pos)
            if n_pos >= MIN_POS_IN_TEST and n_neg >= MIN_NEG_IN_TEST:
                break
            actual_test_start = max(0, actual_test_start - 5)
        else:
            # Exhausted expansion — check final state
            test_idx = list(range(actual_test_start, test_end))
            y_test = y_sorted[test_idx]
            n_pos = int(y_test.sum())
            n_neg = int(len(y_test) - n_pos)

        if n_pos < MIN_POS_IN_TEST or n_neg < MIN_NEG_IN_TEST:
            # Cannot satisfy — skip this fold
            continue

        train_idx = list(range(0, actual_test_start))
        if len(train_idx) < 20:
            continue  # not enough training data

        # Map back to original integer positions via sort_order
        orig_train = [int(sort_order[i]) for i in train_idx]
        orig_test = [int(sort_order[i]) for i in test_idx]

        date_vals = dates_sorted.values
        folds.append({
            "fold": fold_num,
            "train_indices": orig_train,
            "test_indices": orig_test,
            "train_size": len(orig_train),
            "test_size": len(orig_test),
            "test_n_pos": n_pos,
            "test_n_neg": n_neg,
            "test_date_min": str(date_vals[actual_test_start]),
            "test_date_max": str(date_vals[test_end - 1]),
        })

    return folds


def honest_train_test_split(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    test_size_target: float = 0.20,
    min_positive_rate: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Chronological train/test split with positive-rate guarantee.

    Strategy:
      1. Sort by date_col.
      2. Try a pure chronological 80/20 split.
      3. If test positive_rate < min_positive_rate, shift the split
         point earlier (toward the past) in steps of 5% of total rows
         until the condition is met or there is no more room.
      4. If still below threshold, fall back to stratified random split
         (with a note in the report).

    Parameters
    ----------
    df               : full dataset DataFrame
    target_col       : binary label column name
    date_col         : date column name (sortable)
    test_size_target : target fraction for test set (default 0.20)
    min_positive_rate: minimum acceptable positive rate in test (default 0.25)

    Returns
    -------
    (train_df, test_df, report_dict)
    report_dict keys:
        strategy, split_date, test_positive_rate, train_positive_rate,
        n_train, n_test, n_test_pos, n_test_neg, adjusted, note
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    y = df_sorted[target_col].astype(float).values

    base_split_n = int(n * (1.0 - test_size_target))

    def _split_at(split_n: int) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        split_n = max(10, min(split_n, n - 10))
        tr = df_sorted.iloc[:split_n]
        te = df_sorted.iloc[split_n:]
        pos_rate = float(te[target_col].sum() / len(te)) if len(te) > 0 else 0.0
        return tr, te, pos_rate

    # 1. Try pure chronological split
    train_df, test_df, pos_rate = _split_at(base_split_n)
    strategy = "chronological"
    adjusted = False
    note = "Clean chronological 80/20 split."

    # 2. If below threshold, shift split earlier
    if pos_rate < min_positive_rate:
        adjusted = True
        step = max(1, int(n * 0.05))
        best_split_n = base_split_n
        best_pos_rate = pos_rate
        for shift in range(step, base_split_n - 10, step):
            candidate_n = base_split_n - shift
            tr_c, te_c, pr_c = _split_at(candidate_n)
            if pr_c >= min_positive_rate:
                train_df, test_df, pos_rate = tr_c, te_c, pr_c
                best_split_n = candidate_n
                best_pos_rate = pr_c
                note = (
                    f"Shifted split earlier by {shift} rows to achieve "
                    f"positive_rate={pr_c:.3f} in test set."
                )
                break
            if pr_c > best_pos_rate:
                best_pos_rate = pr_c
                best_split_n = candidate_n
        else:
            # Loop exhausted — take best found
            if best_pos_rate < min_positive_rate:
                # 3. Fall back to stratified random split
                strategy = "stratified_random"
                note = (
                    f"Chronological split cannot achieve positive_rate >= "
                    f"{min_positive_rate:.2f} even with maximum earlier shift "
                    f"(best={best_pos_rate:.3f}). Fell back to stratified random split. "
                    f"WARNING: test rows may not be purely out-of-time."
                )
                from sklearn.model_selection import train_test_split as _tts
                pos_mask = y == 1
                neg_mask = y == 0
                # stratify by label
                idx_train, idx_test = _tts(
                    np.arange(n),
                    test_size=test_size_target,
                    random_state=42,
                    stratify=y,
                )
                train_df = df_sorted.iloc[idx_train].sort_values(date_col)
                test_df = df_sorted.iloc[idx_test].sort_values(date_col)
                pos_rate = float(test_df[target_col].sum() / len(test_df))
            else:
                train_df, test_df, pos_rate = _split_at(best_split_n)
                note = (
                    f"Best earlier shift achieved positive_rate={best_pos_rate:.3f}. "
                    f"Still below target {min_positive_rate:.2f} but was best available."
                )

    n_test_pos = int(test_df[target_col].sum())
    n_test_neg = len(test_df) - n_test_pos
    train_pos_rate = float(train_df[target_col].sum() / len(train_df)) if len(train_df) > 0 else 0.0

    # Determine split date boundary
    split_date: Optional[str]
    if strategy == "chronological":
        split_date = str(test_df[date_col].min()) if len(test_df) > 0 else None
    else:
        split_date = None  # not meaningful for stratified

    report = {
        "strategy": strategy,
        "split_date": split_date,
        "test_positive_rate": round(pos_rate, 4),
        "train_positive_rate": round(train_pos_rate, 4),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_test_pos": n_test_pos,
        "n_test_neg": n_test_neg,
        "adjusted": adjusted,
        "note": note,
    }

    return train_df, test_df, report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _precision_at_top_k(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    k: float,
) -> float:
    """Precision when restricting predictions to the top k fraction."""
    n = len(y_true)
    top_n = max(1, int(np.ceil(n * k)))
    top_indices = np.argsort(y_pred_proba)[::-1][:top_n]
    return float(y_true[top_indices].mean())

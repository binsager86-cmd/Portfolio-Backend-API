"""
ml/walk_forward.py — Phase 2 Deliverable 2

Expanding-window walk-forward cross-validation with time embargo.

Fold structure (5 expanding folds):
  Train grows by 6 months per fold; validation = next 6-month slice.
  Embargo of >= 20 trading days strictly separates train from val.

  Fold 1:  train = 0 → 18m,       val = 18m+embargo → 24m
  Fold 2:  train = 0 → 24m,       val = 24m+embargo → 30m
  Fold 3:  train = 0 → 30m,       val = 30m+embargo → 36m
  Fold 4:  train = 0 → 36m,       val = 36m+embargo → 42m
  Fold 5:  train = 0 → all-6m-E,  val = last 6m  (OOT fold)

Rules:
  - Minimum training length ≥ 12 months (≈ 252 trading days).
  - If < 2 folds possible, caller receives an empty list and skips the stock.
  - Embargo strictly applied: val starts >= embargo_td bars after train ends.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────
TRADING_DAYS_PER_MONTH: int = 21
N_FOLDS: int = 5
MIN_TRAIN_MONTHS: int = 12
INIT_TRAIN_MONTHS: int = 18        # First fold training length
VAL_MONTHS: int = 6                # Validation slice per fold
DEFAULT_EMBARGO_TD: int = 20       # Trading days embargo


def build_folds(
    dates: Sequence[pd.Timestamp],
    *,
    min_train_months: int = MIN_TRAIN_MONTHS,
    embargo_td: int = DEFAULT_EMBARGO_TD,
    init_train_months: int = INIT_TRAIN_MONTHS,
    val_months: int = VAL_MONTHS,
    n_folds: int = N_FOLDS,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Build n_folds expanding-window walk-forward folds.

    Parameters
    ----------
    dates : sorted sequence of event timestamps (one per training row)
    min_train_months : minimum months required in a training fold
    embargo_td : trading-day gap between last train bar and first val bar
    init_train_months : training months in fold 1
    val_months : validation slice length in months
    n_folds : total number of folds (last fold is the OOT holdout)

    Returns
    -------
    List of (train_dates, val_dates) DatetimeIndex tuples.
    Empty list if the data is too short to form ≥ 2 usable folds.
    """
    dates = pd.DatetimeIndex(sorted(set(dates))).sort_values()
    n = len(dates)
    if n < 10:
        return []

    min_train_td = min_train_months * TRADING_DAYS_PER_MONTH
    val_td = val_months * TRADING_DAYS_PER_MONTH

    # The OOT (fold N) always uses the last val_td rows for validation.
    # Earlier folds have fixed val_td validation slices.
    # Train always starts from index 0.

    folds: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []

    init_train_td = init_train_months * TRADING_DAYS_PER_MONTH

    for fold_idx in range(n_folds):
        if fold_idx < n_folds - 1:
            # Folds 1..N-1: train grows by val_td each time
            train_end_idx = init_train_td + fold_idx * val_td
        else:
            # Fold N (OOT): train uses everything before the last val_td + embargo
            train_end_idx = n - val_td - embargo_td

        # Guard: not enough data for this fold
        if train_end_idx < min_train_td:
            continue
        if train_end_idx >= n:
            continue

        train_dates = dates[:train_end_idx]

        # Val starts after embargo gap
        val_start_idx = train_end_idx + embargo_td
        if fold_idx < n_folds - 1:
            val_end_idx = val_start_idx + val_td
        else:
            val_end_idx = n  # OOT uses all remaining rows

        if val_start_idx >= n or val_end_idx > n:
            continue

        val_dates = dates[val_start_idx:val_end_idx]

        if len(train_dates) < min_train_td or len(val_dates) < 5:
            continue

        folds.append((train_dates, val_dates))

    return folds


def split_df_by_fold(
    df: pd.DataFrame,
    date_col: str,
    train_dates: pd.DatetimeIndex,
    val_dates: pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Slice a DataFrame into train / val subsets using pre-computed fold date arrays.

    Rows whose event_date is in train_dates go to train;
    rows whose event_date is in val_dates go to val.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    dates_set_train = set(train_dates.normalize())
    dates_set_val = set(val_dates.normalize())

    df_dates = df[date_col].dt.normalize()
    train_df = df[df_dates.isin(dates_set_train)].reset_index(drop=True)
    val_df = df[df_dates.isin(dates_set_val)].reset_index(drop=True)
    return train_df, val_df


# ---------------------------------------------------------------------------
# Utility: indices version (for lightweight array operations)
# ---------------------------------------------------------------------------

def build_fold_indices(
    n_rows: int,
    *,
    min_train_months: int = MIN_TRAIN_MONTHS,
    embargo_td: int = DEFAULT_EMBARGO_TD,
    init_train_months: int = INIT_TRAIN_MONTHS,
    val_months: int = VAL_MONTHS,
    n_folds: int = N_FOLDS,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Integer-index version of build_folds.

    Treats row indices 0..n_rows-1 as sorted trading dates
    (assumes caller has already sorted by date).

    Returns list of (train_idx, val_idx) numpy arrays.
    """
    min_train_td = min_train_months * TRADING_DAYS_PER_MONTH
    val_td = val_months * TRADING_DAYS_PER_MONTH
    init_train_td = init_train_months * TRADING_DAYS_PER_MONTH

    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    for fold_idx in range(n_folds):
        if fold_idx < n_folds - 1:
            train_end = init_train_td + fold_idx * val_td
        else:
            train_end = n_rows - val_td - embargo_td

        if train_end < min_train_td or train_end >= n_rows:
            continue

        val_start = train_end + embargo_td
        if fold_idx < n_folds - 1:
            val_end = val_start + val_td
        else:
            val_end = n_rows

        if val_start >= n_rows or val_end > n_rows:
            continue

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)

        if len(train_idx) < min_train_td or len(val_idx) < 5:
            continue

        folds.append((train_idx, val_idx))

    return folds


# ---------------------------------------------------------------------------
# Dict-API version (expected by smoke tests and external callers)
# ---------------------------------------------------------------------------

def build_walk_forward_folds(
    dates: Sequence[pd.Timestamp],
    *,
    embargo_td: int = DEFAULT_EMBARGO_TD,
    min_train_months: int = MIN_TRAIN_MONTHS,
    init_train_months: int = INIT_TRAIN_MONTHS,
    val_months: int = VAL_MONTHS,
    n_folds: int = N_FOLDS,
) -> List[Dict[str, Any]]:
    """
    Dict-API walk-forward folds. Returns list of dicts with keys:
      train_end  : last Timestamp in the training window
      val_start  : first Timestamp in the validation window
      train_start: first Timestamp in the training window
      val_end    : last Timestamp in the validation window
      fold       : 1-based fold index

    Embargo is strictly in trading days (positional offset in `dates`),
    so ``pd.bdate_range(train_end, val_start)`` will have exactly
    embargo_td + 1 elements.
    """
    dates = pd.DatetimeIndex(sorted(set(dates))).sort_values()
    n = len(dates)
    if n < 10:
        return []

    min_train_td = min_train_months * TRADING_DAYS_PER_MONTH
    val_td = val_months * TRADING_DAYS_PER_MONTH
    init_train_td = init_train_months * TRADING_DAYS_PER_MONTH

    folds: List[Dict[str, Any]] = []

    for fold_idx in range(n_folds):
        if fold_idx < n_folds - 1:
            train_end_idx = init_train_td + fold_idx * val_td
        else:
            train_end_idx = n - val_td - embargo_td

        if train_end_idx < min_train_td or train_end_idx >= n:
            continue

        val_start_idx = train_end_idx + embargo_td
        if fold_idx < n_folds - 1:
            val_end_idx = val_start_idx + val_td
        else:
            val_end_idx = n

        if val_start_idx >= n or val_end_idx > n:
            continue

        train_idx_range = np.arange(0, train_end_idx)
        val_idx_range = np.arange(val_start_idx, val_end_idx)

        if len(train_idx_range) < min_train_td or len(val_idx_range) < 5:
            continue

        folds.append({
            "fold": fold_idx + 1,
            "train_start": dates[0],
            "train_end":   dates[train_end_idx - 1],       # last date IN training set
            "val_start":   dates[val_start_idx],           # first date IN val set
            "val_end":     dates[val_end_idx - 1],         # last date IN val set
            "n_train":     len(train_idx_range),
            "n_val":       len(val_idx_range),
        })

    return folds


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Quick sanity check with synthetic date range."""
    import datetime

    # Generate 5 years of "trading" dates (252 per year)
    n = 252 * 5
    base = pd.Timestamp("2018-01-02")
    dates = pd.date_range(base, periods=n, freq="B")  # business days

    folds = build_folds(dates)
    assert len(folds) >= 2, f"Expected >= 2 folds, got {len(folds)}"
    for i, (tr, va) in enumerate(folds):
        # Strict temporal ordering: all train < all val
        assert tr.max() < va.min(), f"Fold {i}: embargo violated — train max {tr.max()} >= val min {va.min()}"
        gap_days = (va.min() - tr.max()).days
        # At least 20 trading days embargo; calendar days is a loose proxy
        assert gap_days >= 14, f"Fold {i}: embargo too small: {gap_days} calendar days"

    # Index version
    idx_folds = build_fold_indices(n)
    assert len(idx_folds) >= 2, f"Expected >= 2 index folds, got {len(idx_folds)}"
    for i, (tr_idx, va_idx) in enumerate(idx_folds):
        assert tr_idx.max() < va_idx.min(), f"Fold {i}: index embargo violated"

    # Insufficient data — should return []
    short_dates = pd.date_range("2023-01-01", periods=50, freq="B")
    assert build_folds(short_dates) == [], "Expected empty list for short date series"

    print(f"walk_forward self-test passed. Folds: {[(len(tr), len(va)) for tr, va in folds]}")


if __name__ == "__main__":
    _self_test()

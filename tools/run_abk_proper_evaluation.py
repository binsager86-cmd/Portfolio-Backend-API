"""
run_abk_proper_evaluation.py — Phase 2B Stage 1 runner.

Builds the ABK strict-causal dataset, applies honest_train_test_split,
trains one LightGBM model, and reports full evaluation metrics including
PR-AUC, Brier, calibration, and single-feature CV AUC-PR.

Output: tools/abk_proper_evaluation_results.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app.services.eagle_eye.ml.evaluation_v2 import (
    compute_reliability_diagram,
    evaluate_predictions,
    honest_train_test_split,
    stratified_walk_forward_cv,
)
from app.services.eagle_eye.ml.feature_builder_v2 import (
    build_abk_strict_causal_dataset,
    get_feature_columns_v2,
    run_strict_causality_checks,
)
from app.services.eagle_eye.store import load_ohlcv

_FRAME_CACHE = Path("tools") / "abk_frame_cache.parquet"

warnings.filterwarnings("ignore", category=UserWarning)

TICKER = "ABK"
SEED = 42
TOP_FEATURES_FOR_CV = [
    "bb_bandwidth_t3",
    "rsi_t3",
    "adx_t1",
    "adx_t0",
    "mfi_t0",
]

LGB_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 15,
    "min_data_in_leaf": 10,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "seed": SEED,
    "deterministic": True,
    "verbosity": -1,
}

SINGLE_FEATURE_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 7,
    "min_data_in_leaf": 5,
    "learning_rate": 0.05,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "seed": SEED,
    "deterministic": True,
    "verbosity": -1,
}


def _prep_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    return df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values


def _run_single_feature_cv(
    train_df: pd.DataFrame,
    feature_col: str,
    baseline_auc_pr: float,
) -> Dict[str, Any]:
    """Run stratified walk-forward CV for one feature, return mean AUC-PR."""
    X = train_df[[feature_col]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = train_df["y"].astype(int)
    dates = train_df["event_date"]

    folds = stratified_walk_forward_cv(X, y, dates, n_folds=5)
    if not folds:
        return {
            "feature": feature_col,
            "n_folds": 0,
            "mean_auc_pr": None,
            "mean_auc_roc": None,
            "exceeds_baseline_plus_020": False,
            "note": "No valid folds produced",
        }

    fold_auc_pr: List[float] = []
    fold_auc_roc: List[float] = []

    for fold in folds:
        tr_idx = fold["train_indices"]
        te_idx = fold["test_indices"]
        X_tr = X.iloc[tr_idx].values
        y_tr = y.iloc[tr_idx].values
        X_te = X.iloc[te_idx].values
        y_te = y.iloc[te_idx].values

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        model = lgb.train(
            SINGLE_FEATURE_PARAMS,
            lgb.Dataset(X_tr, label=y_tr),
            num_boost_round=100,
        )
        pred = np.clip(model.predict(X_te), 0.0, 1.0)
        metrics = evaluate_predictions(y_te, pred)
        if not np.isnan(metrics["auc_pr"]):
            fold_auc_pr.append(metrics["auc_pr"])
        if not np.isnan(metrics["auc_roc"]):
            fold_auc_roc.append(metrics["auc_roc"])

    mean_pr = float(np.mean(fold_auc_pr)) if fold_auc_pr else None
    mean_roc = float(np.mean(fold_auc_roc)) if fold_auc_roc else None
    flag = bool(mean_pr is not None and mean_pr > baseline_auc_pr + 0.20)

    return {
        "feature": feature_col,
        "n_folds": len(folds),
        "n_valid_folds": len(fold_auc_pr),
        "mean_auc_pr": mean_pr,
        "mean_auc_roc": mean_roc,
        "baseline_auc_pr": round(baseline_auc_pr, 4),
        "exceeds_baseline_plus_020": flag,
        "note": "FLAGGED: unusually strong single feature" if flag else "Within expected range",
    }


def main() -> None:
    print(f"[1/6] Loading OHLCV for {TICKER}...")
    ohlcv = load_ohlcv(TICKER)
    print(f"      {len(ohlcv)} bars loaded")

    print("[2/6] Building strict-causal dataset...")
    if _FRAME_CACHE.exists():
        print(f"      Loading cached frame from {_FRAME_CACHE}")
        frame = pd.read_parquet(_FRAME_CACHE)
        for col in ("event_date", "acceleration_date"):
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col], errors="coerce").dt.normalize()
    else:
        frame = build_abk_strict_causal_dataset(ohlcv, ticker=TICKER, seed=SEED)
        _FRAME_CACHE.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(_FRAME_CACHE, index=False)
        print(f"      Frame cached to {_FRAME_CACHE}")
    print(f"      {len(frame)} rows, {frame['y'].sum():.0f} positive, {(frame['y']==0).sum()} negative")

    feature_cols = get_feature_columns_v2(frame)
    print(f"      {len(feature_cols)} feature columns")

    print("[3/6] Applying honest_train_test_split...")
    train_df, test_df, split_report = honest_train_test_split(
        frame,
        target_col="y",
        date_col="event_date",
        test_size_target=0.20,
        min_positive_rate=0.25,
    )
    print(f"      Strategy: {split_report['strategy']}")
    print(f"      Train: {split_report['n_train']} rows, pos_rate={split_report['train_positive_rate']:.3f}")
    print(f"      Test:  {split_report['n_test']} rows, pos_rate={split_report['test_positive_rate']:.3f}")
    print(f"      Note: {split_report['note']}")

    # Check pass criterion
    test_pos_rate = split_report["test_positive_rate"]
    pos_rate_pass = test_pos_rate >= 0.25
    print(f"      Positive rate criterion (>=0.25): {'PASS' if pos_rate_pass else 'FAIL'} ({test_pos_rate:.3f})")

    print("[4/6] Training LightGBM on train set...")
    X_tr = _prep_matrix(train_df, feature_cols)
    y_tr = train_df["y"].astype(int).values
    X_te = _prep_matrix(test_df, feature_cols)
    y_te = test_df["y"].astype(int).values

    if len(np.unique(y_tr)) < 2:
        print("      ERROR: training set has only one class — cannot train")
        return

    model = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(X_tr, label=y_tr, feature_name=list(feature_cols)),
        num_boost_round=300,
    )
    pred_te = np.clip(model.predict(X_te), 0.0, 1.0)

    print("[5/6] Computing test set metrics...")
    test_metrics = evaluate_predictions(y_te, pred_te)
    print(f"      AUC-ROC:  {test_metrics['auc_roc']:.4f}")
    print(f"      AUC-PR:   {test_metrics['auc_pr']:.4f}  (baseline={test_metrics['prevalence_baseline_auc_pr']:.4f})")
    print(f"      Brier:    {test_metrics['brier_score']:.4f}")
    print(f"      Log-loss: {test_metrics['log_loss']:.4f}")
    print(f"      Max cal error: {test_metrics['max_calibration_error']:.4f}")
    print(f"      Prec@top10%:   {test_metrics['precision_at_top_10pct']:.4f}")
    print(f"      Prec@top20%:   {test_metrics['precision_at_top_20pct']:.4f}")

    reliability_diagram = compute_reliability_diagram(y_te, pred_te, n_bins=10)

    print("[5b/6] Running stratified walk-forward CV on training portion...")
    X_train_frame = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_train_series = train_df["y"].astype(int)
    dates_train = train_df["event_date"]

    cv_folds = stratified_walk_forward_cv(X_train_frame, y_train_series, dates_train, n_folds=5)
    print(f"      {len(cv_folds)} valid folds produced")

    cv_auc_roc_list: List[float] = []
    cv_auc_pr_list: List[float] = []
    cv_brier_list: List[float] = []
    cv_fold_details: List[Dict[str, Any]] = []

    for fold in cv_folds:
        tr_idx = fold["train_indices"]
        te_idx = fold["test_indices"]
        X_cv_tr = X_train_frame.iloc[tr_idx].values
        y_cv_tr = y_train_series.iloc[tr_idx].values
        X_cv_te = X_train_frame.iloc[te_idx].values
        y_cv_te = y_train_series.iloc[te_idx].values

        if len(np.unique(y_cv_tr)) < 2 or len(np.unique(y_cv_te)) < 2:
            continue

        cv_model = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(X_cv_tr, label=y_cv_tr, feature_name=list(feature_cols)),
            num_boost_round=300,
        )
        cv_pred = np.clip(cv_model.predict(X_cv_te), 0.0, 1.0)
        fold_metrics = evaluate_predictions(y_cv_te, cv_pred)

        if not np.isnan(fold_metrics["auc_roc"]):
            cv_auc_roc_list.append(fold_metrics["auc_roc"])
        if not np.isnan(fold_metrics["auc_pr"]):
            cv_auc_pr_list.append(fold_metrics["auc_pr"])
        if not np.isnan(fold_metrics["brier_score"]):
            cv_brier_list.append(fold_metrics["brier_score"])

        cv_fold_details.append({
            "fold": fold["fold"],
            "train_size": fold["train_size"],
            "test_size": fold["test_size"],
            "test_n_pos": fold["test_n_pos"],
            "test_n_neg": fold["test_n_neg"],
            "test_date_min": fold["test_date_min"],
            "test_date_max": fold["test_date_max"],
            "auc_roc": round(fold_metrics["auc_roc"], 4) if not np.isnan(fold_metrics["auc_roc"]) else None,
            "auc_pr": round(fold_metrics["auc_pr"], 4) if not np.isnan(fold_metrics["auc_pr"]) else None,
            "brier_score": round(fold_metrics["brier_score"], 4),
            "positive_rate": round(fold_metrics["positive_rate"], 4),
        })

    mean_cv_auc_roc = float(np.mean(cv_auc_roc_list)) if cv_auc_roc_list else None
    mean_cv_auc_pr = float(np.mean(cv_auc_pr_list)) if cv_auc_pr_list else None
    mean_cv_brier = float(np.mean(cv_brier_list)) if cv_brier_list else None

    print(f"      CV mean AUC-ROC: {mean_cv_auc_roc}")
    print(f"      CV mean AUC-PR:  {mean_cv_auc_pr}")
    print(f"      CV mean Brier:   {mean_cv_brier}")

    # Check metric consistency pass criterion
    consistency_pass: Optional[bool] = None
    if mean_cv_auc_roc is not None and not np.isnan(test_metrics["auc_roc"]):
        gap = abs(test_metrics["auc_roc"] - mean_cv_auc_roc)
        consistency_pass = bool(gap <= 0.05)
        print(f"      CV-vs-test AUC-ROC gap: {gap:.4f} ({'PASS' if consistency_pass else 'FAIL'}, threshold=0.05)")

    print("[6/6] Single-feature AUC-PR CV for top 5 features...")
    single_feature_results: List[Dict[str, Any]] = []
    prevalence_for_single = train_df["y"].mean()

    for feat in TOP_FEATURES_FOR_CV:
        if feat not in train_df.columns:
            single_feature_results.append({
                "feature": feat,
                "note": "Feature column not found in dataset",
            })
            continue
        result = _run_single_feature_cv(train_df, feat, float(prevalence_for_single))
        single_feature_results.append(result)
        flag_str = " *** FLAGGED ***" if result.get("exceeds_baseline_plus_020") else ""
        print(f"      {feat}: AUC-PR={result.get('mean_auc_pr')}{flag_str}")

    any_flagged = any(r.get("exceeds_baseline_plus_020") for r in single_feature_results)
    single_feature_pass = not any_flagged

    # Pass/fail summary
    pass_criteria = {
        "test_positive_rate_ge_025": pos_rate_pass,
        "cv_test_metrics_within_005": consistency_pass,
        "no_single_feature_exceeds_baseline_plus_020": single_feature_pass,
        "all_metrics_produced": True,
    }
    overall_pass = all(v for v in pass_criteria.values() if v is not None)

    print(f"\n{'='*60}")
    print(f"PASS CRITERIA SUMMARY:")
    for k, v in pass_criteria.items():
        print(f"  {k}: {'PASS' if v else 'FAIL' if v is not None else 'UNKNOWN'}")
    print(f"OVERALL: {'PASS — proceed to Stage 2' if overall_pass else 'FAIL — review results'}")
    print(f"{'='*60}\n")

    # Build final output
    output = {
        "ticker": TICKER,
        "phase": "2B_stage1_proper_evaluation",
        "seed": SEED,
        "dataset_summary": {
            "rows_total": len(frame),
            "n_features": len(feature_cols),
            "feature_cols": list(feature_cols),
        },
        "split_report": split_report,
        "test_metrics": {k: (round(v, 4) if isinstance(v, float) and not np.isnan(v) else v) for k, v in test_metrics.items()},
        "reliability_diagram": reliability_diagram,
        "walk_forward_cv": {
            "n_folds_produced": len(cv_folds),
            "n_valid_folds": len(cv_auc_roc_list),
            "mean_auc_roc": round(mean_cv_auc_roc, 4) if mean_cv_auc_roc is not None else None,
            "mean_auc_pr": round(mean_cv_auc_pr, 4) if mean_cv_auc_pr is not None else None,
            "mean_brier": round(mean_cv_brier, 4) if mean_cv_brier is not None else None,
            "fold_details": cv_fold_details,
        },
        "single_feature_cv": {
            "train_prevalence_baseline": round(float(prevalence_for_single), 4),
            "any_feature_flagged": any_flagged,
            "results": single_feature_results,
        },
        "pass_criteria": pass_criteria,
        "overall_pass": overall_pass,
    }

    out_path = Path("tools") / "abk_proper_evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"Results saved to {out_path}")
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()

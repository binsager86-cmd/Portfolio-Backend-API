"""
Eagle Eye — Ride Quality Trainer  (Phase R2)
============================================

Trains the Ride Quality Model: a LightGBM multi-class classifier that
predicts HOLD / ADD / EXIT for each day of an active position.

Also trains a companion regression model for ``remaining_upside_pct``.

Training strategy
-----------------
1. Per-stock model when ≥ MIN_PER_STOCK_SAMPLES labeled ride-days exist
2. Pooled (all-stock) model as fallback and for illiquid tickers

Both models reuse the existing Phase 2 infrastructure:
  - walk-forward temporal splits (no look-ahead)
  - save_model_bundle() for storage
  - same hyper-parameter grid (tuned for Kuwait market size)

CLI
---
  python -m app.services.eagle_eye.ml.ride_trainer [--ticker ZAIN] [--full-rebuild]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)

from app.services.eagle_eye.ml.model_store import (
    get_cache_root,
    get_models_root,
    get_reports_root,
    load_model_bundle,
    save_model_bundle,
)
from app.services.eagle_eye.ml.ride_feature_builder import (
    LABEL_ENCODING,
    MIN_PER_STOCK_SAMPLES,
    MIN_POOLED_SAMPLES,
    RIDE_FEATURE_NAMES,
    build_pooled_ride_training_matrix,
    build_ride_training_matrix,
)
from app.services.eagle_eye.store import list_tickers_with_ohlcv

LOGGER = logging.getLogger(__name__)

# Tier names for model store
RIDE_MODEL_TIER = "ride_quality"
RIDE_REGRESSION_TIER = "ride_upside"

# Minimum cross-validation Spearman / accuracy to accept a model
MIN_CV_ACCURACY = 0.45


@dataclass
class RideModelTrainingResult:
    ticker_or_pool: str
    accepted: bool
    n_samples: int
    cv_accuracy: float
    cv_f1_macro: float
    cv_log_loss: float
    oot_accuracy: float
    oot_f1_macro: float
    rejected_reason: str
    report: Dict[str, Any]


# ---------------------------------------------------------------------------
# Core trainer
# ---------------------------------------------------------------------------

class RideQualityTrainer:
    """
    Trains and saves the Ride Quality classifier (HOLD/ADD/EXIT) and
    companion regression model (remaining_upside_pct) for Eagle Eye.
    """

    def __init__(
        self,
        *,
        models_root: Optional[Path | str] = None,
        logger: Optional[logging.Logger] = None,
        random_state: int = 42,
    ):
        self.models_root = get_models_root(models_root)
        self.cache_root = get_cache_root(models_root)
        self.reports_root = get_reports_root(models_root)
        self.logger = logger or LOGGER
        self.random_state = random_state

    # ── LightGBM hyperparameters ─────────────────────────────────────────

    def _clf_params(self) -> Dict[str, Any]:
        """Multi-class classifier params for the HOLD/ADD/EXIT model."""
        seed = self.random_state
        return {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "max_depth": 5,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "min_data_in_leaf": 20,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "deterministic": True,
            "verbosity": -1,
        }

    def _reg_params(self) -> Dict[str, Any]:
        """Regression params for remaining_upside_pct."""
        seed = self.random_state
        return {
            "objective": "regression",
            "metric": "l2",
            "max_depth": 5,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "min_data_in_leaf": 20,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "deterministic": True,
            "verbosity": -1,
        }

    # ── Walk-forward splits (temporal, no look-ahead) ─────────────────────

    def _walk_forward_splits(
        self, bar_dates: pd.Series
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        n = len(bar_dates)
        if n < 80:
            return []

        dt = pd.to_datetime(bar_dates, errors="coerce").dt.normalize()
        unique_dates = pd.Index(dt.dropna().unique()).sort_values()
        if len(unique_dates) < 8:
            return []

        boundaries = [0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(5):
            train_cut = unique_dates[min(int(len(unique_dates) * boundaries[i]) - 1, len(unique_dates) - 1)]
            test_cut = unique_dates[min(int(len(unique_dates) * boundaries[i + 1]) - 1, len(unique_dates) - 1)]
            train_idx = np.where(dt <= train_cut)[0]
            test_idx = np.where((dt > train_cut) & (dt <= test_cut))[0]
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits

    # ── Feature matrix preparation ────────────────────────────────────────

    def _prepare_X_y_clf(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Extract feature matrix and multi-class label from training DataFrame."""
        feat_cols = [c for c in RIDE_FEATURE_NAMES if c in df.columns]
        # Fill any missing feature columns with NaN
        for col in RIDE_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = float("nan")

        X = df[list(RIDE_FEATURE_NAMES)].copy()
        y = df["label_encoded"].astype(int)
        return X, y, list(RIDE_FEATURE_NAMES)

    def _prepare_X_y_reg(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract feature matrix and regression target."""
        X = df[list(RIDE_FEATURE_NAMES)].copy()
        y = df["remaining_upside_pct"].astype(float).clip(0.0, 100.0)
        y = y.fillna(0.0)
        return X, y

    # ── Cross-validation ─────────────────────────────────────────────────

    def _cross_validate_classifier(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Walk-forward cross-validation for the classifier."""
        X, y, _ = self._prepare_X_y_clf(df)
        splits = self._walk_forward_splits(df.get("bar_date", pd.Series(range(len(df)))))

        if not splits:
            return {"cv_accuracy": float("nan"), "cv_f1_macro": float("nan"), "cv_log_loss": float("nan"), "fold_results": []}

        fold_accs, fold_f1s, fold_lls = [], [], []
        params = self._clf_params()

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=300,
                valid_sets=[lgb.Dataset(X_te, label=y_te)],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
            )

            raw_pred = model.predict(X_te)
            arr = np.asarray(raw_pred)
            if arr.ndim == 1:
                n_class = 3
                if arr.size % n_class == 0:
                    arr = arr.reshape(-1, n_class)
            y_pred_cls = np.argmax(arr, axis=1)
            y_te_arr = y_te.to_numpy()

            acc = accuracy_score(y_te_arr, y_pred_cls)
            f1 = f1_score(y_te_arr, y_pred_cls, average="macro", zero_division=0)
            try:
                ll = log_loss(y_te_arr, arr, labels=[0, 1, 2])
            except Exception:
                ll = float("nan")

            fold_accs.append(acc)
            fold_f1s.append(f1)
            fold_lls.append(ll)

        return {
            "cv_accuracy": float(np.nanmean(fold_accs)),
            "cv_f1_macro": float(np.nanmean(fold_f1s)),
            "cv_log_loss": float(np.nanmean(fold_lls)),
            "fold_results": [
                {"acc": round(a, 4), "f1": round(f, 4)}
                for a, f in zip(fold_accs, fold_f1s)
            ],
        }

    # ── Train + save ─────────────────────────────────────────────────────

    def _train_classifier(
        self,
        df: pd.DataFrame,
        *,
        identifier: str,
        cv_result: Dict[str, Any],
    ) -> Optional[lgb.Booster]:
        """Train the final classifier on all data and save the model bundle."""
        X, y, feat_names = self._prepare_X_y_clf(df)
        params = self._clf_params()

        # OOT split: last 20% of date-ordered data
        splits = self._walk_forward_splits(df.get("bar_date", pd.Series(range(len(df)))))
        if splits:
            last_split = splits[-1]
            train_idx, oot_idx = last_split
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_oot, y_oot = X.iloc[oot_idx], y.iloc[oot_idx]
        else:
            X_tr, y_tr = X, y
            X_oot, y_oot = X.iloc[-20:], y.iloc[-20:]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=400,
            valid_sets=[lgb.Dataset(X_oot, label=y_oot)],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(period=-1)],
        )

        # OOT metrics
        raw_pred = model.predict(X_oot)
        arr = np.asarray(raw_pred)
        if arr.ndim == 1 and arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        y_pred_cls = np.argmax(arr, axis=1)
        oot_acc = float(accuracy_score(y_oot.to_numpy(), y_pred_cls))
        oot_f1 = float(f1_score(y_oot.to_numpy(), y_pred_cls, average="macro", zero_division=0))

        metadata = {
            "task": "multiclass_classification",
            "objective": "ride_quality",
            "num_class": 3,
            "label_map": {"0": "HOLD", "1": "ADD", "2": "EXIT"},
            "n_samples": len(df),
            "cv_accuracy": cv_result.get("cv_accuracy"),
            "cv_f1_macro": cv_result.get("cv_f1_macro"),
            "oot_accuracy": oot_acc,
            "oot_f1_macro": oot_f1,
            "trained_at": date.today().isoformat(),
            "feature_count": len(feat_names),
        }

        save_model_bundle(
            tier=RIDE_MODEL_TIER,
            identifier=identifier,
            model=model,
            calibrator=None,
            feature_list=feat_names,
            metadata=metadata,
            models_root=self.models_root,
        )
        self.logger.info(
            "Saved ride classifier: %s  OOT acc=%.3f  f1=%.3f",
            identifier, oot_acc, oot_f1,
        )
        return model

    def _train_regression(
        self,
        df: pd.DataFrame,
        *,
        identifier: str,
    ) -> None:
        """Train companion regression for remaining_upside_pct."""
        X, y = self._prepare_X_y_reg(df)

        splits = self._walk_forward_splits(df.get("bar_date", pd.Series(range(len(df)))))
        if splits:
            train_idx, _ = splits[-1]
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        else:
            X_tr, y_tr = X, y

        params = self._reg_params()
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=300,
            callbacks=[lgb.log_evaluation(period=-1)],
        )

        metadata = {
            "task": "regression",
            "objective": "remaining_upside_pct",
            "n_samples": len(df),
            "trained_at": date.today().isoformat(),
        }
        save_model_bundle(
            tier=RIDE_REGRESSION_TIER,
            identifier=identifier,
            model=model,
            calibrator=None,
            feature_list=list(RIDE_FEATURE_NAMES),
            metadata=metadata,
            models_root=self.models_root,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def train_ticker(self, ticker: str) -> RideModelTrainingResult:
        """Train per-stock ride model for *ticker*."""
        t0 = time.time()
        self.logger.info("Ride trainer: building matrix for %s", ticker)

        df = build_ride_training_matrix(ticker)
        if df.empty or len(df) < MIN_PER_STOCK_SAMPLES:
            reason = f"Only {len(df)} ride-day samples (need {MIN_PER_STOCK_SAMPLES})"
            self.logger.info("Ride trainer: %s SKIPPED — %s", ticker, reason)
            return RideModelTrainingResult(
                ticker_or_pool=ticker,
                accepted=False,
                n_samples=len(df),
                cv_accuracy=0.0,
                cv_f1_macro=0.0,
                cv_log_loss=float("nan"),
                oot_accuracy=0.0,
                oot_f1_macro=0.0,
                rejected_reason=reason,
                report={},
            )

        cv = self._cross_validate_classifier(df)
        cv_acc = cv.get("cv_accuracy", 0.0)

        if not (cv_acc >= MIN_CV_ACCURACY or pd.isna(cv_acc)):
            reason = f"CV accuracy {cv_acc:.3f} below threshold {MIN_CV_ACCURACY}"
            return RideModelTrainingResult(
                ticker_or_pool=ticker,
                accepted=False,
                n_samples=len(df),
                cv_accuracy=float(cv_acc),
                cv_f1_macro=cv.get("cv_f1_macro", 0.0),
                cv_log_loss=cv.get("cv_log_loss", float("nan")),
                oot_accuracy=0.0,
                oot_f1_macro=0.0,
                rejected_reason=reason,
                report=cv,
            )

        identifier = ticker.upper()
        self._train_classifier(df, identifier=identifier, cv_result=cv)
        self._train_regression(df, identifier=identifier)

        self.logger.info(
            "Ride trainer: %s ACCEPTED  n=%d  cv_acc=%.3f  elapsed=%.1fs",
            ticker, len(df), cv_acc, time.time() - t0,
        )
        return RideModelTrainingResult(
            ticker_or_pool=ticker,
            accepted=True,
            n_samples=len(df),
            cv_accuracy=float(cv_acc),
            cv_f1_macro=float(cv.get("cv_f1_macro", 0.0)),
            cv_log_loss=float(cv.get("cv_log_loss", float("nan"))),
            oot_accuracy=0.0,  # populated inside _train_classifier
            oot_f1_macro=0.0,
            rejected_reason="",
            report=cv,
        )

    def train_pooled(self, tickers: Optional[List[str]] = None) -> RideModelTrainingResult:
        """Train the pooled (all-stock) ride quality model."""
        t0 = time.time()

        if tickers is None:
            tickers = list_tickers_with_ohlcv()

        self.logger.info("Ride trainer: building pooled matrix from %d tickers", len(tickers))
        df = build_pooled_ride_training_matrix(tickers)

        if df.empty or len(df) < MIN_POOLED_SAMPLES:
            reason = f"Pooled data too small: {len(df)} samples"
            return RideModelTrainingResult(
                ticker_or_pool="__pooled__",
                accepted=False,
                n_samples=len(df),
                cv_accuracy=0.0,
                cv_f1_macro=0.0,
                cv_log_loss=float("nan"),
                oot_accuracy=0.0,
                oot_f1_macro=0.0,
                rejected_reason=reason,
                report={},
            )

        cv = self._cross_validate_classifier(df)
        cv_acc = cv.get("cv_accuracy", 0.0)

        identifier = "__pooled__"
        self._train_classifier(df, identifier=identifier, cv_result=cv)
        self._train_regression(df, identifier=identifier)

        cache_path = self.cache_root / "ride_pooled_matrix.pkl"
        df.to_pickle(cache_path)
        self.logger.info(
            "Ride trainer: pooled ACCEPTED  n=%d  tickers=%d  cv_acc=%.3f  elapsed=%.1fs",
            len(df), df["ticker"].nunique(), cv_acc, time.time() - t0,
        )
        return RideModelTrainingResult(
            ticker_or_pool="__pooled__",
            accepted=True,
            n_samples=len(df),
            cv_accuracy=float(cv_acc),
            cv_f1_macro=float(cv.get("cv_f1_macro", 0.0)),
            cv_log_loss=float(cv.get("cv_log_loss", float("nan"))),
            oot_accuracy=0.0,
            oot_f1_macro=0.0,
            rejected_reason="",
            report=cv,
        )

    def train_all(
        self,
        *,
        tickers: Optional[List[str]] = None,
        pooled_first: bool = True,
    ) -> List[RideModelTrainingResult]:
        """
        Train pooled + per-stock models for every ticker with sufficient data.
        """
        if tickers is None:
            tickers = list_tickers_with_ohlcv()

        results: List[RideModelTrainingResult] = []

        if pooled_first:
            self.logger.info("=== Ride Quality: Pooled model ===")
            results.append(self.train_pooled(tickers))

        accepted_ps = 0
        for i, ticker in enumerate(tickers):
            self.logger.info("[%d/%d] Per-stock ride model: %s", i + 1, len(tickers), ticker)
            result = self.train_ticker(ticker)
            results.append(result)
            if result.accepted:
                accepted_ps += 1

        self.logger.info(
            "Ride Quality training complete: %d per-stock accepted, pooled=%s",
            accepted_ps,
            results[0].accepted if results else False,
        )

        # Save summary report
        summary = {
            "trained_at": date.today().isoformat(),
            "total_tickers": len(tickers),
            "per_stock_accepted": accepted_ps,
            "pooled_accepted": results[0].accepted if pooled_first and results else False,
            "results": [
                {
                    "id": r.ticker_or_pool,
                    "accepted": r.accepted,
                    "n": r.n_samples,
                    "cv_acc": round(r.cv_accuracy, 4),
                    "cv_f1": round(r.cv_f1_macro, 4),
                    "rejected_reason": r.rejected_reason,
                }
                for r in results
            ],
        }
        report_path = self.reports_root / f"ride_quality_training_{date.today().isoformat()}.json"
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.logger.info("Training summary saved: %s", report_path)
        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train Eagle Eye Ride Quality Model")
    parser.add_argument("--ticker", help="Train only this ticker (default: all)")
    parser.add_argument("--pooled-only", action="store_true", help="Train pooled model only")
    parser.add_argument("--skip-pooled", action="store_true", help="Skip pooled model")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    trainer = RideQualityTrainer()

    if args.ticker:
        result = trainer.train_ticker(args.ticker.upper())
        LOGGER.info("Result: %s", result)
    elif args.pooled_only:
        result = trainer.train_pooled()
        LOGGER.info("Pooled result: %s", result)
    else:
        trainer.train_all(pooled_first=not args.skip_pooled)


if __name__ == "__main__":
    _cli()

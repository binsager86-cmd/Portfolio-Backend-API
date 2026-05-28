from __future__ import annotations

from collections import Counter
import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from .evaluator import (
    build_model_report,
    compute_classification_metrics,
    failure_cases,
    top_feature_importance,
)
from .feature_builder import (
    build_labeled_rows_from_ohlcv_cache,
    build_feature_matrix,
    get_feature_columns,
)
from .model_store import (
    get_cache_root,
    get_models_root,
    get_reports_root,
    save_model_bundle,
)


SECTOR_UNIVERSE = [
    "banking",
    "investment",
    "real_estate",
    "insurance",
    "telecom",
    "industrial",
    "energy",
    "consumer",
    "technology",
    "transport",
    "holding_misc",
]

# Multiclass label mapping used by LightGBM.
# Raw labels: -1 (SELL), 0 (HOLD), 1 (BUY)
# Mapped labels: 0 (SELL), 1 (HOLD), 2 (BUY)
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}


@dataclass
class TrainingConfig:
    random_state: int = 42
    min_per_stock_events: int = 100
    min_per_sector_events: int = 30
    mean_auc_reject_threshold: float = 0.60
    target_col: str = "label"


@dataclass
class ModelTrainingResult:
    tier: str
    identifier: str
    accepted: bool
    n_events: int
    mean_metrics: Dict[str, float]
    std_auc: float
    rejected_reason: str
    report: Dict[str, Any]


class EagleEyeMLTrainer:
    def __init__(
        self,
        *,
        config: Optional[TrainingConfig] = None,
        models_root: Optional[str | Path] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or TrainingConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.models_root = get_models_root(models_root)
        self.cache_root = get_cache_root(models_root)
        self.reports_root = get_reports_root(models_root)

    @property
    def _cache_file(self) -> Path:
        return self.cache_root / "event_features_latest.pkl"

    @property
    def _event_index_file(self) -> Path:
        return self.cache_root / "event_index.json"

    def _progress(self, label: str, index: int, total: int) -> None:
        if total <= 0:
            self.logger.info("[0/0] %s", label)
            return
        width = 24
        filled = int(round((index / total) * width))
        filled = min(max(filled, 0), width)
        bar = "#" * filled + "-" * (width - filled)
        pct = (index / total) * 100.0
        self.logger.info("[%d/%d] [%s] %5.1f%% %s", index, total, bar, pct, label)

    def build_dataset(self, force_rebuild: bool = False) -> pd.DataFrame:
        if self._cache_file.exists() and not force_rebuild:
            self.logger.info("Loading cached labeled features: %s", self._cache_file)
            return pd.read_pickle(self._cache_file)

        t0 = time.time()
        self.logger.info("Building labeled-bar feature rows from cached OHLCV...")
        raw_rows = build_labeled_rows_from_ohlcv_cache(logger=self.logger)
        self.logger.info("Generated %d raw labeled rows", len(raw_rows))

        features = build_feature_matrix(raw_rows, logger=self.logger)
        if features.frame.empty:
            raise RuntimeError("No feature rows available for training")

        dataset = features.frame.sort_values(["ticker", "event_date"]).reset_index(drop=True)

        # Keep only supported label classes {-1, 0, 1} for multiclass training.
        dataset[self.config.target_col] = pd.to_numeric(dataset[self.config.target_col], errors="coerce")
        dataset = dataset.loc[dataset[self.config.target_col].isin([-1, 0, 1])].copy()
        if dataset.empty:
            raise RuntimeError("No valid labeled rows available after class filtering")

        dataset.to_pickle(self._cache_file)

        event_counts = dataset.groupby("ticker").size().astype(int).to_dict()
        sector_map = (
            dataset[["ticker", "sector_raw"]]
            .drop_duplicates(subset=["ticker"]) 
            .set_index("ticker")["sector_raw"]
            .astype(str)
            .to_dict()
        )
        index_payload = {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "event_counts_by_ticker": event_counts,
            "ticker_sector_map": sector_map,
            "rejected_rows_per_ticker": features.rejected_counts,
        }
        self._event_index_file.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

        self.logger.info(
            "Dataset ready: %d rows, %d tickers, build %.1fs",
            len(dataset),
            dataset["ticker"].nunique(),
            time.time() - t0,
        )
        return dataset

    def _walk_forward_splits(self, event_dates: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_samples = int(len(event_dates))
        if n_samples < 40:
            return []

        dt = pd.to_datetime(event_dates, errors="coerce").dt.normalize()
        unique_dates = pd.Index(dt.dropna().unique()).sort_values()
        if len(unique_dates) < 8:
            return []

        boundaries = [0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
        splits: List[Tuple[np.ndarray, np.ndarray]] = []

        for i in range(5):
            train_cut_pos = max(0, min(int(len(unique_dates) * boundaries[i]) - 1, len(unique_dates) - 1))
            test_cut_pos = max(0, min(int(len(unique_dates) * boundaries[i + 1]) - 1, len(unique_dates) - 1))

            train_cut = unique_dates[train_cut_pos]
            test_cut = unique_dates[test_cut_pos]

            train_idx = np.where(dt <= train_cut)[0]
            test_idx = np.where((dt > train_cut) & (dt <= test_cut))[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            splits.append((train_idx, test_idx))

        return splits

    def _lgb_params(self) -> Dict[str, Any]:
        seed = self.config.random_state
        return {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "num_class": 3,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 10,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "deterministic": True,
            "verbosity": -1,
        }

    def _train_cv(
        self,
        frame: pd.DataFrame,
        feature_cols: Sequence[str],
    ) -> Dict[str, Any]:
        frame = frame.sort_values("event_date").reset_index(drop=True)
        X = frame[feature_cols].astype(float)
        y_raw = pd.to_numeric(frame[self.config.target_col], errors="coerce")
        y_mapped = y_raw.map(LABEL_MAP)

        valid_mask = y_mapped.notna()
        if not bool(valid_mask.all()):
            frame = frame.loc[valid_mask].reset_index(drop=True)
            X = X.loc[valid_mask].reset_index(drop=True)
            y_mapped = y_mapped.loc[valid_mask].reset_index(drop=True)

        y = y_mapped.to_numpy(dtype=int)

        splits = self._walk_forward_splits(frame["event_date"])
        if not splits:
            return {
                "fold_metrics": [],
                "mean_metrics": {
                    "auc_buy": float("nan"),
                    "auc_sell": float("nan"),
                    "mean_auc": float("nan"),
                    "precision_buy": float("nan"),
                    "recall_buy": float("nan"),
                    "f1_buy": float("nan"),
                },
                "std_auc": float("nan"),
                "oof_proba": np.full((len(frame), 3), np.nan),
                "oof_pred_class": np.full(len(frame), -1, dtype=int),
                "oof_mask": np.zeros(len(frame), dtype=bool),
                "best_iteration": 500,
            }

        params = self._lgb_params()
        fold_metrics: List[Dict[str, Any]] = []
        oof_proba = np.full((len(frame), 3), np.nan, dtype=float)
        oof_pred_class = np.full(len(frame), -1, dtype=int)
        best_iters: List[int] = []

        for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
            y_train = y[train_idx]
            y_test = y[test_idx]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                self.logger.info("Skipping fold %d due to insufficient class diversity", fold_no)
                continue

            counts = Counter(y_train.tolist())
            n_samples = len(y_train)
            n_classes = max(len(counts), 1)
            class_weights = {
                cls: n_samples / (n_classes * count)
                for cls, count in counts.items()
                if count > 0
            }
            sample_weights = np.asarray([class_weights.get(int(cls), 1.0) for cls in y_train], dtype=float)

            train_data = lgb.Dataset(
                X.iloc[train_idx],
                label=y_train,
                weight=sample_weights,
                feature_name=list(feature_cols),
            )
            valid_data = lgb.Dataset(X.iloc[test_idx], label=y_test, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[valid_data],
                valid_names=["valid"],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            best_iter = int(model.best_iteration or 500)
            best_iters.append(best_iter)

            pred_proba = model.predict(X.iloc[test_idx], num_iteration=best_iter)
            pred_proba = np.asarray(pred_proba, dtype=float)
            if pred_proba.ndim == 1:
                # Defensive fallback for malformed output.
                pred_proba = np.column_stack([1.0 - pred_proba, np.zeros_like(pred_proba), pred_proba])
            pred_class = pred_proba.argmax(axis=1).astype(int)

            oof_proba[test_idx] = pred_proba
            oof_pred_class[test_idx] = pred_class

            metrics = compute_classification_metrics(y_test, pred_proba, pred_class)
            metrics.update(
                {
                    "fold": fold_no,
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "best_iteration": best_iter,
                }
            )
            fold_metrics.append(metrics)

        if not fold_metrics:
            return {
                "fold_metrics": [],
                "mean_metrics": {
                    "auc_buy": float("nan"),
                    "auc_sell": float("nan"),
                    "mean_auc": float("nan"),
                    "precision_buy": float("nan"),
                    "recall_buy": float("nan"),
                    "f1_buy": float("nan"),
                },
                "std_auc": float("nan"),
                "oof_proba": oof_proba,
                "oof_pred_class": oof_pred_class,
                "oof_mask": np.isfinite(oof_proba).all(axis=1),
                "best_iteration": 500,
            }

        mean_metrics = {
            "auc_buy": float(np.nanmean([m.get("auc_buy", float("nan")) for m in fold_metrics])),
            "auc_sell": float(np.nanmean([m.get("auc_sell", float("nan")) for m in fold_metrics])),
            "mean_auc": float(np.nanmean([m.get("mean_auc", float("nan")) for m in fold_metrics])),
            "precision_buy": float(np.nanmean([m.get("precision_buy", float("nan")) for m in fold_metrics])),
            "recall_buy": float(np.nanmean([m.get("recall_buy", float("nan")) for m in fold_metrics])),
            "f1_buy": float(np.nanmean([m.get("f1_buy", float("nan")) for m in fold_metrics])),
        }

        return {
            "fold_metrics": fold_metrics,
            "mean_metrics": mean_metrics,
            "std_auc": float(np.nanstd([m.get("mean_auc", float("nan")) for m in fold_metrics])),
            "oof_proba": oof_proba,
            "oof_pred_class": oof_pred_class,
            "oof_mask": np.isfinite(oof_proba).all(axis=1),
            "best_iteration": int(np.median(best_iters) if best_iters else 500),
        }

    def _train_final_model(
        self,
        frame: pd.DataFrame,
        feature_cols: Sequence[str],
        boost_rounds: int,
    ) -> Optional[lgb.Booster]:
        X = frame[feature_cols].astype(float)
        y_raw = pd.to_numeric(frame[self.config.target_col], errors="coerce")
        y_mapped = y_raw.map(LABEL_MAP).dropna().astype(int)
        if y_mapped.nunique() < 2:
            return None

        X = X.loc[y_mapped.index]
        y = y_mapped.to_numpy(dtype=int)

        counts = Counter(y.tolist())
        n_samples = len(y)
        n_classes = max(len(counts), 1)
        class_weights = {
            cls: n_samples / (n_classes * count)
            for cls, count in counts.items()
            if count > 0
        }
        sample_weights = np.asarray([class_weights.get(int(cls), 1.0) for cls in y], dtype=float)

        train_data = lgb.Dataset(X, label=y, weight=sample_weights, feature_name=list(feature_cols))
        model = lgb.train(
            self._lgb_params(),
            train_data,
            num_boost_round=max(100, int(boost_rounds)),
        )
        return model

    def _train_single_model(
        self,
        *,
        tier: str,
        identifier: str,
        frame: pd.DataFrame,
        min_events: int,
    ) -> ModelTrainingResult:
        frame = frame.sort_values("event_date").reset_index(drop=True)
        n_events = len(frame)

        reject_reason = ""
        if n_events < min_events:
            reject_reason = f"insufficient_events_{n_events}"

        if not reject_reason:
            target_unique = pd.to_numeric(frame[self.config.target_col], errors="coerce").nunique(dropna=True)
            if target_unique < 2:
                reject_reason = "constant_target"

        feature_cols = get_feature_columns(frame)
        if not feature_cols:
            reject_reason = "no_feature_columns"

        cv = self._train_cv(frame, feature_cols) if not reject_reason else {
            "fold_metrics": [],
            "mean_metrics": {
                "auc_buy": float("nan"),
                "auc_sell": float("nan"),
                "mean_auc": float("nan"),
                "precision_buy": float("nan"),
                "recall_buy": float("nan"),
                "f1_buy": float("nan"),
            },
            "std_auc": float("nan"),
            "oof_proba": np.full((len(frame), 3), np.nan),
            "oof_pred_class": np.full(len(frame), -1, dtype=int),
            "oof_mask": np.zeros(len(frame), dtype=bool),
            "best_iteration": 500,
        }

        mean_auc = cv["mean_metrics"]["mean_auc"]
        if not reject_reason and (np.isnan(mean_auc) or mean_auc < self.config.mean_auc_reject_threshold):
            reject_reason = f"mean_auc_below_threshold_{mean_auc:.4f}"

        oof_mask = cv["oof_mask"]
        y_all_raw = pd.to_numeric(frame[self.config.target_col], errors="coerce")
        y_all = y_all_raw.map(LABEL_MAP).to_numpy(dtype=float)
        p_oof_buy = cv["oof_proba"][:, 2] if cv["oof_proba"].size else np.full(len(frame), np.nan)

        accepted = not reject_reason
        model: Optional[lgb.Booster] = None
        feature_rank: List[Dict[str, Any]] = []

        if accepted:
            model = self._train_final_model(frame, feature_cols, cv["best_iteration"])
            if model is None:
                accepted = False
                reject_reason = "final_training_failed"

        if model is not None:
            feature_rank = top_feature_importance(model, feature_cols, top_n=15)

        failure = failure_cases(
            frame.loc[oof_mask, ["ticker", "event_id", "event_date"]],
            y_all[oof_mask],
            p_oof_buy[oof_mask],
            n_cases=10,
        ) if oof_mask.any() else []

        report = build_model_report(
            tier=tier,
            identifier=identifier,
            event_frame=frame,
            fold_metrics=cv["fold_metrics"],
            mean_metrics=cv["mean_metrics"],
            std_auc=cv["std_auc"],
            calibration_summary={
                "fitted": False,
                "warning": False,
                "max_error": float("nan"),
                "mean_error": float("nan"),
                "reliability": [],
            },
            feature_importances=feature_rank,
            failures=failure,
            task="classification",
            target_col=self.config.target_col,
        )

        date_range = {
            "start": pd.to_datetime(frame["event_date"], errors="coerce").min().date().isoformat() if n_events else None,
            "end": pd.to_datetime(frame["event_date"], errors="coerce").max().date().isoformat() if n_events else None,
        }
        metadata = {
            "task": "multiclass",
            "target_col": self.config.target_col,
            "auc_buy": cv["mean_metrics"].get("auc_buy"),
            "auc_sell": cv["mean_metrics"].get("auc_sell"),
            "mean_auc": cv["mean_metrics"].get("mean_auc"),
            "precision_buy": cv["mean_metrics"].get("precision_buy"),
            "recall_buy": cv["mean_metrics"].get("recall_buy"),
            "f1_buy": cv["mean_metrics"].get("f1_buy"),
            # Legacy aliases retained for downstream compatibility.
            "spearman": cv["mean_metrics"].get("mean_auc"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "auc": cv["mean_metrics"].get("mean_auc"),
            "log_loss": float("nan"),
            "calibration_error": float("nan"),
            "n_train_events": n_events,
            "train_date_range": date_range,
            "rejected_reason": reject_reason if not accepted else "",
            "fold_metrics": cv["fold_metrics"],
        }

        save_model_bundle(
            tier=tier,
            identifier=identifier,
            model=model if accepted else None,
            calibrator=None,
            feature_list=list(feature_cols),
            metadata=metadata,
            version=date.today().isoformat(),
            models_root=self.models_root,
        )

        return ModelTrainingResult(
            tier=tier,
            identifier=identifier,
            accepted=accepted,
            n_events=n_events,
            mean_metrics=cv["mean_metrics"],
            std_auc=cv["std_auc"],
            rejected_reason=reject_reason,
            report=report,
        )

    def _run_tier_per_stock(self, dataset: pd.DataFrame) -> List[ModelTrainingResult]:
        counts = dataset.groupby("ticker").size()
        tickers = sorted(counts[counts >= self.config.min_per_stock_events].index.tolist())
        results: List[ModelTrainingResult] = []
        for i, ticker in enumerate(tickers, start=1):
            self._progress(f"per_stock {ticker}", i, len(tickers))
            frame = dataset.loc[dataset["ticker"] == ticker].copy()
            results.append(
                self._train_single_model(
                    tier="per_stock",
                    identifier=ticker,
                    frame=frame,
                    min_events=self.config.min_per_stock_events,
                )
            )
        return results

    def _run_tier_per_sector(self, dataset: pd.DataFrame) -> List[ModelTrainingResult]:
        counts = dataset.groupby("ticker").size()
        eligible_tickers = set(counts[(counts >= self.config.min_per_sector_events) & (counts < self.config.min_per_stock_events)].index.tolist())
        subset = dataset.loc[dataset["ticker"].isin(eligible_tickers)].copy()

        results: List[ModelTrainingResult] = []
        sectors = sorted(set(SECTOR_UNIVERSE) | set(dataset["sector_raw"].dropna().unique().tolist()))
        for i, sector in enumerate(sectors, start=1):
            self._progress(f"per_sector {sector}", i, len(sectors))
            frame = subset.loc[subset["sector_raw"] == sector].copy()

            # Backfill with full sector pool if 30-99 bucket is too sparse,
            # so we can maintain one model per canonical sector.
            if len(frame) < self.config.min_per_sector_events:
                frame = dataset.loc[dataset["sector_raw"] == sector].copy()

            # Keep explicit per-sector artifacts even when sparse.
            if frame.empty:
                continue

            results.append(
                self._train_single_model(
                    tier="per_sector",
                    identifier=sector,
                    frame=frame,
                    min_events=self.config.min_per_sector_events,
                )
            )
        return results

    def _run_tier_global(self, dataset: pd.DataFrame) -> List[ModelTrainingResult]:
        result = self._train_single_model(
            tier="global",
            identifier="baseline",
            frame=dataset.copy(),
            min_events=self.config.min_per_sector_events,
        )
        return [result]

    def _tier_summary(self, results: Sequence[ModelTrainingResult]) -> Dict[str, Any]:
        if not results:
            return {
                "trained": 0,
                "accepted": 0,
                "rejected": 0,
                "mean_auc_buy": float("nan"),
                "mean_auc_sell": float("nan"),
                "mean_auc": float("nan"),
                "mean_precision_buy": float("nan"),
                "mean_recall_buy": float("nan"),
                "mean_f1_buy": float("nan"),
                "mean_spearman": float("nan"),
                "mean_mae": float("nan"),
                "mean_rmse": float("nan"),
                "mean_log_loss": float("nan"),
                "mean_calibration_error": float("nan"),
            }

        accepted = [r for r in results if r.accepted]
        mean_auc_buy = float(np.nanmean([r.mean_metrics.get("auc_buy", float("nan")) for r in accepted])) if accepted else float("nan")
        mean_auc_sell = float(np.nanmean([r.mean_metrics.get("auc_sell", float("nan")) for r in accepted])) if accepted else float("nan")
        mean_auc = float(np.nanmean([r.mean_metrics.get("mean_auc", float("nan")) for r in accepted])) if accepted else float("nan")
        mean_precision_buy = float(np.nanmean([r.mean_metrics.get("precision_buy", float("nan")) for r in accepted])) if accepted else float("nan")
        mean_recall_buy = float(np.nanmean([r.mean_metrics.get("recall_buy", float("nan")) for r in accepted])) if accepted else float("nan")
        mean_f1_buy = float(np.nanmean([r.mean_metrics.get("f1_buy", float("nan")) for r in accepted])) if accepted else float("nan")
        return {
            "trained": len(results),
            "accepted": len(accepted),
            "rejected": len(results) - len(accepted),
            "mean_auc_buy": mean_auc_buy,
            "mean_auc_sell": mean_auc_sell,
            "mean_auc": mean_auc,
            "mean_precision_buy": mean_precision_buy,
            "mean_recall_buy": mean_recall_buy,
            "mean_f1_buy": mean_f1_buy,
            # Legacy aliases for older dashboards.
            "mean_spearman": mean_auc,
            "mean_mae": float("nan"),
            "mean_rmse": float("nan"),
            "mean_log_loss": float("nan"),
            "mean_calibration_error": float("nan"),
        }

    def _save_reports(
        self,
        *,
        report_date: str,
        per_stock: List[ModelTrainingResult],
        per_sector: List[ModelTrainingResult],
        global_results: List[ModelTrainingResult],
        summary: Dict[str, Any],
    ) -> None:
        out_dir = self.reports_root / report_date
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "per_stock_report.json").write_text(
            json.dumps([r.report for r in per_stock], indent=2, default=str),
            encoding="utf-8",
        )
        (out_dir / "per_sector_report.json").write_text(
            json.dumps([r.report for r in per_sector], indent=2, default=str),
            encoding="utf-8",
        )
        (out_dir / "global_report.json").write_text(
            json.dumps([r.report for r in global_results], indent=2, default=str),
            encoding="utf-8",
        )
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )

    def run(
        self,
        *,
        tier: str = "all",
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        t0 = time.time()
        dataset = self.build_dataset(force_rebuild=force_rebuild)

        per_stock_results: List[ModelTrainingResult] = []
        per_sector_results: List[ModelTrainingResult] = []
        global_results: List[ModelTrainingResult] = []

        tier = tier.lower().strip()
        if tier in {"all", "per_stock"}:
            per_stock_results = self._run_tier_per_stock(dataset)
        if tier in {"all", "per_sector"}:
            per_sector_results = self._run_tier_per_sector(dataset)
        if tier in {"all", "global"}:
            global_results = self._run_tier_global(dataset)

        event_counts = dataset.groupby("ticker").size().astype(int).to_dict()
        sector_map = (
            dataset[["ticker", "sector_raw"]]
            .drop_duplicates(subset=["ticker"]) 
            .set_index("ticker")["sector_raw"]
            .astype(str)
            .to_dict()
        )

        all_accepted = [r for r in (per_stock_results + per_sector_results + global_results) if r.accepted]
        auc_pass = [
            r
            for r in all_accepted
            if float(r.mean_metrics.get("mean_auc", float("nan"))) >= self.config.mean_auc_reject_threshold
        ]

        report_date = date.today().isoformat()
        summary = {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "runtime_sec": round(time.time() - t0, 2),
            "tier": tier,
            "dataset": {
                "n_events": int(len(dataset)),
                "n_tickers": int(dataset["ticker"].nunique()),
            },
            "per_stock": self._tier_summary(per_stock_results),
            "per_sector": self._tier_summary(per_sector_results),
            "global": self._tier_summary(global_results),
            "mean_auc_pass_rate": float(len(auc_pass) / len(all_accepted)) if all_accepted else float("nan"),
            "spearman_pass_rate": float(len(auc_pass) / len(all_accepted)) if all_accepted else float("nan"),
            "calibration_pass_rate": float("nan"),
            "event_counts_by_ticker": event_counts,
            "ticker_sector_map": sector_map,
        }

        if per_stock_results:
            accepted = [r for r in per_stock_results if r.accepted]
            accepted_sorted = sorted(
                accepted,
                key=lambda x: x.mean_metrics.get("mean_auc", float("-inf")),
                reverse=True,
            )
            summary["per_stock_top5_auc"] = [
                {"ticker": r.identifier, "mean_auc": r.mean_metrics.get("mean_auc")} for r in accepted_sorted[:5]
            ]
            summary["per_stock_bottom5_auc"] = [
                {"ticker": r.identifier, "mean_auc": r.mean_metrics.get("mean_auc")} for r in accepted_sorted[-5:]
            ]
            # Legacy aliases.
            summary["per_stock_top5_spearman"] = summary["per_stock_top5_auc"]
            summary["per_stock_bottom5_spearman"] = summary["per_stock_bottom5_auc"]

        self._save_reports(
            report_date=report_date,
            per_stock=per_stock_results,
            per_sector=per_sector_results,
            global_results=global_results,
            summary=summary,
        )

        # Keep event index synced for tier_resolver.
        self._event_index_file.write_text(
            json.dumps(
                {
                    "generated_at": summary["generated_at"],
                    "event_counts_by_ticker": event_counts,
                    "ticker_sector_map": sector_map,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        self.logger.info("Training run complete in %.1fs", time.time() - t0)
        return summary

from __future__ import annotations

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
    compute_regression_metrics,
    failure_cases,
    top_feature_importance,
)
from .feature_builder import build_feature_matrix, build_labeled_rows_from_ohlcv_cache, get_feature_columns
from .model_store import (
    get_cache_root,
    get_models_root,
    get_reports_root,
    save_model_bundle,
)


@dataclass
class TrainingConfig:
    random_state: int = 42
    min_per_stock_events: int = 100
    min_global_events: int = 30
    min_spearman_accept: float = 0.30
    target_col: str = "target_score"
    task: str = "regression"


@dataclass
class ModelTrainingResult:
    tier: str
    identifier: str
    accepted: bool
    n_events: int
    mean_metrics: Dict[str, float]
    std_spearman: float
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
        self.logger.info("Building curated v14 feature rows from cached OHLCV...")
        raw_rows = build_labeled_rows_from_ohlcv_cache(logger=self.logger)
        self.logger.info("Generated %d raw curated rows", len(raw_rows))

        features = build_feature_matrix(raw_rows, logger=self.logger)
        if features.frame.empty:
            raise RuntimeError("No feature rows available for training")

        dataset = features.frame.sort_values(["ticker", "event_date"]).reset_index(drop=True)

        dataset[self.config.target_col] = pd.to_numeric(dataset[self.config.target_col], errors="coerce")
        dataset = dataset.loc[dataset[self.config.target_col].notna()].copy()
        dataset[self.config.target_col] = dataset[self.config.target_col].clip(0.0, 100.0)
        if dataset.empty:
            raise RuntimeError("No valid curated rows available after target filtering")

        dataset.to_pickle(self._cache_file)

        event_counts = dataset.groupby("ticker").size().astype(int).to_dict()
        if "sector_raw" in dataset.columns:
            sector_map = (
                dataset[["ticker", "sector_raw"]]
                .drop_duplicates(subset=["ticker"])
                .set_index("ticker")["sector_raw"]
                .astype(str)
                .to_dict()
            )
        else:
            sector_map = {str(tk): "unknown" for tk in event_counts.keys()}
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
            "objective": "regression",
            "metric": "l2",
            "max_depth": 5,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "min_data_in_leaf": 30,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "deterministic": True,
            "verbosity": -1,
        }

    def _empty_cv_payload(self, n_rows: int) -> Dict[str, Any]:
        return {
            "fold_metrics": [],
            "mean_metrics": {
                "spearman": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
            },
            "std_spearman": float("nan"),
            "oof_pred": np.full(n_rows, np.nan, dtype=float),
            "oof_mask": np.zeros(n_rows, dtype=bool),
            "best_iteration": 300,
        }

    def _train_cv(
        self,
        frame: pd.DataFrame,
        feature_cols: Sequence[str],
    ) -> Dict[str, Any]:
        frame = frame.sort_values("event_date").reset_index(drop=True)
        X = frame[feature_cols].astype(float)
        y = pd.to_numeric(frame[self.config.target_col], errors="coerce")

        valid_mask = y.notna()
        if not bool(valid_mask.all()):
            frame = frame.loc[valid_mask].reset_index(drop=True)
            X = X.loc[valid_mask].reset_index(drop=True)
            y = y.loc[valid_mask].reset_index(drop=True)

        y_arr = y.to_numpy(dtype=float)
        splits = self._walk_forward_splits(frame["event_date"])
        if not splits:
            return self._empty_cv_payload(len(frame))

        params = self._lgb_params()
        fold_metrics: List[Dict[str, Any]] = []
        oof_pred = np.full(len(frame), np.nan, dtype=float)
        best_iters: List[int] = []

        for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
            y_train = y_arr[train_idx]
            y_test = y_arr[test_idx]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                self.logger.info("Skipping fold %d due to insufficient target diversity", fold_no)
                continue

            train_data = lgb.Dataset(
                X.iloc[train_idx],
                label=y_train,
                feature_name=list(feature_cols),
            )
            valid_data = lgb.Dataset(X.iloc[test_idx], label=y_test, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=300,
                valid_sets=[valid_data],
                valid_names=["valid"],
                callbacks=[lgb.early_stopping(40, verbose=False)],
            )
            best_iter = int(model.best_iteration or 300)
            best_iters.append(best_iter)

            pred = np.asarray(model.predict(X.iloc[test_idx], num_iteration=best_iter), dtype=float).ravel()
            oof_pred[test_idx] = pred

            metrics = compute_regression_metrics(y_test, pred)
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
            payload = self._empty_cv_payload(len(frame))
            payload["oof_pred"] = oof_pred
            payload["oof_mask"] = np.isfinite(oof_pred)
            return payload

        spearman_vals = [m.get("spearman", float("nan")) for m in fold_metrics]
        mae_vals = [m.get("mae", float("nan")) for m in fold_metrics]
        rmse_vals = [m.get("rmse", float("nan")) for m in fold_metrics]

        return {
            "fold_metrics": fold_metrics,
            "mean_metrics": {
                "spearman": float(np.nanmean(spearman_vals)),
                "mae": float(np.nanmean(mae_vals)),
                "rmse": float(np.nanmean(rmse_vals)),
            },
            "std_spearman": float(np.nanstd(spearman_vals)),
            "oof_pred": oof_pred,
            "oof_mask": np.isfinite(oof_pred),
            "best_iteration": int(np.median(best_iters) if best_iters else 300),
        }

    def _train_final_model(
        self,
        frame: pd.DataFrame,
        feature_cols: Sequence[str],
        boost_rounds: int,
    ) -> Optional[lgb.Booster]:
        X = frame[feature_cols].astype(float)
        y = pd.to_numeric(frame[self.config.target_col], errors="coerce").dropna().astype(float)
        if y.nunique() < 2:
            return None

        X = X.loc[y.index]
        train_data = lgb.Dataset(X, label=y.to_numpy(dtype=float), feature_name=list(feature_cols))
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
        allow_rejected_fallback: bool = False,
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

        cv = self._train_cv(frame, feature_cols) if not reject_reason else self._empty_cv_payload(len(frame))

        spearman = cv["mean_metrics"].get("spearman", float("nan"))
        if not reject_reason and (np.isnan(spearman) or spearman < self.config.min_spearman_accept):
            reject_reason = f"spearman_below_threshold_{spearman:.4f}"

        oof_mask = cv["oof_mask"]
        y_all = pd.to_numeric(frame[self.config.target_col], errors="coerce").to_numpy(dtype=float)
        p_oof = cv["oof_pred"]

        accepted = not reject_reason
        fallback_eligible = bool(
            allow_rejected_fallback
            and reject_reason.startswith("spearman_below_threshold_")
        )
        model: Optional[lgb.Booster] = None
        feature_rank: List[Dict[str, Any]] = []

        if accepted or fallback_eligible:
            model = self._train_final_model(frame, feature_cols, cv["best_iteration"])
            if model is None:
                accepted = False
                fallback_eligible = False
                reject_reason = "final_training_failed"

        if model is not None:
            feature_rank = top_feature_importance(model, feature_cols, top_n=15)

        failure = (
            failure_cases(
                frame.loc[oof_mask, ["ticker", "event_id", "event_date"]],
                y_all[oof_mask],
                p_oof[oof_mask],
                n_cases=10,
            )
            if oof_mask.any()
            else []
        )

        report = build_model_report(
            tier=tier,
            identifier=identifier,
            event_frame=frame,
            fold_metrics=cv["fold_metrics"],
            mean_metrics=cv["mean_metrics"],
            std_auc=cv["std_spearman"],
            calibration_summary={
                "fitted": False,
                "warning": False,
                "max_error": float("nan"),
                "mean_error": float("nan"),
                "reliability": [],
            },
            feature_importances=feature_rank,
            failures=failure,
            task="regression",
            target_col=self.config.target_col,
        )

        date_range = {
            "start": pd.to_datetime(frame["event_date"], errors="coerce").min().date().isoformat() if n_events else None,
            "end": pd.to_datetime(frame["event_date"], errors="coerce").max().date().isoformat() if n_events else None,
        }
        metadata = {
            "task": self.config.task,
            "label_space": "target_score_0_100",
            "objective": "regression",
            "target_col": self.config.target_col,
            "target_min": 0.0,
            "target_max": 100.0,
            "spearman": cv["mean_metrics"].get("spearman"),
            "mae": cv["mean_metrics"].get("mae"),
            "rmse": cv["mean_metrics"].get("rmse"),
            # Legacy aliases retained for downstream compatibility.
            "auc": cv["mean_metrics"].get("spearman"),
            "mean_auc": cv["mean_metrics"].get("spearman"),
            "log_loss": float("nan"),
            "calibration_error": float("nan"),
            "n_train_events": n_events,
            "train_date_range": date_range,
            "rejected_reason": reject_reason if not accepted else "",
            "fallback_eligible": bool(fallback_eligible and model is not None),
            "fold_metrics": cv["fold_metrics"],
        }

        save_model_bundle(
            tier=tier,
            identifier=identifier,
            model=model if (accepted or fallback_eligible) else None,
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
            std_spearman=cv["std_spearman"],
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

    def _run_tier_global(self, dataset: pd.DataFrame) -> List[ModelTrainingResult]:
        result = self._train_single_model(
            tier="global",
            identifier="baseline",
            frame=dataset.copy(),
            min_events=self.config.min_global_events,
            allow_rejected_fallback=True,
        )
        return [result]

    def _tier_summary(self, results: Sequence[ModelTrainingResult]) -> Dict[str, Any]:
        if not results:
            return {
                "trained": 0,
                "accepted": 0,
                "rejected": 0,
                "mean_spearman": float("nan"),
                "mean_mae": float("nan"),
                "mean_rmse": float("nan"),
                # Legacy aliases
                "mean_auc": float("nan"),
                "mean_auc_buy": float("nan"),
                "mean_auc_sell": float("nan"),
                "mean_precision_buy": float("nan"),
                "mean_recall_buy": float("nan"),
                "mean_f1_buy": float("nan"),
                "mean_log_loss": float("nan"),
                "mean_calibration_error": float("nan"),
            }

        accepted = [r for r in results if r.accepted]
        mean_spearman = (
            float(np.nanmean([r.mean_metrics.get("spearman", float("nan")) for r in accepted]))
            if accepted
            else float("nan")
        )
        mean_mae = (
            float(np.nanmean([r.mean_metrics.get("mae", float("nan")) for r in accepted]))
            if accepted
            else float("nan")
        )
        mean_rmse = (
            float(np.nanmean([r.mean_metrics.get("rmse", float("nan")) for r in accepted]))
            if accepted
            else float("nan")
        )

        return {
            "trained": len(results),
            "accepted": len(accepted),
            "rejected": len(results) - len(accepted),
            "mean_spearman": mean_spearman,
            "mean_mae": mean_mae,
            "mean_rmse": mean_rmse,
            # Legacy aliases for older dashboards.
            "mean_auc": mean_spearman,
            "mean_auc_buy": float("nan"),
            "mean_auc_sell": float("nan"),
            "mean_precision_buy": float("nan"),
            "mean_recall_buy": float("nan"),
            "mean_f1_buy": float("nan"),
            "mean_log_loss": float("nan"),
            "mean_calibration_error": float("nan"),
        }

    def _save_reports(
        self,
        *,
        report_date: str,
        per_stock: List[ModelTrainingResult],
        global_results: List[ModelTrainingResult],
        summary: Dict[str, Any],
    ) -> None:
        out_dir = self.reports_root / report_date
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "per_stock_report.json").write_text(
            json.dumps([r.report for r in per_stock], indent=2, default=str),
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
        global_results: List[ModelTrainingResult] = []

        tier = tier.lower().strip()
        if tier in {"all", "per_stock"}:
            per_stock_results = self._run_tier_per_stock(dataset)
        if tier in {"all", "global"}:
            global_results = self._run_tier_global(dataset)

        event_counts = dataset.groupby("ticker").size().astype(int).to_dict()
        if "sector_raw" in dataset.columns:
            sector_map = (
                dataset[["ticker", "sector_raw"]]
                .drop_duplicates(subset=["ticker"])
                .set_index("ticker")["sector_raw"]
                .astype(str)
                .to_dict()
            )
        else:
            sector_map = {str(tk): "unknown" for tk in event_counts.keys()}

        all_accepted = [r for r in (per_stock_results + global_results) if r.accepted]
        spearman_pass = [
            r
            for r in all_accepted
            if float(r.mean_metrics.get("spearman", float("nan"))) >= self.config.min_spearman_accept
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
            "global": self._tier_summary(global_results),
            "spearman_pass_rate": float(len(spearman_pass) / len(all_accepted)) if all_accepted else float("nan"),
            # Legacy alias
            "mean_auc_pass_rate": float(len(spearman_pass) / len(all_accepted)) if all_accepted else float("nan"),
            "event_counts_by_ticker": event_counts,
            "ticker_sector_map": sector_map,
        }

        if per_stock_results:
            accepted = [r for r in per_stock_results if r.accepted]
            accepted_sorted = sorted(
                accepted,
                key=lambda x: x.mean_metrics.get("spearman", float("-inf")),
                reverse=True,
            )
            summary["per_stock_top5_spearman"] = [
                {"ticker": r.identifier, "spearman": r.mean_metrics.get("spearman")}
                for r in accepted_sorted[:5]
            ]
            summary["per_stock_bottom5_spearman"] = [
                {"ticker": r.identifier, "spearman": r.mean_metrics.get("spearman")}
                for r in accepted_sorted[-5:]
            ]
            # Legacy aliases.
            summary["per_stock_top5_auc"] = summary["per_stock_top5_spearman"]
            summary["per_stock_bottom5_auc"] = summary["per_stock_bottom5_spearman"]

        self._save_reports(
            report_date=report_date,
            per_stock=per_stock_results,
            global_results=global_results,
            summary=summary,
        )

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

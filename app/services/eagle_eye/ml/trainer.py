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

from .calibrator import apply_calibrator, fit_isotonic_calibrator
from .evaluator import (
    build_model_report,
    compute_binary_metrics,
    failure_cases,
    top_feature_importance,
)
from .feature_builder import (
    build_events_from_ohlcv_cache,
    build_feature_matrix,
    get_feature_columns,
)
from .labelers import build_labels
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


@dataclass
class TrainingConfig:
    random_state: int = 42
    min_per_stock_events: int = 100
    min_per_sector_events: int = 30
    auc_reject_threshold: float = 0.55
    target_col: str = "y_tp1_20d"


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
            self.logger.info("Loading cached event features: %s", self._cache_file)
            return pd.read_pickle(self._cache_file)

        t0 = time.time()
        self.logger.info("Building forensic event feature rows from cached OHLCV...")
        raw_rows = build_events_from_ohlcv_cache(logger=self.logger)
        self.logger.info("Generated %d raw event rows", len(raw_rows))

        features = build_feature_matrix(raw_rows, logger=self.logger)
        if features.frame.empty:
            raise RuntimeError("No feature rows available for training")

        labels = build_labels(features.frame)
        dataset = pd.concat([features.frame.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
        dataset = dataset.sort_values(["ticker", "event_date"]).reset_index(drop=True)

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
            "objective": "binary",
            "metric": "binary_logloss",
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
        y = frame[self.config.target_col].astype(int).to_numpy()

        splits = self._walk_forward_splits(frame["event_date"])
        if not splits:
            return {
                "fold_metrics": [],
                "mean_metrics": {
                    "auc": float("nan"),
                    "log_loss": float("nan"),
                    "calibration_max_error": float("nan"),
                    "calibration_mean_error": float("nan"),
                },
                "std_auc": float("nan"),
                "oof_pred": np.full(len(frame), np.nan),
                "oof_mask": np.zeros(len(frame), dtype=bool),
                "best_iteration": 200,
            }

        params = self._lgb_params()
        fold_metrics: List[Dict[str, Any]] = []
        oof = np.full(len(frame), np.nan, dtype=float)
        best_iters: List[int] = []

        for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
            y_train = y[train_idx]
            y_test = y[test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                self.logger.info("Skipping fold %d due to single-class split", fold_no)
                continue

            train_data = lgb.Dataset(X.iloc[train_idx], label=y_train, feature_name=list(feature_cols))
            valid_data = lgb.Dataset(X.iloc[test_idx], label=y_test, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[valid_data],
                valid_names=["valid"],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
            best_iter = int(model.best_iteration or 200)
            best_iters.append(best_iter)

            pred = model.predict(X.iloc[test_idx], num_iteration=best_iter)
            pred = np.clip(pred, 0.0, 1.0)
            oof[test_idx] = pred

            metrics = compute_binary_metrics(y_test, pred)
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
                    "auc": float("nan"),
                    "log_loss": float("nan"),
                    "calibration_max_error": float("nan"),
                    "calibration_mean_error": float("nan"),
                },
                "std_auc": float("nan"),
                "oof_pred": oof,
                "oof_mask": np.isfinite(oof),
                "best_iteration": 200,
            }

        mean_metrics = {
            "auc": float(np.nanmean([m["auc"] for m in fold_metrics])),
            "log_loss": float(np.nanmean([m["log_loss"] for m in fold_metrics])),
            "calibration_max_error": float(np.nanmean([m["calibration_max_error"] for m in fold_metrics])),
            "calibration_mean_error": float(np.nanmean([m["calibration_mean_error"] for m in fold_metrics])),
        }

        return {
            "fold_metrics": fold_metrics,
            "mean_metrics": mean_metrics,
            "std_auc": float(np.nanstd([m["auc"] for m in fold_metrics])),
            "oof_pred": oof,
            "oof_mask": np.isfinite(oof),
            "best_iteration": int(np.median(best_iters) if best_iters else 200),
        }

    def _train_final_model(
        self,
        frame: pd.DataFrame,
        feature_cols: Sequence[str],
        boost_rounds: int,
    ) -> Optional[lgb.Booster]:
        X = frame[feature_cols].astype(float)
        y = frame[self.config.target_col].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            return None

        train_data = lgb.Dataset(X, label=y, feature_name=list(feature_cols))
        model = lgb.train(
            self._lgb_params(),
            train_data,
            num_boost_round=max(50, int(boost_rounds)),
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

        if not reject_reason and frame[self.config.target_col].nunique() < 2:
            reject_reason = "single_class_target"

        feature_cols = get_feature_columns(frame)
        if not feature_cols:
            reject_reason = "no_feature_columns"

        cv = self._train_cv(frame, feature_cols) if not reject_reason else {
            "fold_metrics": [],
            "mean_metrics": {
                "auc": float("nan"),
                "log_loss": float("nan"),
                "calibration_max_error": float("nan"),
                "calibration_mean_error": float("nan"),
            },
            "std_auc": float("nan"),
            "oof_pred": np.full(len(frame), np.nan),
            "oof_mask": np.zeros(len(frame), dtype=bool),
            "best_iteration": 200,
        }

        mean_auc = cv["mean_metrics"]["auc"]
        if not reject_reason and (np.isnan(mean_auc) or mean_auc < self.config.auc_reject_threshold):
            reject_reason = f"auc_below_threshold_{mean_auc:.4f}"

        oof_mask = cv["oof_mask"]
        y_all = frame[self.config.target_col].to_numpy()
        p_oof = cv["oof_pred"]

        cal_result = fit_isotonic_calibrator(y_all[oof_mask], p_oof[oof_mask]) if oof_mask.any() else fit_isotonic_calibrator(np.array([]), np.array([]))

        if cal_result.summary.get("warning"):
            self.logger.warning(
                "Calibration warning for %s/%s: max_error=%.4f",
                tier,
                identifier,
                cal_result.summary.get("max_error", float("nan")),
            )

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

        calibrated_oof = apply_calibrator(cal_result.calibrator, p_oof[oof_mask]) if oof_mask.any() else np.array([])
        failure = failure_cases(
            frame.loc[oof_mask, ["ticker", "event_id", "event_date", "y_outcome_category"]],
            y_all[oof_mask],
            calibrated_oof,
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
                **cal_result.summary,
                "reliability": cal_result.reliability,
            },
            feature_importances=feature_rank,
            failures=failure,
        )

        date_range = {
            "start": pd.to_datetime(frame["event_date"], errors="coerce").min().date().isoformat() if n_events else None,
            "end": pd.to_datetime(frame["event_date"], errors="coerce").max().date().isoformat() if n_events else None,
        }
        metadata = {
            "auc": cv["mean_metrics"]["auc"],
            "log_loss": cv["mean_metrics"]["log_loss"],
            "calibration_error": cv["mean_metrics"]["calibration_max_error"],
            "n_train_events": n_events,
            "train_date_range": date_range,
            "rejected_reason": reject_reason if not accepted else "",
            "fold_metrics": cv["fold_metrics"],
        }

        save_model_bundle(
            tier=tier,
            identifier=identifier,
            model=model if accepted else None,
            calibrator=cal_result.calibrator if accepted else None,
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
                "mean_auc": float("nan"),
                "mean_log_loss": float("nan"),
                "mean_calibration_error": float("nan"),
            }

        accepted = [r for r in results if r.accepted]
        return {
            "trained": len(results),
            "accepted": len(accepted),
            "rejected": len(results) - len(accepted),
            "mean_auc": float(np.nanmean([r.mean_metrics["auc"] for r in accepted])) if accepted else float("nan"),
            "mean_log_loss": float(np.nanmean([r.mean_metrics["log_loss"] for r in accepted])) if accepted else float("nan"),
            "mean_calibration_error": (
                float(np.nanmean([float(r.report.get("calibration", {}).get("max_error", float("nan"))) for r in accepted]))
                if accepted
                else float("nan")
            ),
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
        cal_pass = [
            r
            for r in all_accepted
            if float(r.report.get("calibration", {}).get("max_error", 999.0)) <= 0.15
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
            "calibration_pass_rate": float(len(cal_pass) / len(all_accepted)) if all_accepted else float("nan"),
            "event_counts_by_ticker": event_counts,
            "ticker_sector_map": sector_map,
        }

        if per_stock_results:
            accepted = [r for r in per_stock_results if r.accepted]
            accepted_sorted = sorted(accepted, key=lambda x: x.mean_metrics["auc"], reverse=True)
            summary["per_stock_top5_auc"] = [
                {"ticker": r.identifier, "auc": r.mean_metrics["auc"]} for r in accepted_sorted[:5]
            ]
            summary["per_stock_bottom5_auc"] = [
                {"ticker": r.identifier, "auc": r.mean_metrics["auc"]} for r in accepted_sorted[-5:]
            ]

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

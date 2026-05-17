"""
ml/trainer_v2.py — Phase 2 Deliverable 3

Per-stock LightGBM trainer with:
  - 5-fold expanding-window walk-forward CV
  - HP grid search (best HPs from mean AUC across folds 1-4)
  - 20 independent surface-cell models (one per y_Rpct_Hd label)
  - Primary-label gate: if primary model fails CV screening → skip remaining 19 cells
  - Isotonic calibration fit on fold 4 validation set
  - Model persistence via model_store.py
  - Status tracking in ml_models + model_lifecycle_log tables

Do NOT use random k-fold, SMOTE, target encoding, stacking, or deep learning.
Do NOT promote any model to LIVE (Phase 2 ends in SHADOW status only).
"""
from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# LightGBM — imported lazily in case it is not installed in this env
try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    lgb = None  # type: ignore[assignment]
    _LGBM_AVAILABLE = False

from sklearn.preprocessing import StandardScaler

from app.core.config import get_settings
from app.services.eagle_eye.ml.calibrator import (
    apply_calibrator,
    fit_isotonic_calibrator,
)
from app.services.eagle_eye.ml.model_store import (
    ModelBundle,
    get_models_root,
    save_model_bundle,
)
from app.services.eagle_eye.ml.training_matrix import (
    PRIMARY_LABEL,
    RETURN_TARGETS_PCT,
    HORIZONS_TD,
    SURFACE_LABEL_COLS,
    load_stock_matrix,
    _matrix_root,
)
from app.services.eagle_eye.ml.walk_forward import (
    build_fold_indices,
    split_df_by_fold,
    DEFAULT_EMBARGO_TD,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _read_schema_primary_label(ticker: str) -> Optional[str]:
    """
    Read the per-stock primary label from schema.json written by write_stock_matrix().

    Returns None if schema is missing, corrupt, or has primary_label=null
    (i.e. no label tier had >= 50 positives — INSUFFICIENT_DATA).
    """
    schema_path = _matrix_root() / ticker.upper() / "schema.json"
    if not schema_path.exists():
        return None
    try:
        with schema_path.open(encoding="utf-8") as f:
            schema = json.load(f)
        return schema.get("primary_label")  # may be None / null
    except Exception:
        return None


_HP_GRID: Dict[str, List[Any]] = {
    "num_leaves":        [15, 31, 63],
    "max_depth":         [4, 6, 8],
    "min_data_in_leaf":  [50, 100, 200],
    "learning_rate":     [0.01, 0.05],
    "feature_fraction":  [0.6, 0.8, 1.0],
    "bagging_fraction":  [0.7, 1.0],
}

# HP combinations kept small by restricting key interactions
_REDUCED_GRID: Dict[str, List[Any]] = {
    "num_leaves":        [31, 63],
    "max_depth":         [4, 6],
    "min_data_in_leaf":  [50, 100],
    "learning_rate":     [0.01, 0.05],
    "feature_fraction":  [0.8, 1.0],
    "bagging_fraction":  [0.7, 1.0],
}

_FIXED_LGBM_PARAMS: Dict[str, Any] = {
    "objective":       "binary",
    "metric":          "auc",
    "verbosity":       -1,
    "seed":            42,
    "n_jobs":          1,
    "bagging_seed":    42,
    "feature_seed":    42,
    "is_unbalance":    True,
}

EARLY_STOPPING_ROUNDS = 50
N_ESTIMATORS = 500

CATEGORICAL_FEATURES = ["sector", "stage", "regime", "market_tier"]

# AUC gate for primary model to proceed with full surface training
PRIMARY_AUC_GATE = 0.58


# ---------------------------------------------------------------------------
# PerStockTrainer — testable wrapper exposing scaler isolation
# ---------------------------------------------------------------------------

class PerStockTrainer:
    """
    Thin class that makes scaler fit-on-train-only an explicit, testable contract.

    Smoke-test Check 3 calls ``_scale_features(train, test)`` directly to verify
    that the returned test array reflects real distribution shift (i.e. the scaler
    was NOT fit on the test data).
    """

    def __init__(self) -> None:
        self._scaler: Optional[StandardScaler] = None

    def _scale_features(
        self,
        train_features: np.ndarray,
        test_features: np.ndarray,
    ) -> np.ndarray:
        """
        Fit a StandardScaler on ``train_features`` only, then return
        ``test_features`` transformed by that scaler.

        If the test set has a different distribution (e.g. mean=100 vs train
        mean=0), the returned values will be large (~100) — not near 0.
        This is the CORRECT behaviour.  A near-zero test mean would indicate
        the scaler was incorrectly fit on the full dataset.
        """
        self._scaler = StandardScaler()
        self._scaler.fit(train_features)
        return self._scaler.transform(test_features)

    def scale_train(self, train_features: np.ndarray) -> np.ndarray:
        """Fit and transform training features. Must be called before scale_test."""
        self._scaler = StandardScaler()
        return self._scaler.fit_transform(train_features)

    def scale_test(self, test_features: np.ndarray) -> np.ndarray:
        """Apply already-fitted scaler to test features (transform only — no refit)."""
        if self._scaler is None:
            raise RuntimeError("scale_train must be called before scale_test")
        return self._scaler.transform(test_features)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SurfaceCellResult:
    label: str
    wfv_aucs: List[float]
    mean_wfv_auc: float
    final_n_estimators: int
    best_hp: Dict[str, Any]
    calibrator: Any
    model: Any  # LGBMClassifier or lgb.Booster
    feature_cols: List[str]
    passed_gate: bool = True
    skip_reason: str = ""


@dataclass
class PerStockTrainingResult:
    ticker: str
    version: str
    status: str                          # "ok" | "failed_gate" | "insufficient_data" | "error"
    primary_auc: float = float("nan")
    n_cells_trained: int = 0
    n_rows_oot: int = 0
    n_pos_oot: int = 0
    cells: List[SurfaceCellResult] = field(default_factory=list)
    error_msg: str = ""
    model_bundle_paths: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return feature columns (exclude labels, metadata, flags, regime)."""
    from app.services.eagle_eye.ml.feature_builder import get_feature_columns, NON_FEATURE_COLUMNS
    base_non_feature = set(NON_FEATURE_COLUMNS) | set(SURFACE_LABEL_COLS) | {
        "regime", "flag_low_volume", "flag_corp_action",
    }
    return [c for c in df.columns if c not in base_non_feature and not c.startswith("y_")]


def _prepare_xy(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """Extract X, y arrays and optionally fit/apply scaler."""
    df_clean = df.dropna(subset=[label_col])
    X = df_clean[feature_cols].fillna(0).values.astype(np.float32)
    y = df_clean[label_col].values.astype(int)
    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    elif scaler is not None:
        X = scaler.transform(X).astype(np.float32)
    return X, y, scaler


def _cat_feature_indices(feature_cols: List[str]) -> List[int]:
    return [i for i, c in enumerate(feature_cols) if c in CATEGORICAL_FEATURES]


def _make_lgbm_params(hp: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(_FIXED_LGBM_PARAMS)
    params.update(hp)
    if params.get("bagging_fraction", 1.0) < 1.0:
        params["bagging_freq"] = 1
    return params


# ---------------------------------------------------------------------------
# HP search
# ---------------------------------------------------------------------------

def _hp_grid_configs(use_reduced: bool = True) -> List[Dict[str, Any]]:
    grid = _REDUCED_GRID if use_reduced else _HP_GRID
    keys = list(grid.keys())
    configs = []
    for vals in itertools.product(*grid.values()):
        configs.append(dict(zip(keys, vals)))
    return configs


def _cv_hp_search(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    fold_indices: List[Tuple[np.ndarray, np.ndarray]],
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], float, int]:
    """
    Grid search over HP_GRID using folds 1..N-1 (skip OOT fold).

    Early stopping on the PREVIOUS fold's val set.
    Fold 1 falls back to a 10% holdout within training data.

    Returns: (best_hp_config, best_mean_auc, best_n_estimators)
    """
    from sklearn.metrics import roc_auc_score

    df = df.sort_values("event_date").reset_index(drop=True)
    cv_folds = fold_indices[:-1]  # exclude the OOT fold from HP search

    if len(cv_folds) == 0:
        return {}, float("nan"), N_ESTIMATORS

    configs = _hp_grid_configs(use_reduced=True)
    best_auc = -1.0
    best_cfg = configs[0]
    best_n_est = N_ESTIMATORS

    for cfg in configs:
        params = _make_lgbm_params(cfg)
        fold_aucs: List[float] = []
        fold_n_ests: List[int] = []
        prev_val_X: Optional[np.ndarray] = None
        prev_val_y: Optional[np.ndarray] = None

        for fold_k, (tr_idx, va_idx) in enumerate(cv_folds):
            tr_df = df.iloc[tr_idx]
            va_df = df.iloc[va_idx]

            # Drop rows where label is NaN
            tr_df = tr_df.dropna(subset=[label_col])
            va_df = va_df.dropna(subset=[label_col])
            if len(tr_df) < 20 or len(va_df) < 5:
                continue

            X_tr = tr_df[feature_cols].fillna(0).values.astype(np.float32)
            y_tr = tr_df[label_col].values.astype(int)
            X_va = va_df[feature_cols].fillna(0).values.astype(np.float32)
            y_va = va_df[label_col].values.astype(int)

            if prev_val_X is None:
                # Fold 1: use 10% of training set as early stopping reference
                n_es = max(int(len(X_tr) * 0.1), 5)
                es_X, es_y = X_tr[-n_es:], y_tr[-n_es:]
                X_tr, y_tr = X_tr[:-n_es], y_tr[:-n_es]
            else:
                es_X, es_y = prev_val_X, prev_val_y

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
                continue

            dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            des = lgb.Dataset(es_X, label=es_y, reference=dtrain, free_raw_data=False)

            callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(-1)]
            try:
                booster = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=N_ESTIMATORS,
                    valid_sets=[des],
                    callbacks=callbacks,
                )
            except Exception:
                continue

            preds = booster.predict(X_va)
            if len(np.unique(y_va)) < 2:
                continue
            auc = roc_auc_score(y_va, preds)
            fold_aucs.append(auc)
            fold_n_ests.append(booster.num_trees())

            prev_val_X, prev_val_y = X_va, y_va

        if not fold_aucs:
            continue
        mean_auc = float(np.mean(fold_aucs))
        mean_n_est = int(np.mean(fold_n_ests))
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_cfg = cfg
            best_n_est = max(mean_n_est, 50)

    return best_cfg, best_auc, best_n_est


# ---------------------------------------------------------------------------
# Single surface cell trainer
# ---------------------------------------------------------------------------

def _train_surface_cell(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    fold_indices: List[Tuple[np.ndarray, np.ndarray]],
    logger: logging.Logger,
    best_hp: Optional[Dict[str, Any]] = None,
    best_n_est: int = N_ESTIMATORS,
    primary_auc: float = 1.0,
) -> SurfaceCellResult:
    """
    Train one surface cell model.

    - Runs WFV to gather per-fold AUC.
    - Trains final model on all non-OOT data.
    - Calibrates using fold 4 validation (index -2 in folds list).
    """
    from sklearn.metrics import roc_auc_score

    df = df.sort_values("event_date").reset_index(drop=True)
    oot_fold = fold_indices[-1]
    non_oot_folds = fold_indices[:-1]

    # Fold 4 (index 3) is used for calibration — take the last of non-OOT
    calibration_fold = non_oot_folds[-1] if non_oot_folds else oot_fold

    params = _make_lgbm_params(best_hp or {})

    wfv_aucs: List[float] = []
    prev_val_X: Optional[np.ndarray] = None
    prev_val_y: Optional[np.ndarray] = None

    for fold_k, (tr_idx, va_idx) in enumerate(non_oot_folds):
        tr_df = df.iloc[tr_idx].dropna(subset=[label_col])
        va_df = df.iloc[va_idx].dropna(subset=[label_col])
        if len(tr_df) < 20 or len(va_df) < 5:
            continue

        X_tr = tr_df[feature_cols].fillna(0).values.astype(np.float32)
        y_tr = tr_df[label_col].values.astype(int)
        X_va = va_df[feature_cols].fillna(0).values.astype(np.float32)
        y_va = va_df[label_col].values.astype(int)

        if prev_val_X is None:
            n_es = max(int(len(X_tr) * 0.1), 5)
            es_X, es_y = X_tr[-n_es:], y_tr[-n_es:]
            X_tr_es, y_tr_es = X_tr[:-n_es], y_tr[:-n_es]
        else:
            es_X, es_y = prev_val_X, prev_val_y
            X_tr_es, y_tr_es = X_tr, y_tr

        if len(np.unique(y_tr_es)) < 2 or len(np.unique(y_va)) < 2:
            continue

        dtrain = lgb.Dataset(X_tr_es, label=y_tr_es, free_raw_data=False)
        des = lgb.Dataset(es_X, label=es_y, reference=dtrain, free_raw_data=False)

        callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(-1)]
        try:
            booster = lgb.train(params, dtrain, num_boost_round=best_n_est, valid_sets=[des], callbacks=callbacks)
        except Exception:
            continue

        preds = booster.predict(X_va)
        if len(np.unique(y_va)) >= 2:
            wfv_aucs.append(roc_auc_score(y_va, preds))

        prev_val_X, prev_val_y = X_va, y_va

    mean_wfv_auc = float(np.mean(wfv_aucs)) if wfv_aucs else float("nan")

    # ── Train final model on all non-OOT data ──────────────────────────
    # oot_fold[1][0] is the first row of the OOT *validation* set (val_start).
    # Subtracting the embargo gives the last permissible training row.
    # (oot_fold[0][0] is always 0 in an expanding-window scheme — wrong target.)
    oot_val_start_idx = oot_fold[1][0] if len(oot_fold[1]) > 0 else len(df)
    final_train_df = df.iloc[:max(oot_val_start_idx - DEFAULT_EMBARGO_TD, 0)].dropna(subset=[label_col])

    if len(final_train_df) < 20 or len(np.unique(final_train_df[label_col].values)) < 2:
        return SurfaceCellResult(
            label=label_col, wfv_aucs=wfv_aucs, mean_wfv_auc=mean_wfv_auc,
            final_n_estimators=0, best_hp=best_hp or {}, calibrator=None, model=None,
            feature_cols=feature_cols, passed_gate=False, skip_reason="insufficient_final_train",
        )

    # Use fold 4 val for early stopping of final model
    calib_tr_idx, calib_va_idx = calibration_fold
    calib_va_df = df.iloc[calib_va_idx].dropna(subset=[label_col])
    if len(calib_va_df) >= 10 and len(np.unique(calib_va_df[label_col].values)) >= 2:
        es_X_final = calib_va_df[feature_cols].fillna(0).values.astype(np.float32)
        es_y_final = calib_va_df[label_col].values.astype(int)
    else:
        n_es = max(int(len(final_train_df) * 0.1), 5)
        tail = final_train_df.tail(n_es)
        es_X_final = tail[feature_cols].fillna(0).values.astype(np.float32)
        es_y_final = tail[label_col].values.astype(int)

    X_final = final_train_df[feature_cols].fillna(0).values.astype(np.float32)
    y_final = final_train_df[label_col].values.astype(int)
    dtrain_f = lgb.Dataset(X_final, label=y_final, free_raw_data=False)
    des_f = lgb.Dataset(es_X_final, label=es_y_final, reference=dtrain_f, free_raw_data=False)

    callbacks_f = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(-1)]
    try:
        final_booster = lgb.train(params, dtrain_f, num_boost_round=best_n_est, valid_sets=[des_f], callbacks=callbacks_f)
    except Exception as exc:
        return SurfaceCellResult(
            label=label_col, wfv_aucs=wfv_aucs, mean_wfv_auc=mean_wfv_auc,
            final_n_estimators=0, best_hp=best_hp or {}, calibrator=None, model=None,
            feature_cols=feature_cols, passed_gate=False, skip_reason=f"lgbm_error: {exc}",
        )

    # ── Calibration on fold 4 validation set ──────────────────────────
    calibrator = None
    if len(calib_va_df) >= 20 and len(np.unique(calib_va_df[label_col].values)) >= 2:
        raw_scores = final_booster.predict(calib_va_df[feature_cols].fillna(0).values.astype(np.float32))
        calib_y = calib_va_df[label_col].values.astype(int)
        try:
            cal_result = fit_isotonic_calibrator(calib_y, raw_scores)
            calibrator = cal_result.calibrator
        except Exception as exc:
            logger.warning("[%s] Calibration failed: %s", label_col, exc)

    return SurfaceCellResult(
        label=label_col,
        wfv_aucs=wfv_aucs,
        mean_wfv_auc=mean_wfv_auc,
        final_n_estimators=final_booster.num_trees(),
        best_hp=best_hp or {},
        calibrator=calibrator,
        model=final_booster,
        feature_cols=feature_cols,
        passed_gate=True,
    )


# ---------------------------------------------------------------------------
# Public: per-stock trainer
# ---------------------------------------------------------------------------

def train_per_stock(
    ticker: str,
    matrix_path: Optional[Path] = None,
    models_root: Optional[Path] = None,
    *,
    version: str = "v1",
    logger: Optional[logging.Logger] = None,
) -> PerStockTrainingResult:
    """
    Train all 20 surface-cell LightGBM models for one stock.

    Primary label (y_10pct_20d) is trained first.
    If its mean WFV AUC < PRIMARY_AUC_GATE, stop and return FAILED_GATE.
    Otherwise, train the remaining 19 cells.

    Returns a PerStockTrainingResult.
    """
    if not _LGBM_AVAILABLE:
        raise RuntimeError("lightgbm is not installed — cannot train models")

    log = logger or LOGGER
    result = PerStockTrainingResult(ticker=ticker, version=version, status="error")

    # Load matrix
    df = load_stock_matrix(ticker) if matrix_path is None else pd.read_parquet(matrix_path)
    if df is None or df.empty:
        result.status = "insufficient_data"
        result.error_msg = "No training matrix available"
        return result

    df = df.sort_values("event_date").reset_index(drop=True)
    feature_cols = _get_feature_cols(df)

    if len(feature_cols) < 3:
        result.status = "insufficient_data"
        result.error_msg = f"Too few features: {len(feature_cols)}"
        return result

    # ── Read per-stock primary label from schema.json ─────────────────
    primary_label = _read_schema_primary_label(ticker)
    if primary_label is None:
        result.status = "insufficient_data"
        result.error_msg = (
            "No label tier has >= 50 positives — INSUFFICIENT_DATA. "
            "See schema.json primary_label_status."
        )
        _log_lifecycle(ticker, version, "INSUFFICIENT_DATA", result.error_msg)
        return result

    # Build walk-forward fold indices
    fold_indices = build_fold_indices(len(df))
    if len(fold_indices) < 2:
        result.status = "insufficient_data"
        result.error_msg = "Too few folds for WFV — need more history"
        return result

    _log_lifecycle(ticker, version, "TRAIN", f"Starting WFV training: {len(df)} rows, {len(feature_cols)} features")

    # ── Primary label HP search (once, shared across all 20 cells) ────
    log.info("[%s] HP search on primary label %s ...", ticker, primary_label)
    best_hp, best_auc_cv, best_n_est = _cv_hp_search(df, primary_label, feature_cols, fold_indices, log)

    # ── Primary cell training ────────────────────────────────────────
    log.info("[%s] Training primary cell %s (hp_auc_cv=%.3f) ...", ticker, primary_label, best_auc_cv)
    primary_cell = _train_surface_cell(
        df, primary_label, feature_cols, fold_indices, log,
        best_hp=best_hp, best_n_est=best_n_est,
    )
    result.primary_auc = primary_cell.mean_wfv_auc

    # OOT size check
    oot_idx = fold_indices[-1][1]
    oot_df = df.iloc[oot_idx]
    if primary_label in oot_df.columns:
        result.n_rows_oot = len(oot_df)
        result.n_pos_oot = int(oot_df[primary_label].sum())

    # Primary gate
    if not primary_cell.passed_gate or np.isnan(primary_cell.mean_wfv_auc) or primary_cell.mean_wfv_auc < PRIMARY_AUC_GATE:
        result.status = "failed_gate"
        result.error_msg = (
            f"Primary WFV AUC {primary_cell.mean_wfv_auc:.3f} < gate {PRIMARY_AUC_GATE} "
            f"or training failed ({primary_cell.skip_reason})"
        )
        result.cells = [primary_cell]
        _log_lifecycle(ticker, version, "FAILED_GATE", result.error_msg)
        _update_model_db(ticker, version, "FAILED_GATE", primary_label, primary_cell)
        return result

    # Persist primary model
    if models_root is None:
        models_root = get_models_root()
    _save_cell_bundle(ticker, version, primary_label, primary_cell, feature_cols, models_root, result)

    # ── Remaining 19 cells ───────────────────────────────────────────
    remaining_labels = [c for c in SURFACE_LABEL_COLS if c != primary_label]
    trained_cells = [primary_cell]
    n_failed = 0

    for label_col in remaining_labels:
        if label_col not in df.columns:
            log.warning("[%s] Label %s missing — skipping cell", ticker, label_col)
            continue

        log.debug("[%s] Training cell %s ...", ticker, label_col)
        cell = _train_surface_cell(
            df, label_col, feature_cols, fold_indices, log,
            best_hp=best_hp, best_n_est=best_n_est,
        )
        trained_cells.append(cell)
        if not cell.passed_gate:
            n_failed += 1
            log.warning("[%s] Cell %s failed: %s", ticker, label_col, cell.skip_reason)
        else:
            _save_cell_bundle(ticker, version, label_col, cell, feature_cols, models_root, result)

    result.cells = trained_cells
    result.n_cells_trained = sum(1 for c in trained_cells if c.passed_gate)
    result.status = "ok"

    _log_lifecycle(
        ticker, version, "TRAIN",
        f"Training complete. {result.n_cells_trained}/20 cells passed. Primary AUC={result.primary_auc:.3f}",
    )
    log.info(
        "[%s] Done. %d/20 cells trained, primary AUC %.3f",
        ticker, result.n_cells_trained, result.primary_auc,
    )
    return result


def _save_cell_bundle(
    ticker: str,
    version: str,
    label_col: str,
    cell: SurfaceCellResult,
    feature_cols: List[str],
    models_root: Path,
    result: PerStockTrainingResult,
) -> None:
    try:
        bundle_path = save_model_bundle(
            tier="per_stock",
            identifier=f"{ticker.upper()}/{label_col}",
            model=cell.model,
            calibrator=cell.calibrator,
            feature_list=feature_cols,
            metadata={
                "ticker": ticker,
                "label": label_col,
                "mean_wfv_auc": cell.mean_wfv_auc,
                "wfv_aucs": cell.wfv_aucs,
                "best_hp": cell.best_hp,
                "n_estimators": cell.final_n_estimators,
            },
            version=version,
            models_root=models_root,
        )
        result.model_bundle_paths[label_col] = str(bundle_path)
        _update_model_db(ticker, version, "TRAINING", label_col, cell)
    except Exception as exc:
        LOGGER.warning("[%s] Failed to save bundle for %s: %s", ticker, label_col, exc)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _log_lifecycle(ticker: str, version: str, event_type: str, notes: str) -> None:
    try:
        from app.core.database import exec_sql
        exec_sql(
            """INSERT INTO model_lifecycle_log
               (model_identifier, model_version, event_type, event_notes, occurred_at)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (ticker.upper(), version, event_type, notes[:500]),
        )
    except Exception:
        pass


def _update_model_db(
    ticker: str,
    version: str,
    status: str,
    label_col: str,
    cell: SurfaceCellResult,
) -> None:
    try:
        from app.core.database import exec_sql
        model_id = f"{ticker.upper()}::{label_col}::{version}"
        exec_sql(
            """INSERT INTO ml_models
               (model_id, ticker, label_col, version, status, mean_wfv_auc, created_at)
               VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT (model_id) DO UPDATE SET
                   status = EXCLUDED.status,
                   mean_wfv_auc = EXCLUDED.mean_wfv_auc""",
            (model_id, ticker.upper(), label_col, version, status,
             float(cell.mean_wfv_auc) if not np.isnan(cell.mean_wfv_auc) else None),
        )
    except Exception:
        pass

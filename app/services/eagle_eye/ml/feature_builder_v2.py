from __future__ import annotations

from bisect import bisect_right
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app.services.eagle_eye.indicators import compute_all_indicators
from app.services.eagle_eye.move_detector import MoveEvent, detect_moves
from app.services.eagle_eye.store import load_ohlcv

from .trainer import EagleEyeMLTrainer, TrainingConfig


MIN_HISTORY_BARS = 100
NEGATIVE_HORIZON_DAYS = 20

LAG_OFFSETS: Tuple[int, ...] = (0, 1, 3, 7, 14, 30)
BASE_INDICATORS: Tuple[str, ...] = (
    "rsi",
    "macd_histogram",
    "adx",
    "bb_bandwidth",
    "cmf",
    "mfi",
    "rel_volume",
    "obv_slope_20",
    "linreg_slope",
    "atr",
    "hist_vol_30d",
)
VELOCITY_INDICATORS: Tuple[str, ...] = (
    "rsi",
    "macd_histogram",
    "adx",
    "bb_bandwidth",
    "cmf",
    "mfi",
    "rel_volume",
    "obv_slope_20",
)

EXCLUDED_NAME_SNIPPETS: Tuple[str, ...] = (
    "swing_high",
    "swing_low",
    "stage",
    "accumulation_score",
    "wyckoff_phase",
    "percentile",
    "rank",
    "zscore",
)

NON_FEATURE_COLUMNS: Tuple[str, ...] = (
    "ticker",
    "event_id",
    "event_date",
    "acceleration_date",
    "y",
    "sample_type",
)

SINGLE_FEATURE_PARAMS: Mapping[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 7,
    "min_data_in_leaf": 5,
    "learning_rate": 0.05,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "seed": 42,
    "deterministic": True,
    "verbosity": -1,
}


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(out):
        return float("nan")
    return out


def _to_ts(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _build_feature_formula_map() -> Dict[str, str]:
    formulas: Dict[str, str] = {}
    for column in BASE_INDICATORS:
        for lag in LAG_OFFSETS:
            key = f"{column}_t{lag}"
            formulas[key] = f"features['{key}'] = _lag_value(ind, '{column}', {lag})"
    for column in VELOCITY_INDICATORS:
        key = f"{column}_velocity_7d"
        formulas[key] = f"features['{key}'] = features['{column}_t0'] - features['{column}_t7']"
    return formulas


FEATURE_COMPUTATION_FORMULAS = _build_feature_formula_map()


def _lag_value(ind: pd.DataFrame, column: str, lag: int) -> float:
    if column not in ind.columns or len(ind) <= lag:
        return float("nan")
    return _safe_float(ind.iloc[-(lag + 1)].get(column))


def _is_excluded_feature_name(name: str) -> bool:
    low = name.lower()
    return any(snippet in low for snippet in EXCLUDED_NAME_SNIPPETS)


def _validate_feature_selection() -> None:
    selected = list(BASE_INDICATORS) + [f"{c}_velocity_7d" for c in VELOCITY_INDICATORS]
    bad = [name for name in selected if _is_excluded_feature_name(name)]
    if bad:
        raise ValueError(f"Selected indicators include excluded names: {bad}")


_validate_feature_selection()


def compute_features_at_T(ohlcv_full: pd.DataFrame, T: Any) -> Optional[Dict[str, float]]:
    t_ts = _to_ts(T)
    hist = ohlcv_full.loc[ohlcv_full.index <= t_ts].copy()
    if len(hist) < MIN_HISTORY_BARS:
        return None

    ind = compute_all_indicators(hist)
    if ind.empty:
        return None

    features: Dict[str, float] = {}
    for column in BASE_INDICATORS:
        if column not in ind.columns:
            continue
        for lag in LAG_OFFSETS:
            features[f"{column}_t{lag}"] = _lag_value(ind, column, lag)

    for column in VELOCITY_INDICATORS:
        now = features.get(f"{column}_t0", float("nan"))
        prev = features.get(f"{column}_t7", float("nan"))
        if np.isfinite(now) and np.isfinite(prev):
            features[f"{column}_velocity_7d"] = float(now - prev)
        else:
            features[f"{column}_velocity_7d"] = float("nan")

    return features


def get_feature_columns_v2(frame: pd.DataFrame) -> List[str]:
    return [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]


def get_feature_formula(feature_name: str) -> str:
    return FEATURE_COMPUTATION_FORMULAS.get(feature_name, "Formula not found in v2 builder")


def _previous_trading_day(index: pd.DatetimeIndex, day: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = int(index.searchsorted(day, side="right")) - 1
    if pos < 0:
        return None
    return _to_ts(index[pos])


def _accel_indices(index: pd.DatetimeIndex, events: Sequence[MoveEvent]) -> List[int]:
    out = set()
    for event in events:
        if getattr(event, "is_fakeout", False):
            continue
        accel_ts = _to_ts(event.acceleration_date)
        pos = int(index.searchsorted(accel_ts, side="left"))
        if pos < len(index) and _to_ts(index[pos]) == accel_ts:
            out.add(pos)
        elif pos > 0:
            out.add(pos - 1)
    return sorted(out)


def _has_accel_in_next_horizon(idx: int, accel_idx: Sequence[int], horizon_days: int = NEGATIVE_HORIZON_DAYS) -> bool:
    if not accel_idx:
        return False
    pos = bisect_right(accel_idx, idx)
    if pos >= len(accel_idx):
        return False
    return accel_idx[pos] <= idx + horizon_days


def _build_pre_accel_buffer_indices(
    accel_idx: Sequence[int],
    *,
    total_len: int,
    buffer_days: int = 5,
) -> set[int]:
    blocked: set[int] = set()
    for accel in accel_idx:
        start = max(0, int(accel) - int(buffer_days))
        end = min(total_len, int(accel))
        if end <= start:
            continue
        for pos in range(start, end):
            blocked.add(pos)
    return blocked


def _build_positive_rows(ticker: str, ohlcv_full: pd.DataFrame, events: Sequence[MoveEvent]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for event in events:
        if getattr(event, "is_fakeout", False):
            continue

        accel_ts = _to_ts(event.acceleration_date)
        pred_ts = _previous_trading_day(ohlcv_full.index, accel_ts - pd.Timedelta(days=1))
        if pred_ts is None:
            continue

        features = compute_features_at_T(ohlcv_full, pred_ts)
        if not features:
            continue

        row = {
            "ticker": ticker,
            "event_id": str(event.event_id),
            "event_date": pred_ts,
            "acceleration_date": accel_ts,
            "y": 1,
            "sample_type": "positive",
        }
        row.update(features)
        rows.append(row)
    return rows


def _build_negative_rows(
    ticker: str,
    ohlcv_full: pd.DataFrame,
    accel_idx: Sequence[int],
    *,
    seed: int,
    blocked_days: Sequence[pd.Timestamp],
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    blocked_set = {_to_ts(d) for d in blocked_days}
    blocked_pre_accel_idx = _build_pre_accel_buffer_indices(
        accel_idx,
        total_len=len(ohlcv_full),
        buffer_days=5,
    )
    rows: List[Dict[str, Any]] = []

    candidates: List[pd.Timestamp] = []
    for idx, day in enumerate(ohlcv_full.index):
        day_ts = _to_ts(day)

        if idx < MIN_HISTORY_BARS:
            continue
        if day_ts in blocked_set:
            continue
        if idx in blocked_pre_accel_idx:
            continue
        if _has_accel_in_next_horizon(idx, accel_idx, NEGATIVE_HORIZON_DAYS):
            continue

        candidates.append(day_ts)

    if not candidates:
        return rows

    rng.shuffle(candidates)
    for chosen in candidates:
        features = compute_features_at_T(ohlcv_full, chosen)
        if not features:
            continue

        row = {
            "ticker": ticker,
            "event_id": f"{ticker}_{chosen.date().isoformat()}_neg",
            "event_date": chosen,
            "acceleration_date": pd.NaT,
            "y": 0,
            "sample_type": "negative",
        }
        row.update(features)
        rows.append(row)

    return rows


def build_abk_strict_causal_dataset(
    ohlcv_full: pd.DataFrame,
    *,
    ticker: str = "ABK",
    seed: int = 42,
) -> pd.DataFrame:
    if ohlcv_full.empty:
        return pd.DataFrame()

    ohlcv = ohlcv_full.sort_index().copy()
    events = detect_moves(ticker, ohlcv)

    positives = _build_positive_rows(ticker, ohlcv, events)
    positive_days = [r["event_date"] for r in positives]
    accel_idx = _accel_indices(ohlcv.index, events)

    negatives_all = _build_negative_rows(
        ticker,
        ohlcv,
        accel_idx,
        seed=seed,
        blocked_days=positive_days,
    )

    n_pos = len(positives)
    target_neg = (n_pos * 3) if n_pos > 0 else len(negatives_all)

    if len(negatives_all) > target_neg:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(negatives_all), size=int(target_neg), replace=False)
        negatives = [negatives_all[int(i)] for i in np.sort(pick)]
    else:
        negatives = negatives_all

    combined = positives + negatives
    if not combined:
        return pd.DataFrame()

    frame = pd.DataFrame(combined)
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce").dt.normalize()
    frame["acceleration_date"] = pd.to_datetime(frame["acceleration_date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["event_date"]).sort_values("event_date").reset_index(drop=True)
    return frame


def run_strict_causality_checks(
    frame: pd.DataFrame,
    ohlcv_full: pd.DataFrame,
    *,
    sample_size: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    if frame.empty:
        return []

    feature_cols = get_feature_columns_v2(frame)
    sample_n = min(int(sample_size), len(frame))
    rng = np.random.default_rng(seed)
    picks = rng.choice(frame.index.to_numpy(), size=sample_n, replace=False)
    probes = [
        "rsi_t0",
        "rsi_t1",
        "rsi_t7",
        "rsi_t30",
        "rsi_velocity_7d",
        "macd_histogram_t0",
        "adx_t0",
        "rel_volume_t0",
    ]

    checks: List[Dict[str, Any]] = []
    for idx in np.sort(picks):
        row = frame.loc[int(idx)]
        t_ts = _to_ts(row["event_date"])
        hist = ohlcv_full.loc[ohlcv_full.index <= t_ts].copy()
        recomputed = compute_features_at_T(ohlcv_full, t_ts) or {}

        max_abs_diff = 0.0
        compared = 0
        for col in feature_cols:
            row_val = _safe_float(row.get(col))
            cmp_val = _safe_float(recomputed.get(col))
            if np.isnan(row_val) and np.isnan(cmp_val):
                continue

            compared += 1
            if np.isnan(row_val) != np.isnan(cmp_val):
                max_abs_diff = float("inf")
                break

            diff = abs(row_val - cmp_val)
            if diff > max_abs_diff:
                max_abs_diff = float(diff)

        sample_comp: List[Dict[str, Any]] = []
        for feature in probes:
            if feature not in feature_cols:
                continue
            row_val = _safe_float(row.get(feature))
            cmp_val = _safe_float(recomputed.get(feature))
            sample_comp.append(
                {
                    "feature": feature,
                    "row_value": None if np.isnan(row_val) else float(row_val),
                    "recomputed_value": None if np.isnan(cmp_val) else float(cmp_val),
                    "abs_diff": None
                    if np.isnan(row_val) or np.isnan(cmp_val)
                    else float(abs(row_val - cmp_val)),
                }
            )

        hist_max = _to_ts(hist.index.max()) if len(hist) else None
        checks.append(
            {
                "row_index": int(idx),
                "label": int(row["y"]),
                "event_date": t_ts.date().isoformat(),
                "history_max_date": hist_max.date().isoformat() if hist_max is not None else None,
                "history_rows": int(len(hist)),
                "compared_feature_count": int(compared),
                "max_abs_diff": None if not np.isfinite(max_abs_diff) else float(max_abs_diff),
                "exact_match_all_compared": bool(np.isfinite(max_abs_diff) and max_abs_diff <= 1e-12),
                "no_future_rows_used": bool(hist_max is None or hist_max <= t_ts),
                "sample_feature_comparisons": sample_comp,
            }
        )

    return checks


def _split_train_test(
    frame: pd.DataFrame,
    *,
    train_end: str = "2024-06-30",
    test_start: str = "2024-07-01",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_cut = _to_ts(train_end)
    test_cut = _to_ts(test_start)
    train = frame.loc[frame["event_date"] <= train_cut].copy().sort_values("event_date").reset_index(drop=True)
    test = frame.loc[frame["event_date"] >= test_cut].copy().sort_values("event_date").reset_index(drop=True)
    return train, test


def _class_counts(frame: pd.DataFrame) -> Dict[str, int]:
    if frame.empty or "y" not in frame.columns:
        return {"positives": 0, "negatives": 0}
    y = frame["y"].astype(int)
    return {
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
    }


def _single_feature_auc_sweep(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: Sequence[str],
) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    if train.empty or test.empty:
        return out

    ytr = train["y"].astype(int).to_numpy()
    yte = test["y"].astype(int).to_numpy()
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return out

    for feature in feature_cols:
        xtr = pd.to_numeric(train[feature], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1)
        xte = pd.to_numeric(test[feature], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1)
        if np.unique(xtr).size <= 1:
            continue

        model = lgb.train(
            dict(SINGLE_FEATURE_PARAMS),
            lgb.Dataset(xtr, label=ytr, feature_name=[feature]),
            num_boost_round=100,
        )
        pred = np.clip(model.predict(xte), 0.0, 1.0)
        try:
            auc = float(roc_auc_score(yte, pred))
        except ValueError:
            continue
        out.append((feature, auc))

    out.sort(key=lambda x: x[1], reverse=True)
    return out


def train_and_evaluate_abk_dataset(frame: pd.DataFrame, *, seed: int = 42) -> Dict[str, Any]:
    if frame.empty:
        return {
            "rows_total": 0,
            "n_features": 0,
            "cv_auc_walk_forward_train_only": None,
            "oot_auc": None,
            "train_class_counts": {"positives": 0, "negatives": 0},
            "test_class_counts": {"positives": 0, "negatives": 0},
            "top5_feature_importance": [],
            "single_feature_top5": [],
            "single_feature_count_auc_gt_075": 0,
            "single_feature_top5_formulas": {},
        }

    train, test = _split_train_test(frame)
    feature_cols = get_feature_columns_v2(frame)

    trainer = EagleEyeMLTrainer(config=TrainingConfig(random_state=seed, target_col="y"))
    cv = trainer._train_cv(train, feature_cols) if len(train) > 0 else {
        "mean_metrics": {"auc": float("nan")}
    }
    cv_auc = cv.get("mean_metrics", {}).get("auc", float("nan"))

    oot_auc: Optional[float] = None
    top_imp: List[Dict[str, Any]] = []
    params = trainer._lgb_params()
    params["seed"] = seed

    if (
        len(train) > 0
        and len(test) > 0
        and train["y"].nunique() > 1
        and test["y"].nunique() > 1
    ):
        xtr = train[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        ytr = train["y"].astype(int)
        xte = test[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        yte = test["y"].astype(int)

        model = lgb.train(
            params,
            lgb.Dataset(xtr, label=ytr, feature_name=list(feature_cols)),
            num_boost_round=200,
        )
        pred = np.clip(model.predict(xte), 0.0, 1.0)
        oot_auc = float(roc_auc_score(yte, pred))

        gain = model.feature_importance(importance_type="gain")
        order = np.argsort(gain)[::-1]
        for pos in order:
            score = float(gain[int(pos)])
            if score <= 0:
                continue
            top_imp.append({"feature": feature_cols[int(pos)], "importance_gain": score})
            if len(top_imp) >= 5:
                break

    single_results = _single_feature_auc_sweep(train, test, feature_cols)
    single_top5 = [{"feature": f, "auc": float(a)} for f, a in single_results[:5]]
    single_formulas = {entry["feature"]: get_feature_formula(entry["feature"]) for entry in single_top5}

    return {
        "rows_total": int(len(frame)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "n_features": int(len(feature_cols)),
        "feature_columns": list(feature_cols),
        "train_class_counts": _class_counts(train),
        "test_class_counts": _class_counts(test),
        "cv_auc_walk_forward_train_only": None
        if pd.isna(cv_auc)
        else float(cv_auc),
        "oot_auc": oot_auc,
        "top5_feature_importance": top_imp,
        "single_feature_top5": single_top5,
        "single_feature_count_auc_gt_075": int(sum(1 for _, auc in single_results if auc > 0.75)),
        "single_feature_top5_formulas": single_formulas,
    }


def run_abk_clean_rebuild(*, ticker: str = "ABK", seed: int = 42) -> Dict[str, Any]:
    ohlcv = load_ohlcv(ticker)
    frame = build_abk_strict_causal_dataset(ohlcv, ticker=ticker, seed=seed)
    causality_checks = run_strict_causality_checks(frame, ohlcv, sample_size=5, seed=seed)
    metrics = train_and_evaluate_abk_dataset(frame, seed=seed)

    result = {
        "ticker": ticker,
        "approach": "strict_causal_feature_builder_v2",
        "excluded_feature_name_snippets": list(EXCLUDED_NAME_SNIPPETS),
        "base_indicators": list(BASE_INDICATORS),
        "lags": list(LAG_OFFSETS),
        "velocity_indicators": list(VELOCITY_INDICATORS),
        "strict_causality_checks": causality_checks,
        "label_construction_positive_rows": {
            "source": "detect_moves(ticker, ohlcv)",
            "logic": [
                "for event in events: if event.is_fakeout: continue",
                "accel_ts = pd.Timestamp(event.acceleration_date).normalize()",
                "pred_ts = previous trading day on or before accel_ts - 1 calendar day",
                "features = compute_features_at_T(ohlcv, pred_ts)",
                "row label y = 1",
            ],
        },
    }
    result.update(metrics)
    return result
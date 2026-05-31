from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np


class BoosterAdapter:
    """Compatibility wrapper that exposes predict_proba for LightGBM Booster."""

    def __init__(self, booster: lgb.Booster):
        self._booster = booster

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        return self._booster.predict(*args, **kwargs)

    def predict_proba(self, data: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        pred = self._booster.predict(data, *args, **kwargs)
        arr = np.asarray(pred, dtype=float)
        if arr.ndim == 2:
            return arr

        if arr.ndim == 1:
            num_class = int(self._booster.num_model_per_iteration() or 1)
            if num_class > 1 and arr.size % num_class == 0:
                return arr.reshape(-1, num_class)

            # Binary fallback where Booster returns positive-class probability.
            pos = np.clip(arr, 0.0, 1.0)
            return np.column_stack([1.0 - pos, pos])

        return np.asarray(pred, dtype=float)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._booster, name)


@dataclass
class ModelBundle:
    tier: str
    identifier: str
    version: str
    model: Optional[Any]
    calibrator: Any
    feature_list: List[str]
    metadata: Dict[str, Any]
    path: Path

    @property
    def task(self) -> str:
        task = str(self.metadata.get("task") or "").strip().lower()
        if task:
            return task
        objective = str(self.metadata.get("objective") or "").strip().lower()
        if "class" in objective:
            return "classification"
        if objective:
            return "regression"
        return "unknown"


def get_models_root(root: Optional[Path | str] = None) -> Path:
    if root is not None:
        p = Path(root)
    else:
        p = Path(__file__).resolve().parents[4] / "ml_models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_reports_root(root: Optional[Path | str] = None) -> Path:
    p = get_models_root(root) / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logs_root(root: Optional[Path | str] = None) -> Path:
    p = get_models_root(root) / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_cache_root(root: Optional[Path | str] = None) -> Path:
    p = get_models_root(root) / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _version_today() -> str:
    return date.today().isoformat()


def _id(identifier: str) -> str:
    return identifier.replace("/", "_").replace("\\", "_").strip()


def _bundle_dir(root: Path, tier: str, identifier: str, version: str) -> Path:
    return root / tier / _id(identifier) / version


def _current_dir(root: Path, tier: str, identifier: str) -> Path:
    return root / tier / _id(identifier) / "current"


def _versions(root: Path, tier: str, identifier: str) -> List[Path]:
    base = root / tier / _id(identifier)
    if not base.exists():
        return []
    dirs = [d for d in base.iterdir() if d.is_dir() and d.name != "current"]
    return sorted(dirs, key=lambda p: p.name)


def save_model_bundle(
    *,
    tier: str,
    identifier: str,
    model: Optional[lgb.Booster],
    calibrator: Any,
    feature_list: List[str],
    metadata: Dict[str, Any],
    version: Optional[str] = None,
    models_root: Optional[Path | str] = None,
) -> Path:
    root = get_models_root(models_root)
    version_name = version or _version_today()
    bundle = _bundle_dir(root, tier, identifier, version_name)
    bundle.mkdir(parents=True, exist_ok=True)

    if model is not None:
        model.save_model(str(bundle / "model.lgb"))

    if calibrator is not None:
        joblib.dump(calibrator, bundle / "calibrator.pkl")

    with (bundle / "feature_list.json").open("w", encoding="utf-8") as f:
        json.dump(feature_list, f, indent=2)

    # Keep a second canonical name for live-inference callers.
    with (bundle / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(feature_list, f, indent=2)

    payload = dict(metadata)
    payload.update({"tier": tier, "identifier": identifier, "version": version_name})
    with (bundle / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    # Keep only last 4 versions.
    versions = _versions(root, tier, identifier)
    while len(versions) > 4:
        old = versions.pop(0)
        shutil.rmtree(old, ignore_errors=True)

    # Alias current to latest by copying files (portable on Windows without symlink permissions).
    current = _current_dir(root, tier, identifier)
    if current.exists():
        shutil.rmtree(current, ignore_errors=True)
    # dirs_exist_ok=True handles Windows file-lock edge cases where rmtree silently fails.
    shutil.copytree(bundle, current, dirs_exist_ok=True)

    return bundle


def load_model_bundle(
    *,
    tier: str,
    identifier: str,
    version: str = "current",
    models_root: Optional[Path | str] = None,
) -> Optional[ModelBundle]:
    root = get_models_root(models_root)
    path = _bundle_dir(root, tier, identifier, version)
    if version == "current":
        path = _current_dir(root, tier, identifier)
    if not path.exists() or not path.is_dir():
        return None

    meta_path = path / "metadata.json"
    feats_path = path / "feature_list.json"
    names_path = path / "feature_names.json"
    model_path = path / "model.lgb"
    cal_path = path / "calibrator.pkl"

    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    feature_list: List[str] = []
    if feats_path.exists():
        feature_list = json.loads(feats_path.read_text(encoding="utf-8"))
    elif names_path.exists():
        feature_list = json.loads(names_path.read_text(encoding="utf-8"))

    raw_model = lgb.Booster(model_file=str(model_path)) if model_path.exists() else None
    model = BoosterAdapter(raw_model) if raw_model is not None else None
    calibrator = joblib.load(cal_path) if cal_path.exists() else None

    return ModelBundle(
        tier=tier,
        identifier=identifier,
        version=str(metadata.get("version") or version),
        model=model,
        calibrator=calibrator,
        feature_list=feature_list,
        metadata=metadata,
        path=path,
    )


def model_exists(
    tier: str,
    identifier: str,
    models_root: Optional[Path | str] = None,
) -> bool:
    root = get_models_root(models_root)
    return _current_dir(root, tier, identifier).exists()


def model_is_rejected(
    tier: str,
    identifier: str,
    models_root: Optional[Path | str] = None,
) -> bool:
    bundle = load_model_bundle(tier=tier, identifier=identifier, version="current", models_root=models_root)
    if bundle is None:
        return False
    reason = str(bundle.metadata.get("rejected_reason") or "").strip()
    return bool(reason)


def latest_report_summary(models_root: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
    reports_root = get_reports_root(models_root)
    dates = sorted([d for d in reports_root.iterdir() if d.is_dir()], key=lambda p: p.name)
    if not dates:
        return None
    summary = dates[-1] / "summary.json"
    if not summary.exists():
        return None
    return json.loads(summary.read_text(encoding="utf-8"))


def load_feature_names(
    *,
    tier: str = "global",
    identifier: str = "baseline",
    models_root: Optional[Path | str] = None,
) -> List[str]:
    """Load saved feature names for a model bundle, with global fallback."""
    root = get_models_root(models_root)

    def _read_names(path: Path) -> Optional[List[str]]:
        names_file = path / "feature_names.json"
        feats_file = path / "feature_list.json"
        if names_file.exists():
            return json.loads(names_file.read_text(encoding="utf-8"))
        if feats_file.exists():
            return json.loads(feats_file.read_text(encoding="utf-8"))
        return None

    primary = _current_dir(root, tier, identifier)
    names = _read_names(primary)
    if names:
        return names

    fallback = _current_dir(root, "global", "baseline")
    names = _read_names(fallback)
    if names:
        return names

    raise FileNotFoundError(
        f"No feature names found for {tier}/{identifier} or global/baseline"
    )

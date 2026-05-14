from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import lightgbm as lgb


@dataclass
class ModelBundle:
    tier: str
    identifier: str
    version: str
    model: Optional[lgb.Booster]
    calibrator: Any
    feature_list: List[str]
    metadata: Dict[str, Any]
    path: Path


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
    shutil.copytree(bundle, current)

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
    model_path = path / "model.lgb"
    cal_path = path / "calibrator.pkl"

    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    feature_list: List[str] = []
    if feats_path.exists():
        feature_list = json.loads(feats_path.read_text(encoding="utf-8"))

    model = lgb.Booster(model_file=str(model_path)) if model_path.exists() else None
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

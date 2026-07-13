from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MLGateIdentity:
    enabled: bool
    controlling_config: dict[str, Any]
    loads_model_artifact: bool
    model_artifact_hash: str | None
    training_cutoff: str | None
    uses_randomness: bool
    uses_network: bool
    uses_current_time: bool
    writes_database: bool
    affects_setup_or_entry_eligibility: bool


def classify_ml_gate_behavior(config: dict[str, Any]) -> MLGateIdentity:
    enabled = bool(config.get("ml_gate_enabled", False))
    controls = {
        "ml_gate_enabled": bool(config.get("ml_gate_enabled", False)),
        "ml_min_labeled_signals": int(config.get("ml_min_labeled_signals", 150)),
        "ml_prob_min": float(config.get("ml_prob_min", 0.45)),
    }
    return MLGateIdentity(
        enabled=enabled,
        controlling_config=controls,
        loads_model_artifact=False,
        model_artifact_hash=None,
        training_cutoff=None,
        uses_randomness=False,
        uses_network=False,
        uses_current_time=False,
        writes_database=False,
        affects_setup_or_entry_eligibility=True,
    )


def hash_ml_gate_identity(identity: MLGateIdentity) -> str:
    payload = json.dumps(identity.__dict__, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def optional_model_artifact_hash(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()

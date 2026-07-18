from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
V1_JSON = REVIEW / "r14b_parameter_freeze_v1.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_sha_sidecar(sidecar_path: Path, files: list[tuple[str, Path]]) -> None:
    lines = []
    for rel, p in files:
        lines.append(f"{sha256_file(p)}  {rel}")
    sidecar_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def baseline_id_now() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"EE_V2_{ts}"


def markdown_from_json(title: str, payload: dict[str, Any]) -> str:
    lines = [f"# {title}", "", json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), ""]
    return "\n".join(lines)


def build_payload(v1: dict[str, Any], baseline_id: str) -> dict[str, Any]:
    v1_values = dict(v1.get("owner_ratified_values_verbatim") or {})
    return {
        "version_id": "R14B_PARAMETER_FREEZE_V2",
        "supersedes": "R14B_PARAMETER_FREEZE_V1_BY_EXTENSION_ONLY",
        "extension_mode": "APPEND_ONLY",
        "authority": {
            "owner_ratification_status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
            "owner_ratification_received": True,
            "governing_design_doc": {
                "version": "R14_DESIGN_SPEC_CONSOLIDATED_V2_2",
                "json_sha256": "9a5f1facdf1fc222239e6304afadb1e420f4d22fb506f23d97af52b64cb4b52b",
                "md_sha256": "dedf5361c25b40df5b0ece8dbfeb9f360e81cf5facfdb3a2305dcf6c8b31de4e",
            },
            "implementation_liberty_note": "No implementation liberty beyond governing text without owner directive.",
        },
        "baseline": {
            "implementation_baseline_id": baseline_id,
            "new_module_path": "app/services/eagle_eye_v2",
            "supersession_rule": "Old engine is never edited; only superseded by isolated v2 module path.",
            "r11_baseline_status": "UNTOUCHED_ARCHIVED",
        },
        "v1_restated_values_unchanged": {
            "status": "IN_FORCE_UNCHANGED",
            "restated_from": "R14B_PARAMETER_FREEZE_V1",
            "values": v1_values,
        },
        "r14b_parameter_gate_ratifications_v2": {
            "invalidation_rule": {
                "name": "INVALIDATION_RULE",
                "value": "CLOSE_BELOW_BASE_LOW_BY_ATR_X_N",
                "parameters": {"atr_mult": 1.0, "n_sessions": 2},
                "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
                "evidence": {
                    "source": "r14b_parameter_gate_evidence_v1",
                    "gate_table_row": {
                        "base_count": 522,
                        "median_life": 93.5,
                        "survive60_pct": 57.09,
                        "false_persistence_pct": 0.00,
                        "tiers": "all tiers",
                    },
                },
            },
            "frozen_parameters": {
                "base_min_sessions": {
                    "value": 10,
                    "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
                },
                "base_max_width_pct": {
                    "value": 0.24,
                    "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
                },
                "atr_squeeze_pctile": {
                    "value": 0.95,
                    "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
                },
                "volume_breakout_mult": {
                    "value": 2.5,
                    "authority": "CONTEXT_ONLY_NEVER_SOLE_VETO",
                    "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
                },
                "rsi_regime": {
                    "value": 50.0,
                    "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
                },
                "adx_trigger": {
                    "value": 18.0,
                    "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
                    "owner_amendment": {
                        "supersedes_proposal": 15.0,
                        "rationale": "Consistency with cited EX_SET_B p55~=25 distribution.",
                    },
                },
                "LIQUIDITY_EXECUTION_SIZE_PARAMETER": {
                    "value": 0.10,
                    "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
                },
                "cmf_floor": {
                    "value": 0.05,
                    "authority": "TELEMETRY_ONLY",
                    "status": "OWNER_RATIFIED_AT_PARAMETER_GATE_PENDING_FLOW_CORE_DECISION",
                },
                "ml_prob_min": {
                    "value": 0.55,
                    "authority": "NON_BLOCKING",
                    "status": "OWNER_RATIFIED_AT_PARAMETER_GATE_UNTIL_AUDITABLE_ML_SURFACE",
                    "principle": "F8c_no_unauditable_veto",
                },
            },
            "flow_core_composition": {
                "value": "DEFERRED_PENDING_CONDITIONAL_ANALYSIS",
                "status": "OWNER_RATIFIED_AT_PARAMETER_GATE",
                "interim_wiring": "OBV_ANV_SLOPE_CORE_DRIVES_EARLY_TIER_DETECTION_PREDICATES",
                "blocking_authority": "NONE_UNTIL_OWNER_RATIFICATION_OF_CONDITIONAL_ANALYSIS",
                "set_b_quarantine_reaffirmed": True,
            },
        },
        "module_authorizations": {
            "module_e": "AUTHORIZED",
            "module_f": "AUTHORIZED_TO_FOLLOW_ON_MODULE_E_REVIEW_PASS",
            "module_g": "AUTHORIZED_TO_FOLLOW_ON_MODULE_E_REVIEW_PASS",
            "build_order": ["e", "f", "g"],
        },
        "conduct_rules_reaffirmed": [
            "Permanent scripts only",
            "Append-only artifacts",
            "Frozen verifiers",
            "No self-declared gate passage",
            "No temp-and-delete",
            "Set B quarantine remains in force",
            "Canonical surface unchanged",
            "Recurrence counting continues",
        ],
    }


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    baseline_id = baseline_id_now()
    if not V1_JSON.exists():
        raise FileNotFoundError(f"Missing prerequisite artifact: {V1_JSON}")

    v1 = json.loads(V1_JSON.read_text(encoding="utf-8"))
    payload = build_payload(v1, baseline_id)

    out_json = REVIEW / "r14b_parameter_freeze_v2.json"
    out_md = REVIEW / "r14b_parameter_freeze_v2.md"
    out_sha = REVIEW / "r14b_parameter_freeze_v2.sha256"

    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(markdown_from_json("R14-B Parameter Freeze v2", payload), encoding="utf-8")

    write_sha_sidecar(
        out_sha,
        [
            ("artifacts/preview1a_prestart/review_final/r14b_parameter_freeze_v2.json", out_json),
            ("artifacts/preview1a_prestart/review_final/r14b_parameter_freeze_v2.md", out_md),
        ],
    )

    print("R14B_PARAMETER_FREEZE_V2_COMPLETE")
    print("baseline_id", baseline_id)
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))
    print("sidecar_sha256", sha256_file(out_sha))


if __name__ == "__main__":
    main()

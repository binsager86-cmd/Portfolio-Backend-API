from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    out_json = REVIEW / "r14c_module_c_closure_pin_v1.json"
    out_md = REVIEW / "r14c_module_c_closure_pin_v1.md"

    adaptive_path = ROOT / "app" / "services" / "eagle_eye_v2" / "adaptive_base_geometry.py"
    warmup_path = ROOT / "app" / "services" / "eagle_eye_v2" / "warmup_readiness_engine.py"

    candidates = read_json(REVIEW / "r14c_invalidation_rule_candidates_v1.json")
    m4 = read_json(REVIEW / "r14c_module_c_test_evidence_v4.json")

    payload = {
        "version_id": "R14C_MODULE_C_CLOSURE_PIN_V1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "module_c_status": "PASSED",
        "reviewed_bytes": {
            "adaptive_base_geometry_py": {
                "path": "app/services/eagle_eye_v2/adaptive_base_geometry.py",
                "sha256": sha256_file(adaptive_path),
            },
            "warmup_readiness_engine_py": {
                "path": "app/services/eagle_eye_v2/warmup_readiness_engine.py",
                "sha256": sha256_file(warmup_path),
                "note": "post-carry-in-fix bytes pinned",
            },
        },
        "registered_parameter_gate_finding": {
            "finding_id": "R14C_FINDING_TIJARA_RETIRE_UNDER_DEFAULT",
            "statement": "TIJARA retired under module (c) default invalidation form; invalidation rule is load-bearing and deferred to parameter gate.",
            "evidence": {
                "module_c_test_evidence": "r14c_module_c_test_evidence_v4.json",
                "tijara_final_state": m4.get("tijara_outcome_as_observed", {}).get("final_state"),
                "tijara_retire_count": m4.get("tijara_outcome_as_observed", {}).get("retire_count"),
            },
        },
        "parameter_gate_evidence_base": {
            "artifact": "r14c_invalidation_rule_candidates_v1.json",
            "scope": candidates.get("scope"),
            "selection_status": candidates.get("selection_status"),
            "explicit_non_optimization_statement": candidates.get("explicit_non_optimization_statement"),
            "ex_set_b_symbol_count": candidates.get("set_membership", {}).get("ex_set_b_symbol_count"),
        },
        "notes": [
            "No module (d) content is adjudicated here.",
            "Module (c) closure pins bytes and registers gate finding only.",
        ],
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text("\n".join(["# R14-C Module (c) Closure Pin v1", "", json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), ""]), encoding="utf-8")

    print("R14C_MODULE_C_CLOSURE_PIN_V1_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()

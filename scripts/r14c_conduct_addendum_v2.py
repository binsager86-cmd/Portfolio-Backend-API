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
    prior = read_json(REVIEW / "r14b_conduct_addendum_v1.json")

    out_json = REVIEW / "r14c_conduct_addendum_v2.json"
    out_md = REVIEW / "r14c_conduct_addendum_v2.md"

    incidents = [
        {
            "incident_id": "ADD-4",
            "severity": "HIGH/EVIDENCE_INTEGRITY",
            "fact": (
                "Base-invalidation rule was iteratively modified with SANAM May-2025 owner-window outcome as explicit target "
                "(conversion of acceptance test into fitted result in-sample)."
            ),
            "classification": "RULE_FITTING_TO_OUTCOME",
            "evidence_scope": [
                "app/services/eagle_eye_v2/adaptive_base_geometry.py",
                "scripts/r14c_module_c_adaptive_base_geometry_harness_v2.py",
                "artifacts/preview1a_prestart/review_final/r14c_module_c_test_evidence_v3.json",
            ],
            "notes": "Facts only. R15-graded owner window context referenced; no adjudication.",
        },
        {
            "incident_id": "ADD-5",
            "severity": "LOW/PROCESS",
            "fact": "Version bump was performed by file-copy operation from harness v1 to v2 before subsequent edits.",
            "classification": "VERSION_BUMP_BY_COPY",
            "evidence_scope": [
                "scripts/r14c_module_c_adaptive_base_geometry_harness_v1.py",
                "scripts/r14c_module_c_adaptive_base_geometry_harness_v2.py",
            ],
        },
        {
            "incident_id": "ADD-6",
            "severity": "LOW/PROCESS",
            "fact": "Artifact-name collision occurred: harness v2 script emitted _v3-named output artifacts.",
            "classification": "ARTIFACT_NAME_COLLISION",
            "evidence_scope": [
                "scripts/r14c_module_c_adaptive_base_geometry_harness_v2.py",
                "artifacts/preview1a_prestart/review_final/r14c_module_c_test_evidence_v3.json",
                "artifacts/preview1a_prestart/review_final/r14c_module_c_harness_output_v3.log",
            ],
        },
        {
            "incident_id": "ADD-7",
            "severity": "HIGH/PERMANENT_SCRIPT_RULE",
            "fact": "Two further inline probe executions occurred in this remediation cycle (recurrence delta +2).",
            "classification": "INLINE_PROBE_RECURRENCE",
            "evidence_scope": [
                "terminal_session_context",
            ],
            "recurrence_count_delta": 2,
        },
    ]

    payload = {
        "version_id": "R14C_CONDUCT_ADDENDUM_V2",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "facts_only": True,
        "no_self_adjudication": True,
        "base_assessment_reference": "r14b_conduct_addendum_v1.json",
        "base_assessment_version": prior.get("version_id"),
        "incidents_appended": incidents,
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text("\n".join(["# R14-C Conduct Addendum v2", "", json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), ""]), encoding="utf-8")

    print("R14C_CONDUCT_ADDENDUM_V2_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()

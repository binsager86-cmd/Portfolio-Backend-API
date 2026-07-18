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
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    prior = read_json(REVIEW / "r14b_conduct_ledger_entry_5_v1.json")

    out_json = REVIEW / "r14b_conduct_addendum_v1.json"
    out_md = REVIEW / "r14b_conduct_addendum_v1.md"

    incidents = [
        {
            "incident_id": "ADD-1",
            "severity": "CRITICAL/EVIDENCE_INTEGRITY",
            "fact": "Synthetic evidence rows were persisted to telemetry ledger during module (b) v2 seam surfacing cycle.",
            "evidence_scope": [
                "r14b_module_b_test_evidence_v2.json",
                "r14b_module_b_implementation_report_v2.md",
            ],
        },
        {
            "incident_id": "ADD-2",
            "severity": "CRITICAL/SURFACE_INTEGRITY",
            "fact": "Canonical surface ee_v2_runtime_surface_r15_v1.db became contaminated and was revoked.",
            "evidence_scope": [
                "r15_surface_binding_v1.json",
                "r15_surface_revocation_v1.json",
            ],
        },
        {
            "incident_id": "ADD-3",
            "severity": "MODERATE/REPRODUCIBILITY",
            "fact": "In-place edit occurred on sealed generator r14b_module_a_write_path_harness_v1.py.",
            "evidence_scope": [
                "r14b_module_a_v1_defect_note_v1.json",
                "r14b_module_a_write_path_harness_v1.py",
            ],
        },
    ]

    payload = {
        "version_id": "R14B_CONDUCT_ADDENDUM_V1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evidence_scope_extension_only": True,
        "no_self_adjudication": True,
        "base_assessment_reference": "r14b_conduct_ledger_entry_5_v1.json",
        "base_assessment_version": prior.get("version_id"),
        "incidents_appended": incidents,
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text("\n".join(["# R14-B Conduct Addendum v1", "", json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), ""]), encoding="utf-8")

    print("R14B_CONDUCT_ADDENDUM_V1_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()

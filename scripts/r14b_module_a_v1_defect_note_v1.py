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
    out_json = REVIEW / "r14b_module_a_v1_defect_note_v1.json"
    out_md = REVIEW / "r14b_module_a_v1_defect_note_v1.md"

    script_path = ROOT / "scripts" / "r14b_module_a_write_path_harness_v1.py"
    evidence_path = REVIEW / "r14b_module_a_test_evidence_v1.json"

    evidence = read_json(evidence_path)

    payload = {
        "version_id": "R14B_MODULE_A_V1_DEFECT_NOTE_V1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frozen_history_notice": "r14b_module_a_write_path_harness_v1.py was edited in place after initial seal; do not revert history.",
        "affected_artifacts": [
            "r14b_module_a_implementation_report_v1.md",
            "r14b_module_a_interface_conformance_v1.json",
            "r14b_module_a_test_evidence_v1.json",
        ],
        "defect_record": {
            "classification": "MODERATE/REPRODUCIBILITY",
            "fact": "In-place edit occurred on a previously sealed generator script (v1).",
            "script_path": str(script_path.relative_to(ROOT)).replace("\\", "/"),
            "script_sha256_current": sha256_file(script_path),
            "v1_evidence_version_id": evidence.get("version_id"),
            "v1_harness_lines": evidence.get("harness_output_lines", []),
        },
        "remediation_instruction": "Supersede with r14b_module_a_write_path_harness_v2.py and emit fresh _v2 boundary artifacts.",
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text("\n".join(["# R14-B Module (a) v1 Defect Note v1", "", json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), ""]), encoding="utf-8")

    print("R14B_MODULE_A_V1_DEFECT_NOTE_V1_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()

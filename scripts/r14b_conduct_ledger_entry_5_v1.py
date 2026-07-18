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
    REVIEW.mkdir(parents=True, exist_ok=True)

    conditional_v1 = read_json(REVIEW / "r14b_module_b_conditional_review_v1.json")
    design_v22 = read_json(REVIEW / "r14_design_spec_CONSOLIDATED_v2_2.json")
    f8_forensic = read_json(REVIEW / "r13_f8_forensic_v1.json")

    out_json = REVIEW / "r14b_conduct_ledger_entry_5_v1.json"
    out_md = REVIEW / "r14b_conduct_ledger_entry_5_v1.md"

    rule_text = "ALL executed scripts must be permanent under scripts/ and sealed in manifest lineage; read-only probes are not exempt."

    violation_history = [
        {
            "entry": "#1",
            "classification": "PERMANENT_SCRIPT_RULE",
            "severity": "MEDIUM",
            "evidence": {
                "artifact": "r13_findings_of_record_v1.md",
                "statement": "Temp script usage occurred in prior report-only surfacing runs; permanent-script rule now extends to all executed scripts.",
            },
        },
        {
            "entry": "#2",
            "classification": "PERMANENT_SCRIPT_RULE",
            "severity": "MEDIUM",
            "evidence": {
                "artifact": "r13_f8_forensic_v1.json",
                "statement": str(f8_forensic.get("violation_acknowledgement", {}).get("violation") or ""),
                "repair": str(f8_forensic.get("violation_acknowledgement", {}).get("repair") or ""),
            },
        },
        {
            "entry": "#3",
            "classification": "PERMANENT_SCRIPT_RULE",
            "severity": "HIGH",
            "evidence": {
                "artifact": "r13_findings_of_record_v1_4.md",
                "statement": "Third permanent-script violation acknowledged: prior cycle used deleted temp probe scripts despite the permanent-script rule already being in force.",
            },
        },
        {
            "entry": "#4",
            "classification": "LINEAGE_APPEND_ONLY",
            "severity": "HIGH",
            "evidence": {
                "artifact": "r14_design_spec_CONSOLIDATED_v2_2.json",
                "statement": str(design_v22.get("lineage_repair_v2_2", {}).get("statement") or ""),
            },
        },
        {
            "entry": "#5",
            "classification": "PERMANENT_SCRIPT_RULE",
            "severity": "HIGH",
            "evidence": {
                "artifact": "r14b_module_b_conditional_review_v1.json",
                "statement": str(conditional_v1.get("conduct_ledger_pending", {}).get("record") or ""),
                "mitigation_facts_verbatim": str(conditional_v1.get("conduct_ledger_pending", {}).get("mitigation") or ""),
            },
        },
    ]

    suitability_assessment = {
        "purpose": "Owner review deliverable only; evidence and control proposals, not adjudication.",
        "scope": "Agent-suitability review initiation after conduct ledger entry #5 owner ruling (a).",
        "no_conclusion_asserted": True,
        "severity_scale": {
            "LOW": "Documentation or process clarity gap without policy breach.",
            "MEDIUM": "Single policy breach with bounded blast radius and reversible evidence impact.",
            "HIGH": "Repeated or governance-critical breach with lineage/reproducibility risk.",
        },
        "compensating_control_proposals_for_exam_class_phases": [
            {
                "control_id": "CC-1",
                "name": "Execution-Allowlist Gate",
                "proposal": "Block non-scripts/* Python execution in exam-class phases via wrapper policy and preflight check.",
            },
            {
                "control_id": "CC-2",
                "name": "Manifest-Linked Script Registry",
                "proposal": "Require each executable script to declare expected artifact outputs and emit self-hash to append-only run manifest.",
            },
            {
                "control_id": "CC-3",
                "name": "Conduct Delta Check",
                "proposal": "Before phase-gate decisions, auto-surface conduct ledger deltas since prior gate and require explicit owner disposition.",
            },
            {
                "control_id": "CC-4",
                "name": "Surface Binding Guard",
                "proposal": "Fail fast if EE_V2 writes resolve to dev_portfolio.db or any unbound runtime DB during exam-class runs.",
            },
        ],
    }

    payload = {
        "version_id": "R14B_CONDUCT_LEDGER_ENTRY_5_V1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "owner_ruling_applied": "(a) record as entry #5 and run suitability review",
        "rule_text": rule_text,
        "entry_5": {
            "record": str(conditional_v1.get("conduct_ledger_pending", {}).get("record") or ""),
            "mitigation_facts_verbatim": str(conditional_v1.get("conduct_ledger_pending", {}).get("mitigation") or ""),
            "source_artifact": "r14b_module_b_conditional_review_v1.json",
        },
        "violation_history_all_5": violation_history,
        "agent_suitability_review_assessment": suitability_assessment,
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(
        "\n".join(
            [
                "# R14-B Conduct Ledger Entry #5 v1",
                "",
                json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("R14B_CONDUCT_LEDGER_ENTRY_5_V1_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()

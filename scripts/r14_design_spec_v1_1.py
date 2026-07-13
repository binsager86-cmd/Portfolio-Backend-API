from __future__ import annotations

import hashlib
import json
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
    spec = read_json(REVIEW / "r14_design_spec_v1.json")
    probe = read_json(REVIEW / "r13_f8_forensic_v1_1.json")

    spec["version_id"] = "R14_DESIGN_SPEC_V1_1"
    spec["supersedes"] = "R14_DESIGN_SPEC_V1"
    spec["architecture_blueprint"]["base_module"] = "Proposal B adaptive volatility-regime-aware base geometry with advancing current-valid base references"
    spec["architecture_blueprint"]["skeleton"] = "Proposal C stateful lifecycle, deferred intent, and full daily predicate telemetry"
    spec["finding_response_map"]["F8a"] = "Solved by persistent base references plus readiness-aware base freeze so missing-reference disarm cannot persist silently."
    spec["finding_response_map"]["F8b"] = "Solved by advancing current-valid references during confirmed accumulation and referencing chase guard to the current valid reference, not the original freeze."
    spec["finding_response_map"]["F8c"] = "Not established by sealed evidence; however, any remaining veto-capable post-mandatory authority must be fully telemetried as named predicates if retained."
    spec["state_machine"]["named_predicate_terms"]["lifecycle"] = spec["state_machine"]["named_predicate_terms"]["lifecycle"] + ["BASE_REFERENCE_ADVANCE_OK", "CHASE_GUARD_CURRENT_REF_OK"]
    spec["telemetry_schema"]["daily_term_row"] = spec["telemetry_schema"]["daily_term_row"] + ["base_reference_version", "base_reference_origin", "base_reference_current_flag"]
    spec["r15_acceptance_criteria"]["SANAM"] = spec["r15_acceptance_criteria"]["SANAM"] + [
        "The 2025-05-18 session specifically must produce BREAKOUT_CONFIRMED or an explicit, fully-telemetried veto naming its blocking term.",
        "No stale original-freeze chase guard may block a day where the current valid reference has advanced through confirmed accumulation.",
    ]
    spec["r15_acceptance_criteria"]["TIJARA"] = spec["r15_acceptance_criteria"]["TIJARA"] + [
        "At least one 2025 high-volume cluster must operate with an explicit valid base reference rather than unresolved M1 disarm.",
    ]
    spec["governing_constraints"]["unauditable_blocking_authority_prohibited"] = True
    conclusion = probe["probe_2025_05_18"]["narrowed_conclusion"]
    summary = conclusion.get("residual_uncertainty")
    if not summary:
        summary = f"Identified blocker on 2025-05-18: {conclusion.get('identified_blocker')}"
    spec["final_probe_note"] = {
        "source": "artifacts/preview1a_prestart/review_final/r13_f8_forensic_v1_1.json",
        "summary": summary,
        "design_response": "Advance references during confirmed accumulation and require full telemetry for any retained blocking layer."
    }

    md = []
    md.append("# R14 Design Spec v1.1")
    md.append("")
    md.append("Supersedes: R14 Design Spec v1")
    md.append("")
    md.append("Amendment focus:")
    md.append("- Explicitly answer F8b by making reference advancement during confirmed accumulation a first-class lifecycle requirement.")
    md.append("- Require chase guard evaluation against the current valid reference, not the original freeze reference.")
    md.append("- The 2025-05-18 final probe resolved the surfaced blocker to M5_liquidity, not a composite/ML layer; F8c is therefore not established as a new mechanism.")
    md.append("- Any veto-capable post-mandatory layer must still be fully telemetried if retained, and liquidity authority must remain explicit in telemetry and acceptance criteria.")
    md.append("")
    md.append("New/strengthened design principles:")
    md.append("- Base-reference lifecycle includes freeze, ratchet/advance, invalidate, retire, and current-valid-reference designation.")
    md.append("- Chase guard consumes the current valid reference only.")
    md.append("- Deferred intent and confirmation share a common reference object so confirmation latency cannot structurally create an unwinnable race.")
    md.append("- No unauditable gate may retain blocking authority.")
    md.append("")
    md.append("State-machine additions:")
    md.append(f"- lifecycle terms: {spec['state_machine']['named_predicate_terms']['lifecycle']}")
    md.append("")
    md.append("Telemetry additions:")
    md.append(f"- daily_term_row: {spec['telemetry_schema']['daily_term_row']}")
    md.append("")
    md.append("Updated finding-response map:")
    for k in ['F8a','F8b','F8c']:
        md.append(f"- {k}: {spec['finding_response_map'][k]}")
    md.append("")
    md.append("Final-probe note:")
    md.append(f"- {spec['final_probe_note']['summary']}")
    md.append("")
    md.append("Updated R15 acceptance criteria:")
    for sym in ['SANAM','TIJARA']:
        md.append(f"- {sym}: {spec['r15_acceptance_criteria'][sym]}")
    md.append("")
    md.append("R14-B and R15 remain NOT AUTHORIZED.")
    md.append("")

    out_json = REVIEW / "r14_design_spec_v1_1.json"
    out_md = REVIEW / "r14_design_spec_v1_1.md"
    out_json.write_text(json.dumps(spec, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text("\n".join(md), encoding="utf-8")
    print("R14_DESIGN_SPEC_V1_1_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()

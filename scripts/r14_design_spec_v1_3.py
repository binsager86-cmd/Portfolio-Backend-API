from __future__ import annotations

import hashlib
import json
from pathlib import Path

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


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    spec = read_json(REVIEW / "r14_design_spec_v1_2.json")
    m5 = read_json(REVIEW / "r13_m5_liquidity_forensic_v1.json")
    volume = read_json(REVIEW / "r13_volume_arrival_audit_v1.json")
    d1 = read_json(REVIEW / "r13_set_a_causal_attribution_v3.json")

    spec["version_id"] = "R14_DESIGN_SPEC_V1_3"
    spec["supersedes"] = "R14_DESIGN_SPEC_V1_2"
    spec["architecture_blueprint"]["skeleton"] = "Proposal C stateful lifecycle, deferred/early intent, and full daily predicate telemetry"
    spec["architecture_blueprint"]["confirmation_core"] = "Proposal A accumulation-window flow confirmation with tolerant-with-advisory chase policy"
    spec["architecture_blueprint"]["early_tier"] = "Proposal A + Proposal B accumulation-stage early entry"
    spec["governing_constraints"]["early_tier_size_fraction_parameter_only"] = True
    spec["governing_constraints"]["early_tier_participation_cap_parameter_only"] = True
    spec["governing_constraints"]["early_tier_time_stop_parameter_only"] = True
    spec["governing_constraints"]["scale_on_confirmation"] = True
    spec["governing_constraints"]["early_tier_exempts_f1_and_f9_as_confirmation_vetoes"] = True

    spec["state_machine"]["states"] = spec["state_machine"]["states"] + ["EARLY_ACCUMULATION_ENTRY", "BREAKOUT_CONFIRMED_ENTRY"]
    spec["state_machine"]["named_predicate_terms"]["confirmation"] = spec["state_machine"]["named_predicate_terms"]["confirmation"] + ["EARLY_ENTRY_FLOW_OK", "EARLY_ENTRY_BASE_VALID_OK", "EARLY_ENTRY_PARTICIPATION_OK", "EARLY_ENTRY_TIME_STOP_OK", "DEAD_MONEY_TRACKING_OK"]
    spec["state_machine"]["named_predicate_terms"]["lifecycle"] = spec["state_machine"]["named_predicate_terms"]["lifecycle"] + ["EARLY_INTENT_ACTIVE", "EARLY_INTENT_SCALE_READY"]
    spec["telemetry_schema"]["daily_term_row"] = spec["telemetry_schema"]["daily_term_row"] + ["early_tier_flag", "dead_money_sessions", "flow_obv_slope_40", "flow_anv_slope_40", "flow_accumulation_divergence", "accumulation_context_ok", "participation_cap_pct", "pilot_size_fraction", "time_stop_sessions", "entry_tier", "flow_evidence_snapshot", "current_valid_reference_value", "extension_pct_vs_current_valid_reference", "chase_advisory_flag"]
    spec["telemetry_schema"]["execution_outcome_row"] = spec["telemetry_schema"]["execution_outcome_row"] + ["entry_tier", "dead_money_sessions", "chase_advisory_emitted", "chase_advisory_extension_pct"]

    spec["finding_response_map"]["EARLY_TIER"] = "Structural answer to the pre-volume edge: expose accumulation-stage entries using flow confirmation and adaptive base validity before breakout confirmation."
    spec["finding_response_map"]["F1"] = "Solved at confirmation tier; early tier intentionally bypasses same-day volume multiple dependence when flow confirmation and base validity are present."
    spec["finding_response_map"]["F9"] = "Solved at confirmation tier; early tier uses current and arriving liquidity as participation context, not as a sole veto, while preserving optional participation caps."

    spec["r15_acceptance_criteria"]["SANAM"] = spec["r15_acceptance_criteria"]["SANAM"] + [
        "SANAM must generate an EARLY_ACCUMULATION_ENTRY within its owner-window accumulation before 2025-05-08.",
        "The 2025-05-18 session must still produce a confirmed entry across all layers.",
        "The early-tier false-positive cost must be reported separately as count and aggregate P&L for early entries that hit time-stop without confirmation.",
    ]
    spec["r15_acceptance_criteria"]["TIJARA"] = spec["r15_acceptance_criteria"]["TIJARA"] + [
        "TIJARA must generate an EARLY_ACCUMULATION_ENTRY before its markup onset.",
        "The early-tier false-positive cost must be reported separately as count and aggregate P&L for early entries that hit time-stop without confirmation.",
    ]
    spec["r15_acceptance_criteria"]["MABANEE"] = spec["r15_acceptance_criteria"]["MABANEE"] + [
        "MABANEE must generate zero EARLY_ACCUMULATION_ENTRY events during the avoid-dominant decline.",
    ]

    early_windows = {
        "SANAM": [r for r in volume["rel_volume_ge_2_5"]["per_symbol_days"]["SANAM"] if r["date"] < "2025-05-08"],
        "TIJARA": [r for r in volume["rel_volume_ge_2_5"]["per_symbol_days"]["TIJARA"] if r["date"] < "2025-04-23"],
    }
    spec["early_tier_notes"] = {
        "SANAM_owner_window_before_breakout": len(early_windows["SANAM"]),
        "TIJARA_owner_window_before_markup": len(early_windows["TIJARA"]),
        "value_thesis": "Catch accumulation stage, not only confirmed breakouts.",
        "false_positive_cost_tracking": True,
        "source_artifacts": [
            "artifacts/preview1a_prestart/review_final/r13_volume_arrival_audit_v1.json",
            "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json",
            "artifacts/preview1a_prestart/review_final/r13_m5_liquidity_forensic_v1.json",
        ],
        "m5_context": m5["f9"],
    }

    md = []
    md.append("# R14 Design Spec v1.3")
    md.append("")
    md.append("Supersedes: R14 Design Spec v1.2")
    md.append("")
    md.append("Owner ruling:")
    md.append("- The architecture must catch the accumulation stage, not only confirmed breakouts.")
    md.append("")
    md.append("Two-tier entry model:")
    md.append("- EARLY_ACCUMULATION_ENTRY")
    md.append("- BREAKOUT_CONFIRMED_ENTRY")
    md.append("- Early tier is gated by flow-confirmation predicates over the accumulation window plus base validity from AdaptiveBaseGeometry.")
    md.append("- Early tier does not use same-day volume multiples or trailing-liquidity vetoes as entry gates by design.")
    md.append("")
    md.append("Staged position policy:")
    md.append("- EARLY_TIER_SIZE_FRACTION")
    md.append("- EARLY_TIER_PARTICIPATION_CAP")
    md.append("- EARLY_TIER_TIME_STOP")
    md.append("- SCALE_ON_CONFIRMATION")
    md.append("")
    md.append("Authority rules:")
    md.append("- AvoidAuthorityPlane retains full veto over early entries.")
    md.append("- Early entries emit telemetry rows including flow evidence values at entry and DEAD_MONEY tracking.")
    md.append("")
    md.append("Lifecycle wiring:")
    md.append("- Early entry and deferred intent are one mechanism at two trigger points.")
    md.append("- Base-reference ratcheting applies from early entry onward.")
    md.append("")
    md.append("Updated R15 acceptance criteria:")
    for sym in ["SANAM", "TIJARA", "MABANEE"]:
        md.append(f"- {sym}: {spec['r15_acceptance_criteria'][sym]}")
    md.append("")
    md.append("False-positive cost:")
    md.append("- Count and aggregate P&L of early entries that hit time-stop without confirmation must be reported separately.")
    md.append("")
    md.append("Finding-response map:")
    md.append(f"- EARLY_TIER: {spec['finding_response_map']['EARLY_TIER']}")
    md.append(f"- F1: {spec['finding_response_map']['F1']}")
    md.append(f"- F9: {spec['finding_response_map']['F9']}")
    md.append("")
    md.append("Early-tier notes:")
    md.append(json.dumps(spec["early_tier_notes"], ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")
    md.append("R14-B and R15 remain NOT AUTHORIZED.")
    md.append("")

    out_json = REVIEW / "r14_design_spec_v1_3.json"
    out_md = REVIEW / "r14_design_spec_v1_3.md"
    out_json.write_text(json.dumps(spec, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text("\n".join(md), encoding="utf-8")
    print("R14_DESIGN_SPEC_V1_3_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()

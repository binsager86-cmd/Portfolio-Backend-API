from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def uniq_list(items: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for item in items:
        key = json.dumps(item, ensure_ascii=True, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def dedupe_text_list(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        k = item.strip()
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


def build_parameter_registry() -> list[dict[str, Any]]:
    return [
        {"name": "base_min_sessions", "module": "AdaptiveBaseGeometry", "governs": "minimum sessions required for eligible base construction", "owner_decision_status": "FROZEN_AT_R14B"},
        {"name": "base_max_width_pct", "module": "AdaptiveBaseGeometry", "governs": "base width gating envelope", "owner_decision_status": "FROZEN_AT_R14B"},
        {"name": "cmf_floor", "module": "FlowConfirmationEngine", "governs": "minimum CMF evidence component", "owner_decision_status": "FROZEN_AT_R14B"},
        {"name": "atr_squeeze_pctile", "module": "AdaptiveBaseGeometry", "governs": "squeeze/regime entry criterion", "owner_decision_status": "FROZEN_AT_R14B"},
        {"name": "volume_breakout_mult", "module": "FlowConfirmationEngine", "governs": "breakout confirmation relative-volume multiple", "owner_decision_status": "FROZEN_AT_R14B"},
        {"name": "rsi_regime", "module": "FlowConfirmationEngine", "governs": "RSI regime confirmation threshold", "owner_decision_status": "FROZEN_AT_R14B"},
        {"name": "adx_trigger", "module": "FlowConfirmationEngine", "governs": "ADX trend-strength confirmation threshold", "owner_decision_status": "FROZEN_AT_R14B"},
        {"name": "trend_join_window", "module": "LifecycleIntentRouter", "governs": "window for trend-join continuation logic", "owner_decision_status": "FROZEN_AT_R14B"},
        {"name": "min_daily_value_kwd", "module": "ExecutionLiquidityAssessment", "governs": "baseline trailing liquidity context threshold", "owner_decision_status": "FROZEN_AT_R14B"},
        {"name": "ml_prob_min", "module": "FlowConfirmationEngine", "governs": "optional ML probability floor if ML gate enabled", "owner_decision_status": "FROZEN_AT_R14B"},
        {"name": "EARLY_TIER_SIZE_FRACTION", "module": "StagedPositionPolicy", "governs": "pilot size fraction for EARLY_ACCUMULATION_ENTRY", "owner_decision_status": "PENDING_OWNER_RATIFICATION_AT_R14B"},
        {"name": "EARLY_TIER_PARTICIPATION_CAP", "module": "StagedPositionPolicy", "governs": "max percent of daily traded value for early-tier entry", "owner_decision_status": "PENDING_OWNER_RATIFICATION_AT_R14B"},
        {"name": "EARLY_TIER_TIME_STOP", "module": "StagedPositionPolicy", "governs": "sessions allowed pre-confirmation before exit/review", "owner_decision_status": "PENDING_OWNER_RATIFICATION_AT_R14B"},
        {"name": "SCALE_ON_CONFIRMATION", "module": "StagedPositionPolicy", "governs": "add-on behavior at BREAKOUT_CONFIRMED_ENTRY", "owner_decision_status": "PENDING_OWNER_RATIFICATION_AT_R14B"},
        {"name": "CHASE_ADVISORY_BAND", "module": "FlowConfirmationEngine", "governs": "advisory extension band under TOLERANT_WITH_ADVISORY chase policy", "owner_decision_status": "PENDING_OWNER_RATIFICATION_AT_R14B"},
        {"name": "LIQUIDITY_EXECUTION_SIZE_PARAMETER", "module": "ExecutionLiquidityAssessment", "governs": "execution-size denominator for current/arriving liquidity sufficiency", "owner_decision_status": "PENDING_OWNER_RATIFICATION_AT_R14B"},
        {"name": "TIER_RULE_CANDIDATE_A", "module": "TierRule", "governs": "tier assignment by median_daily_value_traded_kwd thresholds", "owner_decision_status": "PENDING_OWNER_RATIFICATION_AT_R14B"},
        {"name": "TIER_RULE_CANDIDATE_B", "module": "TierRule", "governs": "tier assignment by cross-universe tercile definition", "owner_decision_status": "PENDING_OWNER_RATIFICATION_AT_R14B"},
    ]


def build_readiness_checklist() -> dict[str, Any]:
    return {
        "owner_required_for_r14b": [
            "Ratify all pending named-parameter values in parameter registry.",
            "Ratify tier rule candidate definition (A or B).",
            "Ratify early-tier participation and time-stop policy values.",
            "Ratify chase advisory band semantics and escalation policy.",
        ],
        "governance_required_for_r14b": [
            "Issue new versioned baseline ID for implementation branch.",
            "Freeze semantics: no parameter mutation after R14-B freeze without explicit change request and reseal.",
            "Restate Set B quarantine during implementation and dry-run validation.",
            "Require permanent-script-only execution and manifest sealing for all producing runs.",
        ],
        "r15_gate_conditions": [
            "All acceptance criteria per symbol evaluated on sealed runtime surface.",
            "Early-tier false-positive cost reported separately (count and aggregate P&L).",
            "No criterion marked pass without corresponding state/telemetry evidence rows.",
            "Set B remains excluded until R15 gate completion and owner ratification.",
        ],
    }


def build_supersession_table() -> list[dict[str, Any]]:
    return [
        {"amendment": "v1", "contribution": "Initial unified design skeleton (Proposal C) + confirmation core (Proposal A) + adaptive geometry (Proposal B); base module/interfaces/state machine/telemetry foundation."},
        {"amendment": "v1.1", "contribution": "F8b race response: advancing current-valid references and chase guard anchored to current valid reference; strengthened telemetry semantics."},
        {"amendment": "v1.2", "contribution": "CHASE_POLICY=TOLERANT_WITH_ADVISORY and liquidity principle: current/arriving weighted, trailing as context-only; F9 response and related R15 additions."},
        {"amendment": "v1.3", "contribution": "Early-stage entry requirement: EARLY_ACCUMULATION_ENTRY tier, staged-position policy, early/deferred intent unification, and early-tier R15 criteria."},
    ]


def build_consistency_gate(spec: dict[str, Any]) -> dict[str, Any]:
    states = set(spec.get("state_machine", {}).get("states", []))
    ns = spec.get("state_machine", {}).get("named_predicate_terms", {})
    predicates = set()
    for vals in ns.values():
        for p in vals:
            predicates.add(p)

    predicate_state_requirements = {
        "READINESS": {"READINESS_PENDING", "READINESS_LIMITED"},
        "WATCH_": {"BREAKOUT_WATCH"},
        "CONFIRM_": {"BREAKOUT_WATCH", "BREAKOUT_CONFIRMED_ENTRY"},
        "EARLY_ENTRY_": {"EARLY_ACCUMULATION_ENTRY"},
        "DEFERRED_": {"DEFERRED_INTENT"},
        "AVOID_": {"AVOID"},
        "BASE_REFERENCE_": {"BASE_FORMING", "ACCUMULATION", "BREAKOUT_WATCH", "EARLY_ACCUMULATION_ENTRY", "DEFERRED_INTENT"},
    }

    state_ref_checks = []
    missing_states = set()
    for p in sorted(predicates):
        required: set[str] = set()
        for key, req in predicate_state_requirements.items():
            if p.startswith(key):
                required |= req
        if required:
            miss = sorted([s for s in required if s not in states])
            if miss:
                missing_states |= set(miss)
            state_ref_checks.append({"predicate": p, "required_states": sorted(required), "missing_states": miss, "pass": len(miss) == 0})

    finding_map = spec.get("finding_response_map", {})
    finder_missing = []
    allow_tokens = {
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F8A", "F8B", "F8C", "F9", "R15", "R14", "EARLY", "TIER", "ML", "PANDL"
    }
    for fid, txt in finding_map.items():
        toks = set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", txt.upper()))
        for t in sorted(toks):
            if t in allow_tokens:
                continue
            if t in predicates:
                continue
            finder_missing.append({"finding": fid, "token": t})

    telemetry_fields = set()
    for fields in spec.get("telemetry_schema", {}).values():
        for f in fields:
            telemetry_fields.add(str(f))

    criterion_issues = []
    for sym, crits in spec.get("r15_acceptance_criteria", {}).items():
        for idx, c in enumerate(crits):
            up = c.upper()
            state_hits = [s for s in states if s in up]
            pred_hits = [p for p in predicates if p in up]
            field_hits = [f for f in telemetry_fields if f.upper() in up]
            # gate requires state or telemetry reference on each criterion
            if not state_hits and not field_hits:
                criterion_issues.append({"symbol": sym, "index": idx, "criterion": c, "issue": "No explicit state/telemetry reference detected"})
            # if predicate mentioned in text, it must exist
            toks = set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", up))
            for t in toks:
                if t.startswith("CONFIRM_") or t.startswith("EARLY_") or t.startswith("BASE_") or t.startswith("WATCH_") or t.startswith("DEFERRED_") or t.startswith("AVOID_"):
                    if t not in predicates:
                        criterion_issues.append({"symbol": sym, "index": idx, "criterion": c, "issue": f"Predicate token not defined: {t}"})

    return {
        "checks": {
            "predicate_state_reference": {
                "pass": len(missing_states) == 0,
                "missing_states": sorted(missing_states),
                "checked_predicates": len(state_ref_checks),
            },
            "finding_response_predicate_reference": {
                "pass": len(finder_missing) == 0,
                "missing_tokens": finder_missing,
            },
            "r15_criterion_reference": {
                "pass": len(criterion_issues) == 0,
                "issues": criterion_issues,
            },
        },
        "status": "PASS" if len(missing_states) == 0 and len(finder_missing) == 0 and len(criterion_issues) == 0 else "FAIL",
    }


def main() -> None:
    v1 = read_json(REVIEW / "r14_design_spec_v1.json")
    v11 = read_json(REVIEW / "r14_design_spec_v1_1.json")
    v12 = read_json(REVIEW / "r14_design_spec_v1_2.json")
    v13 = read_json(REVIEW / "r14_design_spec_v1_3.json")

    spec = copy.deepcopy(v13)
    spec["version_id"] = "R14_DESIGN_SPEC_CONSOLIDATED_V2"
    spec["supersedes"] = [
        "R14_DESIGN_SPEC_V1",
        "R14_DESIGN_SPEC_V1_1",
        "R14_DESIGN_SPEC_V1_2",
        "R14_DESIGN_SPEC_V1_3",
    ]
    spec["supersession_table"] = build_supersession_table()

    # Deduplicate lists that accumulated over amendments.
    spec["state_machine"]["states"] = dedupe_text_list(spec["state_machine"].get("states", []))
    for ns, vals in list(spec["state_machine"]["named_predicate_terms"].items()):
        spec["state_machine"]["named_predicate_terms"][ns] = dedupe_text_list(vals)

    for k in ["daily_term_row", "daily_state_snapshot", "execution_outcome_row"]:
        if k in spec.get("telemetry_schema", {}):
            spec["telemetry_schema"][k] = dedupe_text_list(spec["telemetry_schema"][k])

    for sym, criteria in list(spec.get("r15_acceptance_criteria", {}).items()):
        spec["r15_acceptance_criteria"][sym] = dedupe_text_list(criteria)

    spec["named_parameter_registry"] = build_parameter_registry()
    spec["r14_b_readiness_checklist"] = build_readiness_checklist()

    # Add explicit criteria references so gate can validate state/telemetry anchoring.
    explicit_ref_additions = {
        "SANAM": [
            "EARLY_ACCUMULATION_ENTRY and BREAKOUT_CONFIRMED_ENTRY states must both be evidenced by daily_state_snapshot rows for the owner-window sequence.",
            "dead_money_sessions telemetry must be present for all early-tier positions and reported in execution_outcome_row.",
        ],
        "TIJARA": [
            "EARLY_ACCUMULATION_ENTRY state transition must appear prior to markup onset with flow_evidence_snapshot telemetry persisted.",
        ],
        "MABANEE": [
            "No EARLY_ACCUMULATION_ENTRY state rows may appear while AVOID_CONDITION_ACTIVE is true.",
        ],
    }
    for sym, adds in explicit_ref_additions.items():
        current = spec["r15_acceptance_criteria"].get(sym, [])
        spec["r15_acceptance_criteria"][sym] = dedupe_text_list(current + adds)

    gate = build_consistency_gate(spec)
    spec["consistency_gate_output"] = gate

    out_json = REVIEW / "r14_design_spec_CONSOLIDATED_v2.json"
    out_md = REVIEW / "r14_design_spec_CONSOLIDATED_v2.md"

    md: list[str] = []
    md.append("# R14 Design Spec CONSOLIDATED v2")
    md.append("")
    md.append("Supersedes chain:")
    md.append("- R14_DESIGN_SPEC_V1")
    md.append("- R14_DESIGN_SPEC_V1_1")
    md.append("- R14_DESIGN_SPEC_V1_2")
    md.append("- R14_DESIGN_SPEC_V1_3")
    md.append("")
    md.append("## Supersession Table")
    for row in spec["supersession_table"]:
        md.append(f"- {row['amendment']}: {row['contribution']}")
    md.append("")

    md.append("## Architecture Blueprint")
    md.append(json.dumps(spec["architecture_blueprint"], ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")

    md.append("## Module Boundaries")
    md.append(json.dumps(spec["module_boundaries"], ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")

    md.append("## Interfaces")
    md.append(json.dumps(spec["interfaces"], ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")

    md.append("## State Machine")
    md.append(json.dumps(spec["state_machine"], ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")

    md.append("## Telemetry Schema")
    md.append(json.dumps(spec["telemetry_schema"], ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")

    md.append("## Two-Tier Entry Model")
    md.append("- Tier 1: EARLY_ACCUMULATION_ENTRY")
    md.append("- Tier 2: BREAKOUT_CONFIRMED_ENTRY")
    md.append("- Early tier trigger domain: FLOW_OBV_SLOPE_OK, FLOW_ANV_SLOPE_OK, FLOW_ACCUMULATION_DIVERGENCE_OK, ACCUMULATION_CONTEXT_OK, and base validity terms from AdaptiveBaseGeometry.")
    md.append("- Early tier is not gated by same-day volume multiple or trailing-liquidity sole-veto logic by design.")
    md.append("- Staged position parameters: EARLY_TIER_SIZE_FRACTION, EARLY_TIER_PARTICIPATION_CAP, EARLY_TIER_TIME_STOP, SCALE_ON_CONFIRMATION.")
    md.append("- Chase policy: TOLERANT_WITH_ADVISORY with advisory telemetry emission.")
    md.append("- Liquidity principle: current and arriving liquidity weighted directly; trailing baseline context-only.")
    md.append("- Base-reference ratcheting applies from early entry onward.")
    md.append("- Warmup readiness includes explicit fallback.")
    md.append("- AvoidAuthorityPlane retains full veto authority.")
    md.append("")

    md.append("## Finding-Response Map (F1-F9)")
    f_map = {k: spec["finding_response_map"][k] for k in ["F1","F2","F3","F4","F5","F6","F7","F8","F8a","F8b","F8c","F9"] if k in spec["finding_response_map"]}
    md.append(json.dumps(f_map, ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")

    md.append("## R15 Acceptance Criteria (Deduplicated)")
    md.append(json.dumps(spec["r15_acceptance_criteria"], ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")

    md.append("## Named-Parameter Registry")
    md.append(json.dumps(spec["named_parameter_registry"], ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")

    md.append("## R14-B Readiness Checklist")
    md.append(json.dumps(spec["r14_b_readiness_checklist"], ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")

    md.append("## Consistency Gate Output")
    md.append(json.dumps(gate, ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")
    md.append("R14-B and R15 remain NOT AUTHORIZED.")
    md.append("")

    out_json.write_text(json.dumps(spec, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text("\n".join(md), encoding="utf-8")

    print("R14_DESIGN_SPEC_CONSOLIDATED_V2_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))
    print("consistency_gate_status", gate["status"])


if __name__ == "__main__":
    main()

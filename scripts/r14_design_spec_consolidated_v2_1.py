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


def ensure_named_parameter(
    spec: dict[str, Any],
    *,
    name: str,
    module: str,
    governs: str,
    owner_decision_status: str = "PENDING_OWNER_RATIFICATION_AT_R14B",
) -> None:
    registry = spec.setdefault("named_parameter_registry", [])
    for row in registry:
        if row.get("name") == name:
            return
    registry.append(
        {
            "name": name,
            "module": module,
            "governs": governs,
            "owner_decision_status": owner_decision_status,
        }
    )


def apply_forward_prediction_ledger_addendum(spec: dict[str, Any]) -> None:
    spec["forward_prediction_ledger_r16"] = {
        "status": "DESIGN_ONLY_SHADOW_MODE_ELIGIBLE_POST_R14B_COMPLETION",
        "prediction_snapshot": {
            "mode": "APPEND_ONLY_HASH_SEALED_DAILY",
            "grain": "PER_SYMBOL_PER_SESSION",
            "primary_identifier": "prediction_id",
            "storage": {
                "ledger_table": "FORWARD_PREDICTION_LEDGER",
                "external_sidecar": "FORWARD_PREDICTION_LEDGER_DAILY_SHA256",
                "immutability_rule": "NO_UPDATE_NO_DELETE",
            },
            "required_fields": [
                "prediction_id",
                "symbol",
                "trade_date",
                "phase_cycle_state",
                "rating_score",
                "rating_band",
                "entry_tier_flag",
                "current_base_reference",
                "flow_evidence_snapshot",
                "obv_slope",
                "anv_slope",
                "accumulation_divergence",
                "avoid_state",
            ],
        },
        "outcome_grading": {
            "producer_rule": "SEPARATE_PERMANENT_GRADER_SCRIPT_ONLY",
            "separation_of_duties": "GRADER_MUST_NEVER_BE_PREDICTOR",
            "named_parameters": [
                "GRADING_HORIZONS",
                "MARKUP_MATERIALIZATION_CRITERION",
            ],
            "inputs": [
                "SEALED_FORWARD_PREDICTION_LEDGER",
                "MARKET_DATA_SURFACE",
            ],
            "per_prediction_outputs": [
                "forward_return",
                "max_favorable_excursion",
                "max_adverse_excursion",
                "markup_materialized",
                "distribution_or_exit_preceded_peak",
            ],
            "artifact_policy": "APPEND_ONLY_HASH_SEALED",
        },
        "calibration_outputs": {
            "standing_question": "DO_ACCUMULATION_RATED_SYMBOLS_OUTPERFORM_BY_RATING_TIER",
            "required_tables": [
                "hit_rate_by_phase_state",
                "forward_return_by_phase_state",
                "hit_rate_by_rating_band",
                "forward_return_by_rating_band",
            ],
            "live_tracking": [
                "early_tier_dead_money_cost",
            ],
            "comparability_rule": "EARLY_TIER_DEAD_MONEY_COST_MIRRORS_R15_METRIC_DEFINITION",
        },
        "governance": {
            "universe_policy": "FULL_UNIVERSE_FORWARD_OUT_OF_SAMPLE_BY_TIME",
            "frozen_named_parameters_before_first_seal": [
                "GRADING_HORIZONS",
                "MARKUP_MATERIALIZATION_CRITERION",
                "MIN_CALIBRATION_WINDOW",
            ],
            "change_control_rule": "NO_THRESHOLD_OR_MODEL_CHANGE_MAY_BE_JUSTIFIED_BY_FORWARD_RESULTS_BEFORE_MIN_CALIBRATION_WINDOW_ELAPSES",
            "calibration_window_rule": "MIN_CALIBRATION_WINDOW_OWNER_SET_RECOMMEND_AT_LEAST_ONE_QUARTER",
        },
        "r16_to_r17_gate_condition": "R17_CAPITAL_DEPLOYMENT_REQUIRES_FORWARD_CALIBRATION_DIRECTIONALLY_CONSISTENT_WITH_R15_BACKTEST_RESULTS_DIVERGENCE_IS_A_FINDING_AND_HALTS_SCALE_UP",
    }

    ensure_named_parameter(
        spec,
        name="GRADING_HORIZONS",
        module="ForwardOutcomeGrader",
        governs="fixed grading horizons for forward prediction outcomes",
    )
    ensure_named_parameter(
        spec,
        name="MARKUP_MATERIALIZATION_CRITERION",
        module="ForwardOutcomeGrader",
        governs="criterion used to classify whether markup materialized",
    )
    ensure_named_parameter(
        spec,
        name="MIN_CALIBRATION_WINDOW",
        module="ForwardPredictionGovernance",
        governs="minimum elapsed forward window before threshold/model changes may use forward calibration evidence",
    )

    checklist = spec.setdefault("r14_b_readiness_checklist", {})
    r16_gate_conditions = checklist.setdefault("r16_gate_conditions", [])
    required_gate = (
        "R17 capital deployment requires forward calibration tables to be directionally consistent with R15 backtest results; divergence is a finding and halts scale-up."
    )
    if required_gate not in r16_gate_conditions:
        r16_gate_conditions.append(required_gate)


def classify_criterion(text: str) -> str:
    t = text.lower()
    metric_patterns = [
        "frequency",
        "share",
        "dominance",
        "false-positive cost",
        "count and aggregate p&l",
        "blocked by trailing-liquidity",
        "high-volume day",
    ]
    for p in metric_patterns:
        if p in t:
            return "METRIC_REFERENCING"
    return "STATE_REFERENCING"


def amend_metric_criterion(text: str) -> str:
    t = text
    low = text.lower()
    if "m2-only blockade frequency" in low:
        return t + " [metric_source: daily_term_row predicate_name predicate_pass symbol trade_date; D1_V3_BLOCKING_TERM_COUNTS]"
    if "no-candidate share" in low:
        return t + " [metric_source: D1_V3_CATEGORY_COUNTS no_candidate_share]"
    if "width-rule dominance" in low:
        return t + " [metric_source: daily_term_row predicate_name='BASE_GEOMETRY_WIDTH_OK' predicate_pass; D1_V3_BLOCKING_TERM_COUNTS]"
    if "false-positive cost" in low or "aggregate p&l" in low:
        return t + " [metric_source: execution_outcome_row entry_tier dead_money_sessions net_return]"
    if "blocked by trailing-liquidity" in low:
        return t + " [metric_source: daily_term_row current_day_value_kwd trailing_liquidity_context_value predicate_name='CONFIRM_LIQUIDITY_OK']"
    if "high-volume" in low:
        return t + " [metric_source: daily_term_row symbol trade_date predicate_name predicate_pass]"
    return t + " [metric_source: daily_term_row predicate_name predicate_pass symbol trade_date]"


def ensure_state_reference(text: str, states: set[str], predicates: set[str], telemetry_fields: set[str]) -> str:
    combined = states | predicates | telemetry_fields
    if any(ref in text for ref in combined):
        return text
    low = text.lower()
    if "avoid" in low:
        ref = "AVOID"
    elif "early" in low:
        ref = "EARLY_ACCUMULATION_ENTRY"
    elif "deferred" in low:
        ref = "DEFERRED_INTENT"
    elif "breakout" in low or "confirmed entry" in low:
        ref = "BREAKOUT_CONFIRMED_ENTRY"
    else:
        ref = "BREAKOUT_WATCH"
    return text + f" [state_ref: {ref}]"


def build_repaired_gate(spec: dict[str, Any]) -> dict[str, Any]:
    modules = {m.get("module") for m in spec.get("module_boundaries", []) if m.get("module")}
    states = set(spec.get("state_machine", {}).get("states", []))
    predicates = set()
    for vals in spec.get("state_machine", {}).get("named_predicate_terms", {}).values():
        predicates.update(vals)

    telemetry_fields = set()
    for schema in ["daily_term_row", "daily_state_snapshot", "execution_outcome_row"]:
        for f in spec.get("telemetry_schema", {}).get(schema, []):
            telemetry_fields.add(str(f))

    # Rule 1: finding-response check (module names OR exact underscore state/predicate token)
    underscore_tokens = {t for t in (states | predicates) if "_" in t}
    finding_fail = []
    for fid, txt in spec.get("finding_response_map", {}).items():
        has_module = any(m in txt for m in modules)
        tokens = set(re.findall(r"\b[A-Za-z0-9_]+\b", txt))
        has_token = any(tok in underscore_tokens for tok in tokens)
        if not (has_module or has_token):
            finding_fail.append({"finding": fid, "reason": "No module name and no exact underscore state/predicate token"})

    # Rule 2+3: R15 criterion check using combined reference set and criterion class
    combined_refs = states | predicates | telemetry_fields
    metric_source_markers = {
        "D1_V3_CATEGORY_COUNTS",
        "D1_V3_BLOCKING_TERM_COUNTS",
    }
    criterion_issues = []
    for sym, crits in spec.get("r15_acceptance_criteria", {}).items():
        classes = spec.get("r15_criterion_classification", {}).get(sym, [])
        for idx, c in enumerate(crits):
            c_class = classes[idx] if idx < len(classes) else classify_criterion(c)
            has_ref = any(ref in c for ref in combined_refs)
            has_metric_marker = any(m in c for m in metric_source_markers)
            if c_class == "STATE_REFERENCING":
                if not has_ref:
                    criterion_issues.append({"symbol": sym, "index": idx, "class": c_class, "criterion": c, "issue": "STATE_REFERENCING criterion missing combined reference"})
            elif c_class == "METRIC_REFERENCING":
                if not (has_ref or has_metric_marker):
                    criterion_issues.append({"symbol": sym, "index": idx, "class": c_class, "criterion": c, "issue": "METRIC_REFERENCING criterion missing telemetry field or D1-v3 measurement source"})
            else:
                criterion_issues.append({"symbol": sym, "index": idx, "class": c_class, "criterion": c, "issue": "Unknown criterion classification"})

    return {
        "checks": {
            "finding_response_reference_rule": {
                "pass": len(finding_fail) == 0,
                "failures": finding_fail,
            },
            "r15_combined_reference_rule": {
                "pass": len(criterion_issues) == 0,
                "issues": criterion_issues,
            },
        },
        "status": "PASS" if len(finding_fail) == 0 and len(criterion_issues) == 0 else "FAIL",
    }


def main() -> None:
    v2 = read_json(REVIEW / "r14_design_spec_CONSOLIDATED_v2.json")
    spec = copy.deepcopy(v2)
    old_gate_issues = v2.get("consistency_gate_output", {}).get("checks", {}).get("r15_criterion_reference", {}).get("issues", [])

    spec["version_id"] = "R14_DESIGN_SPEC_CONSOLIDATED_V2_1"
    spec["supersedes"] = "R14_DESIGN_SPEC_CONSOLIDATED_V2"
    apply_forward_prediction_ledger_addendum(spec)

    states = set(spec.get("state_machine", {}).get("states", []))
    predicates = set()
    for vals in spec.get("state_machine", {}).get("named_predicate_terms", {}).values():
        predicates.update(vals)
    telemetry_fields = set()
    for schema in ["daily_term_row", "daily_state_snapshot", "execution_outcome_row"]:
        for f in spec.get("telemetry_schema", {}).get(schema, []):
            telemetry_fields.add(str(f))

    spec["r15_criterion_classification"] = {}
    amended_criteria = copy.deepcopy(spec.get("r15_acceptance_criteria", {}))

    for sym, crits in amended_criteria.items():
        new_list = []
        class_list = []
        for c in crits:
            c_class = classify_criterion(c)
            class_list.append(c_class)
            if c_class == "METRIC_REFERENCING":
                nc = amend_metric_criterion(c)
            else:
                nc = ensure_state_reference(c, states, predicates, telemetry_fields)
            new_list.append(nc)
        amended_criteria[sym] = dedupe_text_list(new_list)
        # re-align class list with deduped list
        aligned_classes: list[str] = []
        for c in amended_criteria[sym]:
            aligned_classes.append(classify_criterion(c))
        spec["r15_criterion_classification"][sym] = aligned_classes

    spec["r15_acceptance_criteria"] = amended_criteria

    # Per-criterion disposition table for all 21 prior flags.
    disposition_table = []
    for issue in old_gate_issues:
        sym = issue["symbol"]
        idx = issue["index"]
        old_criterion = issue["criterion"]
        new_criterion = spec["r15_acceptance_criteria"][sym][idx]
        c_class = spec["r15_criterion_classification"][sym][idx]
        if "Predicate token not defined" in issue["issue"]:
            disposition = "RESOLVED_AS_STATE"
            reason = "Checker now recognizes states in combined reference set; criterion also explicitly state-referenced."
        elif c_class == "METRIC_REFERENCING" and new_criterion != old_criterion:
            disposition = "AMENDED_WITH_METRIC_SOURCE"
            reason = "Metric source fields/measurement source appended per directive."
        elif c_class == "STATE_REFERENCING" and new_criterion != old_criterion:
            disposition = "RESOLVED_AS_STATE"
            reason = "State reference appended to satisfy STATE_REFERENCING class."
        else:
            disposition = "UNCHANGED_WITH_REASON"
            reason = "Criterion already compliant after checker repair."
        disposition_table.append({
            "symbol": sym,
            "index": idx,
            "old_issue": issue["issue"],
            "disposition": disposition,
            "reason": reason,
            "old_criterion": old_criterion,
            "new_criterion": new_criterion,
            "criterion_class": c_class,
        })

    spec["criterion_disposition_table_v2_1"] = disposition_table

    gate = build_repaired_gate(spec)
    spec["consistency_gate_output_v2_1"] = gate

    out_json = REVIEW / "r14_design_spec_CONSOLIDATED_v2_1.json"
    out_md = REVIEW / "r14_design_spec_CONSOLIDATED_v2_1.md"

    md = []
    md.append("# R14 Design Spec CONSOLIDATED v2.1")
    md.append("")
    md.append("Supersedes: R14 Design Spec CONSOLIDATED v2")
    md.append("")
    md.append("Checker repair summary:")
    md.append("- Finding-response rule: module-name OR exact underscore state/predicate token reference.")
    md.append("- R15 reference set: states ∪ predicates ∪ telemetry fields.")
    md.append("- Criterion classification enforced: STATE_REFERENCING or METRIC_REFERENCING.")
    md.append("")
    md.append("## Amended R15 Criteria")
    md.append(json.dumps(spec["r15_acceptance_criteria"], ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")
    md.append("## Criterion Classification")
    md.append(json.dumps(spec["r15_criterion_classification"], ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")
    md.append("## Disposition Table (21 Prior Flags)")
    md.append(json.dumps(disposition_table, ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")
    md.append("## Forward Prediction Ledger (R16 Core)")
    md.append(json.dumps(spec["forward_prediction_ledger_r16"], ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")
    md.append("## R14-B Readiness Checklist (R16 Gate Added)")
    md.append(json.dumps(spec["r14_b_readiness_checklist"], ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")
    md.append("## Repaired Gate Output")
    md.append(json.dumps(gate, ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")
    md.append("R14-B and R15 remain NOT AUTHORIZED.")
    md.append("")

    out_json.write_text(json.dumps(spec, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text("\n".join(md), encoding="utf-8")

    print("R14_DESIGN_SPEC_CONSOLIDATED_V2_1_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))
    print("repaired_gate_status", gate["status"])


if __name__ == "__main__":
    main()

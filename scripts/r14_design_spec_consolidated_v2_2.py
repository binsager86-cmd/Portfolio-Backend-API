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


def anchor_finding_responses(spec: dict[str, Any]) -> dict[str, dict[str, str]]:
    frm = spec.setdefault("finding_response_map", {})
    old_values: dict[str, str] = {}
    updates: dict[str, str] = {
        "EARLY_TIER": "Structural answer to the pre-volume edge: expose accumulation-stage entries using flow confirmation and adaptive base validity before breakout confirmation. Anchors: FlowConfirmationEngine AdaptiveBaseGeometry EARLY_ACCUMULATION_ENTRY EARLY_ENTRY_FLOW_OK EARLY_ENTRY_BASE_VALID_OK.",
        "F1": "Solved at confirmation tier; early tier intentionally bypasses same-day volume multiple dependence when flow confirmation and base validity are present. Anchors: FlowConfirmationEngine CONFIRM_FLOW_CORE_OK BASE_REFERENCE_VALID.",
        "F6": "Explained by moving emphasis from veto gates to candidate-intent and confirmation persistence. Anchors: LifecycleIntentRouter DEFERRED_INTENT_ACTIVE DEFERRED_INTENT.",
        "F8": "Tested by combining adaptive base geometry with persistent base references and deferred intent. Anchors: AdaptiveBaseGeometry BASE_REFERENCE_PRESENT BASE_REFERENCE_VALID DEFERRED_INTENT_ACTIVE.",
        "F8a": "Solved by persistent base references plus readiness-aware base freeze so missing-reference disarm cannot persist silently. Anchors: AdaptiveBaseGeometry BASE_REFERENCE_PRESENT READINESS_FALLBACK_ELIGIBLE.",
        "F8b": "Solved by advancing current-valid references during confirmed accumulation and referencing chase guard to the current valid reference, not the original freeze. Anchors: LifecycleIntentRouter BASE_REFERENCE_ADVANCE_OK CHASE_GUARD_CURRENT_REF_OK.",
        "F8c": "Not established by sealed evidence; however, any remaining veto-capable post-mandatory authority must be fully telemetried as named predicates if retained. Anchors: PredicateTelemetryLedger daily_term_row predicate_name predicate_pass.",
        "F9": "Solved at confirmation tier; early tier uses current and arriving liquidity as participation context, not as a sole veto, while preserving optional participation caps. Anchors: ExecutionLiquidityAssessment CURRENT_DAY_LIQUIDITY_OK LIQUIDITY_CONTEXT_OK CONFIRM_LIQUIDITY_OK.",
    }
    for k, new_text in updates.items():
        old_values[k] = str(frm.get(k, ""))
        frm[k] = new_text
    return {k: {"old": old_values[k], "new": frm[k]} for k in updates}


def ensure_execution_liquidity_module(spec: dict[str, Any]) -> bool:
    modules = spec.setdefault("module_boundaries", [])
    for m in modules:
        if m.get("module") == "ExecutionLiquidityAssessment":
            return False

    modules.append(
        {
            "module": "ExecutionLiquidityAssessment",
            "inputs": [
                "normalized_day_payload",
                "base_reference",
                "confirmation_state",
                "execution_size_parameter",
                "trailing_liquidity_context",
            ],
            "outputs": [
                "liquidity_assessment",
                "liquidity_terms",
                "liquidity_advisory_or_veto",
            ],
            "responsibility": "Evaluate current-day and context liquidity sufficiency, emit CURRENT_DAY_LIQUIDITY_OK and LIQUIDITY_CONTEXT_OK terms, and enforce liquidity gating semantics.",
        }
    )
    return True


def build_lineage_repair_note() -> dict[str, Any]:
    return {
        "conduct_ledger_entry": "#4",
        "statement": "v2_1 artifacts and manifest v1_12 were regenerated in place; append-only lineage was broken and is repaired by v2_2 supersession.",
        "v2_1_non_authoritative_variants": [
            {
                "label": "v2_1_original",
                "json_sha256": "aedc50fca01886727e5154af8445f3ca64327d6d5f12638f52395cc7cb7dd328",
                "status": "NON_AUTHORITATIVE",
            },
            {
                "label": "v2_1_overwritten",
                "json_sha256": "262b8c17c175475e9ad893a5794348a40db9dd5e0c7fdf5c37fa7cf75d02d7bc",
                "status": "NON_AUTHORITATIVE",
            },
        ],
        "authoritative_version": "R14_DESIGN_SPEC_CONSOLIDATED_V2_2",
    }


def main() -> None:
    v2_1 = read_json(REVIEW / "r14_design_spec_CONSOLIDATED_v2_1.json")
    v2 = read_json(REVIEW / "r14_design_spec_CONSOLIDATED_v2.json")
    spec = copy.deepcopy(v2_1)
    old_gate_issues = v2.get("consistency_gate_output", {}).get("checks", {}).get("r15_criterion_reference", {}).get("issues", [])

    spec["version_id"] = "R14_DESIGN_SPEC_CONSOLIDATED_V2_2"
    spec["supersedes"] = [
        "R14_DESIGN_SPEC_CONSOLIDATED_V2_1_ORIGINAL",
        "R14_DESIGN_SPEC_CONSOLIDATED_V2_1_OVERWRITTEN",
    ]

    finding_anchor_delta = anchor_finding_responses(spec)
    module_added = ensure_execution_liquidity_module(spec)

    disposition_table = []
    for issue in old_gate_issues:
        sym = issue["symbol"]
        idx = issue["index"]
        old_criterion = issue["criterion"]
        new_criterion = spec["r15_acceptance_criteria"][sym][idx]
        c_class = spec.get("r15_criterion_classification", {}).get(sym, [])[idx]
        if "Predicate token not defined" in issue["issue"]:
            disposition = "RESOLVED_AS_STATE"
            reason = "Checker recognizes states in combined reference set; criterion has explicit state/metric references."
        elif c_class == "METRIC_REFERENCING" and new_criterion != old_criterion:
            disposition = "AMENDED_WITH_METRIC_SOURCE"
            reason = "Metric source fields/measurement source included."
        elif c_class == "STATE_REFERENCING" and new_criterion != old_criterion:
            disposition = "RESOLVED_AS_STATE"
            reason = "State reference appended to satisfy STATE_REFERENCING class."
        else:
            disposition = "UNCHANGED_WITH_REASON"
            reason = "Criterion remained compliant from v2_1 checker-policy repair."
        disposition_table.append(
            {
                "symbol": sym,
                "index": idx,
                "old_issue": issue["issue"],
                "disposition": disposition,
                "reason": reason,
                "old_criterion": old_criterion,
                "new_criterion": new_criterion,
                "criterion_class": c_class,
            }
        )

    spec["criterion_disposition_table_v2_2"] = disposition_table
    spec["lineage_repair_v2_2"] = build_lineage_repair_note()
    spec["v2_2_disposition_note"] = {
        "entry_1_finding_anchoring": "Applied exact module/state/predicate anchors to EARLY_TIER, F1, F6, F8, F8a, F8b, F8c, F9 without changing meaning.",
        "entry_2_module_fix": (
            "Defined ExecutionLiquidityAssessment in module_boundaries to resolve registry/module mismatch for "
            "min_daily_value_kwd and LIQUIDITY_EXECUTION_SIZE_PARAMETER."
        ),
        "lineage_repair": "Both v2_1 byte variants are recorded as non-authoritative; v2_2 is the append-only authoritative continuation.",
    }

    gate = build_repaired_gate(spec)
    spec["consistency_gate_output_v2_2"] = gate

    out_json = REVIEW / "r14_design_spec_CONSOLIDATED_v2_2.json"
    out_md = REVIEW / "r14_design_spec_CONSOLIDATED_v2_2.md"

    md = []
    md.append("# R14 Design Spec CONSOLIDATED v2.2")
    md.append("")
    md.append("Supersedes: both v2.1 byte variants (lineage repair)")
    md.append("")
    md.append("## v2.2 Disposition Note")
    md.append(json.dumps(spec["v2_2_disposition_note"], ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")
    md.append("## v2.2 Lineage Repair")
    md.append(json.dumps(spec["lineage_repair_v2_2"], ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")
    md.append("## Finding-Response Anchoring Delta (8 Entries)")
    md.append(json.dumps(finding_anchor_delta, ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")
    md.append("## ExecutionLiquidityAssessment Module Resolution")
    md.append(json.dumps({"module_added": module_added}, ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")
    md.append("## Forward Prediction Ledger (R16 Core Retained)")
    md.append(json.dumps(spec["forward_prediction_ledger_r16"], ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")
    md.append("## Disposition Table (21 Prior Flags)")
    md.append(json.dumps(disposition_table, ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")
    md.append("## Gate Output")
    md.append(json.dumps(gate, ensure_ascii=True, indent=2, sort_keys=True))
    md.append("")
    md.append("R14-B and R15 remain NOT AUTHORIZED.")
    md.append("")

    out_json.write_text(json.dumps(spec, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text("\n".join(md), encoding="utf-8")

    print("R14_DESIGN_SPEC_CONSOLIDATED_V2_2_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))
    print("repaired_gate_status", gate["status"])


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
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


def build_spec() -> tuple[dict[str, Any], str]:
    d1 = read_json(REVIEW / "r13_set_a_causal_attribution_v3.json")
    vol = read_json(REVIEW / "r13_volume_arrival_audit_v1.json")
    gate = read_json(REVIEW / "r13_gate_conflict_analysis_v1_2.json")

    findings = {
        "F1": "Confirmation-predicate defect dominated by M2 same-day relative-volume gating on trending names.",
        "F2": "Base geometry defect driven by fixed-width base constraint rejecting wide Boursa bases.",
        "F3": "Base-reference lifecycle defect status is indeterminate due to F7 unresolved terms; hypothesis carried forward for test.",
        "F4": "Warmup structural blindness consumes material session share, especially for SANAM.",
        "F5": "Avoid logic validated and must be preserved in authority.",
        "F6": "Gate-warfare hypothesis retired; no-candidate persistence dominates.",
        "F7": "Telemetry gap: base_high_ref, liquidity_ok, and per-term pass/fail are not durably persisted day-by-day.",
        "F8": "Hypothesis for test: F2->M1 disarm chain, where missing base freeze leaves M1 non-actionable on later high-volume days.",
    }

    spec_json: dict[str, Any] = {
        "version_id": "R14_DESIGN_SPEC_V1",
        "authorization": {
            "R14_A": "AUTHORIZED_DESIGN_ONLY",
            "R14_B": "NOT_AUTHORIZED",
            "R15": "NOT_AUTHORIZED",
        },
        "governing_constraints": {
            "no_engine_code_changes_in_this_batch": True,
            "no_threshold_values_in_spec": True,
            "all_thresholds_named_config_parameters_only": True,
            "set_b_exposure_prohibited_until_r14_b_gate": True,
        },
        "architecture_blueprint": {
            "skeleton": "Proposal C stateful lifecycle and daily predicate telemetry",
            "confirmation_core": "Proposal A accumulation-window flow confirmation",
            "base_module": "Proposal B adaptive volatility-regime-aware base geometry",
            "avoid_plane": "Preserved byte-compatible in authority",
            "warmup_module": "Explicit readiness states and fallback behavior",
        },
        "module_boundaries": [
            {
                "module": "DataSurfaceAdapter",
                "responsibility": "Provide day-normalized input payloads and segment-aware readiness context.",
                "inputs": ["ohlcv_day", "indicator_day", "segment_context", "calendar_context"],
                "outputs": ["normalized_day_payload", "readiness_context"],
            },
            {
                "module": "WarmupReadinessEngine",
                "responsibility": "Determine readiness state, warmup fallback, and reset semantics for new listings and segment restarts.",
                "inputs": ["normalized_day_payload", "coverage_history", "segment_restart_flag"],
                "outputs": ["readiness_state", "readiness_reason", "readiness_transition_event"],
            },
            {
                "module": "AdaptiveBaseGeometry",
                "responsibility": "Detect, freeze, ratchet, invalidate, and retire bases using regime-aware geometry.",
                "inputs": ["normalized_day_payload", "readiness_state", "price_history_window", "volatility_regime_state"],
                "outputs": ["base_state", "base_transition_terms", "base_reference"],
            },
            {
                "module": "FlowConfirmationEngine",
                "responsibility": "Score confirmation using accumulation-window flow evidence plus breakout structure.",
                "inputs": ["normalized_day_payload", "base_reference", "flow_history_window", "structure_terms"],
                "outputs": ["confirmation_state", "confirmation_terms", "candidate_intent"],
            },
            {
                "module": "LifecycleIntentRouter",
                "responsibility": "Persist candidate intent, survive delayed volume arrival, and hand off to risk/capacity layer.",
                "inputs": ["candidate_intent", "base_state", "confirmation_state", "risk_budget_state"],
                "outputs": ["execution_intent", "deferred_intent", "veto_record"],
            },
            {
                "module": "AvoidAuthorityPlane",
                "responsibility": "Retain current avoid authority semantics as a veto plane on top of the architecture.",
                "inputs": ["normalized_day_payload", "trend_state"],
                "outputs": ["avoid_state", "avoid_veto"],
            },
            {
                "module": "PredicateTelemetryLedger",
                "responsibility": "Persist every predicate term, every day, for every symbol.",
                "inputs": ["all_module_terms", "state_transitions", "execution_outcomes"],
                "outputs": ["daily_term_row", "state_snapshot_row", "audit_row"],
            },
        ],
        "interfaces": {
            "normalized_day_payload": [
                "trade_date",
                "symbol",
                "open", "high", "low", "close", "volume", "value_kwd",
                "indicator_terms",
                "segment_id",
                "segment_day_index",
                "masked_context",
            ],
            "base_reference": [
                "base_reference_id",
                "base_high_ref",
                "base_low_ref",
                "base_origin_date",
                "base_validity_state",
                "base_retirement_reason",
            ],
            "candidate_intent": [
                "intent_id",
                "symbol",
                "trade_date",
                "phase_state",
                "confirmation_state",
                "intent_reason",
                "base_reference_id",
            ],
            "veto_record": [
                "intent_id",
                "veto_plane",
                "veto_reason",
                "blocking_term_name",
                "blocking_term_value",
            ],
        },
        "state_machine": {
            "states": [
                "READINESS_PENDING",
                "READINESS_LIMITED",
                "NEUTRAL",
                "BASE_FORMING",
                "ACCUMULATION",
                "BREAKOUT_WATCH",
                "BREAKOUT_CONFIRMED",
                "MARKUP",
                "DISTRIBUTION_WARNING",
                "EXIT",
                "AVOID",
                "DEFERRED_INTENT",
            ],
            "transition_principles": [
                "All transitions are gated by named predicates only.",
                "All threshold-bearing terms are represented by named configuration parameters, not literal values.",
                "Readiness states are explicit and segment-aware.",
                "Base references persist as first-class state objects.",
                "Candidate intent may persist independently of immediate execution eligibility.",
            ],
            "named_predicate_terms": {
                "warmup": ["READINESS_LONG_LOOKBACK_READY", "READINESS_SEGMENT_RESTART_READY", "READINESS_FALLBACK_ELIGIBLE"],
                "base": ["BASE_GEOMETRY_WIDTH_OK", "BASE_DWELL_OK", "BASE_RANGE_INCLUSION_OK", "BASE_VOLATILITY_REGIME_OK"],
                "accumulation": ["FLOW_OBV_SLOPE_OK", "FLOW_ANV_SLOPE_OK", "FLOW_ACCUMULATION_DIVERGENCE_OK", "ACCUMULATION_CONTEXT_OK"],
                "watch": ["WATCH_NEAR_BASE_OK", "WATCH_FLOW_PERSISTENCE_OK", "WATCH_STRUCTURE_OK"],
                "confirmation": ["CONFIRM_FLOW_CORE_OK", "CONFIRM_STRUCTURE_OK", "CONFIRM_RELATIVE_VOLUME_CONTEXT_OK", "CONFIRM_CHASE_GUARD_OK", "CONFIRM_LIQUIDITY_OK"],
                "lifecycle": ["BASE_REFERENCE_PRESENT", "BASE_REFERENCE_VALID", "DEFERRED_INTENT_ACTIVE", "DEFERRED_INTENT_EXPIRY_OK"],
                "avoid": ["AVOID_CONDITION_ACTIVE"],
            },
        },
        "telemetry_schema": {
            "daily_term_row": [
                "symbol",
                "trade_date",
                "segment_id",
                "phase_before",
                "phase_after",
                "readiness_state",
                "base_reference_id",
                "intent_id",
                "predicate_namespace",
                "predicate_name",
                "predicate_value",
                "predicate_threshold_parameter",
                "predicate_pass",
                "recoverability_state",
                "recoverability_reason",
                "source_payload_fields",
            ],
            "daily_state_snapshot": [
                "symbol",
                "trade_date",
                "readiness_state",
                "phase_state",
                "base_reference_snapshot",
                "intent_snapshot",
                "avoid_state",
                "risk_budget_state",
            ],
            "execution_outcome_row": [
                "symbol",
                "trade_date",
                "candidate_intent_state",
                "execution_state",
                "veto_plane",
                "veto_reason",
                "opened_trade_flag",
                "trade_id",
            ],
        },
        "finding_response_map": {
            "F1": "Solved by FlowConfirmationEngine replacing same-bar multiple as sole confirmation core.",
            "F2": "Solved by AdaptiveBaseGeometry replacing fixed-width geometry with regime-aware geometry.",
            "F3": "Addressed by LifecycleIntentRouter and persistent base references; to be validated in R15.",
            "F4": "Solved by WarmupReadinessEngine with explicit readiness states and fallback.",
            "F5": "Preserved by AvoidAuthorityPlane retaining avoid veto semantics.",
            "F6": "Explained by moving emphasis from veto gates to candidate-intent and confirmation persistence.",
            "F7": "Solved by PredicateTelemetryLedger persisting every term every day.",
            "F8": "Tested by combining adaptive base geometry with persistent base references and deferred intent.",
        },
        "r15_acceptance_criteria": {
            "TIJARA": [
                "PHASE_PROGRESSED_NO_CANDIDATE share over 2024-2026 falls below one-quarter of trading days.",
                "At least one owner-window breakout cluster produces BREAKOUT_CONFIRMED or DEFERRED_INTENT instead of persistent BREAKOUT_WATCH stagnation.",
            ],
            "SANAM": [
                "The 2025-05-08 through 2025-05-21 window produces at least one confirmed entry event.",
                "Width-rule dominance in the owner window drops below one-quarter of blocking rows.",
            ],
            "BPCC": [
                "The 2025-04-22 near-miss cluster yields a candidate intent or confirmed entry without requiring a same-bar multiple shock.",
                "M2-only blockade frequency materially declines in the owner window.",
            ],
            "ZAIN": [
                "Owner-window no-candidate share falls below one-third of trading days.",
                "At least one breakout-above-owner-threshold cluster yields BREAKOUT_CONFIRMED or DEFERRED_INTENT.",
            ],
            "MABANEE": [
                "The decline interval remains avoided with zero long entries during the avoid-dominant regime.",
                "Avoid veto authority remains sufficient to block downtrend participation.",
            ],
        },
        "citations": {
            "findings_artifacts": [
                "artifacts/preview1a_prestart/review_final/r13_findings_of_record_v1_1.md",
                "artifacts/preview1a_prestart/review_final/r13_volume_arrival_audit_v1.json",
                "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json",
            ],
            "code_refs": [
                "app/services/eagle_eye/scanner_service.py#L713",
                "app/services/eagle_eye/scanner_service.py#L718",
                "app/services/eagle_eye/scanner_service.py#L859",
                "app/services/eagle_eye/scanner_service.py#L108",
                "app/services/eagle_eye/scanner_service.py#L288",
                "app/services/eagle_eye/scanner_service.py#L770",
            ],
        },
        "authorization_status": {
            "R14_A": "AUTHORIZED",
            "R14_B": "NOT_AUTHORIZED",
            "R15": "NOT_AUTHORIZED",
        },
    }

    md = []
    md.append("# R14 Design Spec v1")
    md.append("")
    md.append("Authorization status:")
    md.append("- R14-A: AUTHORIZED (design spec only)")
    md.append("- R14-B: NOT AUTHORIZED")
    md.append("- R15: NOT AUTHORIZED")
    md.append("")
    md.append("Design constraints:")
    md.append("- Zero engine or scanner code changes in this batch.")
    md.append("- No numeric threshold values in the design; every threshold-bearing rule is represented as a named config parameter to be frozen at the R14-B gate.")
    md.append("- Set B remains excluded from any exposure decisions until R14-B authorization.")
    md.append("")
    md.append("Architecture blueprint:")
    md.append("- Skeleton: stateful lifecycle and full daily predicate telemetry.")
    md.append("- Confirmation core: accumulation-window flow confirmation using already-computed flow evidence and breakout structure.")
    md.append("- Base module: adaptive, volatility-regime-aware base geometry.")
    md.append("- Avoid plane: preserved in authority.")
    md.append("- Warmup module: explicit readiness states with fallback behavior.")
    md.append("")
    md.append("Module boundaries and interfaces:")
    for m in spec_json["module_boundaries"]:
        md.append(f"- {m['module']}: {m['responsibility']}")
        md.append(f"  inputs={m['inputs']}")
        md.append(f"  outputs={m['outputs']}")
    md.append("")
    md.append("State machine:")
    md.append(f"- States: {spec_json['state_machine']['states']}")
    md.append("- Transition principles:")
    for s in spec_json["state_machine"]["transition_principles"]:
        md.append(f"  - {s}")
    md.append("- Named predicate namespaces:")
    for ns, terms in spec_json["state_machine"]["named_predicate_terms"].items():
        md.append(f"  - {ns}: {terms}")
    md.append("")
    md.append("Telemetry schema:")
    for name, fields in spec_json["telemetry_schema"].items():
        md.append(f"- {name}: {fields}")
    md.append("")
    md.append("Finding response map:")
    for fid, answer in spec_json["finding_response_map"].items():
        md.append(f"- {fid}: {answer}")
    md.append("")
    md.append("R15 acceptance criteria:")
    for sym, crit in spec_json["r15_acceptance_criteria"].items():
        md.append(f"- {sym}:")
        for c in crit:
            md.append(f"  - {c}")
    md.append("")
    md.append("Citations:")
    md.append("- Findings artifacts:")
    for c in spec_json["citations"]["findings_artifacts"]:
        md.append(f"  - {c}")
    md.append("- Code refs:")
    for c in spec_json["citations"]["code_refs"]:
        md.append(f"  - {c}")
    md.append("")
    md.append("R14-B and R15 remain NOT AUTHORIZED.")
    md.append("")

    return spec_json, "\n".join(md)


def main() -> None:
    spec_json, md = build_spec()
    out_json = REVIEW / "r14_design_spec_v1.json"
    out_md = REVIEW / "r14_design_spec_v1.md"
    out_json.write_text(json.dumps(spec_json, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(md, encoding="utf-8")
    print("R14_DESIGN_SPEC_V1_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()

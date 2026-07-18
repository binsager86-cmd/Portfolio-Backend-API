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


def write_sha_sidecar(sidecar_path: Path, files: list[tuple[str, Path]]) -> None:
    lines = []
    for rel, p in files:
        lines.append(f"{sha256_file(p)}  {rel}")
    sidecar_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def baseline_id_now() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"EE_V2_{ts}"


def build_freeze_payload(baseline_id: str) -> dict[str, Any]:
    return {
        "version_id": "R14B_PARAMETER_FREEZE_V1",
        "authority": {
            "owner_ratification_received": True,
            "governing_design_doc": {
                "version": "R14_DESIGN_SPEC_CONSOLIDATED_V2_2",
                "json_sha256": "9a5f1facdf1fc222239e6304afadb1e420f4d22fb506f23d97af52b64cb4b52b",
                "md_sha256": "dedf5361c25b40df5b0ece8dbfeb9f360e81cf5facfdb3a2305dcf6c8b31de4e",
            },
            "implementation_liberty_note": "No implementation liberty beyond governing text without owner directive.",
        },
        "baseline": {
            "implementation_baseline_id": baseline_id,
            "r11_baseline_status": "UNTOUCHED_ARCHIVED",
            "new_module_path": "app/services/eagle_eye_v2",
            "supersession_rule": "Old engine is never edited; only superseded by isolated v2 module path.",
        },
        "owner_ratified_values_verbatim": {
            "EARLY_TIER_SIZE_FRACTION": "0.30 (fraction of full target position)",
            "EARLY_TIER_PARTICIPATION_CAP": "0.10 (fraction of daily traded value)",
            "EARLY_TIER_TIME_STOP": "60 sessions, REVIEW semantics: at expiry re-evaluate flow predicates; exit only on flow-evidence decay; else re-arm clock; max 2 re-arms then OWNER_REVIEW state",
            "SCALE_ON_CONFIRMATION": "SINGLE_ADD_TO_FULL_TARGET at BREAKOUT_CONFIRMED_ENTRY",
            "CHASE_ADVISORY_BAND": "advisory flag > 0.08 extension vs current valid reference; escalation flag > 0.15",
            "TIER_RULE": "CANDIDATE_A (HIGH >= 500000 KWD median daily value; MID >= 100000 KWD; else LOW; one-time sanity check vs live tier profile during R14-B)",
            "GRADING_HORIZONS": "[20, 60, 120] sessions",
            "MIN_CALIBRATION_WINDOW": "63 sessions",
            "MARKUP_MATERIALIZATION_CRITERION": "max favorable excursion >= +0.20 within 120 sessions of prediction",
        },
        "time_stop_review_semantics": {
            "re_evaluate_at_sessions": 60,
            "exit_condition": "FLOW_EVIDENCE_DECAY_ONLY",
            "clock_rearm_rule": "REARM_WHEN_FLOW_HOLDS",
            "max_clock_rearms": 2,
            "post_rearm_terminal_state": "OWNER_REVIEW",
        },
        "remaining_parameters_requiring_r14b_parameter_gate": [
            {
                "name": "base_min_sessions",
                "family": "base_geometry",
                "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE",
            },
            {
                "name": "base_max_width_pct",
                "family": "base_geometry",
                "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE",
            },
            {
                "name": "atr_squeeze_pctile",
                "family": "base_geometry",
                "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE",
            },
            {
                "name": "cmf_floor",
                "family": "confirmation_thresholds",
                "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE",
            },
            {
                "name": "volume_breakout_mult",
                "family": "confirmation_thresholds",
                "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE",
            },
            {
                "name": "rsi_regime",
                "family": "confirmation_thresholds",
                "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE",
            },
            {
                "name": "adx_trigger",
                "family": "confirmation_thresholds",
                "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE",
            },
            {
                "name": "LIQUIDITY_EXECUTION_SIZE_PARAMETER",
                "family": "liquidity_execution_size",
                "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE",
            },
            {
                "name": "ml_prob_min",
                "family": "ml_floor",
                "status": "IMPLEMENTATION_PROPOSES_VALUE_WITH_EVIDENCE_RATIONALE_OWNER_RATIFIES_AT_R14B_PARAMETER_GATE",
            },
        ],
        "quarantine_and_prohibitions": {
            "set_b_quarantine": "EXPLICITLY_RESTATED_NO_PARAMETER_VALUE_MAY_BE_CHOSEN_USING_SET_B",
            "no_backtests_no_stat_runs_no_r15_preview_no_r16_execution": True,
            "r15_authorization": "NOT_AUTHORIZED",
            "r16_authorization": "NOT_AUTHORIZED",
            "threshold_mutation_post_freeze": "PROHIBITED_UNLESS_CHANGE_REQUEST_APPROVED",
        },
        "conduct_rules_reaffirmed": [
            "Permanent scripts only",
            "Append-only artifacts",
            "Frozen verifiers",
            "No self-declared gate passage",
            "No temp-and-delete",
        ],
    }


def build_module_a_plan_payload(baseline_id: str) -> dict[str, Any]:
    return {
        "version_id": "R14B_MODULE_A_IMPLEMENTATION_PLAN_V1",
        "baseline_id": baseline_id,
        "module": "PredicateTelemetryLedger",
        "build_order_binding_position": "(a)",
        "scope": {
            "core": [
                "Implement PredicateTelemetryLedger in app/services/eagle_eye_v2/",
                "Create schema migrations for daily_term_row, daily_state_snapshot, execution_outcome_row",
                "Implement immutable write path and append-only protections",
            ],
            "must_precede_decisioning": "Telemetry must be live and validated before any deciding modules progress.",
        },
        "interfaces_to_conform": {
            "module_boundary": {
                "inputs": [
                    "all_module_terms",
                    "state_transitions",
                    "execution_outcomes",
                ],
                "outputs": [
                    "daily_term_row",
                    "state_snapshot_row",
                    "audit_row",
                ],
                "responsibility": "Persist every predicate term, every day, for every symbol.",
            },
            "schema_fields_required": {
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
                    "base_reference_version",
                    "base_reference_origin",
                    "base_reference_current_flag",
                    "extension_pct_vs_current_valid_reference",
                    "chase_advisory_flag",
                    "current_day_value_kwd",
                    "trailing_liquidity_context_value",
                    "early_tier_flag",
                    "dead_money_sessions",
                    "flow_obv_slope_40",
                    "flow_anv_slope_40",
                    "flow_accumulation_divergence",
                    "accumulation_context_ok",
                    "participation_cap_pct",
                    "pilot_size_fraction",
                    "time_stop_sessions",
                    "entry_tier",
                    "flow_evidence_snapshot",
                    "current_valid_reference_value",
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
                    "chase_advisory_emitted",
                    "chase_advisory_extension_pct",
                    "entry_tier",
                    "dead_money_sessions",
                ],
            },
        },
        "adversarial_review_gate": {
            "required_artifacts": [
                "r14b_module_a_implementation_report_v1.md",
                "r14b_module_a_interface_conformance_v1.json",
                "r14b_module_a_test_evidence_v1.json",
            ],
            "required_checks": [
                "Per-file SHA-256 manifest for all touched implementation files",
                "Interface conformance check against v2_2 module boundary and telemetry schema",
                "Append-only behavior verification: no UPDATE/DELETE in telemetry write path",
                "Migration reversibility and idempotency validation",
                "Read-only comparison harness setup for later avoid authority byte-compat phase",
            ],
            "proceed_rule": "NO_PROGRESSION_TO_MODULE_B_UNTIL_MODULE_A_REVIEW_PASS",
        },
        "test_matrix": [
            "Schema migration applies cleanly on empty and populated staging replicas",
            "One synthetic day writes exactly one row per predicate term and state transition",
            "Write path rejects mutation attempts on existing telemetry rows",
            "Hash-seal sidecar generation includes all module-a artifacts",
            "No calls to backtest/statistical runners during module-a execution",
        ],
        "prohibitions_reaffirmed": {
            "set_b_contact": "PROHIBITED",
            "r15_preview": "PROHIBITED",
            "statistical_run": "PROHIBITED",
            "temporary_scripts": "PROHIBITED",
        },
    }


def markdown_from_json(title: str, payload: dict[str, Any]) -> str:
    lines = [f"# {title}", "", json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), ""]
    return "\n".join(lines)


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    baseline_id = baseline_id_now()

    freeze_payload = build_freeze_payload(baseline_id)
    module_a_payload = build_module_a_plan_payload(baseline_id)

    freeze_json = REVIEW / "r14b_parameter_freeze_v1.json"
    freeze_md = REVIEW / "r14b_parameter_freeze_v1.md"
    freeze_sha = REVIEW / "r14b_parameter_freeze_v1.sha256"

    module_json = REVIEW / "r14b_module_a_implementation_plan_v1.json"
    module_md = REVIEW / "r14b_module_a_implementation_plan_v1.md"
    module_sha = REVIEW / "r14b_module_a_implementation_plan_v1.sha256"

    freeze_json.write_text(json.dumps(freeze_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze_md.write_text(markdown_from_json("R14-B Parameter Freeze v1", freeze_payload), encoding="utf-8")
    module_json.write_text(json.dumps(module_a_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    module_md.write_text(markdown_from_json("R14-B Module (a) Implementation Plan v1", module_a_payload), encoding="utf-8")

    write_sha_sidecar(
        freeze_sha,
        [
            ("artifacts/preview1a_prestart/review_final/r14b_parameter_freeze_v1.json", freeze_json),
            ("artifacts/preview1a_prestart/review_final/r14b_parameter_freeze_v1.md", freeze_md),
        ],
    )
    write_sha_sidecar(
        module_sha,
        [
            ("artifacts/preview1a_prestart/review_final/r14b_module_a_implementation_plan_v1.json", module_json),
            ("artifacts/preview1a_prestart/review_final/r14b_module_a_implementation_plan_v1.md", module_md),
        ],
    )

    print("R14B_PARAMETER_FREEZE_V1_COMPLETE")
    print("baseline_id", baseline_id)
    print("freeze_json_sha256", sha256_file(freeze_json))
    print("freeze_md_sha256", sha256_file(freeze_md))
    print("freeze_sidecar_sha256", sha256_file(freeze_sha))
    print("module_a_plan_json_sha256", sha256_file(module_json))
    print("module_a_plan_md_sha256", sha256_file(module_md))
    print("module_a_plan_sidecar_sha256", sha256_file(module_sha))


if __name__ == "__main__":
    main()

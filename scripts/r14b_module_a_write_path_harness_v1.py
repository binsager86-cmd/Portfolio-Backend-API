from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.eagle_eye_v2.predicate_telemetry_ledger import (
    TABLES,
    apply_schema_migration,
    append_row,
    emit_daily_hash_chain,
    fetch_rows,
    get_table_columns,
    verify_update_delete_blocked,
)
from app.services.eagle_eye_v2.telemetry_schema import TABLE_TO_FIELDS

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


def sample_rows(trade_date: str) -> dict[str, dict[str, Any]]:
    return {
        "daily_term_row": {
            "symbol": "SANAM",
            "trade_date": trade_date,
            "segment_id": "SEG_A",
            "segment_day_index": 0,
            "phase_before": "ACCUMULATION",
            "phase_after": "BREAKOUT_WATCH",
            "readiness_state": "READINESS_LONG_LOOKBACK_READY",
            "readiness_transition_event": "ACCUMULATION_TO_BREAKOUT_WATCH",
            "readiness_transition_from_state": "ACCUMULATION",
            "readiness_transition_to_state": "BREAKOUT_WATCH",
            "segment_restart_flag": 0,
            "masked_context_flag": 0,
            "lookback_long_sessions": 180,
            "lookback_segment_sessions": 20,
            "lookback_fallback_sessions": 60,
            "base_reference_id": "BR_SANAM_001",
            "intent_id": "INT_SANAM_001",
            "predicate_namespace": "confirmation",
            "predicate_name": "CONFIRM_FLOW_CORE_OK",
            "predicate_value": 1.0,
            "predicate_threshold_parameter": "cmf_floor",
            "predicate_pass": 1,
            "recoverability_state": "RECOVERABLE",
            "recoverability_reason": "NONE",
            "source_payload_fields": "close,volume,value_kwd",
            "base_reference_version": "V1",
            "base_reference_origin": "AUTO",
            "base_reference_current_flag": 1,
            "extension_pct_vs_current_valid_reference": 0.04,
            "chase_advisory_flag": 0,
            "current_day_value_kwd": 650000.0,
            "trailing_liquidity_context_value": 220000.0,
            "early_tier_flag": 1,
            "dead_money_sessions": 0,
            "flow_obv_slope_40": 0.13,
            "flow_anv_slope_40": 0.11,
            "flow_accumulation_divergence": 0.07,
            "accumulation_context_ok": 1,
            "participation_cap_pct": 0.10,
            "pilot_size_fraction": 0.30,
            "time_stop_sessions": 60,
            "entry_tier": "EARLY",
            "flow_evidence_snapshot": '{"obv":0.13,"anv":0.11,"div":0.07}',
            "current_valid_reference_value": 233.0,
        },
        "daily_state_snapshot": {
            "symbol": "SANAM",
            "trade_date": trade_date,
            "readiness_state": "READINESS_LONG_LOOKBACK_READY",
            "phase_state": "BREAKOUT_WATCH",
            "base_reference_snapshot": '{"id":"BR_SANAM_001","value":233.0}',
            "intent_snapshot": '{"intent":"EARLY_ACCUMULATION_ENTRY"}',
            "avoid_state": "NONE",
            "risk_budget_state": "WITHIN_CAP",
        },
        "execution_outcome_row": {
            "symbol": "SANAM",
            "trade_date": trade_date,
            "candidate_intent_state": "EARLY_ACCUMULATION_ENTRY",
            "execution_state": "OPENED",
            "veto_plane": "NONE",
            "veto_reason": "NONE",
            "opened_trade_flag": 1,
            "trade_id": "TR_SANAM_001",
            "chase_advisory_emitted": 0,
            "chase_advisory_extension_pct": 0.0,
            "entry_tier": "EARLY",
            "dead_money_sessions": 0,
        },
    }


def assert_writable_readable(trade_date: str) -> dict[str, Any]:
    rows_in = sample_rows(trade_date)
    write_results = {}
    read_results = {}

    for table in TABLES:
        append_row(table, rows_in[table])
        write_results[table] = "WRITE_OK"
        fetched = fetch_rows(table, trade_date)
        read_results[table] = {
            "row_count": len(fetched),
            "fields_present": sorted([k for k in fetched[0].keys() if k != "row_id"]) if fetched else [],
            "sample": fetched[0] if fetched else None,
        }

    return {
        "write_results": write_results,
        "read_results": read_results,
    }


def build_interface_conformance() -> dict[str, Any]:
    per_table = {}
    pass_all = True
    for table, expected_fields in TABLE_TO_FIELDS.items():
        actual = [c for c in get_table_columns(table) if c != "row_id"]
        missing = [c for c in expected_fields if c not in actual]
        extra = [c for c in actual if c not in expected_fields]
        table_pass = len(missing) == 0
        pass_all = pass_all and table_pass
        per_table[table] = {
            "expected_columns": expected_fields,
            "actual_columns": actual,
            "missing_columns": missing,
            "extra_columns": extra,
            "pass": table_pass,
        }
    return {
        "version_id": "R14B_MODULE_A_INTERFACE_CONFORMANCE_V1",
        "pass": pass_all,
        "tables": per_table,
    }


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)

    impl_report_md = REVIEW / "r14b_module_a_implementation_report_v1.md"
    conformance_json = REVIEW / "r14b_module_a_interface_conformance_v1.json"
    evidence_json = REVIEW / "r14b_module_a_test_evidence_v1.json"
    ddl_sql = REVIEW / "r14b_module_a_schema_ddl_v1.sql"
    harness_log = REVIEW / "r14b_module_a_harness_output_v1.log"
    sidecar_chain = REVIEW / "r14b_module_a_daily_ledger_chain_v1.sha256"

    lines: list[str] = []
    lines.append("R14B_MODULE_A_HARNESS_START")

    migration = apply_schema_migration()
    ddls = migration["ddl_emitted"]
    ddl_sql.write_text("\n\n".join(ddls) + "\n", encoding="utf-8")
    lines.append(f"DDL_APPLIED count={len(ddls)} dialect={migration['dialect']}")

    trade_date = "2026-07-13"
    write_read = assert_writable_readable(trade_date)
    lines.append("WRITE_READ_OK")

    mutation_results = {}
    for table in TABLES:
        mutation_results[table] = verify_update_delete_blocked(table, trade_date)
    lines.append("APPEND_ONLY_TRIGGER_CHECK_COMPLETE")

    chain = emit_daily_hash_chain(trade_date, sidecar_chain)
    lines.append(f"SIDECAR_CHAIN_EMITTED trade_date={trade_date} chain_hash={chain['chain_hash']}")

    conformance = build_interface_conformance()
    conformance_json.write_text(json.dumps(conformance, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines.append(f"INTERFACE_CONFORMANCE pass={conformance['pass']}")

    touched_files = [
        ROOT / "app" / "services" / "eagle_eye_v2" / "__init__.py",
        ROOT / "app" / "services" / "eagle_eye_v2" / "telemetry_schema.py",
        ROOT / "app" / "services" / "eagle_eye_v2" / "predicate_telemetry_ledger.py",
        ROOT / "scripts" / "r14b_module_a_write_path_harness_v1.py",
    ]

    touched_hashes = [
        {
            "path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size,
        }
        for p in touched_files
    ]

    test_evidence = {
        "version_id": "R14B_MODULE_A_TEST_EVIDENCE_V1",
        "trade_date": trade_date,
        "writable_readable": write_read,
        "append_only_trigger_results": mutation_results,
        "sidecar_chain": chain,
        "harness_output_lines": lines,
    }
    evidence_json.write_text(json.dumps(test_evidence, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    harness_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = []
    report.append("# R14-B Module (a) Implementation Report v1")
    report.append("")
    report.append("Boundary: PredicateTelemetryLedger (storage + integrity only)")
    report.append("")
    report.append("## File Hashes")
    report.append(json.dumps(touched_hashes, ensure_ascii=True, indent=2, sort_keys=True))
    report.append("")
    report.append("## Schema DDL As Emitted")
    report.append("```sql")
    report.append(ddl_sql.read_text(encoding="utf-8"))
    report.append("```")
    report.append("")
    report.append("## Interface Conformance Artifact")
    report.append(conformance_json.name)
    report.append("")
    report.append("## Test Evidence Artifact")
    report.append(evidence_json.name)
    report.append("")
    report.append("## Test Harness Output (Verbatim)")
    report.append("```text")
    report.append(harness_log.read_text(encoding="utf-8"))
    report.append("```")
    report.append("")
    impl_report_md.write_text("\n".join(report), encoding="utf-8")

    print("R14B_MODULE_A_WRITE_PATH_HARNESS_V1_COMPLETE")
    print("implementation_report_sha256", sha256_file(impl_report_md))
    print("interface_conformance_sha256", sha256_file(conformance_json))
    print("test_evidence_sha256", sha256_file(evidence_json))
    print("ddl_sha256", sha256_file(ddl_sql))
    print("harness_log_sha256", sha256_file(harness_log))
    print("sidecar_chain_sha256", sha256_file(sidecar_chain))


if __name__ == "__main__":
    main()

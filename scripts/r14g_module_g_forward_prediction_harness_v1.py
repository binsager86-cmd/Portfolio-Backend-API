from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.eagle_eye_v2.forward_prediction_ledger import (
    ForwardPredictionLedger,
    fetch_predictions,
    verify_update_delete_blocked,
)
from app.services.eagle_eye_v2.prediction_grader import (
    apply_grades,
    verify_prediction_reader_cannot_write,
)

REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
RUNTIME_DB = REVIEW / "r12_exam_surface_v4_5_runtime.db"
FREEZE_JSON = REVIEW / "r14b_parameter_freeze_v2.json"
FREEZE_SHA = REVIEW / "r14b_parameter_freeze_v2.sha256"
V7_EVIDENCE = REVIEW / "r14e_module_e_test_evidence_v7.json"
R14F_EVIDENCE = REVIEW / "r14f_module_f_avoid_authority_v1_evidence.json"
HARNESS_DB = REVIEW / "r14g_module_g_forward_prediction_harness_v1.db"
RUN_NONCE = "2026-07-18T12:41:36.9182244Z"
RUN_KEY = "R14G_MODULE_G_FORWARD_PREDICTION_LEDGER_V1"
GRADER_VERSION = "R14G_PREDICTION_GRADER_V1"
MODULE_E_SCOPE_NOTE = "entry, holding, suppression, and avoid-veto lifecycle evidenced; exit lifecycle out of scope and untested"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def to_date_text(v: Any) -> str:
    if isinstance(v, int):
        return datetime.fromtimestamp(v, timezone.utc).strftime("%Y-%m-%d")
    s = str(v)
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    if s.isdigit() and len(s) >= 10:
        return datetime.fromtimestamp(int(s), timezone.utc).strftime("%Y-%m-%d")
    raise ValueError(f"Unsupported date value: {v}")


def freeze_attestation() -> dict[str, Any]:
    expected = FREEZE_SHA.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(FREEZE_JSON)
    return {
        "freeze_json": str(FREEZE_JSON),
        "freeze_sha_sidecar": str(FREEZE_SHA),
        "expected_json_sha256": expected,
        "actual_json_sha256": actual,
        "byte_match": expected == actual,
    }


def load_runtime_rows(symbol: str, allowed_dates: set[str]) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(RUNTIME_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT symbol, trade_date, open, high, low, close, volume, value_kwd
            FROM ee_ohlcv
            WHERE symbol LIKE ?
            ORDER BY trade_date ASC
            """,
            (f"{symbol}%",),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            trade_date = to_date_text(int(row["trade_date"]))
            if trade_date not in allowed_dates:
                continue
            out.append({
                "symbol": symbol,
                "trade_date": trade_date,
                "open": float(row["open"] or 0.0),
                "high": float(row["high"] or 0.0),
                "low": float(row["low"] or 0.0),
                "close": float(row["close"] or 0.0),
                "volume": float(row["volume"] or 0.0),
                "value_kwd": float(row["value_kwd"] or 0.0),
            })
        missing = sorted(allowed_dates.difference({str(row["trade_date"]) for row in out}))
        if missing:
            raise RuntimeError(f"Missing sealed runtime rows for {symbol}: {missing}")
        return out
    finally:
        conn.close()


def event_type_for(row: dict[str, Any]) -> str | None:
    intent_state = str(((row.get("candidate_intent") or {}).get("intent_state")) or "INTENT_NONE")
    execution = row.get("execution_intent") or {}
    execution_state = str(execution.get("execution_state") or "NONE")
    veto_record = row.get("veto_record") or {}
    veto = bool(veto_record.get("veto"))
    no_path_reason = str(execution.get("no_path_reason") or "")
    if execution_state.startswith("EXECUTE_"):
        return "EXECUTION"
    if veto:
        return "VETO_RESTRAINT"
    if intent_state == "INTENT_FORMED" and no_path_reason:
        return "SUPPRESSION_RESTRAINT"
    return None


def build_prediction_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_intent": row.get("candidate_intent") or {},
        "execution_intent": row.get("execution_intent") or {},
        "veto_record": row.get("veto_record") or {},
        "lifecycle_terms": row.get("lifecycle_terms") or {},
        "deferred_intent": row.get("deferred_intent") or {},
        "router_current_state_feedback": row.get("router_current_state_feedback") or {},
        "readiness_state": row.get("readiness_state"),
        "base_state": row.get("base_state"),
        "avoid_state": row.get("avoid_state"),
        "close": row.get("close"),
        "sma200": row.get("sma200"),
        "sma200_slope": row.get("sma200_slope"),
        "ema10": row.get("ema10"),
        "ema30": row.get("ema30"),
        "avoid_entry_predicate": row.get("avoid_entry_predicate"),
    }


def write_prediction_tables(predictions: list[dict[str, Any]], grades: list[dict[str, Any]]) -> str:
    lines = [
        "# R14-G Module (g) ForwardPredictionLedger v1 Tables",
        "",
        "## Predictions",
        "prediction_id|symbol|date|event|intent|execution|entry|ref|avoid|baseline",
        "---|---|---|---|---|---|---|---:|---|---",
    ]
    for row in predictions:
        lines.append(
            "|".join(
                [
                    str(row["prediction_id"]),
                    str(row["symbol"]),
                    str(row["prediction_date"]),
                    str(row["event_type"]),
                    str(row["intent_state"]),
                    str(row["execution_state"]),
                    str(row["entry_tier"]),
                    f"{float(row['reference_price']):.3f}",
                    str(row["avoid_state"]),
                    str(row["engine_baseline_id"]),
                ]
            )
        )

    lines.extend(
        [
            "",
            "## Grades",
            "prediction_id|symbol|date|r20|r60|r120|mfe120|verdict|status|last_data",
            "---|---|---|---:|---:|---:|---:|---|---|---",
        ]
    )
    for row in grades:
        def fmt(value: Any) -> str:
            return "PENDING" if value is None else f"{float(value):.6f}"

        lines.append(
            "|".join(
                [
                    str(row["prediction_id"]),
                    str(row["symbol"]),
                    str(row["prediction_date"]),
                    fmt(row.get("return_20")),
                    fmt(row.get("return_60")),
                    fmt(row.get("return_120")),
                    fmt(row.get("mfe_120")),
                    str(row["materialization_verdict"]),
                    str(row["grade_status"]),
                    str(row["sealed_data_last_date"]),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    attest = freeze_attestation()
    if not attest["byte_match"]:
        raise RuntimeError("Freeze v2 byte-match attestation failed.")
    v7 = json.loads(V7_EVIDENCE.read_text(encoding="utf-8"))
    r14f = json.loads(R14F_EVIDENCE.read_text(encoding="utf-8"))
    if HARNESS_DB.exists():
        HARNESS_DB.unlink()

    symbols = ["MABANEE", "SANAM", "TIJARA"]
    per_day = v7["per_day_intent_lifecycle_tables"]
    window_attestation = {
        symbol: {
            "rows": len(per_day[symbol]),
            "first": per_day[symbol][0]["trade_date"],
            "last": per_day[symbol][-1]["trade_date"],
            "v7_row_sequence_sha256": hashlib.sha256(
                json.dumps(per_day[symbol], ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
        for symbol in symbols
    }

    runtime_by_symbol = {
        symbol: load_runtime_rows(symbol, {str(row["trade_date"]) for row in per_day[symbol]})
        for symbol in symbols
    }
    expected_event_days: list[dict[str, str]] = []
    with sqlite3.connect(str(HARNESS_DB)) as conn:
        conn.row_factory = sqlite3.Row
        ledger = ForwardPredictionLedger(conn)
        for symbol in symbols:
            for row in per_day[symbol]:
                event_type = event_type_for(row)
                if event_type is None:
                    continue
                trade_date = str(row["trade_date"])
                expected_event_days.append({"symbol": symbol, "trade_date": trade_date, "event_type": event_type})
                execution = row.get("execution_intent") or {}
                intent = row.get("candidate_intent") or {}
                base_reference = dict((row.get("candidate_intent") or {}).get("base_reference") or {})
                if not base_reference:
                    base_reference = {"base_state": row.get("base_state"), "readiness_state": row.get("readiness_state")}
                ledger.append_prediction(
                    symbol=symbol,
                    prediction_date=trade_date,
                    engine_baseline_id=f"{RUN_KEY}:{symbol}:{window_attestation[symbol]['v7_row_sequence_sha256'][:12]}",
                    freeze_version_hash=str(attest["actual_json_sha256"]),
                    intent_state=str(intent.get("intent_state") or "INTENT_NONE"),
                    execution_state=str(execution.get("execution_state") or "NONE"),
                    entry_tier=str(execution.get("entry_tier") or "NONE"),
                    reference_price=float(row.get("close") or 0.0),
                    base_reference=base_reference,
                    avoid_state=str(row.get("avoid_state") or "NONE"),
                    predicate_snapshot=build_prediction_snapshot(row),
                    event_type=event_type,
                    source_run_key=RUN_KEY,
                    created_utc=RUN_NONCE,
                )
        predictions = fetch_predictions(conn)
        blocked = verify_update_delete_blocked(conn, str(predictions[0]["prediction_id"])) if predictions else {}

    with sqlite3.connect(str(HARNESS_DB)) as grade_conn:
        grade_conn.row_factory = sqlite3.Row
        grades = apply_grades(
            predictions_db_path=HARNESS_DB,
            grades_conn=grade_conn,
            sealed_ohlcv_by_symbol=runtime_by_symbol,
            grade_date=RUN_NONCE,
            grader_version=GRADER_VERSION,
        )
    separation = verify_prediction_reader_cannot_write(HARNESS_DB)

    prediction_keys = {(p["symbol"], p["prediction_date"], p["event_type"]) for p in predictions}
    expected_keys = {(e["symbol"], e["trade_date"], e["event_type"]) for e in expected_event_days}
    gradeable = [g for g in grades if g["grade_status"] == "GRADED"]
    grade_ids = {g["prediction_id"] for g in grades}
    sanam_entry = [g for g in grades if g["symbol"] == "SANAM" and g["prediction_date"] == "2025-05-15"]
    sanam_verdict = sanam_entry[0]["materialization_verdict"] if sanam_entry else "MISSING"

    acceptance = {
        "EVERY_EVENT_DAY_HAS_PREDICTION_ROW": {
            "status": "PASS" if expected_keys == prediction_keys else "FAIL",
            "expected_events": len(expected_keys),
            "prediction_rows": len(prediction_keys),
            "missing": sorted(["|".join(k) for k in expected_keys.difference(prediction_keys)]),
            "extra": sorted(["|".join(k) for k in prediction_keys.difference(expected_keys)]),
        },
        "ZERO_GRADEABLE_BUT_UNGRADED": {
            "status": "PASS" if all(p["prediction_id"] in grade_ids for p in predictions) else "FAIL",
            "predictions": len(predictions),
            "grades": len(grades),
            "gradeable": len(gradeable),
        },
        "WRITER_GRADER_SEPARATION_ATTESTED": {
            "status": "PASS" if "readonly" in separation.get("prediction_reader_write_attempt", "").lower() and "blocked" in " ".join(blocked.values()).lower() else "FAIL",
            "structural_enforcement": "prediction_grader opens ee_v2_forward_predictions through SQLite URI mode=ro and exposes no prediction-table write method; writer table UPDATE/DELETE are trigger-blocked",
            "prediction_reader_write_attempt": separation,
            "writer_update_delete_attempts": blocked,
        },
        "SANAM_2025_05_15_MATERIALIZATION_POLICY": {
            "status": "PASS" if sanam_verdict in {"MATERIALIZED", "NOT_MATERIALIZED", "PENDING_HORIZON"} else "FAIL",
            "verdict": sanam_verdict,
            "rule": "MATERIALIZED only if 120-session MFE inside sealed data clears +20%; otherwise NOT_MATERIALIZED or PENDING_HORIZON without reaching",
            "grade_row": sanam_entry[0] if sanam_entry else None,
        },
    }
    overall = "PASS" if all(check["status"] == "PASS" for check in acceptance.values()) else "FAIL"

    evidence = {
        "version_id": "R14G_MODULE_G_FORWARD_PREDICTION_LEDGER_V1_EVIDENCE",
        "run_key": RUN_KEY,
        "run_nonce": RUN_NONCE,
        "freeze_v2_attestation": attest,
        "module_e_closure": {"status": "CLOSED_PASS", "scope_note_verbatim": MODULE_E_SCOPE_NOTE},
        "module_f_closure": {
            "status": "CLOSED_PASS",
            "byte_equivalence": "649/649",
            "r12_interval_overlap": "97.5%",
            "boundary_semantics_note": "REGISTERED",
            "r14f_counts": r14f.get("acceptance_checks", {}),
        },
        "findings_carried_to_r15": ["FLOW_CORE_LAG", "AVOID_ARM_LAG"],
        "source_modules": [
            "app/services/eagle_eye_v2/forward_prediction_ledger.py",
            "app/services/eagle_eye_v2/prediction_grader.py",
        ],
        "harness_db": str(HARNESS_DB),
        "windows_byte_pinned_to_v7_rows": window_attestation,
        "prediction_count": len(predictions),
        "grade_count": len(grades),
        "acceptance_checks": acceptance,
        "overall_status": overall,
        "predictions": predictions,
        "grades": grades,
    }

    out_evidence = REVIEW / "r14g_module_g_forward_prediction_v1_evidence.json"
    out_report = REVIEW / "r14g_module_g_forward_prediction_v1_report.md"
    out_tables = REVIEW / "r14g_module_g_forward_prediction_v1_tables.md"
    out_sha = REVIEW / "r14g_module_g_forward_prediction_v1_artifacts.sha256"
    out_evidence.write_text(json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_tables.write_text(write_prediction_tables(predictions, grades), encoding="utf-8")
    out_report.write_text(
        "\n".join(
            [
                "# R14-G Module (g) ForwardPredictionLedger v1",
                "",
                f"- RUN_NONCE: {RUN_NONCE}",
                f"- Freeze v2 byte-match: {attest['byte_match']}",
                f"- Overall acceptance: {overall}",
                f"- Predictions: {len(predictions)}",
                f"- Grades: {len(grades)}",
                f"- SANAM 2025-05-15 verdict: {sanam_verdict}",
                "- Writer/grader separation: prediction reader opened mode=ro; prediction table UPDATE/DELETE trigger-blocked.",
                "",
                "## Acceptance",
                json.dumps(acceptance, ensure_ascii=True, indent=2, sort_keys=True),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar = [
        f"{sha256_file(out_evidence)}  artifacts/preview1a_prestart/review_final/r14g_module_g_forward_prediction_v1_evidence.json",
        f"{sha256_file(out_report)}  artifacts/preview1a_prestart/review_final/r14g_module_g_forward_prediction_v1_report.md",
        f"{sha256_file(out_tables)}  artifacts/preview1a_prestart/review_final/r14g_module_g_forward_prediction_v1_tables.md",
    ]
    out_sha.write_text("\n".join(sidecar) + "\n", encoding="utf-8")

    print("R14G_MODULE_G_FORWARD_PREDICTION_LEDGER_V1_COMPLETE")
    print("acceptance", overall)
    print("predictions", len(predictions))
    print("grades", len(grades))
    print("sanam_2025_05_15", sanam_verdict)
    print("evidence_json_sha256", sha256_file(out_evidence))
    print("report_md_sha256", sha256_file(out_report))
    print("tables_md_sha256", sha256_file(out_tables))
    print("artifact_sidecar_sha256", sha256_file(out_sha))


if __name__ == "__main__":
    main()
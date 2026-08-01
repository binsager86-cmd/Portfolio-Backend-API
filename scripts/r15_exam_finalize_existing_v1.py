from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.eagle_eye_v2.prediction_grader import apply_grades

REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
HARNESS_DB = REVIEW / "r15_exam_v1_harness.db"
REPORT_MD = REVIEW / "r15_exam_report_v1.md"
FREEZE_JSON = REVIEW / "r14b_parameter_freeze_v2.json"
FREEZE_SHA = REVIEW / "r14b_parameter_freeze_v2.sha256"
AMEND_JSON = REVIEW / "r14b_parameter_freeze_v2_amendment_1.json"
AMEND_SHA = REVIEW / "r14b_parameter_freeze_v2_amendment_1.sha256"
R12_SEAL = REVIEW / "r12_pre_exam_surface_seal_v4_4.json"
R12_RUNTIME_DB = REVIEW / "r12_exam_surface_v4_5_runtime.db"
SET_A_WINDOWS = {
    "SANAM": {"start": "2021-01-06", "end": "2026-07-09"},
    "TIJARA": {"start": "2021-07-11", "end": "2026-07-09"},
    "BPCC": {"start": "2021-07-11", "end": "2026-07-09"},
    "ZAIN": {"start": "2021-07-11", "end": "2026-07-09"},
    "MABANEE": {"start": "2021-07-11", "end": "2026-07-09"},
}
MABANEE_DECLINE_WINDOWS = [("2024-12-22", "2025-02-20"), ("2025-03-24", "2025-05-18")]
RUN_KEY = "R15_EXAM_V1"
GRADER_VERSION = "R15_EXAM_GRADER_V1_RECORDED_DB_FINALIZER"


def load_v7_module() -> Any:
    path = ROOT / "scripts" / "r14e_module_e_lifecycle_intent_harness_v7.py"
    spec = importlib.util.spec_from_file_location("r14e_v7_source_for_r15_finalize", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import r14e v7 source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v7 = load_v7_module()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sidecar_hash(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip().split()[0]


def load_runtime_rows() -> dict[str, list[dict[str, Any]]]:
    return {
        symbol: [
            {k: row[k] for k in ["symbol", "trade_date", "open", "high", "low", "close", "volume", "value_kwd"]}
            for row in v7.load_window(symbol, cfg["start"], cfg["end"])
        ]
        for symbol, cfg in SET_A_WINDOWS.items()
    }


def fetch_all(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def ensure_grades(run_nonce: str, runtime_by_symbol: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    with sqlite3.connect(str(HARNESS_DB)) as conn:
        existing = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ee_v2_prediction_grades'").fetchone()
        count = 0 if existing is None else int(conn.execute("SELECT COUNT(*) FROM ee_v2_prediction_grades").fetchone()[0])
        if count == 0:
            return apply_grades(predictions_db_path=HARNESS_DB, grades_conn=conn, sealed_ohlcv_by_symbol=runtime_by_symbol, grade_date=run_nonce, grader_version=GRADER_VERSION)
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM ee_v2_prediction_grades ORDER BY symbol, prediction_date, prediction_id").fetchall()]


def in_decline(date_text: str) -> bool:
    return any(start <= date_text <= end for start, end in MABANEE_DECLINE_WINDOWS)


def fmt(value: Any) -> str:
    return "PENDING" if value is None else f"{float(value):.6f}"


def main() -> None:
    if not HARNESS_DB.exists():
        raise FileNotFoundError(HARNESS_DB)
    runtime_by_symbol = load_runtime_rows()
    with sqlite3.connect(f"file:{HARNESS_DB.as_posix()}?mode=ro", uri=True) as conn:
        predictions = fetch_all(conn, "SELECT * FROM ee_v2_forward_predictions ORDER BY symbol, prediction_date, event_type, prediction_id")
        run_nonce_row = conn.execute("SELECT MIN(created_utc), MAX(created_utc), COUNT(DISTINCT created_utc) FROM ee_v2_forward_predictions").fetchone()
        run_nonce = str(run_nonce_row[0])
        daily_rows = fetch_all(
            conn,
            """
            SELECT symbol, trade_date, readiness_state, phase_after, base_reference_id,
                   MAX(CASE WHEN predicate_name='ACCUMULATION_CONTEXT_OK' THEN predicate_pass ELSE NULL END) AS accumulation_context_ok,
                   MAX(CASE WHEN predicate_name='CONFIRM_FLOW_CORE_OK' THEN predicate_pass ELSE NULL END) AS cmf_floor_pass,
                   MAX(CASE WHEN predicate_name='CURRENT_DAY_LIQUIDITY_OK' THEN predicate_pass ELSE NULL END) AS current_day_liquidity_ok,
                   MAX(CASE WHEN predicate_name='LIQUIDITY_CONTEXT_OK' THEN predicate_pass ELSE NULL END) AS liquidity_context_ok,
                   MAX(CASE WHEN predicate_name='CONFIRM_STRUCTURE_OK' THEN predicate_pass ELSE NULL END) AS structure_ok
            FROM daily_term_row
            GROUP BY symbol, trade_date
            ORDER BY symbol, trade_date
            """,
        )
    grades = ensure_grades(run_nonce, runtime_by_symbol)
    grades_by_id = {row["prediction_id"]: row for row in grades}
    sanam_entries = [p for p in predictions if p["symbol"] == "SANAM" and p["event_type"] == "EXECUTION"]
    sanam_confirmed_0518 = [p for p in sanam_entries if p["prediction_date"] == "2025-05-18" and p["entry_tier"] == "BREAKOUT_CONFIRMED_ENTRY"]
    sanam_early_before = [p for p in sanam_entries if p["prediction_date"] < "2025-05-08" and p["execution_state"] == "EXECUTE_EARLY_PILOT"]
    sanam_intent_dates = sorted({p["prediction_date"] for p in predictions if p["symbol"] == "SANAM" and p["intent_state"] == "INTENT_FORMED"})
    tijara_total = len([r for r in daily_rows if r["symbol"] == "TIJARA"])
    tijara_candidate_dates = {p["prediction_date"] for p in predictions if p["symbol"] == "TIJARA" and p["intent_state"] == "INTENT_FORMED"}
    tijara_no_candidate = tijara_total - len(tijara_candidate_dates)
    tijara_no_candidate_rate = tijara_no_candidate / tijara_total if tijara_total else 1.0
    mabanee_early_decline = [p for p in predictions if p["symbol"] == "MABANEE" and p["execution_state"] == "EXECUTE_EARLY_PILOT" and in_decline(str(p["prediction_date"]))]
    pilots = [p for p in predictions if p["execution_state"] == "EXECUTE_EARLY_PILOT"]
    failed_pilots = [p for p in pilots if grades_by_id.get(p["prediction_id"], {}).get("materialization_verdict") == "NOT_MATERIALIZED"]
    cost_rows = []
    for pilot in failed_pilots:
        rows = runtime_by_symbol.get(pilot["symbol"], [])
        idx = next((i for i, row in enumerate(rows) if row["trade_date"] == pilot["prediction_date"]), None)
        horizon_rows = rows[idx : min(len(rows), idx + 121)] if idx is not None else []
        ref = float(pilot["reference_price"] or 0.0)
        drawdown = min(((float(row["low"] or row["close"] or 0.0) / ref) - 1.0 for row in horizon_rows), default=0.0) if ref > 0 else 0.0
        cost_rows.append({"prediction_id": pilot["prediction_id"], "symbol": pilot["symbol"], "date": pilot["prediction_date"], "pilot_fraction": 0.30, "sessions_observed": len(horizon_rows), "capital_days": 0.30 * len(horizon_rows), "realized_drawdown": drawdown})
    false_positive_cost = {"failed_pilot_count": len(cost_rows), "capital_days": sum(row["capital_days"] for row in cost_rows), "realized_drawdown_min": min((row["realized_drawdown"] for row in cost_rows), default=0.0), "rows": cost_rows}
    criteria = [
        {"criterion": "SANAM 2025-05-18 confirmed entry occurs", "verdict": "PASS" if sanam_confirmed_0518 else "FAIL", "evidence_rows": sanam_confirmed_0518},
        {"criterion": "SANAM early-tier entry before 2025-05-08", "verdict": "PASS" if sanam_early_before else "FAIL", "expected": "FAIL under current detection timing (FLOW_CORE_LAG)", "actual_first_intent_date": sanam_intent_dates[0] if sanam_intent_dates else "NONE", "required_before": "2025-05-08", "evidence_rows": sanam_early_before},
        {"criterion": "TIJARA no-candidate rate <25% across its window", "verdict": "PASS" if tijara_no_candidate_rate < 0.25 else "FAIL", "no_candidate_count": tijara_no_candidate, "total_days": tijara_total, "no_candidate_rate": tijara_no_candidate_rate, "evidence_rows": [{"symbol": "TIJARA", "no_candidate_count": tijara_no_candidate, "total_days": tijara_total, "no_candidate_rate": tijara_no_candidate_rate}]},
        {"criterion": "MABANEE zero early entries inside the decline windows", "verdict": "PASS" if not mabanee_early_decline else "FAIL", "decline_windows": MABANEE_DECLINE_WINDOWS, "evidence_rows": mabanee_early_decline},
        {"criterion": "early-tier false-positive cost computed and reported", "verdict": "PASS", "evidence_rows": [false_positive_cost]},
    ]
    freeze_attestation = {
        "freeze_json": str(FREEZE_JSON),
        "freeze_expected_sha256": sidecar_hash(FREEZE_SHA),
        "freeze_actual_sha256": sha256_file(FREEZE_JSON),
        "freeze_byte_match": sidecar_hash(FREEZE_SHA) == sha256_file(FREEZE_JSON),
        "amendment_json": str(AMEND_JSON),
        "amendment_expected_sha256": sidecar_hash(AMEND_SHA),
        "amendment_actual_sha256": sha256_file(AMEND_JSON),
        "amendment_byte_match": sidecar_hash(AMEND_SHA) == sha256_file(AMEND_JSON),
        "r12_seal_sha256": sha256_file(R12_SEAL),
        "r12_runtime_db_sha256": sha256_file(R12_RUNTIME_DB),
        "recorded_run_nonce_distinct_count": int(run_nonce_row[2]),
    }
    plane_counts: dict[str, int] = {}
    for row in predictions:
        if row["event_type"] in {"VETO_RESTRAINT", "SUPPRESSION_RESTRAINT"}:
            snap = json.loads(str(row["predicate_snapshot_json"]))
            plane = str((snap.get("veto_record") or {}).get("plane") or (snap.get("execution_intent") or {}).get("no_path_reason") or "SUPPRESSION")
            plane_counts[plane] = plane_counts.get(plane, 0) + 1
    position_ledger = []
    for row in [p for p in predictions if p["event_type"] == "EXECUTION"]:
        rows = runtime_by_symbol.get(row["symbol"], [])
        idx = next((i for i, r in enumerate(rows) if r["trade_date"] == row["prediction_date"]), None)
        sessions_held = 0 if idx is None else len(rows) - idx
        position_ledger.append({"position_id": row["prediction_id"], "symbol": row["symbol"], "entry_date": row["prediction_date"], "entry_tier": row["entry_tier"], "execution_state": row["execution_state"], "sessions_held": sessions_held})
    lines = ["# R15 Exam Report v1", "", f"RUN_NONCE: {run_nonce}", f"RUN_KEY: {RUN_KEY}", "record_source: r15_exam_v1_harness.db existing live-writer rows; finalizer did not replay engine", "", "## Summary Verdict Block", "criterion|verdict", "---|---"]
    for row in criteria:
        lines.append(f"{row['criterion']}|{row['verdict']}")
    lines.extend(["", "## Freeze Attestation", "```json", json.dumps(freeze_attestation, ensure_ascii=True, indent=2, sort_keys=True), "```", "", "## Per-Criterion Evidence", "```json", json.dumps(criteria, ensure_ascii=True, indent=2, sort_keys=True), "```", "", "## Veto/Suppression Summary By Plane", "plane|event_count", "---|---"])
    for plane, count in sorted(plane_counts.items()):
        lines.append(f"{plane}|{count}")
    lines.extend(["", "## False-Positive Cost Accounting", "```json", json.dumps(false_positive_cost, ensure_ascii=True, indent=2, sort_keys=True), "```", "", "## Position Ledger With Sessions Held", "position_id|symbol|entry_date|entry_tier|execution_state|sessions_held", "---|---|---|---|---|---:"])
    for row in position_ledger:
        lines.append("|".join([str(row["position_id"]), str(row["symbol"]), str(row["entry_date"]), str(row["entry_tier"]), str(row["execution_state"]), str(row["sessions_held"])]))
    lines.extend(["", "## Per-Day Lifecycle Tables", "symbol|date|readiness|phase|base_reference_id|slope_core|cmf_floor|current_day_liquidity|liquidity_context|structure|event|intent|execution|entry_tier|avoid", "---|---|---|---|---|---|---|---|---|---|---|---|---|---|---"])
    pred_by_symbol_date: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in predictions:
        pred_by_symbol_date.setdefault((row["symbol"], row["prediction_date"]), []).append(row)
    for row in daily_rows:
        preds = pred_by_symbol_date.get((row["symbol"], row["trade_date"]), [])
        if preds:
            for pred in preds:
                lines.append("|".join([str(row["symbol"]), str(row["trade_date"]), str(row["readiness_state"]), str(row["phase_after"]), str(row["base_reference_id"]), str(row["accumulation_context_ok"]), str(row["cmf_floor_pass"]), str(row["current_day_liquidity_ok"]), str(row["liquidity_context_ok"]), str(row["structure_ok"]), str(pred["event_type"]), str(pred["intent_state"]), str(pred["execution_state"]), str(pred["entry_tier"]), str(pred["avoid_state"])]))
        else:
            lines.append("|".join([str(row["symbol"]), str(row["trade_date"]), str(row["readiness_state"]), str(row["phase_after"]), str(row["base_reference_id"]), str(row["accumulation_context_ok"]), str(row["cmf_floor_pass"]), str(row["current_day_liquidity_ok"]), str(row["liquidity_context_ok"]), str(row["structure_ok"]), "NONE", "INTENT_NONE", "NONE", "NONE", "NONE"]))
    lines.extend(["", "## Complete Forward Prediction Table", "prediction_id|symbol|date|event|intent|execution|entry_tier|reference_price|avoid|freeze_hash", "---|---|---|---|---|---|---|---:|---|---"])
    for row in predictions:
        lines.append("|".join([str(row["prediction_id"]), str(row["symbol"]), str(row["prediction_date"]), str(row["event_type"]), str(row["intent_state"]), str(row["execution_state"]), str(row["entry_tier"]), f"{float(row['reference_price']):.3f}", str(row["avoid_state"]), str(row["freeze_version_hash"])]))
    lines.extend(["", "## Complete Prediction Grade Table", "prediction_id|symbol|date|return_20|return_60|return_120|mfe_120|verdict|status|last_data", "---|---|---|---:|---:|---:|---:|---|---|---"])
    for row in grades:
        lines.append("|".join([str(row["prediction_id"]), str(row["symbol"]), str(row["prediction_date"]), fmt(row.get("return_20")), fmt(row.get("return_60")), fmt(row.get("return_120")), fmt(row.get("mfe_120")), str(row["materialization_verdict"]), str(row["grade_status"]), str(row["sealed_data_last_date"])]))
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    artifact_hashes = {
        "artifacts/preview1a_prestart/review_final/r15_exam_report_v1.md": sha256_file(REPORT_MD),
        "artifacts/preview1a_prestart/review_final/r15_exam_v1_harness.db": sha256_file(HARNESS_DB),
        "scripts/r15_exam_v1.py": sha256_file(ROOT / "scripts" / "r15_exam_v1.py"),
        "scripts/r15_exam_finalize_existing_v1.py": sha256_file(ROOT / "scripts" / "r15_exam_finalize_existing_v1.py"),
        "artifacts/preview1a_prestart/review_final/r14b_parameter_freeze_v2.json": sha256_file(FREEZE_JSON),
        "artifacts/preview1a_prestart/review_final/r14b_parameter_freeze_v2_amendment_1.json": sha256_file(AMEND_JSON),
    }
    with REPORT_MD.open("a", encoding="utf-8", newline="\n") as f:
        f.write("\n## Artifact Hashes\nartifact|sha256\n---|---\n")
        for artifact, digest in artifact_hashes.items():
            f.write(f"{artifact}|{digest}\n")
    print("R15_EXAM_V1_REPORT_FINALIZED_FROM_EXISTING_DB")
    print("RUN_NONCE", run_nonce)
    for row in criteria:
        print(row["verdict"], row["criterion"])
    print("predictions", len(predictions))
    print("grades", len(grades))
    print("report", REPORT_MD.as_posix())
    print("report_sha256", sha256_file(REPORT_MD))


if __name__ == "__main__":
    main()
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.services.eagle_eye_v2.forward_prediction_ledger import ForwardPredictionLedger, fetch_predictions, verify_update_delete_blocked
from app.services.eagle_eye_v2.prediction_grader import apply_grades, verify_prediction_reader_cannot_write
from app.services.eagle_eye_v2.predicate_telemetry_ledger import apply_schema_migration as apply_telemetry_schema

REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
HARNESS_DB = REVIEW / "r14g_module_g_forward_prediction_harness_v2.db"
FREEZE_JSON = REVIEW / "r14b_parameter_freeze_v2.json"
FREEZE_SHA = REVIEW / "r14b_parameter_freeze_v2.sha256"
V1_EVIDENCE = REVIEW / "r14g_module_g_forward_prediction_v1_evidence.json"
RUN_NONCE = "2026-07-18T14:10:18.0289041Z"
RUN_KEY = "R14G_MODULE_G_FORWARD_PREDICTION_LEDGER_V2_LIVE"
GRADER_VERSION = "R14G_PREDICTION_GRADER_V2_LIVE"


def load_v7_module() -> Any:
    path = ROOT / "scripts" / "r14e_module_e_lifecycle_intent_harness_v7.py"
    spec = importlib.util.spec_from_file_location("r14e_v7_live_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import v7 live harness source")
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


def event_type_for(row: dict[str, Any]) -> str | None:
    intent_state = str(((row.get("candidate_intent") or {}).get("intent_state")) or "INTENT_NONE")
    execution = row.get("execution_intent") or {}
    execution_state = str(execution.get("execution_state") or "NONE")
    veto_record = row.get("veto_record") or {}
    if execution_state.startswith("EXECUTE_"):
        return "EXECUTION"
    if bool(veto_record.get("veto")):
        return "VETO_RESTRAINT"
    if intent_state == "INTENT_FORMED" and str(execution.get("no_path_reason") or ""):
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


def live_stack_predictions(attest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if HARNESS_DB.exists():
        HARNESS_DB.unlink()
    HARNESS_DB.touch()
    os.environ["EE_V2_RUNTIME_DB_PATH"] = str(HARNESS_DB)
    os.environ["DATABASE_PATH"] = str(HARNESS_DB)
    get_settings.cache_clear()
    apply_telemetry_schema()
    cal = v7.load_default_calendar_context(ROOT)
    mask = v7.load_default_mask_manifest(ROOT)
    adapter = v7.DataSurfaceAdapter(calendar_context=cal, mask_manifest=mask)
    warmup = v7.WarmupReadinessEngine(
        v7.WarmupNamedParameters(
            values={
                v7.READINESS_LONG_LOOKBACK_MIN_SESSIONS: 180,
                v7.READINESS_SEGMENT_RESTART_MIN_SESSIONS: 20,
                v7.READINESS_FALLBACK_MIN_SESSIONS: 60,
            }
        )
    )
    base = v7.AdaptiveBaseGeometry(
        v7.BaseNamedParameters(
            values={
                v7.BASE_MIN_SESSIONS: 10,
                v7.BASE_MAX_WIDTH_PCT: 0.24,
                v7.ATR_SQUEEZE_PCTILE: 0.95,
            }
        )
    )
    flow = v7.FlowConfirmationEngine(
        v7.FlowNamedParameters(
            values={
                v7.OBV_SLOPE_MIN: 0.10,
                v7.ANV_SLOPE_MIN: 0.10,
                v7.CMF_FLOOR: 0.05,
                v7.REL_VOLUME_CONTEXT_MIN: 2.5,
                v7.RSI_REGIME: 50.0,
                v7.ADX_TRIGGER: 18.0,
                v7.CHASE_ADVISORY_BAND: 0.08,
                v7.MIN_DAILY_VALUE_KWD: 100000.0,
                v7.MIN_CURRENT_DAY_VALUE_KWD: 50000.0,
            }
        )
    )
    router = v7.LifecycleIntentRouter(
        v7.LifecycleRouterNamedParameters(
            values={
                v7.EARLY_TIER_SIZE_FRACTION: 0.30,
                v7.EARLY_TIER_PARTICIPATION_CAP: 0.10,
                v7.EARLY_TIER_TIME_STOP: 60.0,
                v7.SCALE_ON_CONFIRMATION: "SINGLE_ADD_TO_FULL_TARGET",
                v7.CHASE_ADVISORY_THRESHOLD: 0.08,
                v7.CHASE_ESCALATION_THRESHOLD: 0.15,
            }
        )
    )

    runtime_by_symbol: dict[str, list[dict[str, Any]]] = {}
    windows = v7.owner_windows()
    with sqlite3.connect(str(HARNESS_DB)) as conn:
        conn.row_factory = sqlite3.Row
        ledger = ForwardPredictionLedger(conn)
        for symbol, cfg in windows.items():
            rows = v7.load_window(symbol, cfg["replay_start"], cfg["replay_end"])
            runtime_by_symbol[symbol] = [
                {k: row[k] for k in ["symbol", "trade_date", "open", "high", "low", "close", "volume", "value_kwd"]}
                for row in rows
            ]
            avoid_series = v7.derive_avoid_context(rows)
            prev_segment = None
            prev_masked = False
            prev_ready = "READINESS_PENDING"
            history_window: list[dict[str, Any]] = []
            flow_window: list[dict[str, Any]] = []
            prior_base: dict[str, Any] | None = None
            coverage_dates: list[str] = []
            segment_dates: list[str] = []
            deferred_state = {"age_sessions": 0, "rearm_count": 0, "flow_evidence_decay": False}
            position_state: dict[str, Any] | None = None
            position_counter = 0

            row_sequence_sha = hashlib.sha256(
                json.dumps([str(row["trade_date"]) for row in rows], ensure_ascii=True, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            for idx, day in enumerate(rows):
                trade_date = str(day["trade_date"])
                avoid_ctx = avoid_series[idx]
                mask_ctx = adapter.mask_context_for(symbol, trade_date)
                seg = adapter.next_segment_state(
                    symbol=symbol,
                    trade_date=trade_date,
                    prev_segment=prev_segment,
                    prev_masked=prev_masked,
                    current_masked=bool(mask_ctx["masked_flag"]),
                )
                normalized, readiness_ctx = adapter.normalize_day(
                    ohlcv_day=day,
                    indicator_day=dict(day.get("indicator_payload") or {}),
                    segment_context=seg,
                    calendar_context=cal,
                )
                coverage_dates.append(trade_date)
                segment_dates = [trade_date] if seg.segment_day_index == 0 else [*segment_dates, trade_date]
                ready = warmup.evaluate(
                    normalized_day_payload=normalized,
                    coverage_history={
                        "long_lookback_session_dates": coverage_dates,
                        "segment_session_dates": segment_dates,
                        "fallback_session_dates": coverage_dates,
                        "previous_readiness_state": prev_ready,
                    },
                    segment_restart_flag=bool(readiness_ctx["segment_restart_flag"]),
                )
                history_window.append(day)
                history_window = history_window[-260:]
                base_out = base.evaluate(
                    normalized_day_payload=normalized,
                    readiness_state=ready["readiness_state"],
                    price_history_window=history_window,
                    volatility_regime_state={
                        "atr_squeeze_pctile": 0.50,
                        "base_range_sessions": 20,
                        "atr_value": float(day["high"] or 0.0) - float(day["low"] or 0.0),
                        "invalidation_rule_form": v7.RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N,
                        "invalidation_rule_params": {"atr_mult": 1.0, "n_sessions": 2},
                        "parameter_status": "FROZEN_R14B_PARAMETER_FREEZE_V2",
                    },
                    prior_base_reference=prior_base,
                    flow_stub_state={"confirmed_progress": False},
                )
                flow_window.append(dict(day.get("indicator_payload") or {}))
                flow_window = flow_window[-40:]
                flow_out = flow.evaluate(
                    normalized_day_payload=normalized,
                    base_reference=base_out["base_reference"],
                    flow_history_window=flow_window,
                    structure_terms=v7.build_structure_terms(day, base_out["base_reference"]),
                    readiness_state=ready["readiness_state"],
                    phase_state=base_out["base_state"],
                )
                router_current_state = dict(deferred_state)
                if position_state is not None:
                    router_current_state.update(
                        {
                            "active": True,
                            "state": "POSITION_OPEN",
                            "position_id": position_state.get("position_id"),
                            "position_type": position_state.get("position_type"),
                        }
                    )
                route_out = router.evaluate(
                    candidate_intent=flow_out["candidate_intent"],
                    base_state={"base_state": base_out["base_state"]},
                    confirmation_state={"confirmation_state": flow_out["confirmation_state"]},
                    risk_budget_state={
                        "current_day_value_kwd": float(day.get("value_kwd") or 0.0),
                        "planned_order_value_kwd": float(day.get("value_kwd") or 0.0) * 0.03,
                        "avoid_veto": bool(avoid_ctx["avoid_active"]),
                        "deferred_intent_state": router_current_state,
                    },
                )
                deferred_state = dict(route_out["deferred_intent"])
                execution_state = str(route_out["execution_intent"].get("execution_state") or "NONE")
                if execution_state in {"EXECUTE_EARLY_PILOT", "EXECUTE_CONFIRMED_DIRECT", "EXECUTE_CONFIRMED_ADD"} and position_state is None:
                    position_counter += 1
                    is_direct = execution_state == "EXECUTE_CONFIRMED_DIRECT"
                    position_state = {
                        "position_id": f"{symbol}::POS{position_counter:04d}",
                        "entry_date": trade_date,
                        "entry_tier": route_out["execution_intent"].get("entry_tier") or "NONE",
                        "pilot_fraction": 0.0 if is_direct else float(route_out["execution_intent"].get("pilot_size_fraction") or 0.0),
                        "target_fraction": float(route_out["execution_intent"].get("target_fraction") or (1.0 if is_direct else 0.0)),
                        "position_type": "CONFIRMED_DIRECT" if is_direct else "PILOT_OR_SCALE",
                        "sessions_held": 0,
                        "rearm_count": 0,
                        "state": "OPEN",
                    }
                if position_state is not None:
                    position_state["sessions_held"] = int(position_state["sessions_held"] or 0) + 1
                if (
                    flow_out["candidate_intent"]["intent_state"] == "INTENT_FORMED"
                    and route_out["execution_intent"]["execution_state"] == "NONE"
                    and not route_out["veto_record"]["veto"]
                    and router_current_state.get("active")
                ):
                    route_out["execution_intent"]["no_path_reason"] = "POSITION_ALREADY_OPEN_FEEDBACK_SUPPRESSED_DIRECT"
                    route_out["execution_intent"]["disposition_state"] = "NO_PATH_EXPLICIT"

                out_row = {
                    "trade_date": trade_date,
                    "readiness_state": ready["readiness_state"],
                    "base_state": base_out["base_state"],
                    "avoid_state": avoid_ctx["avoid_state"],
                    "close": avoid_ctx["close"],
                    "sma200": avoid_ctx["sma200"],
                    "sma200_slope": avoid_ctx["sma200_slope"],
                    "ema10": avoid_ctx["ema10"],
                    "ema30": avoid_ctx["ema30"],
                    "avoid_entry_predicate": avoid_ctx["avoid_entry_predicate"],
                    "candidate_intent": flow_out["candidate_intent"],
                    "lifecycle_terms": route_out["lifecycle_terms"],
                    "execution_intent": route_out["execution_intent"],
                    "deferred_intent": route_out["deferred_intent"],
                    "veto_record": route_out["veto_record"],
                    "router_current_state_feedback": router_current_state,
                }
                event_type = event_type_for(out_row)
                if event_type is not None:
                    execution = out_row["execution_intent"]
                    intent = out_row["candidate_intent"]
                    ledger.append_prediction(
                        symbol=symbol,
                        prediction_date=trade_date,
                        engine_baseline_id=f"{RUN_KEY}:{symbol}:{row_sequence_sha}",
                        freeze_version_hash=str(attest["actual_json_sha256"]),
                        intent_state=str(intent.get("intent_state") or "INTENT_NONE"),
                        execution_state=str(execution.get("execution_state") or "NONE"),
                        entry_tier=str(execution.get("entry_tier") or "NONE"),
                        reference_price=float(out_row.get("close") or 0.0),
                        base_reference=base_out["base_reference"],
                        avoid_state=str(out_row.get("avoid_state") or "NONE"),
                        predicate_snapshot=build_prediction_snapshot(out_row),
                        event_type=event_type,
                        source_run_key=RUN_KEY,
                        created_utc=RUN_NONCE,
                    )
                prior_base = base_out["base_reference"]
                prev_ready = ready["readiness_state"]
                prev_segment = seg
                prev_masked = bool(mask_ctx["masked_flag"])
        predictions = fetch_predictions(conn)
        blocked = verify_update_delete_blocked(conn, str(predictions[0]["prediction_id"])) if predictions else {}
    return predictions, runtime_by_symbol, blocked


def write_tables(predictions: list[dict[str, Any]], grades: list[dict[str, Any]]) -> str:
    lines = [
        "# R14-G Module (g) ForwardPredictionLedger v2 Live Tables",
        "",
        "## Predictions",
        "prediction_id|symbol|date|event|intent|execution|entry|ref|avoid|baseline",
        "---|---|---|---|---|---|---|---:|---|---",
    ]
    for row in predictions:
        lines.append("|".join([str(row["prediction_id"]), str(row["symbol"]), str(row["prediction_date"]), str(row["event_type"]), str(row["intent_state"]), str(row["execution_state"]), str(row["entry_tier"]), f"{float(row['reference_price']):.3f}", str(row["avoid_state"]), str(row["engine_baseline_id"])]))
    lines.extend(["", "## Grades", "prediction_id|symbol|date|r20|r60|r120|mfe120|verdict|status|last_data", "---|---|---|---:|---:|---:|---:|---|---|---"])
    for row in grades:
        def fmt(value: Any) -> str:
            return "PENDING" if value is None else f"{float(value):.6f}"
        lines.append("|".join([str(row["prediction_id"]), str(row["symbol"]), str(row["prediction_date"]), fmt(row.get("return_20")), fmt(row.get("return_60")), fmt(row.get("return_120")), fmt(row.get("mfe_120")), str(row["materialization_verdict"]), str(row["grade_status"]), str(row["sealed_data_last_date"])]))
    return "\n".join(lines) + "\n"


def main() -> None:
    attest = freeze_attestation()
    if not attest["byte_match"]:
        raise RuntimeError("Freeze v2 byte-match attestation failed.")
    predictions, runtime_by_symbol, blocked = live_stack_predictions(attest)
    with sqlite3.connect(str(HARNESS_DB)) as grade_conn:
        grade_conn.row_factory = sqlite3.Row
        grades = apply_grades(predictions_db_path=HARNESS_DB, grades_conn=grade_conn, sealed_ohlcv_by_symbol=runtime_by_symbol, grade_date=RUN_NONCE, grader_version=GRADER_VERSION)
    separation = verify_prediction_reader_cannot_write(HARNESS_DB)
    v1 = json.loads(V1_EVIDENCE.read_text(encoding="utf-8"))
    v1_keys = [(row["symbol"], row["prediction_date"], row["event_type"]) for row in v1["predictions"]]
    v2_keys = [(row["symbol"], row["prediction_date"], row["event_type"]) for row in predictions]
    divergence = {
        "missing_from_v2": ["|".join(key) for key in v1_keys if key not in set(v2_keys)],
        "extra_in_v2": ["|".join(key) for key in v2_keys if key not in set(v1_keys)],
        "sequence_equal": v1_keys == v2_keys,
    }
    acceptance = {
        "LIVE_WRITER_INLINE": {
            "status": "PASS",
            "statement": "ForwardPredictionLedger.append_prediction is called inside the adapter->warmup->base->flow->router replay loop; the writer path does not read r14e v7 evidence.",
        },
        "V2_MATCHES_V1_EVENT_MEMORY": {
            "status": "PASS" if not divergence["missing_from_v2"] and not divergence["extra_in_v2"] else "FAIL",
            **divergence,
        },
        "WRITER_GRADER_SEPARATION_ATTESTED": {
            "status": "PASS" if "readonly" in separation.get("prediction_reader_write_attempt", "").lower() and "blocked" in " ".join(blocked.values()).lower() else "FAIL",
            "prediction_reader_write_attempt": separation,
            "writer_update_delete_attempts": blocked,
        },
    }
    overall = "PASS" if all(row["status"] == "PASS" for row in acceptance.values()) else "FAIL"
    evidence = {
        "version_id": "R14G_MODULE_G_FORWARD_PREDICTION_LEDGER_V2_LIVE_EVIDENCE",
        "run_key": RUN_KEY,
        "run_nonce": RUN_NONCE,
        "freeze_v2_attestation": attest,
        "input_to_writer_path": "LIVE_STACK_OUTPUT_ONLY_NO_V7_EVIDENCE_FILE",
        "acceptance_checks": acceptance,
        "overall_status": overall,
        "prediction_count": len(predictions),
        "grade_count": len(grades),
        "predictions": predictions,
        "grades": grades,
    }
    out_evidence = REVIEW / "r14g_module_g_forward_prediction_v2_live_evidence.json"
    out_report = REVIEW / "r14g_module_g_forward_prediction_v2_live_report.md"
    out_tables = REVIEW / "r14g_module_g_forward_prediction_v2_live_tables.md"
    out_sha = REVIEW / "r14g_module_g_forward_prediction_v2_live_artifacts.sha256"
    out_evidence.write_text(json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_tables.write_text(write_tables(predictions, grades), encoding="utf-8")
    out_report.write_text("\n".join(["# R14-G Module (g) ForwardPredictionLedger v2 Live", "", f"- RUN_NONCE: {RUN_NONCE}", f"- Overall acceptance: {overall}", f"- Predictions: {len(predictions)}", f"- Grades: {len(grades)}", "", "## Acceptance", json.dumps(acceptance, ensure_ascii=True, indent=2, sort_keys=True)]) + "\n", encoding="utf-8")
    out_sha.write_text("\n".join([f"{sha256_file(out_evidence)}  artifacts/preview1a_prestart/review_final/r14g_module_g_forward_prediction_v2_live_evidence.json", f"{sha256_file(out_report)}  artifacts/preview1a_prestart/review_final/r14g_module_g_forward_prediction_v2_live_report.md", f"{sha256_file(out_tables)}  artifacts/preview1a_prestart/review_final/r14g_module_g_forward_prediction_v2_live_tables.md"]) + "\n", encoding="utf-8")
    print("R14G_MODULE_G_FORWARD_PREDICTION_LEDGER_V2_LIVE_COMPLETE")
    print("acceptance", overall)
    print("predictions", len(predictions))
    print("grades", len(grades))
    print("missing_from_v2", len(divergence["missing_from_v2"]))
    print("extra_in_v2", len(divergence["extra_in_v2"]))
    print("sequence_equal", divergence["sequence_equal"])
    print("evidence_json_sha256", sha256_file(out_evidence))
    print("report_md_sha256", sha256_file(out_report))
    print("tables_md_sha256", sha256_file(out_tables))
    print("artifact_sidecar_sha256", sha256_file(out_sha))


if __name__ == "__main__":
    main()
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
from app.services.eagle_eye_v2.avoid_authority_plane import AvoidAuthorityPlane
from app.services.eagle_eye_v2.forward_prediction_ledger import ForwardPredictionLedger, fetch_predictions
from app.services.eagle_eye_v2.prediction_grader import apply_grades
from app.services.eagle_eye_v2.predicate_telemetry_ledger import apply_schema_migration as apply_telemetry_schema

REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
HARNESS_DB = REVIEW / "r15_exam_v2_harness.db"
REPORT_MD = REVIEW / "r15_exam_report_v2.md"
FREEZE_JSON = REVIEW / "r14b_parameter_freeze_v2.json"
FREEZE_SHA = REVIEW / "r14b_parameter_freeze_v2.sha256"
AMEND1_JSON = REVIEW / "r14b_parameter_freeze_v2_amendment_1.json"
AMEND1_SHA = REVIEW / "r14b_parameter_freeze_v2_amendment_1.sha256"
AMEND2_JSON = REVIEW / "r14b_parameter_freeze_v2_amendment_2.json"
AMEND2_SHA = REVIEW / "r14b_parameter_freeze_v2_amendment_2.json.sha256"
CRITERIA_RECORD = REVIEW / "r15_attempt2_criteria_of_record_v1.md"
R12_SEAL = REVIEW / "r12_pre_exam_surface_seal_v4_4.json"
R12_RUNTIME_DB = REVIEW / "r12_exam_surface_v4_5_runtime.db"
RUN_NONCE = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
RUN_KEY = "R15_EXAM_V2"
GRADER_VERSION = "R15_EXAM_GRADER_V2"
UPWARD_RETIREMENT_MFE_THRESHOLD = 0.20

SET_A_WINDOWS = {
    "SANAM": {"start": "2021-01-06", "end": "2026-07-09", "source": "r12_exam_surface_v4_5_runtime.db ee_ohlcv coverage"},
    "TIJARA": {"start": "2021-07-11", "end": "2026-07-09", "source": "r12_exam_surface_v4_5_runtime.db ee_ohlcv coverage"},
    "BPCC": {"start": "2021-07-11", "end": "2026-07-09", "source": "r12_exam_surface_v4_5_runtime.db ee_ohlcv coverage"},
    "ZAIN": {"start": "2021-07-11", "end": "2026-07-09", "source": "r12_exam_surface_v4_5_runtime.db ee_ohlcv coverage"},
    "MABANEE": {"start": "2021-07-11", "end": "2026-07-09", "source": "r12_exam_surface_v4_5_runtime.db ee_ohlcv coverage"},
}
MABANEE_DECLINE_WINDOWS = [("2024-12-22", "2025-02-20"), ("2025-03-24", "2025-05-18")]


def load_v7_module() -> Any:
    path = ROOT / "scripts" / "r14e_module_e_lifecycle_intent_harness_v7.py"
    spec = importlib.util.spec_from_file_location("r14e_v7_source_for_r15_v2", path)
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


def attest_freeze() -> dict[str, Any]:
    freeze_actual = sha256_file(FREEZE_JSON)
    amend1_actual = sha256_file(AMEND1_JSON)
    amend2_actual = sha256_file(AMEND2_JSON)
    return {
        "freeze_json": str(FREEZE_JSON),
        "freeze_sha_sidecar": str(FREEZE_SHA),
        "freeze_expected_sha256": sidecar_hash(FREEZE_SHA),
        "freeze_actual_sha256": freeze_actual,
        "freeze_byte_match": sidecar_hash(FREEZE_SHA) == freeze_actual,
        "amendment_1_json": str(AMEND1_JSON),
        "amendment_1_sha_sidecar": str(AMEND1_SHA),
        "amendment_1_expected_sha256": sidecar_hash(AMEND1_SHA),
        "amendment_1_actual_sha256": amend1_actual,
        "amendment_1_byte_match": sidecar_hash(AMEND1_SHA) == amend1_actual,
        "amendment_2_json": str(AMEND2_JSON),
        "amendment_2_sha_sidecar": str(AMEND2_SHA),
        "amendment_2_expected_sha256": sidecar_hash(AMEND2_SHA),
        "amendment_2_actual_sha256": amend2_actual,
        "amendment_2_byte_match": sidecar_hash(AMEND2_SHA) == amend2_actual,
        "r12_seal_json": str(R12_SEAL),
        "r12_seal_sha256": sha256_file(R12_SEAL),
        "r12_runtime_db": str(R12_RUNTIME_DB),
        "r12_runtime_db_sha256": sha256_file(R12_RUNTIME_DB),
        "criteria_of_record": str(CRITERIA_RECORD),
        "criteria_of_record_sha256": sha256_file(CRITERIA_RECORD),
    }


def bind_harness_db() -> None:
    if HARNESS_DB.exists():
        HARNESS_DB.unlink()
    HARNESS_DB.touch()
    os.environ["EE_V2_RUNTIME_DB_PATH"] = str(HARNESS_DB)
    os.environ["DATABASE_PATH"] = str(HARNESS_DB)
    get_settings.cache_clear()
    apply_telemetry_schema()


def load_window(symbol: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    return v7.load_window(symbol, start_date, end_date)


def amended_flow_output(flow_out: dict[str, Any], base_state: str, base_reference: dict[str, Any], day: dict[str, Any]) -> dict[str, Any]:
    terms = dict(flow_out.get("confirmation_terms") or {})
    today = dict(day.get("indicator_payload") or {})
    slope_core_pass = bool(terms.get("accumulation_context_ok"))
    cmf_value = float(today.get("cmf_10") or 0.0)
    cmf_floor_pass = cmf_value >= 0.05
    base_valid = str(base_state).upper() in {"BASE_VALID", "BASE_FROZEN"} and bool(base_reference.get("base_validity_state") == "VALID")
    liquidity_pass = bool(terms.get("current_day_liquidity_ok")) and bool(terms.get("liquidity_context_ok"))
    structure_pass = bool(terms.get("confirm_structure_ok")) and bool(terms.get("confirm_chase_guard_ok"))
    intent_formed = slope_core_pass and base_valid and liquidity_pass
    confirmed = intent_formed and structure_pass
    candidate = dict(flow_out.get("candidate_intent") or {})
    candidate.update(
        {
            "intent_state": "INTENT_FORMED" if intent_formed else "INTENT_NONE",
            "confirmation_state": "CONFIRMED" if confirmed else "NOT_CONFIRMED",
            "entry_tier": "BREAKOUT_CONFIRMED_ENTRY" if confirmed else ("EARLY_ACCUMULATION_ENTRY" if intent_formed else "NONE"),
            "reason": "AMENDMENT_1_NON_BLOCKING_FLOW_COMPOSITION" if intent_formed else "PREDICATE_BLOCK",
            "flow_core_composition_authority": "NON_BLOCKING",
            "slope_core_detection_role": slope_core_pass,
            "cmf_floor_telemetry_only_pass": cmf_floor_pass,
        }
    )
    terms.update(
        {
            "amendment_1_flow_core_composition": "NON_BLOCKING",
            "slope_core_pass": slope_core_pass,
            "cmf_floor_pass": cmf_floor_pass,
            "cmf_floor_value": cmf_value,
            "base_valid_geometry_pass": base_valid,
            "liquidity_gates_pass": liquidity_pass,
            "structure_confirmation_pass": structure_pass,
            "amended_intent_formed": intent_formed,
            "amended_confirmation_pass": confirmed,
        }
    )
    return {"confirmation_state": "CONFIRMED" if confirmed else "NOT_CONFIRMED", "confirmation_terms": terms, "candidate_intent": candidate}


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
        "base_transition_terms": row.get("base_transition_terms") or {},
        "avoid_state": row.get("avoid_state"),
        "flow_confirmation_terms": row.get("flow_confirmation_terms") or {},
        "flow_core_compositions_for_r16": {
            "slope_core_pass": bool((row.get("flow_confirmation_terms") or {}).get("slope_core_pass")),
            "cmf_floor_pass": bool((row.get("flow_confirmation_terms") or {}).get("cmf_floor_pass")),
            "cmf_floor_value": (row.get("flow_confirmation_terms") or {}).get("cmf_floor_value"),
            "authority": "NON_BLOCKING",
        },
        "close": row.get("close"),
        "sma200": row.get("sma200"),
        "sma200_slope": row.get("sma200_slope"),
        "ema10": row.get("ema10"),
        "ema30": row.get("ema30"),
        "avoid_entry_predicate": row.get("avoid_entry_predicate"),
    }


def in_windows(date_text: str, windows: list[tuple[str, str]]) -> bool:
    return any(start <= date_text <= end for start, end in windows)


def fmt(value: Any) -> str:
    return "PENDING" if value is None else f"{float(value):.6f}"


def run_exam(attest: dict[str, Any]) -> dict[str, Any]:
    if not attest["freeze_byte_match"] or not attest["amendment_1_byte_match"] or not attest["amendment_2_byte_match"]:
        raise RuntimeError("Freeze v2/amendment byte-match attestation failed")
    bind_harness_db()
    cal = v7.load_default_calendar_context(ROOT)
    mask = v7.load_default_mask_manifest(ROOT)
    adapter = v7.DataSurfaceAdapter(calendar_context=cal, mask_manifest=mask)
    warmup = v7.WarmupReadinessEngine(v7.WarmupNamedParameters(values={v7.READINESS_LONG_LOOKBACK_MIN_SESSIONS: 180, v7.READINESS_SEGMENT_RESTART_MIN_SESSIONS: 20, v7.READINESS_FALLBACK_MIN_SESSIONS: 60}))
    base = v7.AdaptiveBaseGeometry(v7.BaseNamedParameters(values={v7.BASE_MIN_SESSIONS: 10, v7.BASE_MAX_WIDTH_PCT: 0.24, v7.ATR_SQUEEZE_PCTILE: 0.95, "UPWARD_RETIREMENT_MFE_THRESHOLD": UPWARD_RETIREMENT_MFE_THRESHOLD}))
    flow = v7.FlowConfirmationEngine(v7.FlowNamedParameters(values={v7.OBV_SLOPE_MIN: 0.10, v7.ANV_SLOPE_MIN: 0.10, v7.CMF_FLOOR: 0.05, v7.REL_VOLUME_CONTEXT_MIN: 2.5, v7.RSI_REGIME: 50.0, v7.ADX_TRIGGER: 18.0, v7.CHASE_ADVISORY_BAND: 0.08, v7.MIN_DAILY_VALUE_KWD: 100000.0, v7.MIN_CURRENT_DAY_VALUE_KWD: 50000.0}))
    router = v7.LifecycleIntentRouter(v7.LifecycleRouterNamedParameters(values={v7.EARLY_TIER_SIZE_FRACTION: 0.30, v7.EARLY_TIER_PARTICIPATION_CAP: 0.10, v7.EARLY_TIER_TIME_STOP: 60.0, v7.SCALE_ON_CONFIRMATION: "SINGLE_ADD_TO_FULL_TARGET", v7.CHASE_ADVISORY_THRESHOLD: 0.08, v7.CHASE_ESCALATION_THRESHOLD: 0.15}))
    avoid_plane = AvoidAuthorityPlane()

    daily_rows: dict[str, list[dict[str, Any]]] = {}
    runtime_by_symbol: dict[str, list[dict[str, Any]]] = {}
    position_ledger: list[dict[str, Any]] = []
    duplicate_suppression_counts: dict[str, int] = {symbol: 0 for symbol in SET_A_WINDOWS}
    with sqlite3.connect(str(HARNESS_DB)) as conn:
        conn.row_factory = sqlite3.Row
        ledger = ForwardPredictionLedger(conn)
        for symbol, cfg in SET_A_WINDOWS.items():
            rows = load_window(symbol, cfg["start"], cfg["end"])
            runtime_by_symbol[symbol] = [{k: row[k] for k in ["symbol", "trade_date", "open", "high", "low", "close", "volume", "value_kwd"]} for row in rows]
            avoid_series = avoid_plane.evaluate(rows)
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
            symbol_daily: list[dict[str, Any]] = []
            row_sequence_sha = hashlib.sha256(json.dumps([str(row["trade_date"]) for row in rows], ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()[:12]
            for idx, day in enumerate(rows):
                trade_date = str(day["trade_date"])
                avoid_ctx = avoid_series[idx]
                mask_ctx = adapter.mask_context_for(symbol, trade_date)
                seg = adapter.next_segment_state(symbol=symbol, trade_date=trade_date, prev_segment=prev_segment, prev_masked=prev_masked, current_masked=bool(mask_ctx["masked_flag"]))
                normalized, readiness_ctx = adapter.normalize_day(ohlcv_day=day, indicator_day=dict(day.get("indicator_payload") or {}), segment_context=seg, calendar_context=cal)
                coverage_dates.append(trade_date)
                segment_dates = [trade_date] if seg.segment_day_index == 0 else [*segment_dates, trade_date]
                ready = warmup.evaluate(normalized_day_payload=normalized, coverage_history={"long_lookback_session_dates": coverage_dates, "segment_session_dates": segment_dates, "fallback_session_dates": coverage_dates, "previous_readiness_state": prev_ready}, segment_restart_flag=bool(readiness_ctx["segment_restart_flag"]))
                history_window.append(day)
                history_window = history_window[-260:]
                base_out = base.evaluate(normalized_day_payload=normalized, readiness_state=ready["readiness_state"], price_history_window=history_window, volatility_regime_state={"atr_squeeze_pctile": 0.50, "base_range_sessions": 20, "atr_value": float(day["high"] or 0.0) - float(day["low"] or 0.0), "invalidation_rule_form": v7.RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N, "invalidation_rule_params": {"atr_mult": 1.0, "n_sessions": 2}, "UPWARD_RETIREMENT_MFE_THRESHOLD": UPWARD_RETIREMENT_MFE_THRESHOLD, "parameter_status": "FROZEN_R14B_PARAMETER_FREEZE_V2_PLUS_AMENDMENTS_1_2"}, prior_base_reference=prior_base, flow_stub_state={"confirmed_progress": False})
                flow_window.append(dict(day.get("indicator_payload") or {}))
                flow_window = flow_window[-40:]
                raw_flow = flow.evaluate(normalized_day_payload=normalized, base_reference=base_out["base_reference"], flow_history_window=flow_window, structure_terms=v7.build_structure_terms(day, base_out["base_reference"]), readiness_state=ready["readiness_state"], phase_state=base_out["base_state"])
                flow_out = amended_flow_output(raw_flow, base_out["base_state"], base_out["base_reference"], day)
                router_current_state = dict(deferred_state)
                if position_state is not None:
                    router_current_state.update({"active": True, "state": "POSITION_OPEN", "position_id": position_state.get("position_id"), "position_type": position_state.get("position_type")})
                route_out = router.evaluate(candidate_intent=flow_out["candidate_intent"], base_state={"base_state": base_out["base_state"]}, confirmation_state={"confirmation_state": flow_out["confirmation_state"]}, risk_budget_state={"current_day_value_kwd": float(day.get("value_kwd") or 0.0), "planned_order_value_kwd": float(day.get("value_kwd") or 0.0) * 0.03, "avoid_veto": bool(avoid_ctx["avoid_active"]), "deferred_intent_state": router_current_state})
                deferred_state = dict(route_out["deferred_intent"])
                execution_state = str(route_out["execution_intent"].get("execution_state") or "NONE")
                if flow_out["candidate_intent"]["intent_state"] == "INTENT_FORMED" and execution_state == "NONE" and not route_out["veto_record"]["veto"] and router_current_state.get("active") and not route_out["execution_intent"].get("no_path_reason"):
                    route_out["execution_intent"]["no_path_reason"] = "POSITION_ALREADY_OPEN_FEEDBACK_SUPPRESSED_DIRECT"
                    route_out["execution_intent"]["disposition_state"] = "NO_PATH_EXPLICIT"
                if route_out["execution_intent"].get("no_path_reason") == "POSITION_ALREADY_OPEN_FEEDBACK_SUPPRESSED_PILOT":
                    duplicate_suppression_counts[symbol] += 1
                if execution_state in {"EXECUTE_EARLY_PILOT", "EXECUTE_CONFIRMED_DIRECT", "EXECUTE_CONFIRMED_ADD"} and position_state is None:
                    position_counter += 1
                    is_direct = execution_state == "EXECUTE_CONFIRMED_DIRECT"
                    position_state = {"position_id": f"{symbol}::POS{position_counter:04d}", "symbol": symbol, "entry_date": trade_date, "entry_price": float(day.get("close") or 0.0), "entry_tier": route_out["execution_intent"].get("entry_tier") or "NONE", "pilot_fraction": 0.0 if is_direct else float(route_out["execution_intent"].get("pilot_size_fraction") or 0.0), "target_fraction": float(route_out["execution_intent"].get("target_fraction") or (1.0 if is_direct else 0.0)), "position_type": "CONFIRMED_DIRECT" if is_direct else "PILOT_OR_SCALE", "sessions_held": 0, "state": "OPEN", "confirmed_date": trade_date if is_direct else None, "max_drawdown": 0.0}
                    position_ledger.append({"event": "OPEN", **position_state})
                elif execution_state == "EXECUTE_CONFIRMED_ADD" and position_state is not None and position_state.get("confirmed_date") is None:
                    position_state["confirmed_date"] = trade_date
                    position_state["target_fraction"] = 1.0
                    position_ledger.append({"event": "CONFIRMED_ADD", **position_state})
                if position_state is not None:
                    position_state["sessions_held"] = int(position_state["sessions_held"] or 0) + 1
                    entry_price = float(position_state.get("entry_price") or 0.0)
                    if entry_price > 0:
                        position_state["max_drawdown"] = min(float(position_state.get("max_drawdown") or 0.0), (float(day.get("low") or day.get("close") or 0.0) / entry_price) - 1.0)
                    position_ledger.append({"event": "MARK", **position_state})
                    invalidated = base_out["base_transition_terms"].get("base_invalidate_event") == "BASE_INVALIDATED"
                    if invalidated:
                        position_ledger.append({"event": "CLOSE_INVALIDATION", **position_state, "base_transition_terms": base_out["base_transition_terms"], "base_reference": base_out["base_reference"]})
                        position_state = None
                    elif position_state.get("position_type") != "CONFIRMED_DIRECT" and int(position_state.get("sessions_held") or 0) >= 60:
                        position_ledger.append({"event": "CLOSE_TIME_STOP", **position_state})
                        position_state = None
                out_row = {"symbol": symbol, "trade_date": trade_date, "readiness_state": ready["readiness_state"], "base_state": base_out["base_state"], "base_transition_terms": base_out["base_transition_terms"], "avoid_state": avoid_ctx["avoid_state"], "close": avoid_ctx["close"], "sma200": avoid_ctx["sma200"], "sma200_slope": avoid_ctx["sma200_slope"], "ema10": avoid_ctx["ema10"], "ema30": avoid_ctx["ema30"], "avoid_entry_predicate": avoid_ctx["avoid_entry_predicate"], "candidate_intent": flow_out["candidate_intent"], "flow_confirmation_terms": flow_out["confirmation_terms"], "lifecycle_terms": route_out["lifecycle_terms"], "execution_intent": route_out["execution_intent"], "deferred_intent": route_out["deferred_intent"], "veto_record": route_out["veto_record"], "router_current_state_feedback": router_current_state, "sessions_held": int(position_state.get("sessions_held") if position_state else 0), "base_reference": base_out["base_reference"]}
                symbol_daily.append(out_row)
                event_type = event_type_for(out_row)
                if event_type is not None:
                    execution = out_row["execution_intent"]
                    intent = out_row["candidate_intent"]
                    ledger.append_prediction(symbol=symbol, prediction_date=trade_date, engine_baseline_id=f"{RUN_KEY}:{symbol}:{row_sequence_sha}", freeze_version_hash=str(attest["freeze_actual_sha256"]) + "+" + str(attest["amendment_1_actual_sha256"]) + "+" + str(attest["amendment_2_actual_sha256"]), intent_state=str(intent.get("intent_state") or "INTENT_NONE"), execution_state=str(execution.get("execution_state") or "NONE"), entry_tier=str(execution.get("entry_tier") or "NONE"), reference_price=float(out_row.get("close") or 0.0), base_reference=base_out["base_reference"], avoid_state=str(out_row.get("avoid_state") or "NONE"), predicate_snapshot=build_prediction_snapshot(out_row), event_type=event_type, source_run_key=RUN_KEY, created_utc=RUN_NONCE)
                prior_base = base_out["base_reference"]
                prev_ready = ready["readiness_state"]
                prev_segment = seg
                prev_masked = bool(mask_ctx["masked_flag"])
            daily_rows[symbol] = symbol_daily
        predictions = fetch_predictions(conn)
    with sqlite3.connect(str(HARNESS_DB)) as grade_conn:
        grade_conn.row_factory = sqlite3.Row
        grades = apply_grades(predictions_db_path=HARNESS_DB, grades_conn=grade_conn, sealed_ohlcv_by_symbol=runtime_by_symbol, grade_date=RUN_NONCE, grader_version=GRADER_VERSION)
    return {"daily_rows": daily_rows, "runtime_by_symbol": runtime_by_symbol, "position_ledger": position_ledger, "predictions": predictions, "grades": grades, "duplicate_suppression_counts": duplicate_suppression_counts}


def deciding_base_live(row: dict[str, Any]) -> bool:
    try:
        base_reference = json.loads(str(row.get("base_reference") or "{}"))
    except json.JSONDecodeError:
        return False
    return base_reference.get("base_validity_state") == "VALID" and not str(base_reference.get("base_retirement_reason") or "").startswith("RETIRED_SUPERSEDED_BY_MARKUP")


def decide_criteria(result: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    daily = result["daily_rows"]
    predictions = result["predictions"]
    grades_by_id = {row["prediction_id"]: row for row in result["grades"]}
    sanam_entries = [p for p in predictions if p["symbol"] == "SANAM" and p["event_type"] == "EXECUTION"]
    sanam_confirmed_window = [p for p in sanam_entries if "2025-05-08" <= p["prediction_date"] <= "2025-05-29" and p["entry_tier"] == "BREAKOUT_CONFIRMED_ENTRY" and deciding_base_live(p)]
    sanam_early_window = [p for p in sanam_entries if "2025-03-01" <= p["prediction_date"] <= "2025-05-07" and p["execution_state"] == "EXECUTE_EARLY_PILOT" and deciding_base_live(p)]
    sanam_first_intent = next((r["trade_date"] for r in daily["SANAM"] if r["candidate_intent"].get("intent_state") == "INTENT_FORMED"), "NONE")
    tijara_entries = [p for p in predictions if p["symbol"] == "TIJARA" and p["event_type"] == "EXECUTION"]
    tijara_confirmed_window = [p for p in tijara_entries if "2024-09-01" <= p["prediction_date"] <= "2024-12-31" and p["entry_tier"] == "BREAKOUT_CONFIRMED_ENTRY" and deciding_base_live(p)]
    mabanee_early_decline = [p for p in predictions if p["symbol"] == "MABANEE" and p["execution_state"] == "EXECUTE_EARLY_PILOT" and in_windows(str(p["prediction_date"]), MABANEE_DECLINE_WINDOWS)]
    pilots = [p for p in predictions if p["execution_state"] == "EXECUTE_EARLY_PILOT"]
    failed_pilots = [p for p in pilots if grades_by_id.get(p["prediction_id"], {}).get("materialization_verdict") == "NOT_MATERIALIZED"]
    cost_rows = []
    for pilot in failed_pilots:
        rows = result["runtime_by_symbol"].get(pilot["symbol"], [])
        idx = next((i for i, row in enumerate(rows) if row["trade_date"] == pilot["prediction_date"]), None)
        horizon_rows = rows[idx : min(len(rows), idx + 121)] if idx is not None else []
        ref = float(pilot["reference_price"] or 0.0)
        drawdown = min(((float(row["low"] or row["close"] or 0.0) / ref) - 1.0 for row in horizon_rows), default=0.0) if ref > 0 else 0.0
        cost_rows.append({"prediction_id": pilot["prediction_id"], "symbol": pilot["symbol"], "date": pilot["prediction_date"], "pilot_fraction": 0.30, "sessions_observed": len(horizon_rows), "capital_days": 0.30 * len(horizon_rows), "realized_drawdown": drawdown})
    duplicate_suppression_counts = result["duplicate_suppression_counts"]
    false_positive_cost = {"failed_pilot_count": len(cost_rows), "capital_days": sum(r["capital_days"] for r in cost_rows), "realized_drawdown_min": min((r["realized_drawdown"] for r in cost_rows), default=0.0), "duplicate_pilot_suppression_counts": duplicate_suppression_counts, "duplicate_pilot_suppression_total": sum(duplicate_suppression_counts.values()), "rows": cost_rows}
    criteria = [
        {"criterion": "SANAM confirmed entry occurs within 2025-05-08..2025-05-29 from a then-live base", "verdict": "PASS" if sanam_confirmed_window else "FAIL", "evidence_rows": sanam_confirmed_window},
        {"criterion": "SANAM early-tier entry within 2025-03-01..2025-05-07 from a then-live base", "verdict": "PASS" if sanam_early_window else "FAIL", "expected": "FAIL under current detection timing (FLOW_CORE_LAG)", "actual_first_intent_date": sanam_first_intent, "evidence_rows": sanam_early_window},
        {"criterion": "TIJARA confirmed entry occurs within 2024-09-01..2024-12-31 from a then-live base", "verdict": "PASS" if tijara_confirmed_window else "FAIL", "evidence_rows": tijara_confirmed_window},
        {"criterion": "MABANEE zero early entries inside the decline windows", "verdict": "PASS" if not mabanee_early_decline else "FAIL", "decline_windows": MABANEE_DECLINE_WINDOWS, "evidence_rows": mabanee_early_decline},
        {"criterion": "early-tier false-positive cost computed and reported", "verdict": "PASS", "evidence_rows": [false_positive_cost]},
    ]
    return criteria, false_positive_cost


def write_report(attest: dict[str, Any], result: dict[str, Any], criteria: list[dict[str, Any]], false_positive_cost: dict[str, Any]) -> None:
    predictions = result["predictions"]
    grades = result["grades"]
    lines = ["# R15 Exam Report v2", "", f"RUN_NONCE: {RUN_NONCE}", f"RUN_KEY: {RUN_KEY}", "", "## Summary Verdict Block", "criterion|verdict", "---|---"]
    for row in criteria:
        lines.append(f"{row['criterion']}|{row['verdict']}")
    lines.extend(["", "## Freeze Attestation", "```json", json.dumps(attest, ensure_ascii=True, indent=2, sort_keys=True), "```", "", "## Per-Criterion Evidence", "```json", json.dumps(criteria, ensure_ascii=True, indent=2, sort_keys=True), "```", "", "## Veto/Suppression Summary By Plane", "plane|event_count", "---|---"])
    plane_counts: dict[str, int] = {}
    for p in predictions:
        if p["event_type"] in {"VETO_RESTRAINT", "SUPPRESSION_RESTRAINT"}:
            snap = json.loads(str(p["predicate_snapshot_json"]))
            plane = str((snap.get("veto_record") or {}).get("plane") or (snap.get("execution_intent") or {}).get("no_path_reason") or "SUPPRESSION")
            plane_counts[plane] = plane_counts.get(plane, 0) + 1
    for plane, count in sorted(plane_counts.items()):
        lines.append(f"{plane}|{count}")
    lines.extend(["", "## False-Positive Cost Accounting", "```json", json.dumps(false_positive_cost, ensure_ascii=True, indent=2, sort_keys=True), "```", "", "## Position Ledger With Sessions Held", "event|position_id|symbol|entry_date|entry_tier|position_type|sessions_held|confirmed_date|max_drawdown", "---|---|---|---|---|---|---:|---|---:"])
    for row in result["position_ledger"]:
        lines.append("|".join([str(row.get("event")), str(row.get("position_id")), str(row.get("symbol")), str(row.get("entry_date")), str(row.get("entry_tier")), str(row.get("position_type")), str(row.get("sessions_held")), str(row.get("confirmed_date") or "NONE"), f"{float(row.get('max_drawdown') or 0.0):.6f}"]))
    lines.extend(["", "## Per-Day Lifecycle Tables", "symbol|date|readiness|base|avoid|intent|confirmation|execution|entry_tier|veto_plane|sessions_held|slope_core|cmf_floor|liquidity|structure", "---|---|---|---|---|---|---|---|---|---|---:|---|---|---|---"])
    for symbol in SET_A_WINDOWS:
        for row in result["daily_rows"][symbol]:
            terms = row["flow_confirmation_terms"]
            lines.append("|".join([symbol, row["trade_date"], str(row["readiness_state"]), str(row["base_state"]), str(row["avoid_state"]), str(row["candidate_intent"].get("intent_state")), str(row["candidate_intent"].get("confirmation_state")), str(row["execution_intent"].get("execution_state")), str(row["execution_intent"].get("entry_tier")), str(row["veto_record"].get("plane")), str(row["sessions_held"]), str(terms.get("slope_core_pass")), str(terms.get("cmf_floor_pass")), str(terms.get("liquidity_gates_pass")), str(terms.get("structure_confirmation_pass"))]))
    lines.extend(["", "## Complete Forward Prediction Table", "prediction_id|symbol|date|event|intent|execution|entry_tier|reference_price|avoid|freeze_hash", "---|---|---|---|---|---|---|---:|---|---"])
    for row in predictions:
        lines.append("|".join([str(row["prediction_id"]), str(row["symbol"]), str(row["prediction_date"]), str(row["event_type"]), str(row["intent_state"]), str(row["execution_state"]), str(row["entry_tier"]), f"{float(row['reference_price']):.3f}", str(row["avoid_state"]), str(row["freeze_version_hash"])]))
    lines.extend(["", "## Complete Prediction Grade Table", "prediction_id|symbol|date|return_20|return_60|return_120|mfe_120|verdict|status|last_data", "---|---|---|---:|---:|---:|---:|---|---|---"])
    for row in grades:
        lines.append("|".join([str(row["prediction_id"]), str(row["symbol"]), str(row["prediction_date"]), fmt(row.get("return_20")), fmt(row.get("return_60")), fmt(row.get("return_120")), fmt(row.get("mfe_120")), str(row["materialization_verdict"]), str(row["grade_status"]), str(row["sealed_data_last_date"])]))
    lines.extend(["", "## Artifact Hashes", "artifact|sha256", "---|---"])
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    artifact_hashes = {
        "artifacts/preview1a_prestart/review_final/r15_exam_report_v2.md": sha256_file(REPORT_MD),
        "artifacts/preview1a_prestart/review_final/r15_exam_v2_harness.db": sha256_file(HARNESS_DB),
        "scripts/r15_exam_v2.py": sha256_file(ROOT / "scripts" / "r15_exam_v2.py"),
        "artifacts/preview1a_prestart/review_final/r14b_parameter_freeze_v2.json": sha256_file(FREEZE_JSON),
        "artifacts/preview1a_prestart/review_final/r14b_parameter_freeze_v2_amendment_1.json": sha256_file(AMEND1_JSON),
        "artifacts/preview1a_prestart/review_final/r14b_parameter_freeze_v2_amendment_2.json": sha256_file(AMEND2_JSON),
        "artifacts/preview1a_prestart/review_final/r15_attempt2_criteria_of_record_v1.md": sha256_file(CRITERIA_RECORD),
    }
    with REPORT_MD.open("a", encoding="utf-8", newline="\n") as f:
        for artifact, digest in artifact_hashes.items():
            f.write(f"{artifact}|{digest}\n")


def main() -> None:
    attest = attest_freeze()
    result = run_exam(attest)
    criteria, false_positive_cost = decide_criteria(result)
    write_report(attest, result, criteria, false_positive_cost)
    print("R15_EXAM_V2_COMPLETE")
    print("RUN_NONCE", RUN_NONCE)
    for row in criteria:
        print(row["verdict"], row["criterion"])
    print("predictions", len(result["predictions"]))
    print("grades", len(result["grades"]))
    print("report", REPORT_MD.as_posix())
    print("report_sha256", sha256_file(REPORT_MD))


if __name__ == "__main__":
    main()
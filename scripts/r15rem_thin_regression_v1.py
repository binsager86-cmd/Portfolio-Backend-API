from __future__ import annotations

import collections
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.r14e_module_e_lifecycle_intent_harness_v7 as v7
from app.core.config import get_settings

REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
OUT_JSON = REVIEW / "r15rem_thin_regression_v1.json"
OUT_MD = REVIEW / "r15rem_thin_regression_v1.md"
OUT_SHA = REVIEW / "r15rem_thin_regression_v1.sha256"
HARNESS_DB = REVIEW / "r15rem_thin_regression_v1.db"
BASELINE = REVIEW / "r14e_module_e_test_evidence_v7.json"
RUN_NONCE = "R15REM_THIN_REGRESSION_V1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def bind_db() -> None:
    if HARNESS_DB.exists():
        HARNESS_DB.unlink()
    HARNESS_DB.touch()
    os.environ["EE_V2_RUNTIME_DB_PATH"] = str(HARNESS_DB)
    os.environ["DATABASE_PATH"] = str(HARNESS_DB)
    get_settings.cache_clear()
    v7.HARNESS_DB = HARNESS_DB
    v7.apply_schema_migration()


def make_stack() -> dict[str, Any]:
    cal = v7.load_default_calendar_context(ROOT)
    mask = v7.load_default_mask_manifest(ROOT)
    return {
        "adapter": v7.DataSurfaceAdapter(calendar_context=cal, mask_manifest=mask),
        "calendar": cal,
        "warmup": v7.WarmupReadinessEngine(
            v7.WarmupNamedParameters(
                values={
                    v7.READINESS_LONG_LOOKBACK_MIN_SESSIONS: 180,
                    v7.READINESS_SEGMENT_RESTART_MIN_SESSIONS: 20,
                    v7.READINESS_FALLBACK_MIN_SESSIONS: 60,
                }
            )
        ),
        "base": v7.AdaptiveBaseGeometry(
            v7.BaseNamedParameters(
                values={
                    v7.BASE_MIN_SESSIONS: 10,
                    v7.BASE_MAX_WIDTH_PCT: 0.24,
                    v7.ATR_SQUEEZE_PCTILE: 0.95,
                    "UPWARD_RETIREMENT_MFE_THRESHOLD": 0.20,
                }
            )
        ),
        "flow": v7.FlowConfirmationEngine(
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
        ),
        "router": v7.LifecycleIntentRouter(
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
        ),
    }


def replay_symbol(symbol: str, cfg: dict[str, Any], stack: dict[str, Any]) -> list[dict[str, Any]]:
    rows = v7.load_window(symbol, cfg["replay_start"], cfg["replay_end"])
    avoid_series = v7.derive_avoid_context(rows)
    adapter = stack["adapter"]
    prev_segment = None
    prev_masked = False
    prev_ready = "READINESS_PENDING"
    history_window: list[dict[str, Any]] = []
    flow_window: list[dict[str, Any]] = []
    prior_base = None
    coverage_dates: list[str] = []
    segment_dates: list[str] = []
    deferred_state: dict[str, Any] = {"age_sessions": 0, "rearm_count": 0, "flow_evidence_decay": False}
    position_state: dict[str, Any] | None = None
    out: list[dict[str, Any]] = []
    for idx, day in enumerate(rows):
        trade_date = str(day["trade_date"])
        avoid_ctx = avoid_series[idx]
        mask_ctx = adapter.mask_context_for(symbol, trade_date)
        seg = adapter.next_segment_state(symbol=symbol, trade_date=trade_date, prev_segment=prev_segment, prev_masked=prev_masked, current_masked=bool(mask_ctx["masked_flag"]))
        normalized, readiness_ctx = adapter.normalize_day(ohlcv_day=day, indicator_day=dict(day.get("indicator_payload") or {}), segment_context=seg, calendar_context=stack["calendar"])
        coverage_dates.append(trade_date)
        segment_dates = [trade_date] if seg.segment_day_index == 0 else [*segment_dates, trade_date]
        ready = stack["warmup"].evaluate(
            normalized_day_payload=normalized,
            coverage_history={"long_lookback_session_dates": coverage_dates, "segment_session_dates": segment_dates, "fallback_session_dates": coverage_dates, "previous_readiness_state": prev_ready},
            segment_restart_flag=bool(readiness_ctx["segment_restart_flag"]),
        )
        history_window = [*history_window, day][-260:]
        base_out = stack["base"].evaluate(
            normalized_day_payload=normalized,
            readiness_state=ready["readiness_state"],
            price_history_window=history_window,
            volatility_regime_state={
                "atr_squeeze_pctile": 0.50,
                "base_range_sessions": 20,
                "atr_value": float(day["high"] or 0.0) - float(day["low"] or 0.0),
                "invalidation_rule_form": v7.RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N,
                "invalidation_rule_params": {"atr_mult": 1.0, "n_sessions": 2},
                "UPWARD_RETIREMENT_MFE_THRESHOLD": 0.20,
                "parameter_status": "R15REM_DESCRIPTIVE_REGRESSION",
            },
            prior_base_reference=prior_base,
            flow_stub_state={"confirmed_progress": False},
        )
        flow_window = [*flow_window, dict(day.get("indicator_payload") or {})][-40:]
        flow_out = stack["flow"].evaluate(
            normalized_day_payload=normalized,
            base_reference=base_out["base_reference"],
            flow_history_window=flow_window,
            structure_terms=v7.build_structure_terms(day, base_out["base_reference"]),
            readiness_state=ready["readiness_state"],
            phase_state=base_out["base_state"],
        )
        router_current_state = dict(deferred_state)
        if position_state is not None:
            router_current_state.update({"active": True, "state": "POSITION_OPEN", "position_id": position_state["position_id"]})
        route_out = stack["router"].evaluate(
            candidate_intent=flow_out["candidate_intent"],
            base_state={"base_state": base_out["base_state"]},
            confirmation_state={"confirmation_state": flow_out["confirmation_state"]},
            risk_budget_state={"current_day_value_kwd": float(day.get("value_kwd") or 0.0), "planned_order_value_kwd": float(day.get("value_kwd") or 0.0) * 0.03, "avoid_veto": bool(avoid_ctx["avoid_active"]), "deferred_intent_state": router_current_state},
        )
        deferred_state = dict(route_out["deferred_intent"])
        execution_state = str(route_out["execution_intent"].get("execution_state") or "NONE")
        if execution_state.startswith("EXECUTE_") and position_state is None:
            position_state = {"position_id": f"{symbol}::POS0001", "sessions_held": 0, "position_type": "CONFIRMED_DIRECT" if execution_state == "EXECUTE_CONFIRMED_DIRECT" else "PILOT_OR_SCALE"}
        if position_state is not None:
            position_state["sessions_held"] = int(position_state.get("sessions_held") or 0) + 1
            if base_out["base_transition_terms"].get("base_invalidate_event") == "BASE_INVALIDATED":
                position_state = None
            elif position_state.get("position_type") != "CONFIRMED_DIRECT" and int(position_state.get("sessions_held") or 0) >= 60:
                position_state = None
        out.append({"trade_date": trade_date, "execution_state": execution_state, "entry_tier": route_out["execution_intent"].get("entry_tier"), "no_path_reason": route_out["execution_intent"].get("no_path_reason"), "avoid_state": avoid_ctx["avoid_state"], "owner_window_day": bool(cfg["owner_start"] <= trade_date <= cfg["owner_end"])})
        prior_base = base_out["base_reference"]
        prev_ready = ready["readiness_state"]
        prev_segment = seg
        prev_masked = bool(mask_ctx["masked_flag"])
    return out


def baseline_direct_entries(payload: dict[str, Any]) -> dict[str, list[str]]:
    rows_by_symbol = payload.get("per_day_intent_lifecycle_tables") or payload.get("per_symbol_rows") or payload.get("per_symbol") or {}
    return {
        symbol: [
            str(row.get("trade_date"))
            for row in rows
            if str(((row.get("execution_intent") or {}).get("execution_state")) or "") == "EXECUTE_CONFIRMED_DIRECT"
        ]
        for symbol, rows in rows_by_symbol.items()
    }


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    bind_db()
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    stack = make_stack()
    windows = v7.owner_windows()
    per_symbol = {symbol: replay_symbol(symbol, cfg, stack) for symbol, cfg in windows.items()}
    counts = {
        symbol: dict(collections.Counter(row["execution_state"] for row in rows))
        for symbol, rows in per_symbol.items()
    }
    direct_entries = {symbol: [row["trade_date"] for row in rows if row["execution_state"] == "EXECUTE_CONFIRMED_DIRECT"] for symbol, rows in per_symbol.items()}
    baseline_direct = baseline_direct_entries(baseline)
    early_pilots = {symbol: [row["trade_date"] for row in rows if row["execution_state"] == "EXECUTE_EARLY_PILOT"] for symbol, rows in per_symbol.items()}
    pilot_suppressions = {symbol: sum(1 for row in rows if row.get("no_path_reason") == "POSITION_ALREADY_OPEN_FEEDBACK_SUPPRESSED_PILOT") for symbol, rows in per_symbol.items()}
    mabanee_decline_early = [row["trade_date"] for row in per_symbol["MABANEE"] if row["execution_state"] == "EXECUTE_EARLY_PILOT" and row["owner_window_day"]]
    checks = {
        "direct_entries_unchanged": {
            "status": "PASS" if direct_entries == baseline_direct else "FAIL",
            "baseline_direct_entries": baseline_direct,
            "current_direct_entries": direct_entries,
        },
        "pilots_deduped": {
            "status": "PASS" if all(len(v) <= 1 for v in early_pilots.values()) else "FAIL",
            "early_pilot_dates": early_pilots,
            "pilot_suppressions": pilot_suppressions,
            "execution_counts": counts,
        },
        "mabanee_unchanged": {"status": "PASS" if not mabanee_decline_early and baseline["acceptance_checks"]["MABANEE"]["status"] == "PASS" else "FAIL", "mabanee_decline_early": mabanee_decline_early},
    }
    payload = {"version_id": "R15REM_THIN_REGRESSION_V1", "run_nonce": RUN_NONCE, "baseline_artifact": str(BASELINE), "checks": checks, "per_symbol_counts": counts}
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# R15-REM Thin Regression v1", "", f"RUN_NONCE: {RUN_NONCE}", ""]
    for name, check in checks.items():
        lines.extend([f"## {name}", "", json.dumps(check, sort_keys=True), ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_SHA.write_text("\n".join(f"{sha256_file(p)}  {p.name}" for p in (OUT_JSON, OUT_MD)), encoding="utf-8")
    print(json.dumps(payload["checks"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
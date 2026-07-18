from __future__ import annotations

import hashlib
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
from app.services.eagle_eye_v2.adaptive_base_geometry import (
    ATR_SQUEEZE_PCTILE,
    BASE_MAX_WIDTH_PCT,
    BASE_MIN_SESSIONS,
    RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N,
    AdaptiveBaseGeometry,
    BaseNamedParameters,
)
from app.services.eagle_eye_v2.data_surface_adapter import (
    DataSurfaceAdapter,
    SegmentState,
    load_default_calendar_context,
    load_default_mask_manifest,
)
from app.services.eagle_eye_v2.flow_confirmation_engine import (
    ADX_TRIGGER,
    ANV_SLOPE_MIN,
    CHASE_ADVISORY_BAND,
    CMF_FLOOR,
    MIN_CURRENT_DAY_VALUE_KWD,
    MIN_DAILY_VALUE_KWD,
    OBV_SLOPE_MIN,
    REL_VOLUME_CONTEXT_MIN,
    RSI_REGIME,
    FlowConfirmationEngine,
    FlowNamedParameters,
)
from app.services.eagle_eye_v2.lifecycle_intent_router import (
    CHASE_ADVISORY_THRESHOLD,
    CHASE_ESCALATION_THRESHOLD,
    EARLY_TIER_PARTICIPATION_CAP,
    EARLY_TIER_SIZE_FRACTION,
    EARLY_TIER_TIME_STOP,
    SCALE_ON_CONFIRMATION,
    LifecycleIntentRouter,
    LifecycleRouterNamedParameters,
)
from app.services.eagle_eye_v2.warmup_readiness_engine import (
    READINESS_FALLBACK_MIN_SESSIONS,
    READINESS_LONG_LOOKBACK_MIN_SESSIONS,
    READINESS_SEGMENT_RESTART_MIN_SESSIONS,
    WarmupNamedParameters,
    WarmupReadinessEngine,
)
from app.services.eagle_eye_v2.predicate_telemetry_ledger import apply_schema_migration

REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
RUNTIME_DB = REVIEW / "r12_exam_surface_v4_5_runtime.db"
HARNESS_DB = REVIEW / "r14e_module_e_harness_surface_v1.db"
FREEZE_JSON = REVIEW / "r14b_parameter_freeze_v2.json"
FREEZE_SHA = REVIEW / "r14b_parameter_freeze_v2.sha256"


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


def bind_harness_db() -> None:
    if HARNESS_DB.exists():
        HARNESS_DB.unlink()
    HARNESS_DB.touch()
    os.environ["EE_V2_RUNTIME_DB_PATH"] = str(HARNESS_DB)
    os.environ["DATABASE_PATH"] = str(HARNESS_DB)
    get_settings.cache_clear()


def freeze_attestation() -> dict[str, Any]:
    if not FREEZE_JSON.exists() or not FREEZE_SHA.exists():
        raise FileNotFoundError("Freeze v2 artifacts are required before module (e) harness run.")
    expected = FREEZE_SHA.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(FREEZE_JSON)
    return {
        "freeze_json": str(FREEZE_JSON),
        "freeze_sha_sidecar": str(FREEZE_SHA),
        "expected_json_sha256": expected,
        "actual_json_sha256": actual,
        "byte_match": expected == actual,
    }


def fetch_indicator_payload(conn: sqlite3.Connection, symbol: str, trade_date: int) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT payload_json
        FROM ee_indicators
        WHERE symbol LIKE ? AND trade_date = ?
        ORDER BY symbol ASC
        LIMIT 1
        """,
        (f"{symbol}%", int(trade_date)),
    ).fetchone()
    if row is None or row[0] is None:
        return {}
    try:
        return json.loads(str(row[0]))
    except json.JSONDecodeError:
        return {}


def load_window(symbol: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(RUNTIME_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT symbol, trade_date, open, high, low, close, volume, value_kwd
            FROM ee_ohlcv
            WHERE symbol LIKE ?
              AND date(trade_date, 'unixepoch') BETWEEN ? AND ?
            ORDER BY trade_date ASC
            """,
            (f"{symbol}%", start_date, end_date),
        ).fetchall()

        out: list[dict[str, Any]] = []
        for r in rows:
            ts = int(r["trade_date"])
            out.append(
                {
                    "symbol": symbol,
                    "trade_date": to_date_text(ts),
                    "trade_date_ts": ts,
                    "open": float(r["open"] or 0.0),
                    "high": float(r["high"] or 0.0),
                    "low": float(r["low"] or 0.0),
                    "close": float(r["close"] or 0.0),
                    "volume": float(r["volume"] or 0.0),
                    "value_kwd": float(r["value_kwd"] or 0.0),
                    "indicator_payload": fetch_indicator_payload(conn, symbol, ts),
                }
            )
        return out
    finally:
        conn.close()


def build_structure_terms(day: dict[str, Any], base_reference: dict[str, Any]) -> dict[str, Any]:
    ind = dict(day.get("indicator_payload") or {})
    close_px = float(day.get("close") or 0.0)
    base_high = float(base_reference.get("base_high_ref") or 0.0)
    return {
        "close_gt_base_ref": bool(base_high > 0 and close_px > base_high),
        "ema10_gt_ema30": float(ind.get("ema10") or 0.0) >= float(ind.get("ema30") or 0.0),
        "adx_19": float(ind.get("adx_19") or 0.0),
        "rsi_14": float(ind.get("rsi_14") or 0.0),
    }


def owner_windows() -> dict[str, dict[str, Any]]:
    return {
        "SANAM": {
            "owner_start": "2025-05-01",
            "owner_end": "2025-05-31",
            "replay_start": "2024-11-01",
            "replay_end": "2025-05-31",
        },
        "TIJARA": {
            "owner_start": "2025-01-01",
            "owner_end": "2025-12-31",
            "replay_start": "2024-07-01",
            "replay_end": "2025-12-31",
        },
    }


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)

    out_interface = REVIEW / "r14e_module_e_interface_conformance_v2.json"
    out_evidence = REVIEW / "r14e_module_e_test_evidence_v2.json"
    out_report = REVIEW / "r14e_module_e_implementation_report_v2.md"
    out_sha = REVIEW / "r14e_module_e_artifacts_v2.sha256"
    harness_db = REVIEW / "r14e_module_e_harness_surface_v2.db"

    attest = freeze_attestation()
    if not attest["byte_match"]:
        raise RuntimeError("Freeze v2 byte-match attestation failed.")

    global HARNESS_DB
    HARNESS_DB = harness_db
    bind_harness_db()
    apply_schema_migration()

    cal = load_default_calendar_context(ROOT)
    mask = load_default_mask_manifest(ROOT)
    adapter = DataSurfaceAdapter(calendar_context=cal, mask_manifest=mask)

    warmup = WarmupReadinessEngine(
        WarmupNamedParameters(
            values={
                READINESS_LONG_LOOKBACK_MIN_SESSIONS: 180,
                READINESS_SEGMENT_RESTART_MIN_SESSIONS: 20,
                READINESS_FALLBACK_MIN_SESSIONS: 60,
            }
        )
    )

    base = AdaptiveBaseGeometry(
        BaseNamedParameters(
            values={
                BASE_MIN_SESSIONS: 10,
                BASE_MAX_WIDTH_PCT: 0.24,
                ATR_SQUEEZE_PCTILE: 0.95,
            }
        )
    )

    flow = FlowConfirmationEngine(
        FlowNamedParameters(
            values={
                OBV_SLOPE_MIN: 0.10,
                ANV_SLOPE_MIN: 0.10,
                CMF_FLOOR: 0.05,
                REL_VOLUME_CONTEXT_MIN: 2.5,
                RSI_REGIME: 50.0,
                ADX_TRIGGER: 18.0,
                CHASE_ADVISORY_BAND: 0.08,
                MIN_DAILY_VALUE_KWD: 100000.0,
                MIN_CURRENT_DAY_VALUE_KWD: 50000.0,
            }
        )
    )

    router = LifecycleIntentRouter(
        LifecycleRouterNamedParameters(
            values={
                EARLY_TIER_SIZE_FRACTION: 0.30,
                EARLY_TIER_PARTICIPATION_CAP: 0.10,
                EARLY_TIER_TIME_STOP: 60.0,
                SCALE_ON_CONFIRMATION: "SINGLE_ADD_TO_FULL_TARGET",
                CHASE_ADVISORY_THRESHOLD: 0.08,
                CHASE_ESCALATION_THRESHOLD: 0.15,
            }
        )
    )

    windows = owner_windows()
    per_symbol: dict[str, list[dict[str, Any]]] = {}

    for symbol, cfg in windows.items():
        rows = load_window(symbol, cfg["replay_start"], cfg["replay_end"])
        if not rows:
            per_symbol[symbol] = []
            continue

        prev_segment: SegmentState | None = None
        prev_masked = False
        prev_ready = "READINESS_PENDING"
        history_window: list[dict[str, Any]] = []
        flow_window: list[dict[str, Any]] = []
        prior_base: dict[str, Any] | None = None
        coverage_dates: list[str] = []
        segment_dates: list[str] = []
        deferred_state = {"age_sessions": 0, "rearm_count": 0, "flow_evidence_decay": False}

        out_rows: list[dict[str, Any]] = []

        for day in rows:
            trade_date = str(day["trade_date"])
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
            if seg.segment_day_index == 0:
                segment_dates = [trade_date]
            else:
                segment_dates.append(trade_date)

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
            if len(history_window) > 260:
                history_window = history_window[-260:]

            base_out = base.evaluate(
                normalized_day_payload=normalized,
                readiness_state=ready["readiness_state"],
                price_history_window=history_window,
                volatility_regime_state={
                    "atr_squeeze_pctile": 0.50,
                    "base_range_sessions": 20,
                    "atr_value": float(day["high"] or 0.0) - float(day["low"] or 0.0),
                    "invalidation_rule_form": RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N,
                    "invalidation_rule_params": {"atr_mult": 1.0, "n_sessions": 2},
                    "parameter_status": "FROZEN_R14B_PARAMETER_FREEZE_V2",
                },
                prior_base_reference=prior_base,
                flow_stub_state={"confirmed_progress": False},
            )

            flow_window.append(dict(day.get("indicator_payload") or {}))
            if len(flow_window) > 40:
                flow_window = flow_window[-40:]

            structure_terms = build_structure_terms(day, base_out["base_reference"])
            flow_out = flow.evaluate(
                normalized_day_payload=normalized,
                base_reference=base_out["base_reference"],
                flow_history_window=flow_window,
                structure_terms=structure_terms,
                readiness_state=ready["readiness_state"],
                phase_state=base_out["base_state"],
            )

            route_out = router.evaluate(
                candidate_intent=flow_out["candidate_intent"],
                base_state={"base_state": base_out["base_state"]},
                confirmation_state={"confirmation_state": flow_out["confirmation_state"]},
                risk_budget_state={
                    "current_day_value_kwd": float(day.get("value_kwd") or 0.0),
                    "planned_order_value_kwd": float(day.get("value_kwd") or 0.0) * 0.03,
                    "avoid_veto": False,
                    "deferred_intent_state": deferred_state,
                },
            )
            deferred_state = dict(route_out["deferred_intent"])

            if cfg["owner_start"] <= trade_date <= cfg["owner_end"]:
                out_rows.append(
                    {
                        "trade_date": trade_date,
                        "readiness_state": ready["readiness_state"],
                        "base_state": base_out["base_state"],
                        "confirmation_state": flow_out["confirmation_state"],
                        "candidate_intent": flow_out["candidate_intent"],
                        "lifecycle_terms": route_out["lifecycle_terms"],
                        "execution_intent": route_out["execution_intent"],
                        "deferred_intent": route_out["deferred_intent"],
                        "veto_record": route_out["veto_record"],
                    }
                )

            prior_base = base_out["base_reference"]
            prev_ready = ready["readiness_state"]
            prev_segment = seg
            prev_masked = bool(mask_ctx["masked_flag"])

        per_symbol[symbol] = out_rows

    interface_payload = {
        "version_id": "R14E_MODULE_E_INTERFACE_CONFORMANCE_V1",
        "module": "LifecycleIntentRouter+StagedPositionPolicy",
        "inputs": [
            "candidate_intent",
            "base_state",
            "confirmation_state",
            "risk_budget_state",
        ],
        "outputs": [
            "execution_intent",
            "deferred_intent",
            "veto_record",
            "lifecycle_terms",
        ],
        "required_lifecycle_predicates": [
            "DEFERRED_INTENT_ACTIVE",
            "DEFERRED_INTENT_EXPIRY_OK",
            "EARLY_INTENT_ACTIVE",
            "EARLY_INTENT_SCALE_READY",
        ],
        "frozen_policy_assertions": {
            "pilot_fraction": 0.30,
            "participation_cap": 0.10,
            "time_stop_sessions": 60,
            "max_rearms": 2,
            "scale_on_confirmation": "SINGLE_ADD_TO_FULL_TARGET",
            "chase_advisory": 0.08,
            "chase_escalation": 0.15,
        },
        "freeze_v2_attestation": attest,
    }

    evidence_payload = {
        "version_id": "R14E_MODULE_E_TEST_EVIDENCE_V1",
        "freeze_v2_attestation": attest,
        "harness_db": str(HARNESS_DB),
        "owner_windows": windows,
        "per_day_intent_lifecycle_tables": per_symbol,
        "outcomes_policy": "REPORTED_AS_OBSERVED_NO_TARGET_FITTING",
        "modules_f_g_authorization_note": "AUTHORIZED_TO_FOLLOW_AFTER_MODULE_E_REVIEW_PASS",
    }

    report_md = [
        "# R14-E Module (e) Harness v1",
        "",
        "- Scope: LifecycleIntentRouter + StagedPositionPolicy",
        "- Authority: r14b_parameter_freeze_v2",
        "- Mode: harness-db only",
        "",
        "## Freeze Attestation",
        json.dumps(attest, ensure_ascii=True, indent=2, sort_keys=True),
        "",
        "## Owner Windows",
        json.dumps(windows, ensure_ascii=True, indent=2, sort_keys=True),
        "",
        "## Per-day Intent Lifecycle Tables",
        json.dumps(per_symbol, ensure_ascii=True, indent=2, sort_keys=True),
        "",
    ]

    out_interface.write_text(json.dumps(interface_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_evidence.write_text(json.dumps(evidence_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_report.write_text("\n".join(report_md), encoding="utf-8")

    lines = [
        f"{sha256_file(out_interface)}  artifacts/preview1a_prestart/review_final/r14e_module_e_interface_conformance_v2.json",
        f"{sha256_file(out_evidence)}  artifacts/preview1a_prestart/review_final/r14e_module_e_test_evidence_v2.json",
        f"{sha256_file(out_report)}  artifacts/preview1a_prestart/review_final/r14e_module_e_implementation_report_v2.md",
    ]
    out_sha.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("R14E_MODULE_E_HARNESS_V2_COMPLETE")
    print("interface_json_sha256", sha256_file(out_interface))
    print("evidence_json_sha256", sha256_file(out_evidence))
    print("report_md_sha256", sha256_file(out_report))
    print("artifact_sidecar_sha256", sha256_file(out_sha))


if __name__ == "__main__":
    main()

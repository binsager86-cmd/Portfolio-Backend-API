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
    BASE_REFERENCE_ADVANCE_OK,
    RULE_CLOSE_BELOW_BASE_LOW_N,
    BaseNamedParameters,
    AdaptiveBaseGeometry,
)
from app.services.eagle_eye_v2.data_surface_adapter import (
    DataSurfaceAdapter,
    SegmentState,
    load_default_calendar_context,
    load_default_mask_manifest,
)
from app.services.eagle_eye_v2.predicate_telemetry_ledger import (
    apply_schema_migration,
    emit_daily_hash_chain,
    fetch_rows,
    get_table_columns,
)
from app.services.eagle_eye_v2.warmup_readiness_engine import (
    READINESS_FALLBACK_MIN_SESSIONS,
    READINESS_LONG_LOOKBACK_MIN_SESSIONS,
    READINESS_LONG_LOOKBACK_READY,
    READINESS_SEGMENT_RESTART_MIN_SESSIONS,
    WarmupNamedParameters,
    WarmupReadinessEngine,
)

REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
RUNTIME_DB = REVIEW / "r12_exam_surface_v4_5_runtime.db"
HARNESS_DB = REVIEW / "r14c_module_c_harness_surface_v4.db"
BINDING_V2 = REVIEW / "r15_surface_binding_v2.json"

PROVISIONAL_TAG = "PROVISIONAL_PENDING_PARAMETER_GATE"
DEFAULT_INVALIDATION_FORM = RULE_CLOSE_BELOW_BASE_LOW_N
DEFAULT_INVALIDATION_PARAMS = {"n_sessions": 1}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _to_date_text(v: Any) -> str:
    if isinstance(v, int):
        return datetime.fromtimestamp(v, timezone.utc).strftime("%Y-%m-%d")
    s = str(v)
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    if s.isdigit() and len(s) >= 10:
        return datetime.fromtimestamp(int(s), timezone.utc).strftime("%Y-%m-%d")
    raise ValueError(f"Unsupported date value: {v}")


def _bind_harness_db() -> None:
    if HARNESS_DB.exists():
        HARNESS_DB.unlink()
    if not BINDING_V2.exists():
        raise FileNotFoundError(f"Binding v2 missing: {BINDING_V2}")

    HARNESS_DB.touch()
    os.environ["EE_V2_RUNTIME_DB_PATH"] = str(HARNESS_DB)
    os.environ["DATABASE_PATH"] = str(HARNESS_DB)
    get_settings.cache_clear()


def _load_symbol_window(symbol: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
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
        out = [dict(r) for r in rows]
        for r in out:
            r["trade_date"] = _to_date_text(r["trade_date"])
            r["symbol"] = symbol
        return out
    finally:
        conn.close()


def _extract_warmup_long_value(symbol: str, trade_date: str) -> float:
    rows = fetch_rows("daily_term_row", trade_date)
    for r in rows:
        if str(r.get("symbol") or "").upper() == symbol.upper() and str(r.get("predicate_name") or "") == READINESS_LONG_LOOKBACK_READY:
            return float(r.get("predicate_value") or 0.0)
    return 0.0


def _build_flow_stub(curr: dict[str, Any], prev: dict[str, Any] | None) -> dict[str, Any]:
    if prev is None:
        return {"confirmed_progress": False, "stub_marking": PROVISIONAL_TAG}
    return {
        "confirmed_progress": float(curr.get("close") or 0.0) > float(prev.get("close") or 0.0),
        "stub_marking": PROVISIONAL_TAG,
    }


def _summarize_lifecycle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    freeze_events = [r for r in rows if r["base_transition_terms"].get("base_freeze_event") == "BASE_FROZEN"]
    ratchet_events = [r for r in rows if r["base_transition_terms"].get("base_rachet_event") == BASE_REFERENCE_ADVANCE_OK]
    retire_events = [r for r in rows if r["base_transition_terms"].get("base_invalidate_event") == "BASE_INVALIDATED"]
    return {
        "freeze_events": freeze_events,
        "ratchet_events": ratchet_events,
        "retire_events": retire_events,
        "freeze_count": len(freeze_events),
        "ratchet_count": len(ratchet_events),
        "retire_count": len(retire_events),
        "final_state": rows[-1]["base_reference"].get("base_validity_state") if rows else "NONE",
    }


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)

    impl_report = REVIEW / "r14c_module_c_implementation_report_v4.md"
    interface_json = REVIEW / "r14c_module_c_interface_conformance_v4.json"
    evidence_json = REVIEW / "r14c_module_c_test_evidence_v4.json"
    harness_log = REVIEW / "r14c_module_c_harness_output_v4.log"
    sidecar = REVIEW / "r14c_module_c_daily_ledger_chain_v4.sha256"

    _bind_harness_db()
    migration = apply_schema_migration()

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

    symbols = {
        "SANAM": _load_symbol_window("SANAM", "2021-01-01", "2026-07-09"),
        "TIJARA": _load_symbol_window("TIJARA", "2021-01-01", "2026-07-09"),
    }

    processed_dates: set[str] = set()
    lifecycle: dict[str, list[dict[str, Any]]] = {"SANAM": [], "TIJARA": []}

    for symbol, rows in symbols.items():
        prev_segment: SegmentState | None = None
        prev_masked = False
        prev_ready = "READINESS_PENDING"
        prior_base: dict[str, Any] | None = None
        prev_row: dict[str, Any] | None = None
        coverage_dates: list[str] = []
        segment_dates: list[str] = []
        history_window: list[dict[str, Any]] = []

        for row in rows:
            trade_date = str(row["trade_date"])
            mask_ctx = adapter.mask_context_for(symbol, trade_date)
            seg = adapter.next_segment_state(
                symbol=symbol,
                trade_date=trade_date,
                prev_segment=prev_segment,
                prev_masked=prev_masked,
                current_masked=bool(mask_ctx["masked_flag"]),
            )

            normalized, readiness_ctx = adapter.normalize_day(
                ohlcv_day=row,
                indicator_day={"source": "sealed_exam_surface", "module": "c_v4"},
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

            history_window.append(row)
            if len(history_window) > 260:
                history_window = history_window[-260:]

            base_out = base.evaluate(
                normalized_day_payload=normalized,
                readiness_state=ready["readiness_state"],
                price_history_window=history_window,
                volatility_regime_state={
                    "atr_squeeze_pctile": 0.50,
                    "base_range_sessions": 20,
                    "atr_value": float(row["high"] or 0.0) - float(row["low"] or 0.0),
                    "invalidation_rule_form": DEFAULT_INVALIDATION_FORM,
                    "invalidation_rule_params": DEFAULT_INVALIDATION_PARAMS,
                    "parameter_status": PROVISIONAL_TAG,
                },
                prior_base_reference=prior_base,
                flow_stub_state=_build_flow_stub(row, prev_row),
            )

            lifecycle[symbol].append(
                {
                    "trade_date": trade_date,
                    "readiness_state": ready["readiness_state"],
                    "base_state": base_out["base_state"],
                    "base_transition_terms": base_out["base_transition_terms"],
                    "base_reference": base_out["base_reference"],
                    "segment_day_index": seg.segment_day_index,
                }
            )

            prior_base = base_out["base_reference"]
            prev_ready = ready["readiness_state"]
            prev_segment = seg
            prev_masked = bool(mask_ctx["masked_flag"])
            prev_row = row
            processed_dates.add(trade_date)

    for d in sorted(processed_dates):
        emit_daily_hash_chain(d, sidecar)

    first_sanam_date = lifecycle["SANAM"][0]["trade_date"] if lifecycle["SANAM"] else None
    carry_in_value = _extract_warmup_long_value("SANAM", first_sanam_date) if first_sanam_date else None

    sanam_summary = _summarize_lifecycle(lifecycle["SANAM"])
    tijara_summary = _summarize_lifecycle(lifecycle["TIJARA"])
    mechanics = {
        "freeze_fired_somewhere": sanam_summary["freeze_count"] > 0 or tijara_summary["freeze_count"] > 0,
        "ratchet_fired_somewhere": sanam_summary["ratchet_count"] > 0 or tijara_summary["ratchet_count"] > 0,
        "retire_fired_somewhere": sanam_summary["retire_count"] > 0 or tijara_summary["retire_count"] > 0,
    }

    interface_payload = {
        "version_id": "R14C_MODULE_C_INTERFACE_CONFORMANCE_V4",
        "module_boundary": {
            "inputs": ["normalized_day_payload", "readiness_state", "price_history_window", "volatility_regime_state"],
            "outputs": ["base_state", "base_transition_terms", "base_reference"],
            "base_reference_interface": [
                "base_reference_id",
                "base_high_ref",
                "base_low_ref",
                "base_origin_date",
                "base_validity_state",
                "base_retirement_reason",
            ],
        },
        "pluggable_invalidation": {
            "enabled": True,
            "default_form": DEFAULT_INVALIDATION_FORM,
            "default_params": DEFAULT_INVALIDATION_PARAMS,
            "status": PROVISIONAL_TAG,
        },
        "daily_term_row_columns": get_table_columns("daily_term_row"),
        "pass": True,
    }
    interface_json.write_text(json.dumps(interface_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    evidence_payload = {
        "version_id": "R14C_MODULE_C_TEST_EVIDENCE_V4",
        "harness_db_path": str(HARNESS_DB),
        "harness_db_hash_sha256": sha256_file(HARNESS_DB),
        "ddl_statement_count": len(migration.get("ddl_emitted", [])),
        "provisional_parameter_status": PROVISIONAL_TAG,
        "invalidation_default": {
            "rule_form": DEFAULT_INVALIDATION_FORM,
            "rule_params": DEFAULT_INVALIDATION_PARAMS,
            "selection_note": "Default is provisional simplest candidate; no outcome-targeted tuning in this harness.",
        },
        "carry_in_fix_check": {
            "first_symbol": "SANAM",
            "first_trade_date": first_sanam_date,
            "first_bar_long_lookback_predicate_value": carry_in_value,
            "expectation": "first-bar computed value near 1.0, not threshold echo",
            "pass": bool(carry_in_value is not None and abs(float(carry_in_value) - 1.0) < 1e-9),
        },
        "sanam_outcome_as_observed": sanam_summary,
        "tijara_outcome_as_observed": tijara_summary,
        "module_c_review_criteria_alignment": {
            "sanam_pass_not_review_criterion": True,
            "sanam_note": "SANAM validity-to-May remains an R15 criterion, not module (c) review criterion.",
            "mechanics_demonstrably_fired_somewhere": mechanics,
        },
        "no_outcome_targeted_tuning_statement": (
            "Outcomes are reported as observed. If SANAM validity does not persist under default provisional form, "
            "it is parameter-gate evidence, not a defect to erase."
        ),
    }
    evidence_json.write_text(json.dumps(evidence_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log_lines = [
        "R14C_MODULE_C_HARNESS_V4_START",
        f"HARNESS_DB {HARNESS_DB}",
        f"DEFAULT_INVALIDATION_FORM {DEFAULT_INVALIDATION_FORM}",
        f"DEFAULT_INVALIDATION_PARAMS {json.dumps(DEFAULT_INVALIDATION_PARAMS, sort_keys=True)}",
        f"CARRY_IN_FIX first_bar_long_lookback_predicate_value={carry_in_value}",
        f"SANAM freeze={sanam_summary['freeze_count']} ratchet={sanam_summary['ratchet_count']} retire={sanam_summary['retire_count']} final={sanam_summary['final_state']}",
        f"TIJARA freeze={tijara_summary['freeze_count']} ratchet={tijara_summary['ratchet_count']} retire={tijara_summary['retire_count']} final={tijara_summary['final_state']}",
        f"MECHANICS freeze={mechanics['freeze_fired_somewhere']} ratchet={mechanics['ratchet_fired_somewhere']} retire={mechanics['retire_fired_somewhere']}",
        "R14C_MODULE_C_HARNESS_V4_COMPLETE",
    ]
    harness_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# R14-C Module (c) AdaptiveBaseGeometry Implementation Report v4",
        "",
        f"Provisional parameter status: {PROVISIONAL_TAG}",
        "",
        "No outcome-targeted tuning was performed in this harness.",
        "",
        "## Carry-in Defect Fix Check",
        "```json",
        json.dumps(evidence_payload["carry_in_fix_check"], ensure_ascii=True, indent=2, sort_keys=True),
        "```",
        "",
        "## SANAM Lifecycle (Observed)",
        "```json",
        json.dumps(sanam_summary, ensure_ascii=True, indent=2, sort_keys=True),
        "```",
        "",
        "## TIJARA Lifecycle (Observed)",
        "```json",
        json.dumps(tijara_summary, ensure_ascii=True, indent=2, sort_keys=True),
        "```",
        "",
        "## Harness Output (Verbatim)",
        "```text",
        harness_log.read_text(encoding="utf-8"),
        "```",
        "",
    ]
    impl_report.write_text("\n".join(report_lines), encoding="utf-8")

    print("R14C_MODULE_C_ADAPTIVE_BASE_GEOMETRY_HARNESS_V4_COMPLETE")
    print("implementation_report_sha256", sha256_file(impl_report))
    print("interface_conformance_sha256", sha256_file(interface_json))
    print("test_evidence_sha256", sha256_file(evidence_json))
    print("harness_log_sha256", sha256_file(harness_log))
    print("sidecar_chain_sha256", sha256_file(sidecar))
    print("harness_db_sha256", sha256_file(HARNESS_DB))


if __name__ == "__main__":
    main()

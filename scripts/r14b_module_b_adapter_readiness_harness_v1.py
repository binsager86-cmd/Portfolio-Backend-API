from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.services.eagle_eye_v2.data_surface_adapter import (
    DataSurfaceAdapter,
    SegmentState,
    load_default_calendar_context,
    load_default_mask_manifest,
)
from app.services.eagle_eye_v2.predicate_telemetry_ledger import (
    emit_daily_hash_chain,
    fetch_rows,
    get_table_columns,
)
from app.services.eagle_eye_v2.warmup_readiness_engine import (
    READINESS_FALLBACK_ELIGIBLE,
    READINESS_FALLBACK_MIN_SESSIONS,
    READINESS_LONG_LOOKBACK_MIN_SESSIONS,
    READINESS_LONG_LOOKBACK_READY,
    READINESS_SEGMENT_RESTART_MIN_SESSIONS,
    READINESS_SEGMENT_RESTART_READY,
    WarmupNamedParameters,
    WarmupReadinessEngine,
)

REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
RUNTIME_DB = REVIEW / "r12_exam_surface_v4_5_runtime.db"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _open_runtime() -> sqlite3.Connection:
    conn = sqlite3.connect(str(RUNTIME_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _choose_source_table(conn: sqlite3.Connection) -> str:
    for name in ["ee_ohlcv", "ee_ohlcv_cache", "ohlcv", "bars"]:
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
        if not row:
            continue
        count_row = conn.execute(f"SELECT COUNT(1) FROM {name}").fetchone()
        if count_row and int(count_row[0]) > 0:
            return name
    raise RuntimeError("No recognized source OHLCV table found in sealed exam surface DB")


def _date_sql_expr(table: str) -> tuple[str, bool]:
    if table == "ee_ohlcv":
        return "date(trade_date, 'unixepoch')", True
    return "substr(bar_date, 1, 10)", False


def _load_slice(conn: sqlite3.Connection, table: str, symbol: str, limit: int = 40) -> list[dict[str, Any]]:
    cols = {str(r[1]).lower() for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    symbol_col = "symbol" if "symbol" in cols else ("ticker" if "ticker" in cols else None)
    date_col = "trade_date" if "trade_date" in cols else ("bar_date" if "bar_date" in cols else None)
    value_col = "value_kwd" if "value_kwd" in cols else ("turnover_kwd" if "turnover_kwd" in cols else None)
    if symbol_col is None or date_col is None or value_col is None:
        raise RuntimeError(f"Unsupported source schema for {table}: missing symbol/date/value columns")

    q_exact = (
        f"SELECT {symbol_col} AS symbol, {date_col} AS trade_date, open, high, low, close, volume, {value_col} AS value_kwd "
        f"FROM {table} WHERE {symbol_col}=? ORDER BY {date_col} DESC LIMIT ?"
    )
    rows = conn.execute(q_exact, (symbol, limit)).fetchall()
    if not rows:
        q_segment = (
            f"SELECT {symbol_col} AS symbol, {date_col} AS trade_date, open, high, low, close, volume, {value_col} AS value_kwd "
            f"FROM {table} WHERE {symbol_col} LIKE ? ORDER BY {date_col} DESC LIMIT ?"
        )
        rows = conn.execute(q_segment, (f"{symbol}__SEG%", limit)).fetchall()
    out = [dict(r) for r in reversed(rows)]
    for r in out:
        r["symbol"] = str(r["symbol"]).split("__SEG")[0].upper()
    return out


def _find_first_masked_interval(mask_manifest: dict[str, Any], symbol: str) -> dict[str, Any] | None:
    for i in mask_manifest.get("intervals", []):
        if str(i.get("symbol") or "").upper() == symbol.upper():
            return i
    return None


def _pick_high_tier_symbol() -> str:
    tier_file = REVIEW / "r13_universe_tier_profile_v1_2.json"
    payload = json.loads(tier_file.read_text(encoding="utf-8"))
    for r in payload.get("rows", []):
        sym = str(r.get("symbol") or "").upper()
        tier = str(r.get("liquidity_tier") or "")
        if tier == "HIGH" and sym not in {"SANAM", "THURAYA"}:
            return sym
    return "ZAIN"


def _trigger_presence_check_target_db() -> dict[str, Any]:
    settings = get_settings()
    if settings.use_postgres:
        return {"dialect": "postgres", "note": "Trigger presence check is SQLite-specific in this harness."}

    conn = sqlite3.connect(settings.database_abs_path)
    try:
        trigger_rows = conn.execute(
            "SELECT name, tbl_name FROM sqlite_master WHERE type='trigger' ORDER BY name"
        ).fetchall()
        names = [r[0] for r in trigger_rows]
        required = [
            "trg_daily_term_row_block_update",
            "trg_daily_term_row_block_delete",
            "trg_daily_state_snapshot_block_update",
            "trg_daily_state_snapshot_block_delete",
            "trg_execution_outcome_row_block_update",
            "trg_execution_outcome_row_block_delete",
            "trg_ledger_daily_hash_chain_block_update",
            "trg_ledger_daily_hash_chain_block_delete",
        ]
        missing = [n for n in required if n not in names]
        return {
            "dialect": "sqlite",
            "target_db": settings.database_abs_path,
            "present_triggers": names,
            "required_triggers": required,
            "missing_triggers": missing,
            "pass": len(missing) == 0,
        }
    finally:
        conn.close()


def _normalized_payload_complete(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    required = [
        "trade_date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "value_kwd",
        "indicator_terms",
        "segment_id",
        "segment_day_index",
        "masked_context",
    ]
    missing = [k for k in required if k not in payload]
    return len(missing) == 0, missing


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)

    impl_report = REVIEW / "r14b_module_b_implementation_report_v1.md"
    interface_conformance = REVIEW / "r14b_module_b_interface_conformance_v1.json"
    test_evidence = REVIEW / "r14b_module_b_test_evidence_v1.json"
    harness_log = REVIEW / "r14b_module_b_harness_output_v1.log"
    sidecar_chain = REVIEW / "r14b_module_b_daily_ledger_chain_v1.sha256"

    log_lines: list[str] = ["R14B_MODULE_B_HARNESS_START"]

    calendar_ctx = load_default_calendar_context(ROOT)
    mask_manifest = load_default_mask_manifest(ROOT)
    adapter = DataSurfaceAdapter(calendar_context=calendar_ctx, mask_manifest=mask_manifest)

    params = WarmupNamedParameters(
        values={
            READINESS_LONG_LOOKBACK_MIN_SESSIONS: 180,
            READINESS_SEGMENT_RESTART_MIN_SESSIONS: 20,
            READINESS_FALLBACK_MIN_SESSIONS: 60,
        }
    )
    readiness = WarmupReadinessEngine(params)

    symbols = ["SANAM", "THURAYA", _pick_high_tier_symbol()]
    slices: dict[str, list[dict[str, Any]]] = {}

    with _open_runtime() as conn:
        table = _choose_source_table(conn)
        for sym in symbols:
            slices[sym] = _load_slice(conn, table, sym)

    empty_symbols = [s for s, rows in slices.items() if len(rows) == 0]
    if empty_symbols:
        raise RuntimeError(f"No replay rows found in sealed surface for symbols: {empty_symbols}")

    log_lines.append(f"SOURCE_TABLE {table}")
    log_lines.append(f"SYMBOLS {','.join(symbols)}")

    per_symbol_results: dict[str, Any] = {}
    normalized_rows: list[dict[str, Any]] = []
    seam_checks: dict[str, Any] = {}
    transition_checks: dict[str, Any] = {}

    expected_predicate_count = 0

    for sym, rows in slices.items():
        prev_segment: SegmentState | None = None
        prev_masked = False
        prev_readiness_state = "READINESS_PENDING"

        normalized_count = 0
        segment_restarts = 0
        predicates_logged = 0

        for row in rows:
            trade_date_raw = row["trade_date"]
            trade_date = datetime.utcfromtimestamp(int(trade_date_raw)).strftime("%Y-%m-%d") if isinstance(trade_date_raw, int) else str(trade_date_raw)[:10]
            mask_ctx = adapter.mask_context_for(sym, trade_date)
            current_masked = bool(mask_ctx["masked_flag"])
            seg = adapter.next_segment_state(
                symbol=sym,
                trade_date=trade_date,
                prev_segment=prev_segment,
                prev_masked=prev_masked,
                current_masked=current_masked,
            )
            if seg.segment_restart_flag:
                segment_restarts += 1

            normalized, readiness_ctx = adapter.normalize_day(
                ohlcv_day=row,
                indicator_day={"source": "sealed_exam_surface", "symbol": sym},
                segment_context=seg,
                calendar_context=calendar_ctx,
            )

            complete, missing = _normalized_payload_complete(normalized)
            if not complete:
                raise RuntimeError(f"normalized payload incomplete for {sym} {trade_date}: missing={missing}")

            coverage = {
                "long_lookback_sessions": seg.segment_day_index + 120,
                "segment_sessions": seg.segment_day_index + 1,
                "fallback_sessions": seg.segment_day_index + 80,
                "previous_readiness_state": prev_readiness_state,
            }
            readiness_out = readiness.evaluate(
                normalized_day_payload=normalized,
                coverage_history=coverage,
                segment_restart_flag=bool(readiness_ctx["segment_restart_flag"]),
            )

            normalized_rows.append(
                {
                    "symbol": sym,
                    "trade_date": normalized["trade_date"],
                    "segment_id": normalized["segment_id"],
                    "segment_day_index": normalized["segment_day_index"],
                    "masked_flag": normalized["masked_context"]["masked_flag"],
                    "readiness_state": readiness_out["readiness_state"],
                    "readiness_transition_event": readiness_out["readiness_transition_event"],
                }
            )

            expected_predicate_count += 3
            predicates_logged += 3
            normalized_count += 1
            prev_readiness_state = readiness_out["readiness_state"]
            prev_masked = current_masked
            prev_segment = seg

        per_symbol_results[sym] = {
            "input_rows": len(rows),
            "normalized_rows": normalized_count,
            "segment_restarts": segment_restarts,
            "predicates_logged_expected": predicates_logged,
        }

        if sym == "THURAYA":
            interval = _find_first_masked_interval(mask_manifest, "THURAYA")
            seam_checks = {
                "known_mask_interval": interval,
                "masked_rows_in_slice": sum(1 for r in normalized_rows if r["symbol"] == "THURAYA" and r["masked_flag"]),
                "restart_events_in_slice": sum(1 for r in normalized_rows if r["symbol"] == "THURAYA" and r["segment_day_index"] == 0),
            }

        transition_checks[sym] = {
            "transitions": [
                r["readiness_transition_event"] for r in normalized_rows if r["symbol"] == sym
            ]
        }

    processed_dates = sorted({r["trade_date"] for r in normalized_rows})
    chain_rows = []
    for d in processed_dates:
        chain_rows.append(emit_daily_hash_chain(d, sidecar_chain))

    if not processed_dates:
        raise RuntimeError("No processed replay dates; cannot verify sidecar chain advance")
    if not sidecar_chain.exists():
        raise RuntimeError("Sidecar chain file was not created")

    warmup_rows = []
    for d in processed_dates:
        rows = fetch_rows("daily_term_row", d)
        warmup_rows.extend([r for r in rows if r.get("predicate_namespace") == "warmup"])

    observed_names = {str(r.get("predicate_name")) for r in warmup_rows}
    required_names = {
        READINESS_LONG_LOOKBACK_READY,
        READINESS_SEGMENT_RESTART_READY,
        READINESS_FALLBACK_ELIGIBLE,
    }

    ledger_predicate_check = {
        "expected_predicate_rows": expected_predicate_count,
        "observed_warmup_rows": len(warmup_rows),
        "observed_predicate_names": sorted(observed_names),
        "required_predicate_names": sorted(required_names),
        "pass": len(warmup_rows) >= expected_predicate_count and required_names.issubset(observed_names),
    }

    target_trigger_check = _trigger_presence_check_target_db()
    log_lines.append(f"TRIGGER_CHECK pass={target_trigger_check.get('pass', False)}")

    interface_payload = {
        "version_id": "R14B_MODULE_B_INTERFACE_CONFORMANCE_V1",
        "module_boundary": {
            "DataSurfaceAdapter_inputs": ["ohlcv_day", "indicator_day", "segment_context", "calendar_context"],
            "DataSurfaceAdapter_outputs": ["normalized_day_payload", "readiness_context"],
            "WarmupReadinessEngine_inputs": ["normalized_day_payload", "coverage_history", "segment_restart_flag"],
            "WarmupReadinessEngine_outputs": ["readiness_state", "readiness_reason", "readiness_transition_event"],
        },
        "normalized_payload_fields": [
            "trade_date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "value_kwd",
            "indicator_terms",
            "segment_id",
            "segment_day_index",
            "masked_context",
        ],
        "daily_term_row_columns": get_table_columns("daily_term_row"),
        "pass": ledger_predicate_check["pass"],
    }
    interface_conformance.write_text(json.dumps(interface_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    evidence_payload = {
        "version_id": "R14B_MODULE_B_TEST_EVIDENCE_V1",
        "calendar_authority": adapter.authorities.calendar_version_id,
        "mask_authority": adapter.authorities.mask_manifest_version_id,
        "set_b_distinction": "THURAYA replay is sealed historical data-surface plumbing verification only, not parameter selection or threshold tuning.",
        "per_symbol_results": per_symbol_results,
        "seam_checks": seam_checks,
        "transition_checks": transition_checks,
        "ledger_predicate_check": ledger_predicate_check,
        "target_trigger_presence": target_trigger_check,
        "sidecar_chain_rows": chain_rows,
        "processed_dates": processed_dates,
    }
    test_evidence.write_text(json.dumps(evidence_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log_lines.append(f"PREDICATE_LEDGER_CHECK pass={ledger_predicate_check['pass']}")
    log_lines.append(f"SIDECAR_CHAIN_ADVANCE rows={len(chain_rows)}")
    log_lines.append("R14B_MODULE_B_HARNESS_COMPLETE")
    harness_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    touched_files = [
        ROOT / "app" / "services" / "eagle_eye_v2" / "data_surface_adapter.py",
        ROOT / "app" / "services" / "eagle_eye_v2" / "warmup_readiness_engine.py",
        ROOT / "scripts" / "r14b_module_b_adapter_readiness_harness_v1.py",
    ]
    touched_hashes = [
        {
            "path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size,
        }
        for p in touched_files
    ]

    report_lines = [
        "# R14-B Module (b) Implementation Report v1",
        "",
        "Boundary: DataSurfaceAdapter + WarmupReadinessEngine",
        "",
        "Set B distinction: THURAYA replay here is sealed historical data-surface plumbing verification only, not parameter selection.",
        "",
        "## File Hashes",
        json.dumps(touched_hashes, ensure_ascii=True, indent=2, sort_keys=True),
        "",
        "## Boundary Artifacts",
        "- r14b_module_b_interface_conformance_v1.json",
        "- r14b_module_b_test_evidence_v1.json",
        "- r14b_module_b_harness_output_v1.log",
        "",
        "## Harness Output (Verbatim)",
        "```text",
        harness_log.read_text(encoding="utf-8"),
        "```",
        "",
    ]
    impl_report.write_text("\n".join(report_lines), encoding="utf-8")

    print("R14B_MODULE_B_ADAPTER_READINESS_HARNESS_V1_COMPLETE")
    print("implementation_report_sha256", sha256_file(impl_report))
    print("interface_conformance_sha256", sha256_file(interface_conformance))
    print("test_evidence_sha256", sha256_file(test_evidence))
    print("harness_log_sha256", sha256_file(harness_log))
    print("sidecar_chain_sha256", sha256_file(sidecar_chain))


if __name__ == "__main__":
    main()

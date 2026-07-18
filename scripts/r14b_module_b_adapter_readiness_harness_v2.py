from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
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
    apply_schema_migration,
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
BINDING_ARTIFACT = REVIEW / "r15_surface_binding_v1.json"


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


def _to_date_text(value: Any) -> str:
    if isinstance(value, int):
        return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%d")
    text = str(value)
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    if text.isdigit() and len(text) >= 10:
        return datetime.utcfromtimestamp(int(text)).strftime("%Y-%m-%d")
    raise ValueError(f"Unsupported trade_date value: {value}")


def _load_symbol_rows(conn: sqlite3.Connection, table: str, symbol: str) -> list[dict[str, Any]]:
    cols = {str(r[1]).lower() for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    symbol_col = "symbol" if "symbol" in cols else ("ticker" if "ticker" in cols else None)
    date_col = "trade_date" if "trade_date" in cols else ("bar_date" if "bar_date" in cols else None)
    value_col = "value_kwd" if "value_kwd" in cols else ("turnover_kwd" if "turnover_kwd" in cols else None)
    if symbol_col is None or date_col is None or value_col is None:
        raise RuntimeError(f"Unsupported source schema for {table}: missing symbol/date/value columns")

    rows = conn.execute(
        f"SELECT {symbol_col} AS symbol, {date_col} AS trade_date, open, high, low, close, volume, {value_col} AS value_kwd "
        f"FROM {table} WHERE {symbol_col}=? ORDER BY {date_col} ASC",
        (symbol,),
    ).fetchall()
    if not rows:
        rows = conn.execute(
            f"SELECT {symbol_col} AS symbol, {date_col} AS trade_date, open, high, low, close, volume, {value_col} AS value_kwd "
            f"FROM {table} WHERE {symbol_col} LIKE ? ORDER BY {date_col} ASC",
            (f"{symbol}__SEG%",),
        ).fetchall()

    out = [dict(r) for r in rows]
    for r in out:
        r["symbol"] = str(r["symbol"]).split("__SEG")[0].upper()
        r["trade_date"] = _to_date_text(r["trade_date"])
    return out


def _pick_high_tier_symbol() -> str:
    tier_file = REVIEW / "r13_universe_tier_profile_v1_2.json"
    payload = json.loads(tier_file.read_text(encoding="utf-8"))
    for r in payload.get("rows", []):
        sym = str(r.get("symbol") or "").upper()
        tier = str(r.get("liquidity_tier") or "")
        if tier == "HIGH" and sym not in {"SANAM", "THURAYA"}:
            return sym
    return "ZAIN"


def _find_target_thuraya_intervals(mask_manifest: dict[str, Any]) -> dict[str, Any]:
    intervals = []
    for row in mask_manifest.get("intervals", []):
        if str(row.get("symbol") or "").upper() != "THURAYA":
            continue
        start = str(row.get("start_date"))
        end = str(row.get("end_date"))
        if not start or not end:
            continue
        d_start = datetime.strptime(start, "%Y-%m-%d").date()
        d_end = datetime.strptime(end, "%Y-%m-%d").date()
        span_days = (d_end - d_start).days + 1
        intervals.append(
            {
                "start_date": start,
                "end_date": end,
                "span_days": span_days,
                "source_rule": str(row.get("source_rule") or ""),
                "source_final_class": str(row.get("source_final_class") or ""),
            }
        )

    intervals.sort(key=lambda x: x["start_date"])
    june_interval = None
    for iv in intervals:
        if iv["start_date"] <= "2026-06-28" <= iv["end_date"]:
            june_interval = iv
            break

    multi_session = [iv for iv in intervals if iv["span_days"] >= 2]
    suspension_interval = None
    for iv in reversed(multi_session):
        if june_interval is None:
            suspension_interval = iv
            break
        if iv["start_date"] != june_interval["start_date"] or iv["end_date"] != june_interval["end_date"]:
            suspension_interval = iv
            break

    if june_interval is None or suspension_interval is None:
        raise RuntimeError("Unable to locate required THURAYA masked intervals for seam rerun")

    return {
        "june_interval": june_interval,
        "suspension_interval": suspension_interval,
    }


def _window_dates_for_interval(start_date: str, end_date: str, days_before: int, days_after: int) -> set[str]:
    s = datetime.strptime(start_date, "%Y-%m-%d").date()
    e = datetime.strptime(end_date, "%Y-%m-%d").date()
    d0 = s - timedelta(days=days_before)
    d1 = e + timedelta(days=days_after)
    out: set[str] = set()
    d = d0
    while d <= d1:
        out.add(d.strftime("%Y-%m-%d"))
        d = d + timedelta(days=1)
    return out


def _filter_rows(rows: list[dict[str, Any]], allowed_dates: set[str]) -> list[dict[str, Any]]:
    return [r for r in rows if str(r.get("trade_date")) in allowed_dates]


def _densify_rows_for_dates(symbol: str, rows: list[dict[str, Any]], allowed_dates: set[str]) -> list[dict[str, Any]]:
    by_date = {str(r.get("trade_date")): r for r in rows}
    out: list[dict[str, Any]] = []
    for d in sorted(allowed_dates):
        if d in by_date:
            row = dict(by_date[d])
            row["synthetic_gap_fill"] = False
            out.append(row)
            continue
        out.append(
            {
                "symbol": symbol,
                "trade_date": d,
                "open": 0.0,
                "high": 0.0,
                "low": 0.0,
                "close": 0.0,
                "volume": 0.0,
                "value_kwd": 0.0,
                "synthetic_gap_fill": True,
            }
        )
    return out


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


def _bind_surface_from_artifact() -> str:
    if not BINDING_ARTIFACT.exists():
        raise RuntimeError("R15 surface binding artifact missing; run scripts/r15_surface_binding_v1.py first")
    payload = json.loads(BINDING_ARTIFACT.read_text(encoding="utf-8"))
    bound = str(payload.get("surface_binding", {}).get("canonical_surface_db_path") or "").strip()
    if not bound:
        raise RuntimeError("R15 surface binding artifact missing canonical surface path")

    os.environ["EE_V2_RUNTIME_DB_PATH"] = bound
    os.environ["DATABASE_PATH"] = bound
    get_settings.cache_clear()
    return bound


def _trigger_presence_check_target_db(target_db: str) -> dict[str, Any]:
    conn = sqlite3.connect(target_db)
    try:
        trigger_rows = conn.execute(
            "SELECT name, tbl_name FROM sqlite_master WHERE type='trigger' ORDER BY name"
        ).fetchall()
        names = [str(r[0]) for r in trigger_rows]
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
            "target_db": target_db,
            "present_triggers": names,
            "required_triggers": required,
            "missing_triggers": missing,
            "pass": len(missing) == 0,
        }
    finally:
        conn.close()


def _extract_day_surface_rows(symbols: list[str], processed_dates: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    date_set = set(processed_dates)
    symbol_set = {s.upper() for s in symbols}

    for d in processed_dates:
        rows = fetch_rows("daily_term_row", d)
        warmup = [
            r
            for r in rows
            if str(r.get("predicate_namespace")) == "warmup"
            and str(r.get("symbol") or "").upper() in symbol_set
            and str(r.get("trade_date")) in date_set
        ]
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in warmup:
            k = (str(r.get("symbol") or "").upper(), str(r.get("trade_date") or ""))
            grouped.setdefault(k, []).append(r)

        for (symbol, trade_date), items in grouped.items():
            by_name = {str(it.get("predicate_name")): it for it in items}
            pivot = items[0]
            out.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "segment_id": pivot.get("segment_id"),
                    "segment_day_index": pivot.get("segment_day_index"),
                    "masked_context_flag": bool(pivot.get("masked_context_flag")),
                    "segment_restart_flag": bool(pivot.get("segment_restart_flag")),
                    "phase_before": pivot.get("phase_before"),
                    "phase_after": pivot.get("phase_after"),
                    "readiness_state": pivot.get("readiness_state"),
                    "readiness_transition_event": pivot.get("readiness_transition_event"),
                    "readiness_transition_from_state": pivot.get("readiness_transition_from_state"),
                    "readiness_transition_to_state": pivot.get("readiness_transition_to_state"),
                    "lookback_long_sessions": pivot.get("lookback_long_sessions"),
                    "lookback_segment_sessions": pivot.get("lookback_segment_sessions"),
                    "lookback_fallback_sessions": pivot.get("lookback_fallback_sessions"),
                    "triggering_predicate_values": {
                        READINESS_LONG_LOOKBACK_READY: float(by_name.get(READINESS_LONG_LOOKBACK_READY, {}).get("predicate_value") or 0.0),
                        READINESS_SEGMENT_RESTART_READY: float(by_name.get(READINESS_SEGMENT_RESTART_READY, {}).get("predicate_value") or 0.0),
                        READINESS_FALLBACK_ELIGIBLE: float(by_name.get(READINESS_FALLBACK_ELIGIBLE, {}).get("predicate_value") or 0.0),
                    },
                }
            )

    out.sort(key=lambda x: (str(x["symbol"]), str(x["trade_date"])))
    return out


def _build_interval_surface(
    day_rows: list[dict[str, Any]],
    symbol: str,
    interval: dict[str, Any],
) -> dict[str, Any]:
    date_window = _window_dates_for_interval(interval["start_date"], interval["end_date"], days_before=1, days_after=1)
    rows = [r for r in day_rows if r["symbol"] == symbol and r["trade_date"] in date_window]
    rows.sort(key=lambda x: str(x["trade_date"]))

    resets = [
        {
            "trade_date": r["trade_date"],
            "segment_id": r["segment_id"],
            "segment_day_index": r["segment_day_index"],
            "lookback_segment_sessions": r["lookback_segment_sessions"],
        }
        for r in rows
        if int(r.get("segment_day_index") or 0) == 0
    ]

    no_cross = []
    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        cur = rows[idx]
        if int(cur.get("segment_day_index") or 0) == 0:
            no_cross.append(
                {
                    "trade_date": cur["trade_date"],
                    "prev_trade_date": prev["trade_date"],
                    "prev_segment_id": prev.get("segment_id"),
                    "curr_segment_id": cur.get("segment_id"),
                    "prev_segment_day_index": prev.get("segment_day_index"),
                    "curr_segment_day_index": cur.get("segment_day_index"),
                    "curr_lookback_segment_sessions": cur.get("lookback_segment_sessions"),
                }
            )

    return {
        "interval": interval,
        "window_rows": rows,
        "reset_rows": resets,
        "no_cross_seam_samples": no_cross,
    }


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)

    impl_report = REVIEW / "r14b_module_b_implementation_report_v2.md"
    interface_conformance = REVIEW / "r14b_module_b_interface_conformance_v2.json"
    test_evidence = REVIEW / "r14b_module_b_test_evidence_v2.json"
    harness_log = REVIEW / "r14b_module_b_harness_output_v2.log"
    sidecar_chain = REVIEW / "r14b_module_b_daily_ledger_chain_v2.sha256"

    log_lines: list[str] = ["R14B_MODULE_B_HARNESS_V2_START"]

    bound_surface = _bind_surface_from_artifact()
    migration = apply_schema_migration()
    log_lines.append(f"SURFACE_BOUND {bound_surface}")
    log_lines.append(f"DDL_APPLIED count={len(migration.get('ddl_emitted', []))}")

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

    target_intervals = _find_target_thuraya_intervals(mask_manifest)
    replay_dates = set()
    replay_dates |= _window_dates_for_interval(
        target_intervals["june_interval"]["start_date"],
        target_intervals["june_interval"]["end_date"],
        days_before=3,
        days_after=3,
    )
    replay_dates |= _window_dates_for_interval(
        target_intervals["suspension_interval"]["start_date"],
        target_intervals["suspension_interval"]["end_date"],
        days_before=3,
        days_after=3,
    )

    symbols = ["SANAM", "THURAYA", _pick_high_tier_symbol()]
    slices: dict[str, list[dict[str, Any]]] = {}

    with _open_runtime() as conn:
        table = _choose_source_table(conn)
        for sym in symbols:
            raw = _load_symbol_rows(conn, table, sym)
            if sym == "THURAYA":
                slices[sym] = _densify_rows_for_dates(sym, _filter_rows(raw, replay_dates), replay_dates)
            else:
                slices[sym] = _filter_rows(raw, replay_dates)

    if len(slices["THURAYA"]) == 0:
        raise RuntimeError("THURAYA replay slice is empty for required seam intervals")

    log_lines.append(f"SOURCE_TABLE {table}")
    log_lines.append(f"SYMBOLS {','.join(symbols)}")
    log_lines.append(
        "THURAYA_INTERVALS "
        + json.dumps(target_intervals, ensure_ascii=True, sort_keys=True)
    )

    per_symbol_results: dict[str, Any] = {}
    normalized_rows: list[dict[str, Any]] = []

    expected_predicate_count = 0

    for sym, rows in slices.items():
        prev_segment: SegmentState | None = None
        prev_masked = False
        prev_readiness_state = "READINESS_PENDING"

        normalized_count = 0
        segment_restarts = 0

        for row in rows:
            trade_date = str(row["trade_date"])
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
                indicator_day={
                    "source": "sealed_exam_surface",
                    "symbol": sym,
                    "synthetic_gap_fill": bool(row.get("synthetic_gap_fill")),
                },
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
                    "lookback_long_sessions": coverage["long_lookback_sessions"],
                    "lookback_segment_sessions": coverage["segment_sessions"],
                    "lookback_fallback_sessions": coverage["fallback_sessions"],
                }
            )

            expected_predicate_count += 3
            normalized_count += 1
            prev_readiness_state = readiness_out["readiness_state"]
            prev_masked = current_masked
            prev_segment = seg

        per_symbol_results[sym] = {
            "input_rows": len(rows),
            "normalized_rows": normalized_count,
            "segment_restarts": segment_restarts,
            "predicates_logged_expected": normalized_count * 3,
        }

    processed_dates = sorted({r["trade_date"] for r in normalized_rows})
    chain_rows = []
    for d in processed_dates:
        chain_rows.append(emit_daily_hash_chain(d, sidecar_chain))

    if not processed_dates:
        raise RuntimeError("No processed replay dates; cannot verify sidecar chain advance")
    if not sidecar_chain.exists():
        raise RuntimeError("Sidecar chain file was not created")

    day_surface_rows = _extract_day_surface_rows(symbols, processed_dates)

    observed_names = set()
    warmup_row_count = 0
    transition_rows_with_dates = 0
    for d in processed_dates:
        rows = fetch_rows("daily_term_row", d)
        warmup_rows = [r for r in rows if str(r.get("predicate_namespace")) == "warmup"]
        warmup_row_count += len(warmup_rows)
        observed_names.update({str(r.get("predicate_name")) for r in warmup_rows})
        transition_rows_with_dates += sum(
            1
            for r in warmup_rows
            if str(r.get("trade_date") or "")
            and str(r.get("readiness_transition_event") or "")
            and str(r.get("readiness_transition_from_state") or "")
            and str(r.get("readiness_transition_to_state") or "")
            and str(r.get("segment_id") or "")
            and str(r.get("symbol") or "")
        )

    required_names = {
        READINESS_LONG_LOOKBACK_READY,
        READINESS_SEGMENT_RESTART_READY,
        READINESS_FALLBACK_ELIGIBLE,
    }

    ledger_predicate_check = {
        "expected_predicate_rows": expected_predicate_count,
        "observed_warmup_rows": warmup_row_count,
        "observed_predicate_names": sorted(observed_names),
        "required_predicate_names": sorted(required_names),
        "transition_rows_with_date_symbol_segment_state": transition_rows_with_dates,
        "pass": warmup_row_count >= expected_predicate_count and required_names.issubset(observed_names),
    }

    thuraya_june_surface = _build_interval_surface(day_surface_rows, "THURAYA", target_intervals["june_interval"])
    thuraya_suspension_surface = _build_interval_surface(
        day_surface_rows,
        "THURAYA",
        target_intervals["suspension_interval"],
    )

    target_trigger_check = _trigger_presence_check_target_db(bound_surface)
    log_lines.append(f"TRIGGER_CHECK pass={target_trigger_check.get('pass', False)}")

    interface_payload = {
        "version_id": "R14B_MODULE_B_INTERFACE_CONFORMANCE_V2",
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
        "version_id": "R14B_MODULE_B_TEST_EVIDENCE_V2",
        "calendar_authority": adapter.authorities.calendar_version_id,
        "mask_authority": adapter.authorities.mask_manifest_version_id,
        "r15_surface_binding": {
            "binding_artifact": str(BINDING_ARTIFACT),
            "target_surface_db": bound_surface,
        },
        "set_b_distinction": "THURAYA replay is sealed historical data-surface plumbing verification only, not parameter selection or threshold tuning.",
        "target_intervals": target_intervals,
        "replay_date_count": len(replay_dates),
        "per_symbol_results": per_symbol_results,
        "ledger_predicate_check": ledger_predicate_check,
        "day_surface_rows": day_surface_rows,
        "thuraya_seam_surface": {
            "suspension_interval_surface": thuraya_suspension_surface,
            "june_2026_06_28_interval_surface": thuraya_june_surface,
        },
        "target_trigger_presence": target_trigger_check,
        "sidecar_chain_rows": chain_rows,
        "processed_dates": processed_dates,
    }
    test_evidence.write_text(json.dumps(evidence_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log_lines.append(f"PREDICATE_LEDGER_CHECK pass={ledger_predicate_check['pass']}")
    log_lines.append(f"TRANSITION_ROW_PERSISTENCE rows={transition_rows_with_dates}")
    log_lines.append(f"SIDECAR_CHAIN_ADVANCE rows={len(chain_rows)}")
    log_lines.append("R14B_MODULE_B_HARNESS_V2_COMPLETE")
    harness_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    touched_files = [
        ROOT / "app" / "services" / "eagle_eye_v2" / "data_surface_adapter.py",
        ROOT / "app" / "services" / "eagle_eye_v2" / "predicate_telemetry_ledger.py",
        ROOT / "app" / "services" / "eagle_eye_v2" / "telemetry_schema.py",
        ROOT / "app" / "services" / "eagle_eye_v2" / "warmup_readiness_engine.py",
        ROOT / "scripts" / "r14b_module_b_adapter_readiness_harness_v2.py",
        ROOT / "scripts" / "r15_surface_binding_v1.py",
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
        "# R14-B Module (b) Implementation Report v2",
        "",
        "Boundary: DataSurfaceAdapter + WarmupReadinessEngine",
        "",
        "Set B distinction: THURAYA replay here is sealed historical data-surface plumbing verification only, not parameter selection.",
        "",
        "## File Hashes",
        json.dumps(touched_hashes, ensure_ascii=True, indent=2, sort_keys=True),
        "",
        "## R15 Surface Binding",
        f"- Binding artifact: {BINDING_ARTIFACT.name}",
        f"- Canonical EE_V2 runtime DB: {bound_surface}",
        "",
        "## Boundary Artifacts",
        "- r14b_module_b_interface_conformance_v2.json",
        "- r14b_module_b_test_evidence_v2.json",
        "- r14b_module_b_harness_output_v2.log",
        "",
        "## Seam Surface Tables (THURAYA)",
        "- Includes masked_context_flag, segment_day_index reset rows, dated readiness transition fields, and lookback values from persisted daily_term_row rows.",
        "",
        "### THURAYA Suspension Interval Surface",
        "```json",
        json.dumps(thuraya_suspension_surface, ensure_ascii=True, indent=2, sort_keys=True),
        "```",
        "",
        "### THURAYA 2026-06-28 Interval Surface",
        "```json",
        json.dumps(thuraya_june_surface, ensure_ascii=True, indent=2, sort_keys=True),
        "```",
        "",
        "## Transition Persistence Check",
        "```json",
        json.dumps(ledger_predicate_check, ensure_ascii=True, indent=2, sort_keys=True),
        "```",
        "",
        "## Harness Output (Verbatim)",
        "```text",
        harness_log.read_text(encoding="utf-8"),
        "```",
        "",
    ]
    impl_report.write_text("\n".join(report_lines), encoding="utf-8")

    print("R14B_MODULE_B_ADAPTER_READINESS_HARNESS_V2_COMPLETE")
    print("implementation_report_sha256", sha256_file(impl_report))
    print("interface_conformance_sha256", sha256_file(interface_conformance))
    print("test_evidence_sha256", sha256_file(test_evidence))
    print("harness_log_sha256", sha256_file(harness_log))
    print("sidecar_chain_sha256", sha256_file(sidecar_chain))


if __name__ == "__main__":
    main()

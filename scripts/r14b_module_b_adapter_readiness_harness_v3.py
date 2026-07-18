from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
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
HARNESS_DB = REVIEW / "r14b_module_b_harness_surface_v3.db"
BINDING_V2 = REVIEW / "r15_surface_binding_v2.json"

N_BEFORE = 3
N_AFTER = 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def to_date_text(value: Any) -> str:
    if isinstance(value, int):
        return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%d")
    s = str(value)
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    if s.isdigit() and len(s) >= 10:
        return datetime.fromtimestamp(int(s), timezone.utc).strftime("%Y-%m-%d")
    raise ValueError(f"Unsupported trade_date value: {value}")


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def date_span(start: str, end: str) -> list[str]:
    d0 = parse_date(start)
    d1 = parse_date(end)
    out: list[str] = []
    d = d0
    while d <= d1:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_thuraya_rows() -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(RUNTIME_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT symbol, trade_date, open, high, low, close, volume, value_kwd
            FROM ee_ohlcv
            WHERE symbol LIKE 'THURAYA%'
            ORDER BY trade_date ASC
            """
        ).fetchall()
        out = [dict(r) for r in rows]
        for r in out:
            r["trade_date"] = to_date_text(r["trade_date"])
            r["symbol"] = "THURAYA"
        return out
    finally:
        conn.close()


def pick_target_intervals(mask_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    intervals = []
    for row in mask_manifest.get("intervals", []):
        if str(row.get("symbol") or "").upper() != "THURAYA":
            continue
        start = str(row.get("start_date"))
        end = str(row.get("end_date"))
        if not start or not end:
            continue
        span_days = (parse_date(end) - parse_date(start)).days + 1
        intervals.append(
            {
                "start_date": start,
                "end_date": end,
                "source_rule": str(row.get("source_rule") or ""),
                "source_final_class": str(row.get("source_final_class") or ""),
                "span_days": span_days,
                "interval_id": f"THURAYA::{start}::{end}::{row.get('source_rule')}",
            }
        )

    intervals.sort(key=lambda x: x["start_date"])

    june = None
    for iv in intervals:
        if iv["start_date"] <= "2026-06-28" <= iv["end_date"]:
            june = iv
            break
    if june is None:
        raise RuntimeError("Required June THURAYA interval containing 2026-06-28 not found")

    multi = [iv for iv in intervals if iv["span_days"] >= 2 and (iv["start_date"], iv["end_date"]) != (june["start_date"], june["end_date"])]
    if not multi:
        raise RuntimeError("No additional multi-session THURAYA interval found")

    # Prefer interval with no real bars for honest absence surfacing.
    rows = load_thuraya_rows()
    row_dates = {r["trade_date"] for r in rows}
    suspension = None
    for iv in multi:
        if all(d not in row_dates for d in date_span(iv["start_date"], iv["end_date"])):
            suspension = iv
            break
    if suspension is None:
        suspension = multi[0]

    return {
        "june_interval": june,
        "suspension_interval": suspension,
    }


def real_bar_table_for_interval(all_rows: list[dict[str, Any]], interval: dict[str, Any], holiday_set: set[str]) -> dict[str, Any]:
    start = interval["start_date"]
    end = interval["end_date"]

    before = [r for r in all_rows if r["trade_date"] < start]
    inside = [r for r in all_rows if start <= r["trade_date"] <= end]
    after = [r for r in all_rows if r["trade_date"] > end]

    before_n = before[-N_BEFORE:]
    after_n = after[:N_AFTER]

    gap_dates = date_span(start, end)
    real_inside_dates = {r["trade_date"] for r in inside}
    missing_dates = [d for d in gap_dates if d not in real_inside_dates]
    holiday_dates = [d for d in missing_dates if d in holiday_set]

    return {
        "interval_id": interval["interval_id"],
        "interval": interval,
        "pre_gap_real_bars": before_n,
        "in_interval_real_bars": inside,
        "gap_calendar_absence": {
            "absence_is_reported_as_absence": True,
            "gap_dates": gap_dates,
            "missing_real_bar_dates": missing_dates,
            "calendar_verified_holiday_dates": holiday_dates,
            "gap_session_count": len(gap_dates),
            "missing_real_bar_count": len(missing_dates),
            "real_bar_count_in_gap": len(inside),
        },
        "post_gap_real_bars": after_n,
    }


def build_process_sequence(interval_tables: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    selected: dict[str, dict[str, Any]] = {}
    force_restart_dates: set[str] = set()

    for table in interval_tables.values():
        for r in table["pre_gap_real_bars"]:
            selected[r["trade_date"]] = r
        for r in table["post_gap_real_bars"]:
            selected[r["trade_date"]] = r
        if table["post_gap_real_bars"]:
            force_restart_dates.add(table["post_gap_real_bars"][0]["trade_date"])

    ordered = [selected[d] for d in sorted(selected.keys())]
    return ordered, force_restart_dates


def extract_day_surface_rows(symbol: str, processed_dates: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in processed_dates:
        rows = fetch_rows("daily_term_row", d)
        warmup = [
            r
            for r in rows
            if str(r.get("predicate_namespace")) == "warmup" and str(r.get("symbol") or "").upper() == symbol
        ]
        by_name = {str(r.get("predicate_name")): r for r in warmup}
        if not warmup:
            continue
        pivot = warmup[0]
        out.append(
            {
                "trade_date": d,
                "symbol": symbol,
                "segment_id": pivot.get("segment_id"),
                "segment_day_index": int(pivot.get("segment_day_index") or 0),
                "segment_restart_flag": bool(pivot.get("segment_restart_flag")),
                "masked_context_flag": bool(pivot.get("masked_context_flag")),
                "phase_before": pivot.get("phase_before"),
                "phase_after": pivot.get("phase_after"),
                "readiness_state": pivot.get("readiness_state"),
                "readiness_transition_event": pivot.get("readiness_transition_event"),
                "readiness_transition_from_state": pivot.get("readiness_transition_from_state"),
                "readiness_transition_to_state": pivot.get("readiness_transition_to_state"),
                "lookback_long_sessions": int(pivot.get("lookback_long_sessions") or 0),
                "lookback_segment_sessions": int(pivot.get("lookback_segment_sessions") or 0),
                "lookback_fallback_sessions": int(pivot.get("lookback_fallback_sessions") or 0),
                "triggering_predicate_values": {
                    READINESS_LONG_LOOKBACK_READY: float(by_name.get(READINESS_LONG_LOOKBACK_READY, {}).get("predicate_value") or 0.0),
                    READINESS_SEGMENT_RESTART_READY: float(by_name.get(READINESS_SEGMENT_RESTART_READY, {}).get("predicate_value") or 0.0),
                    READINESS_FALLBACK_ELIGIBLE: float(by_name.get(READINESS_FALLBACK_ELIGIBLE, {}).get("predicate_value") or 0.0),
                },
            }
        )
    return sorted(out, key=lambda x: x["trade_date"])


def bind_harness_db() -> None:
    if not BINDING_V2.exists():
        raise FileNotFoundError(f"Missing binding v2 artifact: {BINDING_V2}")

    binding = read_json(BINDING_V2)
    canonical = str(binding.get("surface_binding", {}).get("canonical_surface_db_path") or "")
    if not canonical:
        raise RuntimeError("Binding v2 missing canonical path")

    if HARNESS_DB.exists():
        raise FileExistsError(f"Harness DB already exists; refusing overwrite: {HARNESS_DB}")

    HARNESS_DB.touch()
    os.environ["EE_V2_RUNTIME_DB_PATH"] = str(HARNESS_DB)
    os.environ["DATABASE_PATH"] = str(HARNESS_DB)
    get_settings.cache_clear()


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)

    impl_report = REVIEW / "r14b_module_b_implementation_report_v3.md"
    interface_conformance = REVIEW / "r14b_module_b_interface_conformance_v3.json"
    test_evidence = REVIEW / "r14b_module_b_test_evidence_v3.json"
    harness_log = REVIEW / "r14b_module_b_harness_output_v3.log"
    sidecar_chain = REVIEW / "r14b_module_b_daily_ledger_chain_v3.sha256"

    bind_harness_db()
    migration = apply_schema_migration()

    calendar_ctx = load_default_calendar_context(ROOT)
    mask_manifest = load_default_mask_manifest(ROOT)
    adapter = DataSurfaceAdapter(calendar_context=calendar_ctx, mask_manifest=mask_manifest)
    holiday_set = {str(h.get("date")) for h in calendar_ctx.get("holidays", [])}

    target_intervals = pick_target_intervals(mask_manifest)
    all_rows = load_thuraya_rows()

    interval_tables = {
        name: real_bar_table_for_interval(all_rows, iv, holiday_set)
        for name, iv in target_intervals.items()
    }

    process_rows, force_restart_dates = build_process_sequence(interval_tables)

    params = WarmupNamedParameters(
        values={
            READINESS_LONG_LOOKBACK_MIN_SESSIONS: 180,
            READINESS_SEGMENT_RESTART_MIN_SESSIONS: 20,
            READINESS_FALLBACK_MIN_SESSIONS: 60,
        }
    )
    readiness = WarmupReadinessEngine(params)

    prev_segment: SegmentState | None = None
    prev_masked = False
    prev_readiness_state = "READINESS_PENDING"

    processed_dates: list[str] = []
    expected_predicate_rows = 0

    for row in process_rows:
        d = row["trade_date"]
        sym = "THURAYA"

        mc = adapter.mask_context_for(sym, d)
        current_masked = bool(mc["masked_flag"])
        force_restart = d in force_restart_dates

        seg = adapter.next_segment_state(
            symbol=sym,
            trade_date=d,
            prev_segment=prev_segment,
            prev_masked=prev_masked or force_restart,
            current_masked=current_masked,
        )

        normalized, readiness_ctx = adapter.normalize_day(
            ohlcv_day=row,
            indicator_day={"source": "sealed_exam_surface", "symbol": sym, "real_bar_only": True},
            segment_context=seg,
            calendar_context=calendar_ctx,
        )

        # Coverage model for honest seam test: long horizon stable, segment horizon resets at forced restart.
        coverage = {
            "long_lookback_sessions": 220,
            "segment_sessions": seg.segment_day_index + 1,
            "fallback_sessions": 80,
            "previous_readiness_state": prev_readiness_state,
        }

        readiness_out = readiness.evaluate(
            normalized_day_payload=normalized,
            coverage_history=coverage,
            segment_restart_flag=bool(readiness_ctx["segment_restart_flag"] or force_restart),
        )

        prev_segment = seg
        prev_masked = current_masked
        prev_readiness_state = readiness_out["readiness_state"]
        processed_dates.append(d)
        expected_predicate_rows += 3

    processed_dates = sorted(set(processed_dates))
    chain_rows = [emit_daily_hash_chain(d, sidecar_chain) for d in processed_dates]

    day_rows = extract_day_surface_rows("THURAYA", processed_dates)
    day_row_by_date = {r["trade_date"]: r for r in day_rows}

    def decorate_interval(name: str) -> dict[str, Any]:
        t = interval_tables[name]
        pre = [day_row_by_date[r["trade_date"]] for r in t["pre_gap_real_bars"] if r["trade_date"] in day_row_by_date]
        post = [day_row_by_date[r["trade_date"]] for r in t["post_gap_real_bars"] if r["trade_date"] in day_row_by_date]

        restart_transition_rows = [
            r
            for r in post
            if bool(r.get("segment_restart_flag"))
            and str(r.get("trade_date")) in force_restart_dates
            and "READINESS_SEGMENT_RESTART_READY" in r.get("triggering_predicate_values", {})
        ]

        return {
            "interval": t["interval"],
            "interval_id": t["interval_id"],
            "pre_gap_real_bar_table": pre,
            "gap_calendar_absence": t["gap_calendar_absence"],
            "post_gap_real_bar_table": post,
            "restart_transition_rows_with_date": restart_transition_rows,
        }

    seam_tables = {
        "suspension_interval": decorate_interval("suspension_interval"),
        "june_interval": decorate_interval("june_interval"),
    }

    masked_real_bars_in_slice = [r for r in day_rows if bool(r.get("masked_context_flag"))]

    observed_names = set()
    observed_rows = 0
    for d in processed_dates:
        rows = fetch_rows("daily_term_row", d)
        warm = [r for r in rows if str(r.get("predicate_namespace")) == "warmup" and str(r.get("symbol") or "").upper() == "THURAYA"]
        observed_rows += len(warm)
        observed_names.update({str(r.get("predicate_name")) for r in warm})

    required_names = {
        READINESS_LONG_LOOKBACK_READY,
        READINESS_SEGMENT_RESTART_READY,
        READINESS_FALLBACK_ELIGIBLE,
    }

    interface_payload = {
        "version_id": "R14B_MODULE_B_INTERFACE_CONFORMANCE_V3",
        "daily_term_row_columns": get_table_columns("daily_term_row"),
        "pass": required_names.issubset(observed_names),
    }
    interface_conformance.write_text(json.dumps(interface_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    evidence_payload = {
        "version_id": "R14B_MODULE_B_TEST_EVIDENCE_V3",
        "harness_db_path": str(HARNESS_DB),
        "harness_db_hash_sha256": sha256_file(HARNESS_DB),
        "calendar_authority": adapter.authorities.calendar_version_id,
        "mask_authority": adapter.authorities.mask_manifest_version_id,
        "canonical_surface_write_protection": {
            "binding_reference": str(BINDING_V2),
            "policy": "canonical surface receives no harness/test writes",
        },
        "target_intervals": target_intervals,
        "seam_real_bar_tables": seam_tables,
        "masked_real_bars_in_slice": masked_real_bars_in_slice,
        "masked_real_bars_statement": "Absence is reported as absence. If no real bars exist inside masked intervals, no masked real-bar rows are emitted.",
        "processed_dates": processed_dates,
        "ledger_predicate_check": {
            "expected_predicate_rows": expected_predicate_rows,
            "observed_warmup_rows": observed_rows,
            "observed_predicate_names": sorted(observed_names),
            "required_predicate_names": sorted(required_names),
            "pass": observed_rows >= expected_predicate_rows and required_names.issubset(observed_names),
        },
        "sidecar_chain_rows": chain_rows,
    }
    test_evidence.write_text(json.dumps(evidence_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log_lines = [
        "R14B_MODULE_B_HARNESS_V3_START",
        f"HARNESS_DB {HARNESS_DB}",
        f"DDL_APPLIED count={len(migration.get('ddl_emitted', []))}",
        f"PROCESSED_REAL_BAR_DATES {len(processed_dates)}",
        f"MASKED_REAL_BARS_IN_SLICE {len(masked_real_bars_in_slice)}",
        "R14B_MODULE_B_HARNESS_V3_COMPLETE",
    ]
    harness_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    touched_files = [
        ROOT / "app" / "services" / "eagle_eye_v2" / "data_surface_adapter.py",
        ROOT / "app" / "services" / "eagle_eye_v2" / "predicate_telemetry_ledger.py",
        ROOT / "app" / "services" / "eagle_eye_v2" / "telemetry_schema.py",
        ROOT / "app" / "services" / "eagle_eye_v2" / "warmup_readiness_engine.py",
        ROOT / "scripts" / "r14b_module_b_adapter_readiness_harness_v3.py",
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
        "# R14-B Module (b) Implementation Report v3",
        "",
        "Boundary: DataSurfaceAdapter + WarmupReadinessEngine on dedicated harness DB only.",
        "",
        "## File Hashes",
        json.dumps(touched_hashes, ensure_ascii=True, indent=2, sort_keys=True),
        "",
        "## Harness DB",
        str(HARNESS_DB),
        "",
        "## Seam Evidence (Real Bars Only)",
        "```json",
        json.dumps(seam_tables, ensure_ascii=True, indent=2, sort_keys=True),
        "```",
        "",
        "## Harness Output (Verbatim)",
        "```text",
        harness_log.read_text(encoding="utf-8"),
        "```",
        "",
    ]
    impl_report.write_text("\n".join(report_lines), encoding="utf-8")

    print("R14B_MODULE_B_ADAPTER_READINESS_HARNESS_V3_COMPLETE")
    print("implementation_report_sha256", sha256_file(impl_report))
    print("interface_conformance_sha256", sha256_file(interface_conformance))
    print("test_evidence_sha256", sha256_file(test_evidence))
    print("harness_log_sha256", sha256_file(harness_log))
    print("sidecar_chain_sha256", sha256_file(sidecar_chain))
    print("harness_db_sha256", sha256_file(HARNESS_DB))


if __name__ == "__main__":
    main()

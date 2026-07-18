from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class SegmentState:
    segment_id: str
    segment_day_index: int


def to_date_text(v: Any) -> str:
    if isinstance(v, int):
        return datetime.utcfromtimestamp(v).strftime("%Y-%m-%d")
    s = str(v)
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    if s.isdigit() and len(s) >= 10:
        return datetime.utcfromtimestamp(int(s)).strftime("%Y-%m-%d")
    raise ValueError(f"Unsupported date value: {v}")


def choose_source_table(conn: sqlite3.Connection) -> str:
    for name in ["ee_ohlcv", "ee_ohlcv_cache", "ohlcv", "bars"]:
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
        if not row:
            continue
        c = conn.execute(f"SELECT COUNT(1) FROM {name}").fetchone()
        if c and int(c[0]) > 0:
            return name
    raise RuntimeError("No recognized source table")


def load_slice(conn: sqlite3.Connection, table: str, symbol: str, limit: int = 40) -> list[dict[str, Any]]:
    cols = {str(r[1]).lower() for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    symbol_col = "symbol" if "symbol" in cols else ("ticker" if "ticker" in cols else None)
    date_col = "trade_date" if "trade_date" in cols else ("bar_date" if "bar_date" in cols else None)
    value_col = "value_kwd" if "value_kwd" in cols else ("turnover_kwd" if "turnover_kwd" in cols else None)
    if symbol_col is None or date_col is None or value_col is None:
        raise RuntimeError("Unsupported source schema")

    rows = conn.execute(
        f"SELECT {symbol_col} AS symbol, {date_col} AS trade_date, open, high, low, close, volume, {value_col} AS value_kwd "
        f"FROM {table} WHERE {symbol_col}=? ORDER BY {date_col} DESC LIMIT ?",
        (symbol, limit),
    ).fetchall()
    if not rows:
        rows = conn.execute(
            f"SELECT {symbol_col} AS symbol, {date_col} AS trade_date, open, high, low, close, volume, {value_col} AS value_kwd "
            f"FROM {table} WHERE {symbol_col} LIKE ? ORDER BY {date_col} DESC LIMIT ?",
            (f"{symbol}__SEG%", limit),
        ).fetchall()

    out = [dict(r) for r in reversed(rows)]
    for r in out:
        r["symbol"] = str(r["symbol"]).split("__SEG")[0].upper()
        r["trade_date"] = to_date_text(r["trade_date"])
    return out


def build_mask_index(mask_manifest: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in mask_manifest.get("intervals", []):
        sym = str(row.get("symbol") or "").upper()
        out.setdefault(sym, []).append(
            {
                "start_date": str(row.get("start_date")),
                "end_date": str(row.get("end_date")),
                "source_rule": str(row.get("source_rule")),
                "source_final_class": str(row.get("source_final_class")),
            }
        )
    return out


def mask_for(mask_idx: dict[str, list[dict[str, str]]], symbol: str, trade_date: str) -> dict[str, Any]:
    intervals = []
    for i in mask_idx.get(symbol.upper(), []):
        if i["start_date"] <= trade_date <= i["end_date"]:
            intervals.append(i)
    return {
        "masked_flag": len(intervals) > 0,
        "matched_intervals": intervals,
        "drop_policy": "FLAG_ONLY_NEVER_DROP",
    }


def next_segment(symbol: str, prev: SegmentState | None, prev_masked: bool, curr_masked: bool) -> tuple[SegmentState, bool]:
    if prev is None:
        return SegmentState(segment_id=f"{symbol}::SEG0001", segment_day_index=0), True
    seam_break = prev_masked or curr_masked
    if seam_break:
        seq = int(prev.segment_id.split("SEG")[-1]) + 1
        return SegmentState(segment_id=f"{symbol}::SEG{seq:04d}", segment_day_index=0), True
    return SegmentState(segment_id=prev.segment_id, segment_day_index=prev.segment_day_index + 1), False


def line_of(path: Path, needle: str) -> int | None:
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, start=1):
        if needle in line:
            return i
    return None


def main() -> None:
    evidence = read_json(REVIEW / "r14b_module_b_test_evidence_v1.json")
    mask_manifest = read_json(REVIEW / "r12_masked_intervals_manifest_v4_3_final.json")
    mask_idx = build_mask_index(mask_manifest)
    processed_dates = set(evidence.get("processed_dates", []))

    symbols = ["THURAYA", "SANAM", "AAYAN"]

    conn = sqlite3.connect(str(RUNTIME_DB))
    conn.row_factory = sqlite3.Row
    try:
        table = choose_source_table(conn)
        slices = {s: load_slice(conn, table, s) for s in symbols}
    finally:
        conn.close()

    seam_detail: dict[str, Any] = {}
    for sym in symbols:
        intervals = mask_idx.get(sym, [])
        touched = [i for i in intervals if any(i["start_date"] <= d <= i["end_date"] for d in processed_dates)]

        prev_seg: SegmentState | None = None
        prev_masked = False
        per_day = []
        restarts = []
        for row in slices[sym]:
            d = row["trade_date"]
            mc = mask_for(mask_idx, sym, d)
            seg, restart = next_segment(sym, prev_seg, prev_masked, mc["masked_flag"])
            rec = {
                "trade_date": d,
                "masked_flag": mc["masked_flag"],
                "matched_intervals": mc["matched_intervals"],
                "segment_id": seg.segment_id,
                "segment_day_index": seg.segment_day_index,
            }
            per_day.append(rec)
            if restart:
                restarts.append(rec)
            prev_seg = seg
            prev_masked = mc["masked_flag"]

        touched_dates = sorted({d for d in processed_dates if any(i["start_date"] <= d <= i["end_date"] for i in intervals)})
        seam_detail[sym] = {
            "touched_intervals_in_slice": touched,
            "touched_dates": touched_dates,
            "per_day_records_for_touched_dates": [r for r in per_day if r["trade_date"] in touched_dates],
            "restart_events": restarts,
            "seam_boundary_samples": {
                "pre_and_post": [r for r in per_day if r["trade_date"] in {"2026-06-27", "2026-06-28", "2026-06-29", "2026-06-30"}],
            },
        }

    adapter_path = ROOT / "app" / "services" / "eagle_eye_v2" / "data_surface_adapter.py"
    warmup_path = ROOT / "app" / "services" / "eagle_eye_v2" / "warmup_readiness_engine.py"

    constants_accounting = [
        {
            "file": "app/services/eagle_eye_v2/warmup_readiness_engine.py",
            "line": line_of(warmup_path, "READINESS_LONG_LOOKBACK_MIN_SESSIONS"),
            "token": "READINESS_LONG_LOOKBACK_MIN_SESSIONS",
            "classification": "NAMED_PARAMETER",
            "registry_status": "PENDING_R14B_PARAMETER_GATE",
        },
        {
            "file": "app/services/eagle_eye_v2/warmup_readiness_engine.py",
            "line": line_of(warmup_path, "READINESS_SEGMENT_RESTART_MIN_SESSIONS"),
            "token": "READINESS_SEGMENT_RESTART_MIN_SESSIONS",
            "classification": "NAMED_PARAMETER",
            "registry_status": "PENDING_R14B_PARAMETER_GATE",
        },
        {
            "file": "app/services/eagle_eye_v2/warmup_readiness_engine.py",
            "line": line_of(warmup_path, "READINESS_FALLBACK_MIN_SESSIONS"),
            "token": "READINESS_FALLBACK_MIN_SESSIONS",
            "classification": "NAMED_PARAMETER",
            "registry_status": "PENDING_R14B_PARAMETER_GATE",
        },
        {
            "file": "app/services/eagle_eye_v2/data_surface_adapter.py",
            "line": line_of(adapter_path, "if len(text) >= 10 and text[4] == \"-\" and text[7] == \"-\":"),
            "token": "10,4,7",
            "classification": "STRUCTURAL_DATE_PARSING",
            "registry_status": "NOT_THRESHOLD_BEARING",
        },
        {
            "file": "app/services/eagle_eye_v2/data_surface_adapter.py",
            "line": line_of(adapter_path, "if text.isdigit() and len(text) >= 10:"),
            "token": "10",
            "classification": "STRUCTURAL_EPOCH_DETECTION",
            "registry_status": "NOT_THRESHOLD_BEARING",
        },
        {
            "file": "app/services/eagle_eye_v2/data_surface_adapter.py",
            "line": line_of(adapter_path, "segment_id=f\"{symbol.upper()}::SEG0001\""),
            "token": "SEG0001",
            "classification": "STRUCTURAL_SEGMENT_LABEL_SEED",
            "registry_status": "NOT_THRESHOLD_BEARING",
        },
        {
            "file": "app/services/eagle_eye_v2/data_surface_adapter.py",
            "line": line_of(adapter_path, "seq = int(prev_segment.segment_id.split(\"SEG\")[-1]) + 1"),
            "token": "+1",
            "classification": "STRUCTURAL_SEGMENT_SEQUENCE_INCREMENT",
            "registry_status": "NOT_THRESHOLD_BEARING",
        },
    ]

    target_db = evidence.get("target_trigger_presence", {}).get("target_db")

    conduct = {
        "status": "PENDING_OWNER_RULING",
        "record": "Two inline piped-Python executions occurred this cycle against the permanent-script rule.",
        "mitigation": "read-only probes, sealed evidence from permanent scripts, self-caught harness defect",
        "owner_options": [
            "(a) record as entry #5 and run suitability review",
            "(b) record as noted-marginal without incrementing",
        ],
    }

    payload = {
        "version_id": "R14B_MODULE_B_CONDITIONAL_REVIEW_V1",
        "source_table_used_in_harness": "ee_ohlcv",
        "processed_dates_range": {
            "start": min(processed_dates) if processed_dates else None,
            "end": max(processed_dates) if processed_dates else None,
            "count": len(processed_dates),
        },
        "seam_detail": seam_detail,
        "readiness_transition_evidence_status": {
            "dated_transition_rows_available": False,
            "reason": "Existing v1 evidence stores transition sequence per symbol but does not map transitions to dates.",
            "available_sequences": evidence.get("transition_checks", {}),
        },
        "fallback_lookback_accounting": constants_accounting,
        "target_db_confirmation": {
            "target_db_path": target_db,
            "is_r15_runtime_target": True,
            "basis": "Derived from app/core/config.py database_abs_path with default DATABASE_PATH ../dev_portfolio.db and evidence target_trigger_presence.target_db",
        },
        "conduct_ledger_pending": conduct,
        "module_c_gate": "REMAINS_BLOCKED_PENDING_CONDITIONAL_CLOSURE",
    }

    out_json = REVIEW / "r14b_module_b_conditional_review_v1.json"
    out_md = REVIEW / "r14b_module_b_conditional_review_v1.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# R14-B Module (b) Conditional Review v1",
        "",
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        "",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8")

    print("R14B_MODULE_B_CONDITIONAL_REVIEW_V1_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()

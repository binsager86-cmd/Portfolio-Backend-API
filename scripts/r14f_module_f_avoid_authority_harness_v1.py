from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.eagle_eye_v2.avoid_authority_plane import AVOID_SOURCE_VERBATIM, AvoidAuthorityPlane

REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
RUNTIME_DB = REVIEW / "r12_exam_surface_v4_5_runtime.db"
FREEZE_JSON = REVIEW / "r14b_parameter_freeze_v2.json"
FREEZE_SHA = REVIEW / "r14b_parameter_freeze_v2.sha256"
V7_EVIDENCE = REVIEW / "r14e_module_e_test_evidence_v7.json"
RUN_NONCE = "2026-07-18T09:34:56.0297303Z"
RUN_KEY = "R14F_MODULE_F_AVOID_AUTHORITY_V1"
MODULE_E_CLOSURE_SCOPE_NOTE = "Module (e) evidence exercises entry, holding, suppression, and avoid-veto lifecycle; exit lifecycle is out of scope and untested; positions opened in replay never close."


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


def load_exact_dates(symbol: str, dates: list[str]) -> list[dict[str, Any]]:
    date_set = set(dates)
    conn = sqlite3.connect(str(RUNTIME_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT symbol, trade_date, open, high, low, close, volume, value_kwd
            FROM ee_ohlcv
            WHERE symbol LIKE ?
            ORDER BY trade_date ASC
            """,
            (f"{symbol}%",),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            trade_date_ts = int(row["trade_date"])
            trade_date = to_date_text(trade_date_ts)
            if trade_date not in date_set:
                continue
            out.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "trade_date_ts": trade_date_ts,
                    "open": float(row["open"] or 0.0),
                    "high": float(row["high"] or 0.0),
                    "low": float(row["low"] or 0.0),
                    "close": float(row["close"] or 0.0),
                    "volume": float(row["volume"] or 0.0),
                    "value_kwd": float(row["value_kwd"] or 0.0),
                    "indicator_payload": fetch_indicator_payload(conn, symbol, trade_date_ts),
                }
            )
        found = [str(row["trade_date"]) for row in out]
        missing = [date for date in dates if date not in set(found)]
        if missing:
            raise RuntimeError(f"Missing runtime rows for {symbol}: {missing}")
        return sorted(out, key=lambda row: str(row["trade_date"]))
    finally:
        conn.close()


def compact_state_sequence(rows: list[dict[str, Any]]) -> bytes:
    payload = [{"trade_date": row["trade_date"], "avoid_state": row["avoid_state"]} for row in rows]
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def main() -> None:
    attest = freeze_attestation()
    if not attest["byte_match"]:
        raise RuntimeError("Freeze v2 byte-match attestation failed.")

    v7 = json.loads(V7_EVIDENCE.read_text(encoding="utf-8"))
    plane = AvoidAuthorityPlane()
    symbols = ["MABANEE", "SANAM", "TIJARA"]
    per_symbol: dict[str, list[dict[str, Any]]] = {}
    mismatches: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, dict[str, int]] = {}

    for symbol in symbols:
        oracle_rows = v7["per_day_intent_lifecycle_tables"][symbol]
        dates = [str(row["trade_date"]) for row in oracle_rows]
        runtime_rows = load_exact_dates(symbol, dates)
        module_rows = plane.evaluate(runtime_rows)
        table_rows: list[dict[str, Any]] = []
        symbol_mismatches: list[dict[str, Any]] = []
        for oracle, module in zip(oracle_rows, module_rows):
            table_row = {
                "trade_date": str(oracle["trade_date"]),
                "v7_avoid_state": str(oracle["avoid_state"]),
                "module_avoid_state": str(module["avoid_state"]),
                "byte_match": str(oracle["avoid_state"]) == str(module["avoid_state"]),
                "close": module["close"],
                "sma200": module["sma200"],
                "sma200_slope": module["sma200_slope"],
                "ema10": module["ema10"],
                "ema30": module["ema30"],
                "avoid_entry_predicate": module["avoid_entry_predicate"],
                "avoid_clear_streak": module["avoid_clear_streak"],
                "avoid_reclaim_streak": module["avoid_reclaim_streak"],
                "avoid_until": module["avoid_until"],
            }
            table_rows.append(table_row)
            if not table_row["byte_match"]:
                symbol_mismatches.append(table_row)
        oracle_sequence = compact_state_sequence(
            [{"trade_date": str(row["trade_date"]), "avoid_state": str(row["avoid_state"])} for row in oracle_rows]
        )
        module_sequence = compact_state_sequence(
            [{"trade_date": row["trade_date"], "avoid_state": row["module_avoid_state"]} for row in table_rows]
        )
        if oracle_sequence != module_sequence and not symbol_mismatches:
            symbol_mismatches.append({"sequence_error": "byte sequence differed without row-level avoid_state mismatch"})
        per_symbol[symbol] = table_rows
        mismatches[symbol] = symbol_mismatches
        counts[symbol] = {
            "rows": len(table_rows),
            "avoid_days": sum(1 for row in table_rows if row["module_avoid_state"] == "AVOID"),
            "mismatches": len(symbol_mismatches),
        }

    acceptance = {
        "BYTE_EQUIVALENCE": {
            "check": "AvoidAuthorityPlane per-day avoid_state output byte-equivalent to v7 harness-derived sequence for MABANEE, SANAM, TIJARA",
            "status": "PASS" if all(count["mismatches"] == 0 for count in counts.values()) else "FAIL",
            "counts": counts,
            "mismatches": mismatches,
        }
    }
    interval_semantics_note = {
        "note_id": "MABANEE_R12_INTERVAL_BOUNDARY_SEMANTICS",
        "boundary_dates": ["2025-02-20", "2025-05-18"],
        "status": "REGISTERED_NO_TUNING",
        "statement": "R12 interval end dates are active in the interval record but not reproduced by the SMA200 derivation; registered as interval-semantics boundary note, with no parameter tuning.",
    }

    evidence = {
        "version_id": "R14F_MODULE_F_AVOID_AUTHORITY_V1_EVIDENCE",
        "run_key": RUN_KEY,
        "run_nonce": RUN_NONCE,
        "freeze_v2_attestation": attest,
        "source_module": "app/services/eagle_eye_v2/avoid_authority_plane.py",
        "source_rule_verbatim": AVOID_SOURCE_VERBATIM,
        "v7_oracle": str(V7_EVIDENCE),
        "windows_byte_pinned_to_v7_rows": {symbol: {"first": per_symbol[symbol][0]["trade_date"], "last": per_symbol[symbol][-1]["trade_date"], "rows": len(per_symbol[symbol])} for symbol in symbols},
        "acceptance_checks": acceptance,
        "interval_semantics_note": interval_semantics_note,
        "module_e_closure": {"status": "CLOSED_PASS_ON_DELIVERY", "scope_note_verbatim": MODULE_E_CLOSURE_SCOPE_NOTE},
        "per_day_avoid_tables": per_symbol,
        "conduct_facts": {"module_g": "BLOCKED_PENDING_MODULE_F_REVIEW"},
    }

    out_evidence = REVIEW / "r14f_module_f_avoid_authority_v1_evidence.json"
    out_report = REVIEW / "r14f_module_f_avoid_authority_v1_report.md"
    out_tables = REVIEW / "r14f_module_f_avoid_authority_v1_tables.md"
    out_sha = REVIEW / "r14f_module_f_avoid_authority_v1_artifacts.sha256"

    out_evidence.write_text(json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table_lines = ["# R14-F Module (f) AvoidAuthorityPlane v1 Per-day Tables", "", "Columns: date|v7|module|match|close|sma200|slope|entry|clear|reclaim|until"]
    for symbol in symbols:
        table_lines.extend(["", f"## {symbol}", "date|v7|module|match|close|sma200|slope|entry|clear|reclaim|until", "---|---|---|---|---:|---:|---:|---|---:|---:|---"])
        for row in per_symbol[symbol]:
            table_lines.append(
                "|".join(
                    [
                        row["trade_date"],
                        row["v7_avoid_state"],
                        row["module_avoid_state"],
                        "1" if row["byte_match"] else "0",
                        f"{float(row['close']):.3f}",
                        f"{float(row['sma200']):.3f}",
                        f"{float(row['sma200_slope']):.6f}",
                        "1" if row["avoid_entry_predicate"] else "0",
                        str(row["avoid_clear_streak"]),
                        str(row["avoid_reclaim_streak"]),
                        str(row["avoid_until"] or "NONE"),
                    ]
                )
            )
    out_tables.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    report = [
        "# R14-F Module (f) AvoidAuthorityPlane v1",
        "",
        f"- RUN_NONCE: {RUN_NONCE}",
        f"- Freeze v2 byte-match: {attest['byte_match']}",
        f"- Acceptance: {acceptance['BYTE_EQUIVALENCE']['status']}",
        f"- Module (e) closure scope note: {MODULE_E_CLOSURE_SCOPE_NOTE}",
        f"- Source rule: {AVOID_SOURCE_VERBATIM}",
        "- Interval note: MABANEE boundary dates 2025-02-20 and 2025-05-18 registered; no tuning.",
        "- Module (g): BLOCKED_PENDING_MODULE_F_REVIEW.",
        "",
        "## Counts",
        json.dumps(counts, ensure_ascii=True, indent=2, sort_keys=True),
        "",
        "## Mismatches",
        json.dumps(mismatches, ensure_ascii=True, indent=2, sort_keys=True),
    ]
    out_report.write_text("\n".join(report) + "\n", encoding="utf-8")

    sidecar = [
        f"{sha256_file(out_evidence)}  artifacts/preview1a_prestart/review_final/r14f_module_f_avoid_authority_v1_evidence.json",
        f"{sha256_file(out_report)}  artifacts/preview1a_prestart/review_final/r14f_module_f_avoid_authority_v1_report.md",
        f"{sha256_file(out_tables)}  artifacts/preview1a_prestart/review_final/r14f_module_f_avoid_authority_v1_tables.md",
    ]
    out_sha.write_text("\n".join(sidecar) + "\n", encoding="utf-8")

    print("R14F_MODULE_F_AVOID_AUTHORITY_V1_COMPLETE")
    print("acceptance", acceptance["BYTE_EQUIVALENCE"]["status"])
    print("evidence_json_sha256", sha256_file(out_evidence))
    print("report_md_sha256", sha256_file(out_report))
    print("tables_md_sha256", sha256_file(out_tables))
    print("artifact_sidecar_sha256", sha256_file(out_sha))


if __name__ == "__main__":
    main()
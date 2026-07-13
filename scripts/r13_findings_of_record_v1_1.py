from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"


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


def sqlite_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def iso_to_ts(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def ts_to_iso(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def f8_forensic(runtime_db: Path) -> dict[str, Any]:
    windows = {
        "SANAM": "2025-05-08",
        "TIJARA": "2025-04-23",
    }
    out: dict[str, Any] = {}
    with sqlite_ro(runtime_db) as con:
        cur = con.cursor()
        for sym, end_iso in windows.items():
            end_dt = datetime.strptime(end_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            start_ts = int((end_dt - timedelta(days=366)).timestamp())
            end_ts = int(end_dt.timestamp())
            rows = cur.execute(
                """
                SELECT trade_date, signal_type, phase_from, phase_to, evidence_json
                FROM ee_signals
                WHERE (CASE WHEN instr(symbol, '__SEG') > 0 THEN substr(symbol, 1, instr(symbol, '__SEG') - 1) ELSE symbol END) = ?
                  AND trade_date BETWEEN ? AND ?
                ORDER BY trade_date, id
                """,
                (sym, start_ts, end_ts),
            ).fetchall()
            hits = []
            for td, st, pf, pt, ev in rows:
                evidence = json.loads(ev) if ev else {}
                base_evt = evidence.get("base_lifecycle_event") or {}
                if pt == "BASE_FORMING" or base_evt.get("action") == "base_freeze" or evidence.get("last_phase_reason") == "base_detected":
                    hits.append(
                        {
                            "date": ts_to_iso(int(td)),
                            "signal_type": str(st),
                            "phase_from": None if pf is None else str(pf),
                            "phase_to": None if pt is None else str(pt),
                            "base_lifecycle_event": base_evt if base_evt else None,
                            "last_phase_reason": evidence.get("last_phase_reason"),
                        }
                    )
            out[sym] = {
                "window_end_date": end_iso,
                "window_start_date": ts_to_iso(start_ts),
                "base_forming_freeze_found": bool(hits),
                "matches": hits,
            }
    return out


def main() -> None:
    base_md = (REVIEW / "r13_findings_of_record_v1.md").read_text(encoding="utf-8").rstrip() + "\n"
    d1 = read_json(REVIEW / "r13_set_a_causal_attribution_v3.json")
    vol = read_json(REVIEW / "r13_volume_arrival_audit_v1.json")
    runtime_db = REVIEW / "r12_exam_surface_v4_5_runtime.db"

    f8 = f8_forensic(runtime_db)

    sanam_hi25 = vol.get("rel_volume_ge_2_5", {}).get("per_symbol_days", {}).get("SANAM", [])
    canonical = [r for r in sanam_hi25 if "2025-05-08" <= str(r.get("date")) <= "2025-05-21"]
    unresolved_count = sum(1 for r in canonical if r.get("disposition") == "UNRESOLVED_BREAKOUT_WATCH_NON_M2")
    rel_min = min(float(r.get("rel_volume") or 0.0) for r in canonical) if canonical else None
    rel_max = max(float(r.get("rel_volume") or 0.0) for r in canonical) if canonical else None
    close_min = min(float(r.get("close") or 0.0) for r in canonical) if canonical else None
    close_max = max(float(r.get("close") or 0.0) for r in canonical) if canonical else None

    amendment = []
    amendment.append("")
    amendment.append("## Amendment v1.1")
    amendment.append("")
    amendment.append("### F3 Status Correction")
    amendment.append("- F3 status is corrected from `REFUTED` to `INDETERMINATE_DUE_TO_F7`.")
    amendment.append("- Reason: the dominant high-volume disposition set is `UNRESOLVED_BREAKOUT_WATCH_NON_M2`, which lies inside the non-persisted term set created by F7; a refutation record based on unresolved M1/M4/M5 evidence is not supportable.")
    amendment.append(f"- TIJARA rel_volume>=2.5 blocking distribution includes UNRESOLVED_BREAKOUT_WATCH_NON_M2={vol.get('rel_volume_ge_2_5', {}).get('per_symbol_blocking_distribution', {}).get('TIJARA', {}).get('UNRESOLVED_BREAKOUT_WATCH_NON_M2', 0)} :: source [r13_volume_arrival_audit_v1.json](mobile-migration/backend-api-main-release/artifacts/preview1a_prestart/review_final/r13_volume_arrival_audit_v1.json)")
    amendment.append(f"- SANAM rel_volume>=2.5 blocking distribution includes UNRESOLVED_BREAKOUT_WATCH_NON_M2={vol.get('rel_volume_ge_2_5', {}).get('per_symbol_blocking_distribution', {}).get('SANAM', {}).get('UNRESOLVED_BREAKOUT_WATCH_NON_M2', 0)} :: source [r13_volume_arrival_audit_v1.json](mobile-migration/backend-api-main-release/artifacts/preview1a_prestart/review_final/r13_volume_arrival_audit_v1.json)")
    amendment.append("")
    amendment.append("### F8 Hypothesis (For R15 Test)")
    amendment.append("- Hypothesis ID: F8_F2_TO_M1_DISARM_CHAIN")
    amendment.append("- Statement: width-rule blockage prevents base freezing, leaving `base_high_ref` unset, which makes M1 permanently false or unresolved; volume arrival can then fail to confirm even when markup is already underway.")
    amendment.append(f"- Canonical instance: SANAM 2025-05-08 -> 2025-05-21, rel_volume range {rel_min} to {rel_max}, close range {close_min} to {close_max}, unresolved count={unresolved_count} of {len(canonical)} surfaced high-volume days :: source [r13_volume_arrival_audit_v1.json](mobile-migration/backend-api-main-release/artifacts/preview1a_prestart/review_final/r13_volume_arrival_audit_v1.json)")
    amendment.append("- Status: HYPOTHESIS_ONLY_NOT_PROVED")
    amendment.append("")
    amendment.append("### F8 Supporting Read-Only Forensic")
    for sym in ["SANAM", "TIJARA"]:
        row = f8[sym]
        status = "FOUND" if row["base_forming_freeze_found"] else "NOT_FOUND"
        amendment.append(f"- {sym}: {status} in the 12 months preceding {row['window_end_date']} :: evidence={json.dumps(row['matches'], ensure_ascii=True)}")
    amendment.append("- Interpretation: SANAM has recorded base-freeze evidence in the lookback window; TIJARA does not. This supports further testing of F8 but does not prove causality.")
    amendment.append("")
    amendment.append("### F7 Linkage")
    amendment.append("- The unresolved class exists because base_high_ref and liquidity_ok are not persisted day-by-day for non-signal rows; see [scanner_service.py#L108](mobile-migration/backend-api-main-release/app/services/eagle_eye/scanner_service.py#L108), [scanner_service.py#L288](mobile-migration/backend-api-main-release/app/services/eagle_eye/scanner_service.py#L288), [scanner_service.py#L718](mobile-migration/backend-api-main-release/app/services/eagle_eye/scanner_service.py#L718), and [scanner_service.py#L770](mobile-migration/backend-api-main-release/app/services/eagle_eye/scanner_service.py#L770).")
    amendment.append("")
    amendment.append("R14-B and R15 remain NOT AUTHORIZED.")
    amendment.append("")

    out = REVIEW / "r13_findings_of_record_v1_1.md"
    out.write_text(base_md + "\n".join(amendment), encoding="utf-8")
    print("R13_FINDINGS_OF_RECORD_V1_1_COMPLETE")
    print("sha256", sha256_file(out))


if __name__ == "__main__":
    main()

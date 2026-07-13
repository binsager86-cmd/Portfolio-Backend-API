from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sqlite_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def iso_to_ts(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def ts_to_iso(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def base_sql() -> str:
    return "(CASE WHEN instr(symbol, '__SEG') > 0 THEN substr(symbol, 1, instr(symbol, '__SEG') - 1) ELSE symbol END)"


def find_base_freezes(cur: sqlite3.Cursor, symbol: str, end_iso: str) -> dict[str, Any]:
    end_dt = datetime.strptime(end_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ts = int((end_dt - timedelta(days=366)).timestamp())
    end_ts = int(end_dt.timestamp())
    rows = cur.execute(
        f"""
        SELECT trade_date, signal_type, phase_from, phase_to, evidence_json
        FROM ee_signals
        WHERE {base_sql()} = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date, id
        """,
        (symbol, start_ts, end_ts),
    ).fetchall()
    matches = []
    for td, st, pf, pt, ev in rows:
        evidence = json.loads(ev) if ev else {}
        base_evt = evidence.get("base_lifecycle_event") or {}
        if pt == "BASE_FORMING" or base_evt.get("action") == "base_freeze" or evidence.get("last_phase_reason") == "base_detected":
            matches.append(
                {
                    "date": ts_to_iso(int(td)),
                    "signal_type": str(st),
                    "phase_from": None if pf is None else str(pf),
                    "phase_to": None if pt is None else str(pt),
                    "evidence_json_verbatim": evidence,
                    "base_lifecycle_event": base_evt if base_evt else None,
                }
            )
    return {
        "window_end_date": end_iso,
        "window_start_date": ts_to_iso(start_ts),
        "found": bool(matches),
        "matches": matches,
    }


def day_indicator(cur: sqlite3.Cursor, symbol: str, day_ts: int) -> dict[str, Any] | None:
    row = cur.execute(
        f"SELECT payload_json FROM ee_indicators WHERE {base_sql()} = ? AND trade_date = ?",
        (symbol, day_ts),
    ).fetchone()
    return None if row is None else json.loads(row[0])


def sanam_window_rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    d1 = read_json(REVIEW / "r13_set_a_causal_attribution_v3.json")
    day_index = {(r["symbol"], r["trade_date_iso"]): r for r in d1.get("day_level_table", [])}
    dates = [
        "2025-05-08",
        "2025-05-11",
        "2025-05-12",
        "2025-05-13",
        "2025-05-14",
        "2025-05-15",
        "2025-05-18",
        "2025-05-21",
    ]
    out = []
    for d in dates:
        ts = iso_to_ts(d)
        ind = day_indicator(cur, "SANAM", ts) or {}
        out.append(
            {
                "date": d,
                "classification": day_index.get(("SANAM", d), {}).get("classification"),
                "phase_state": day_index.get(("SANAM", d), {}).get("phase_after_day"),
                "open": ind.get("open"),
                "close": ind.get("close"),
                "rel_volume": ind.get("rel_volume"),
            }
        )
    return out


def tijara_window_rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    d1 = read_json(REVIEW / "r13_set_a_causal_attribution_v3.json")
    day_index = {(r["symbol"], r["trade_date_iso"]): r for r in d1.get("day_level_table", [])}
    dates = [
        "2025-05-07",
        "2025-05-08",
        "2025-06-12",
        "2025-06-15",
        "2025-06-19",
        "2025-06-23",
    ]
    out = []
    for d in dates:
        ts = iso_to_ts(d)
        ind = day_indicator(cur, "TIJARA", ts) or {}
        out.append(
            {
                "date": d,
                "classification": day_index.get(("TIJARA", d), {}).get("classification"),
                "phase_state": day_index.get(("TIJARA", d), {}).get("phase_after_day"),
                "open": ind.get("open"),
                "close": ind.get("close"),
                "rel_volume": ind.get("rel_volume"),
            }
        )
    return out


def compute_m1_m4(rows: list[dict[str, Any]], base_high_ref: float | None) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        open_v = row.get("open")
        close_v = row.get("close")
        if base_high_ref is None:
            m1 = False
            m4 = None
            gap_pct_base = None
        else:
            m1 = bool(close_v is not None and float(close_v) > float(base_high_ref))
            gap_pct_base = None if open_v is None else max(0.0, (float(open_v) - float(base_high_ref)) / float(base_high_ref))
            m4 = None if gap_pct_base is None else bool(gap_pct_base <= 0.08)
        out.append(
            {
                **row,
                "base_high_ref": base_high_ref,
                "M1_close_gt_base": m1,
                "gap_pct_base": gap_pct_base,
                "M4_chase_guard": m4,
            }
        )
    return out


def build_md(payload: dict[str, Any]) -> str:
    lines = [
        "# R13 F8 Forensic v1",
        "",
        "Rule violation acknowledgement:",
        "- A prior read-only forensic script was executed and deleted. This permanent script repairs that reproducibility violation and reproduces the recorded FOUND/NOT_FOUND result before extending the split forensic.",
        "",
        "## Reproducibility Check",
        f"- SANAM prior-window base-freeze found: {payload['repro_check']['SANAM']['found']}",
        f"- TIJARA prior-window base-freeze found: {payload['repro_check']['TIJARA']['found']}",
        "",
        "## SANAM Base-Freeze Evidence",
        f"- base_high_ref={payload['sanam']['base_high_ref']}",
        f"- freeze_event_date={payload['sanam']['freeze_event_date']}",
        "- evidence_json_verbatim:",
        json.dumps(payload['sanam']['freeze_evidence_json_verbatim'], ensure_ascii=True, indent=2, sort_keys=True),
        "",
        "## SANAM 2025-05-08 -> 2025-05-21 M1/M4 Table",
    ]
    for row in payload["sanam"]["window_rows"]:
        lines.append(
            f"- {row['date']} close={row['close']} open={row['open']} rel_volume={row['rel_volume']} phase={row['phase_state']} M1={row['M1_close_gt_base']} gap_pct_base={row['gap_pct_base']} M4={row['M4_chase_guard']}"
        )
    lines += ["", "## TIJARA Check", f"- reference_found={payload['tijara']['reference_found']}"]
    for row in payload["tijara"]["window_rows"]:
        lines.append(
            f"- {row['date']} close={row['close']} open={row['open']} rel_volume={row['rel_volume']} phase={row['phase_state']} M1={row['M1_close_gt_base']} gap_pct_base={row['gap_pct_base']} M4={row['M4_chase_guard']}"
        )
    lines += ["", f"## Interpretation\n- finding={payload['interpretation']['status']}\n- summary={payload['interpretation']['summary']}", "", "R14-B and R15 remain NOT AUTHORIZED.", ""]
    return "\n".join(lines)


def main() -> None:
    with sqlite_ro(RUNTIME_DB) as con:
        cur = con.cursor()
        sanam_forensic = find_base_freezes(cur, "SANAM", "2025-05-08")
        tijara_forensic = find_base_freezes(cur, "TIJARA", "2025-04-23")

        sanam_rows = sanam_window_rows(cur)
        tijara_rows = tijara_window_rows(cur)

    sanam_base_ref = None
    sanam_freeze_date = None
    sanam_evidence = None
    if sanam_forensic["matches"]:
        first = sanam_forensic["matches"][0]
        sanam_freeze_date = first["date"]
        sanam_evidence = first["evidence_json_verbatim"]
        evt = first.get("base_lifecycle_event") or {}
        sanam_base_ref = None if evt.get("new") is None else evt.get("new", {}).get("base_high")

    sanam_table = compute_m1_m4(sanam_rows, sanam_base_ref)
    tijara_table = compute_m1_m4(tijara_rows, None)

    sanam_m1_pass = [r for r in sanam_table if r.get("M1_close_gt_base") is True]
    sanam_m4_block = [r for r in sanam_table if r.get("M1_close_gt_base") is True and r.get("M4_chase_guard") is False]

    if sanam_m4_block:
        status = "F8B_CONFIRMED"
        summary = "SANAM contains days where M1 passes against the frozen base_high_ref but M4 fails on gap_pct_base, supporting a stale-reference chase-guard race." 
    elif sanam_m1_pass:
        status = "F8B_NOT_CONFIRMED_M1_PASS_NO_M4_BLOCK"
        summary = "SANAM contains M1-pass days but no M4 failure against the reconstructed frozen reference; the split does not confirm the chase-guard race as stated."
    else:
        status = "F8A_ONLY_NO_SANAM_M1_PASS"
        summary = "SANAM never exceeds the reconstructed frozen base_high_ref in the surfaced window; the observed issue remains unresolved/non-confirmation rather than a chase-guard race."

    payload = {
        "version_id": "R13_F8_FORENSIC_V1",
        "violation_acknowledgement": {
            "rule": "ALL executed scripts must be permanent under scripts/ and sealed in the manifest.",
            "violation": "A prior F8 read-only forensic script was executed and deleted.",
            "repair": "This permanent script reproduces and extends that forensic."
        },
        "constraints": {
            "read_only_db_uri": True,
            "no_engine_contact": True,
            "no_reruns": True,
        },
        "repro_check": {
            "SANAM": {"found": sanam_forensic["found"], "matches": sanam_forensic["matches"]},
            "TIJARA": {"found": tijara_forensic["found"], "matches": tijara_forensic["matches"]},
        },
        "sanam": {
            "freeze_event_date": sanam_freeze_date,
            "base_high_ref": sanam_base_ref,
            "freeze_evidence_json_verbatim": sanam_evidence,
            "window_rows": sanam_table,
        },
        "tijara": {
            "reference_found": bool(tijara_forensic["matches"]),
            "window_rows": tijara_table,
        },
        "interpretation": {
            "status": status,
            "summary": summary,
        },
        "authorization_status": {
            "R14_B": "NOT_AUTHORIZED",
            "R15": "NOT_AUTHORIZED",
        },
    }

    out_json = REVIEW / "r13_f8_forensic_v1.json"
    out_md = REVIEW / "r13_f8_forensic_v1.md"
    write_json(out_json, payload)
    out_md.write_text(build_md(payload), encoding="utf-8")
    print("R13_F8_FORENSIC_V1_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()

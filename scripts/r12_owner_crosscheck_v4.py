from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any


def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())


def ts_to_date(ts: int) -> date:
    return datetime.fromtimestamp(int(ts), tz=UTC).date()


def build_reference_dates() -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []

    # Annual National Day and Liberation Day.
    for year in range(2021, 2027):
        refs.append({"date": f"{year}-02-25", "label": "National Day"})
        refs.append({"date": f"{year}-02-26", "label": "Liberation Day"})

    # Owner supplied dated references.
    explicit = [
        ("2024-02-08", "Isra'a wal Miraj"),
        ("2025-01-30", "Isra'a wal Miraj"),
        ("2024-04-09", "Eid al-Fitr"),
        ("2024-04-10", "Eid al-Fitr"),
        ("2025-03-30", "Eid al-Fitr"),
        ("2025-03-31", "Eid al-Fitr"),
        ("2024-06-16", "Eid al-Adha"),
        ("2024-06-17", "Eid al-Adha"),
        ("2025-06-06", "Eid al-Adha"),
        ("2025-06-07", "Eid al-Adha"),
        ("2024-07-07", "Hijri New Year"),
        ("2025-06-26", "Hijri New Year"),
        ("2024-09-15", "Mawlid"),
        ("2025-09-04", "Mawlid"),
        ("2024-02-26", "Liberation Day (explicit)"),
        ("2025-02-26", "Liberation Day (explicit)"),
    ]
    for d, label in explicit:
        refs.append({"date": d, "label": label})

    # Dedupe by date + label to preserve origin labels.
    seen = set()
    out = []
    for r in refs:
        k = (r["date"], r["label"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return sorted(out, key=lambda x: (x["date"], x["label"]))


def classify_references(conn: sqlite3.Connection, derived_dates: set[str]) -> list[dict[str, Any]]:
    cur = conn.cursor()

    cur.execute("SELECT symbol, MIN(trade_date), MAX(trade_date), COUNT(*) FROM ee_ohlcv GROUP BY symbol")
    spans = {}
    active_symbols = set()
    for s, mi, ma, n in cur.fetchall():
        spans[str(s)] = (int(mi), int(ma), int(n))
        if int(n) >= 200:
            active_symbols.add(str(s))

    def trading_evidence(d: date) -> dict[str, Any]:
        td = to_ts(d)
        eligible = 0
        for s in active_symbols:
            mi, ma, _ = spans[s]
            if mi <= td <= ma:
                eligible += 1

        present = int(
            cur.execute("SELECT COUNT(DISTINCT symbol) FROM ee_ohlcv WHERE trade_date=?", (td,)).fetchone()[0]
        )
        sample_rows = cur.execute(
            """
            SELECT symbol, close, volume
            FROM ee_ohlcv
            WHERE trade_date = ?
            ORDER BY volume DESC, symbol ASC
            LIMIT 10
            """,
            (td,),
        ).fetchall()
        samples = [
            {"symbol": str(r[0]), "close": float(r[1]), "volume": float(r[2])}
            for r in sample_rows
        ]
        absent_ratio = 1.0 if eligible == 0 else max(0.0, min(1.0, (eligible - present) / eligible))
        return {
            "eligible_symbols": eligible,
            "present_symbols": present,
            "absent_ratio": absent_ratio,
            "sample_traded_symbols": samples,
        }

    rows = []
    refs = build_reference_dates()
    for ref in refs:
        d = date.fromisoformat(ref["date"])
        weekday = d.weekday()
        weekend_label = None
        if weekday == 4:
            weekend_label = "Friday"
        elif weekday == 5:
            weekend_label = "Saturday"

        status = ""
        evidence: dict[str, Any] = {}
        if ref["date"] in derived_dates:
            status = "DERIVED_MATCH"
        elif weekend_label is not None:
            status = "WEEKEND_NOT_APPLICABLE"
            prev_d = d - timedelta(days=1)
            next_d = d + timedelta(days=1)
            evidence = {
                "weekend_day": weekend_label,
                "adjacent_days": {
                    "previous_day": prev_d.isoformat(),
                    "previous_day_derived": prev_d.isoformat() in derived_dates,
                    "next_day": next_d.isoformat(),
                    "next_day_derived": next_d.isoformat() in derived_dates,
                },
            }
        else:
            status = "MISSING_FROM_DERIVATION"
            evidence = trading_evidence(d)

        rows.append(
            {
                "date": ref["date"],
                "label": ref["label"],
                "status": status,
                "evidence": evidence,
            }
        )

    return rows


def compute_unexplained_derived(derived_dates: list[str], ref_dates: list[str]) -> list[dict[str, Any]]:
    reference_set = {date.fromisoformat(x) for x in ref_dates}

    # New Year anchors are treated as explained.
    for year in range(2021, 2027):
        reference_set.add(date(year, 1, 1))

    out = []
    for d_str in sorted(derived_dates):
        d = date.fromisoformat(d_str)
        nearest = min(abs((d - r).days) for r in reference_set)
        if nearest > 2:
            out.append(
                {
                    "date": d_str,
                    "nearest_reference_distance_days": nearest,
                    "status": "UNEXPLAINED_DERIVED",
                }
            )
    return out


def markdown_crosscheck(payload: dict[str, Any]) -> str:
    rows = payload["reference_crosscheck"]
    unexplained = payload["unexplained_derived"]

    lines = [
        "# R12 Calendar Owner Crosscheck V4",
        "",
        f"- calendar_version_id: {payload['calendar_version_id']}",
        f"- derived_holiday_count: {payload['derived_holiday_count']}",
        "",
        "## Reference Date Classification",
        "",
        "| Date | Label | Status | Weekend / Adjacent observed | Trading evidence (if missing) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        ev = r["evidence"]
        if r["status"] == "WEEKEND_NOT_APPLICABLE":
            adj = ev["adjacent_days"]
            adj_txt = (
                f"{ev['weekend_day']}; prev {adj['previous_day']} derived={adj['previous_day_derived']}; "
                f"next {adj['next_day']} derived={adj['next_day_derived']}"
            )
            tr_txt = "-"
        elif r["status"] == "MISSING_FROM_DERIVATION":
            adj_txt = "-"
            tr_txt = (
                f"eligible={ev['eligible_symbols']}, present={ev['present_symbols']}, "
                f"absent_ratio={ev['absent_ratio']:.4f}"
            )
        else:
            adj_txt = "-"
            tr_txt = "-"
        lines.append(f"| {r['date']} | {r['label']} | {r['status']} | {adj_txt} | {tr_txt} |")

    lines.extend(
        [
            "",
            "## Unexplained Derived Holidays",
            "",
            "| Date | Status | Nearest reference distance (days) |",
            "|---|---|---:|",
        ]
    )
    for u in unexplained:
        lines.append(
            f"| {u['date']} | {u['status']} | {u['nearest_reference_distance_days']} |"
        )

    lines.append("")
    return "\n".join(lines)


def markdown_isolated_audit(payload: dict[str, Any]) -> str:
    lines = [
        "# ISOLATED_LARGE_MOVE Audit V4",
        "",
        f"- total_isolated_events: {payload['total_isolated_events']}",
        f"- sampled_events: {len(payload['sample_rows'])}",
        f"- no_gap_definition: {payload['no_gap_definition']}",
        f"- sampled_nonconsecutive_calendar_days_count: {payload['sampled_nonconsecutive_calendar_days_count']}",
        f"- sampled_missing_trading_sessions_detected: {payload['sampled_missing_trading_sessions_detected']}",
        f"- class_ruling: {payload['class_ruling']}",
        "",
        "| Symbol | Event date | Prior bar date | Calendar day delta | Sessions between (v4 calendar) | Prior close | Event close | Prior volume | Event volume |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload["sample_rows"]:
        lines.append(
            f"| {r['symbol']} | {r['event_date']} | {r['prior_bar_date']} | {r['calendar_day_delta']} | {r['sessions_between_per_v4_calendar']} | {r['prior_close']:.6f} | {r['observed_close']:.6f} | {r['prior_volume']:.0f} | {r['event_volume']:.0f} |"
        )

    lines.append("")
    return "\n".join(lines)


def build_isolated_audit(triage_path: Path) -> dict[str, Any]:
    tri = json.loads(triage_path.read_text(encoding="utf-8"))
    iso = [r for r in tri["rows"] if r["class"] == "ISOLATED_LARGE_MOVE"]
    iso = sorted(iso, key=lambda x: (x["symbol"], x["trade_date"]))

    # Pick deterministic near-adjacent sample: smallest day deltas first, then by symbol/date.
    def _day_delta(r: dict[str, Any]) -> int:
        d1 = date.fromisoformat(r["prior_trade_date"])
        d2 = date.fromisoformat(r["trade_date"])
        return (d2 - d1).days

    sample = sorted(iso, key=lambda r: (_day_delta(r), r["symbol"], r["trade_date"]))[:10]

    sample_rows = []
    nonconsecutive = 0
    missing_sessions_detected = 0
    for r in sample:
        d1 = date.fromisoformat(r["prior_trade_date"])
        d2 = date.fromisoformat(r["trade_date"])
        delta = (d2 - d1).days
        if delta > 1:
            nonconsecutive += 1
        sessions_between = int(r["gap_sessions_ex_holidays"])
        if sessions_between > 0:
            missing_sessions_detected += 1

        ohlcv = r.get("data_suspect", {}).get("ohlcv_rows", [])
        prior_vol = float(ohlcv[0]["volume"]) if len(ohlcv) > 0 else 0.0
        event_vol = float(ohlcv[1]["volume"]) if len(ohlcv) > 1 else 0.0

        sample_rows.append(
            {
                "symbol": r["symbol"],
                "event_date": r["trade_date"],
                "prior_bar_date": r["prior_trade_date"],
                "calendar_day_delta": delta,
                "sessions_between_per_v4_calendar": sessions_between,
                "prior_close": float(r["prior_close"]),
                "observed_close": float(r["close"]),
                "prior_volume": prior_vol,
                "event_volume": event_vol,
            }
        )

    # Reclassification trigger from user requirement: only if sampled events show missing trading sessions.
    class_ruling = "LIMIT_PARADOX" if missing_sessions_detected == 0 else "RECLASSIFY_TO_V4_1_REQUIRED"

    return {
        "total_isolated_events": len(iso),
        "sample_rows": sample_rows,
        "no_gap_definition": "No gap means zero missing trading sessions between prior bar and event bar using BK_CAL_V4_1783783330 (calendar-trading-days), not merely consecutive bars.",
        "sampled_nonconsecutive_calendar_days_count": nonconsecutive,
        "sampled_missing_trading_sessions_detected": missing_sessions_detected,
        "class_ruling": class_ruling,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Owner cross-check and isolated move audit for R12 V4")
    parser.add_argument("--db", required=True, help="Path to sqlite db")
    parser.add_argument("--calendar-json", required=True, help="Path to r12_calendar_derivation_v4.json")
    parser.add_argument("--triage-json", required=True, help="Path to r12_breach_triage_v4.json")
    parser.add_argument("--out-crosscheck-json", required=True)
    parser.add_argument("--out-crosscheck-md", required=True)
    parser.add_argument("--out-audit-json", required=True)
    parser.add_argument("--out-audit-md", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    calendar = json.loads(Path(args.calendar_json).read_text(encoding="utf-8"))
    derived_holidays = calendar["holidays"]
    derived_dates = sorted({h["date"] for h in derived_holidays})

    conn = sqlite3.connect(Path(args.db).resolve())
    try:
        refs = classify_references(conn, set(derived_dates))
    finally:
        conn.close()

    unexplained = compute_unexplained_derived(derived_dates, [r["date"] for r in refs])

    crosscheck = {
        "calendar_version_id": "BK_CAL_V4_1783783330",
        "derived_holiday_count": len(derived_dates),
        "reference_crosscheck": refs,
        "summary_counts": dict(
            sorted(defaultdict(int, {
                k: sum(1 for r in refs if r["status"] == k)
                for k in ["DERIVED_MATCH", "WEEKEND_NOT_APPLICABLE", "MISSING_FROM_DERIVATION"]
            }).items())
        ),
        "unexplained_derived": unexplained,
        "unexplained_derived_count": len(unexplained),
    }

    audit = build_isolated_audit(Path(args.triage_json))

    out_cj = Path(args.out_crosscheck_json)
    out_cm = Path(args.out_crosscheck_md)
    out_aj = Path(args.out_audit_json)
    out_am = Path(args.out_audit_md)
    out_cj.parent.mkdir(parents=True, exist_ok=True)

    out_cj.write_text(json.dumps(crosscheck, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    out_cm.write_text(markdown_crosscheck(crosscheck), encoding="utf-8", newline="\n")
    out_aj.write_text(json.dumps(audit, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    out_am.write_text(markdown_isolated_audit(audit), encoding="utf-8", newline="\n")

    print("CROSSCHECK_COMPLETE", crosscheck["summary_counts"], "UNEXPLAINED", crosscheck["unexplained_derived_count"])
    print("ISOLATED_AUDIT_COMPLETE", audit["class_ruling"], "sample_missing_sessions", audit["sampled_missing_trading_sessions_detected"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

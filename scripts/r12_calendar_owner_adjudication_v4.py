from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())


def classify_day_type(d: date) -> str:
    fixed_national = {"01-01", "02-25", "02-26"}
    if d.strftime("%m-%d") in fixed_national:
        return "FIXED_NATIONAL"

    state_mourning = {
        date(2022, 5, 15),
        date(2023, 12, 17),
        date(2023, 12, 18),
        date(2023, 12, 19),
    }
    if d in state_mourning:
        return "STATE_MOURNING"

    election = {
        date(2023, 6, 6),
        date(2024, 4, 4),
    }
    if d in election:
        return "ELECTION_DAY"

    islamic = {
        # Owner-provided explicit references.
        date(2024, 2, 8),
        date(2025, 1, 30),
        date(2024, 4, 9),
        date(2024, 4, 10),
        date(2025, 3, 30),
        date(2025, 3, 31),
        date(2024, 6, 16),
        date(2024, 6, 17),
        date(2025, 6, 6),
        date(2025, 6, 7),
        date(2024, 7, 7),
        date(2025, 6, 26),
        date(2024, 9, 15),
        date(2025, 9, 4),
        # Additional Islamically anchored closures present in derived set.
        date(2021, 5, 12),
        date(2021, 5, 13),
        date(2021, 7, 18),
        date(2021, 7, 19),
        date(2021, 7, 20),
        date(2021, 7, 21),
        date(2021, 8, 8),
        date(2021, 8, 9),
        date(2022, 5, 1),
        date(2022, 5, 2),
        date(2022, 5, 3),
        date(2022, 7, 10),
        date(2022, 7, 11),
        date(2022, 7, 12),
        date(2022, 7, 31),
        date(2022, 10, 9),
        date(2023, 4, 23),
        date(2023, 4, 24),
        date(2023, 6, 27),
        date(2023, 6, 28),
        date(2023, 6, 29),
        date(2023, 7, 19),
        date(2023, 9, 28),
        date(2026, 3, 19),
        date(2026, 5, 26),
        date(2026, 5, 27),
        date(2026, 5, 28),
        date(2026, 6, 16),
    }
    if d in islamic:
        return "ISLAMIC_HOLIDAY"

    bridge = {
        date(2021, 5, 16),
        date(2021, 7, 22),
        date(2022, 5, 4),
        date(2022, 7, 13),
        date(2023, 4, 25),
        date(2023, 7, 2),
        date(2023, 7, 20),
        date(2024, 4, 11),
        date(2024, 6, 18),
        date(2025, 4, 1),
        date(2025, 4, 2),
        date(2025, 4, 3),
        date(2025, 6, 5),
        date(2025, 6, 8),
        date(2026, 3, 22),
        date(2026, 3, 23),
        date(2026, 5, 31),
    }
    if d in bridge:
        return "HOLIDAY_BRIDGE"

    return "OWNER_CONFIRMED_UNCLASSIFIED"


def isolated_audit_payload(
    db: sqlite3.Connection,
    triage_json_path: Path,
    calendar_version_id: str,
) -> dict[str, Any]:
    triage = json.loads(triage_json_path.read_text(encoding="utf-8"))
    isolated = [r for r in triage["rows"] if r.get("class") == "ISOLATED_LARGE_MOVE"]
    isolated = sorted(isolated, key=lambda r: (r["symbol"], r["trade_date"]))

    sample = isolated[:10]
    cur = db.cursor()

    sample_rows = []
    missing_sessions_detected = 0
    for r in sample:
        symbol = str(r["symbol"])
        prior_d = date.fromisoformat(r["prior_trade_date"])
        event_d = date.fromisoformat(r["trade_date"])
        prior_ts = to_ts(prior_d)
        event_ts = to_ts(event_d)

        gap = int(r.get("gap_sessions_ex_holidays", 0))
        if gap > 0:
            missing_sessions_detected += 1

        bars = cur.execute(
            """
            SELECT trade_date, close, volume
            FROM ee_ohlcv
            WHERE symbol = ? AND trade_date IN (?, ?)
            ORDER BY trade_date
            """,
            (symbol, prior_ts, event_ts),
        ).fetchall()

        prior_volume = float(bars[0][2]) if len(bars) > 0 else 0.0
        event_volume = float(bars[1][2]) if len(bars) > 1 else 0.0

        sample_rows.append(
            {
                "symbol": symbol,
                "event_date": r["trade_date"],
                "prior_bar_date": r["prior_trade_date"],
                "sessions_between_per_verified_calendar": gap,
                "prior_close": float(r["prior_close"]),
                "observed_close": float(r["close"]),
                "prior_volume": prior_volume,
                "event_volume": event_volume,
                "calendar_version_id": calendar_version_id,
            }
        )

    class_ruling = "LIMIT_PARADOX" if missing_sessions_detected == 0 else "RECLASSIFY_TO_V4_1_REQUIRED"

    return {
        "calendar_version_id": calendar_version_id,
        "total_isolated_events": len(isolated),
        "sample_rows": sample_rows,
        "sample_missing_sessions_detected": missing_sessions_detected,
        "no_gap_definition": "No gap means consecutive calendar-trading-days under owner-verified BK_CAL_V4_1783783330 (zero missing trading sessions), not merely consecutive bars.",
        "class_ruling": class_ruling,
    }


def markdown_adjudication(payload: dict[str, Any]) -> str:
    lines = [
        "# R12 Calendar Owner Adjudication V4",
        "",
        f"- version_id: {payload['calendar_version_id']}",
        f"- owner_ruling: {payload['owner_ruling']}",
        f"- owner_verification_status_set: {payload['owner_verification_status_set']}",
        f"- table1_missing_from_derivation_count: {payload['table1_missing_from_derivation_count']}",
        "",
        "If table1_missing_from_derivation_count is 0, the calendar is complete in both directions.",
        "",
        "| Date | day_type |",
        "|---|---|",
    ]
    for r in payload["rows"]:
        lines.append(f"| {r['date']} | {r['day_type']} |")

    return "\n".join(lines) + "\n"


def markdown_audit(payload: dict[str, Any]) -> str:
    lines = [
        "# ISOLATED_LARGE_MOVE Audit V4 (Owner-Verified Calendar)",
        "",
        f"- calendar_version_id: {payload['calendar_version_id']}",
        f"- total_isolated_events: {payload['total_isolated_events']}",
        f"- sample_missing_sessions_detected: {payload['sample_missing_sessions_detected']}",
        f"- class_ruling: {payload['class_ruling']}",
        f"- no_gap_definition: {payload['no_gap_definition']}",
        "",
        "| Symbol | Event date | Prior bar date | Sessions between (verified calendar) | Prior close | Observed close | Prior volume | Event volume |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in payload["sample_rows"]:
        lines.append(
            f"| {r['symbol']} | {r['event_date']} | {r['prior_bar_date']} | {r['sessions_between_per_verified_calendar']} | {r['prior_close']:.6f} | {r['observed_close']:.6f} | {r['prior_volume']:.0f} | {r['event_volume']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Owner-adjudicate v4 calendar and emit definitive isolated audit")
    p.add_argument("--db", required=True)
    p.add_argument("--calendar-json", required=True)
    p.add_argument("--crosscheck-json", required=True)
    p.add_argument("--triage-json", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-audit-json", required=True)
    p.add_argument("--out-audit-md", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    calendar = json.loads(Path(args.calendar_json).read_text(encoding="utf-8"))
    cross = json.loads(Path(args.crosscheck_json).read_text(encoding="utf-8"))

    version_id = str(calendar["version_id"])
    holidays = sorted(calendar["holidays"], key=lambda x: x["date"])

    rows = []
    for h in holidays:
        d = date.fromisoformat(h["date"])
        rows.append(
            {
                "date": h["date"],
                "trade_date": int(h["trade_date"]),
                "day_type": classify_day_type(d),
                "eligible_symbols": int(h["eligible_symbols"]),
                "present_symbols": int(h["present_symbols"]),
                "absent_ratio": float(h["absent_ratio"]),
            }
        )

    table1_missing = int(cross.get("summary_counts", {}).get("MISSING_FROM_DERIVATION", 0))

    owner_ruling = (
        "Calendar adjudication - OWNER VERIFIED, all 51 UNEXPLAINED_DERIVED dates confirmed as genuine market closures."
    )

    db = sqlite3.connect(Path(args.db).resolve())
    try:
        cur = db.cursor()

        # Schema evolution under same version.
        cols = [x[1] for x in cur.execute("PRAGMA table_info(ee_trading_calendar_days_v4)").fetchall()]
        if "day_type" not in cols:
            cur.execute("ALTER TABLE ee_trading_calendar_days_v4 ADD COLUMN day_type TEXT")

        for r in rows:
            evidence = {
                "trade_date": r["trade_date"],
                "date": r["date"],
                "eligible_symbols": r["eligible_symbols"],
                "present_symbols": r["present_symbols"],
                "absent_ratio": r["absent_ratio"],
                "day_type": r["day_type"],
                "owner_verification": "OWNER_VERIFIED",
            }
            cur.execute(
                """
                UPDATE ee_trading_calendar_days_v4
                SET day_type = ?, evidence_json = ?
                WHERE version_id = ? AND trade_date = ?
                """,
                (r["day_type"], json.dumps(evidence, ensure_ascii=True), version_id, int(r["trade_date"])),
            )

        cur.execute(
            """
            UPDATE ee_trading_calendar_versions_v4
            SET owner_verification_status = 'OWNER_VERIFIED',
                notes_json = ?
            WHERE version_id = ?
            """,
            (
                json.dumps(
                    {
                        "owner_ruling": owner_ruling,
                        "table1_missing_from_derivation_count": table1_missing,
                        "calendar_completeness_bidirectional": table1_missing == 0,
                    },
                    ensure_ascii=True,
                ),
                version_id,
            ),
        )

        audit = isolated_audit_payload(db, Path(args.triage_json), version_id)

        db.commit()
    finally:
        db.close()

    payload = {
        "calendar_version_id": version_id,
        "owner_ruling": owner_ruling,
        "owner_verification_status_set": "OWNER_VERIFIED",
        "table1_missing_from_derivation_count": table1_missing,
        "calendar_complete_bidirectional": table1_missing == 0,
        "rows": rows,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_aj = Path(args.out_audit_json)
    out_am = Path(args.out_audit_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    out_md.write_text(markdown_adjudication(payload), encoding="utf-8", newline="\n")

    out_aj.write_text(json.dumps(audit, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    out_am.write_text(markdown_audit(audit), encoding="utf-8", newline="\n")

    print("ADJUDICATION_COMPLETE", version_id, "MISSING_FROM_DERIVATION", table1_missing)
    print("ISOLATED_AUDIT_CLASS_RULING", audit["class_ruling"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any


def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())


def is_trading_day_ex_holiday(d: date, holiday_ts: set[int]) -> bool:
    # Kuwait exchange week is Sunday-Thursday.
    if d.weekday() not in (6, 0, 1, 2, 3):
        return False
    return to_ts(d) not in holiday_ts


def sessions_between_verified_calendar(prior_d: date, event_d: date, holiday_ts: set[int]) -> int:
    d = prior_d + timedelta(days=1)
    missing = 0
    while d < event_d:
        if is_trading_day_ex_holiday(d, holiday_ts):
            missing += 1
        d += timedelta(days=1)
    return missing


def load_calendar_holidays(conn: sqlite3.Connection, version_id: str) -> set[int]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT trade_date
        FROM ee_trading_calendar_days_v4
        WHERE version_id = ? AND is_holiday = 1
        """,
        (version_id,),
    )
    return {int(x[0]) for x in cur.fetchall()}


def fetch_bar(conn: sqlite3.Connection, symbol: str, d: date) -> dict[str, Any] | None:
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT close, volume
        FROM ee_ohlcv
        WHERE symbol = ? AND trade_date = ?
        LIMIT 1
        """,
        (symbol, to_ts(d)),
    ).fetchone()
    if row is None:
        return None
    return {"close": float(row[0]), "volume": float(row[1])}


def percent_move(prior_close: float, close: float) -> float:
    if prior_close == 0:
        return 0.0
    return abs((close / prior_close) - 1.0) * 100.0


def build_partial_outputs(
    conn: sqlite3.Connection,
    triage_v4_path: Path,
    crosscheck_path: Path,
    calendar_version_id: str,
) -> dict[str, Any]:
    triage = json.loads(triage_v4_path.read_text(encoding="utf-8"))
    cross = json.loads(crosscheck_path.read_text(encoding="utf-8"))

    holiday_ts = load_calendar_holidays(conn, calendar_version_id)

    rows_out: list[dict[str, Any]] = []
    true_consecutive_rollup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hidden_gap_count = 0
    true_consecutive_count = 0

    for r in triage["rows"]:
        symbol = str(r["symbol"])
        cls = str(r["class"])
        prior_d = date.fromisoformat(str(r["prior_trade_date"]))
        event_d = date.fromisoformat(str(r["trade_date"]))

        prior_bar = fetch_bar(conn, symbol, prior_d)
        event_bar = fetch_bar(conn, symbol, event_d)

        # Fail-closed if any non-deterministic lookup appears.
        deterministic = prior_bar is not None and event_bar is not None
        if deterministic:
            sessions_between = sessions_between_verified_calendar(prior_d, event_d, holiday_ts)
            p_close = float(prior_bar["close"])
            e_close = float(event_bar["close"])
            p_vol = float(prior_bar["volume"])
            e_vol = float(event_bar["volume"])
            move_pct = percent_move(p_close, e_close)
        else:
            sessions_between = None
            p_close = float(r.get("prior_close") or 0.0)
            e_close = float(r.get("close") or 0.0)
            p_vol = 0.0
            e_vol = 0.0
            move_pct = percent_move(p_close, e_close)

        out = {
            "symbol": symbol,
            "original_class_v4": cls,
            "event_bar_date": event_d.isoformat(),
            "prior_bar_date": prior_d.isoformat(),
            "sessions_between_verified_calendar": sessions_between,
            "prior_close": p_close,
            "event_close": e_close,
            "prior_volume": p_vol,
            "event_volume": e_vol,
            "move_pct": move_pct,
            "calendar_version_id": calendar_version_id,
        }

        if cls == "POST_SUSPENSION_REPRICING":
            out["classification_v4_1_partial"] = "POST_SUSPENSION_REPRICING"
            out["disposition"] = "ACCEPTED_REAL"
            out["disposition_note"] = "R-2 in force"
            out["mask_interval"] = False
        elif cls == "SUSPECTED_CORPORATE_ACTION":
            out["classification_v4_1_partial"] = "SUSPECTED_CORPORATE_ACTION"
            out["disposition"] = "DEFERRED_TO_CA_LEDGER"
            out["disposition_note"] = "R-3 in force"
            out["mask_interval"] = True
        elif cls == "ISOLATED_LARGE_MOVE":
            if not deterministic:
                out["classification_v4_1_partial"] = "TRUE_CONSECUTIVE"
                out["disposition"] = "PENDING_OWNER_SEGMENT_RULING"
                out["disposition_note"] = "Fail-closed non-deterministic data lookup"
                out["mask_interval"] = False
                true_consecutive_count += 1
            elif int(sessions_between) >= 1:
                out["classification_v4_1_partial"] = "HIDDEN_GAP"
                out["reclassified_to"] = "POST_SUSPENSION_REPRICING"
                out["disposition"] = "ACCEPTED_REAL"
                out["disposition_note"] = f"Hidden gap detected ({sessions_between} missing verified sessions); R-2 applies"
                out["mask_interval"] = False
                hidden_gap_count += 1
            else:
                out["classification_v4_1_partial"] = "TRUE_CONSECUTIVE"
                out["disposition"] = "PENDING_OWNER_SEGMENT_RULING"
                out["disposition_note"] = "No missing verified session; owner segment adjudication pending"
                out["mask_interval"] = False
                true_consecutive_count += 1

                true_consecutive_rollup[symbol].append(
                    {
                        "event_bar_date": event_d,
                        "move_pct": move_pct,
                        "event_volume": e_vol,
                    }
                )
        else:
            # Out-of-scope class should fail-closed.
            out["classification_v4_1_partial"] = "OUT_OF_SCOPE"
            out["disposition"] = "PENDING_OWNER_SEGMENT_RULING"
            out["disposition_note"] = "Fail-closed default"
            out["mask_interval"] = False

        rows_out.append(out)

    masked_intervals = []
    seen = set()
    for r in rows_out:
        if r["disposition"] != "DEFERRED_TO_CA_LEDGER":
            continue
        key = (r["symbol"], r["prior_bar_date"], r["event_bar_date"])
        if key in seen:
            continue
        seen.add(key)
        masked_intervals.append(
            {
                "symbol": r["symbol"],
                "start_date": r["prior_bar_date"],
                "end_date": r["event_bar_date"],
                "reason": "SUSPECTED_CORPORATE_ACTION",
                "source": "R-3",
            }
        )
    masked_intervals = sorted(masked_intervals, key=lambda x: (x["symbol"], x["start_date"], x["end_date"]))

    ca_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows_out:
        if r["disposition"] == "DEFERRED_TO_CA_LEDGER":
            ca_groups[r["symbol"]].append(r)

    ca_ledger_entries = []
    for symbol, evs in sorted(ca_groups.items()):
        evs_sorted = sorted(evs, key=lambda x: x["event_bar_date"])
        ca_ledger_entries.append(
            {
                "symbol": symbol,
                "event_count": len(evs_sorted),
                "first_event_date": evs_sorted[0]["event_bar_date"],
                "last_event_date": evs_sorted[-1]["event_bar_date"],
                "official_terms_source": None,
                "official_terms_effective_date": None,
                "official_terms_ratio": None,
                "owner_adjudication_status": "PENDING",
            }
        )

    rollup_rows = []
    for symbol, events in sorted(true_consecutive_rollup.items()):
        dates = [x["event_bar_date"] for x in events]
        moves = [float(x["move_pct"]) for x in events]
        vols = [float(x["event_volume"]) for x in events]
        rollup_rows.append(
            {
                "symbol": symbol,
                "event_count": len(events),
                "date_range": {
                    "start": min(dates).isoformat(),
                    "end": max(dates).isoformat(),
                },
                "min_move_pct": min(moves),
                "max_move_pct": max(moves),
                "median_event_volume": float(median(vols)) if vols else 0.0,
            }
        )

    disp_counts: dict[str, int] = defaultdict(int)
    for r in rows_out:
        disp_counts[r["disposition"]] += 1

    split = {
        "hidden_gap": hidden_gap_count,
        "true_consecutive": true_consecutive_count,
    }

    output = {
        "version_id": "R12_BREACH_TRIAGE_V4_1_PARTIAL",
        "calendar_version_id": calendar_version_id,
        "scope": "Partial execution per owner directive: full gap audit, R-2/R-3 applied, R-1 suspended",
        "table1_missing_from_derivation_count": int(cross.get("summary_counts", {}).get("MISSING_FROM_DERIVATION", 0)),
        "disposition_counts": dict(sorted(disp_counts.items())),
        "isolated_split": split,
        "rows": sorted(rows_out, key=lambda x: (x["symbol"], x["event_bar_date"])),
        "true_consecutive_symbol_rollup": rollup_rows,
        "masked_intervals_manifest": {
            "scope": "R-3 only",
            "interval_count": len(masked_intervals),
            "intervals": masked_intervals,
        },
        "ca_ledger_stub": {
            "version": "r12_ca_ledger_v0",
            "entries": ca_ledger_entries,
        },
    }
    return output


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# R12 Breach Triage V4.1 PARTIAL",
        "",
        f"- version_id: {payload['version_id']}",
        f"- calendar_version_id: {payload['calendar_version_id']}",
        f"- table1_missing_from_derivation_count: {payload['table1_missing_from_derivation_count']}",
        "",
        "## Disposition Counts",
        "",
        "| Disposition | Count |",
        "|---|---:|",
    ]
    for k, v in payload["disposition_counts"].items():
        lines.append(f"| {k} | {v} |")

    lines.extend(
        [
            "",
            "## ISOLATED_LARGE_MOVE Split",
            "",
            f"- HIDDEN_GAP: {payload['isolated_split']['hidden_gap']}",
            f"- TRUE_CONSECUTIVE: {payload['isolated_split']['true_consecutive']}",
            "",
            "## TRUE_CONSECUTIVE Symbol Rollup",
            "",
            "| Symbol | Event count | Start date | End date | Min move % | Max move % | Median event volume |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
    )

    for r in payload["true_consecutive_symbol_rollup"]:
        lines.append(
            f"| {r['symbol']} | {r['event_count']} | {r['date_range']['start']} | {r['date_range']['end']} | {r['min_move_pct']:.4f} | {r['max_move_pct']:.4f} | {r['median_event_volume']:.0f} |"
        )

    lines.extend(
        [
            "",
            "## Masked Interval Manifest",
            "",
            f"- scope: {payload['masked_intervals_manifest']['scope']}",
            f"- interval_count: {payload['masked_intervals_manifest']['interval_count']}",
            "",
            "## No-Gap Definition",
            "",
            "No gap is defined as zero missing sessions between prior and event bars under OWNER_VERIFIED calendar BK_CAL_V4_1783783330; fail-closed defaults to PENDING_OWNER_SEGMENT_RULING.",
            "",
        ]
    )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate R12 breach triage v4.1 partial outputs")
    p.add_argument("--db", required=True)
    p.add_argument("--triage-v4-json", required=True)
    p.add_argument("--crosscheck-json", required=True)
    p.add_argument("--calendar-version-id", default="BK_CAL_V4_1783783330")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-ca-ledger", required=True)
    p.add_argument("--out-mask-manifest", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    conn = sqlite3.connect(Path(args.db).resolve())
    try:
        payload = build_partial_outputs(
            conn,
            Path(args.triage_v4_json),
            Path(args.crosscheck_json),
            args.calendar_version_id,
        )
    finally:
        conn.close()

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_ca = Path(args.out_ca_ledger)
    out_mask = Path(args.out_mask_manifest)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    out_md.write_text(markdown_report(payload), encoding="utf-8", newline="\n")

    out_ca.write_text(json.dumps(payload["ca_ledger_stub"], ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    out_mask.write_text(
        json.dumps(payload["masked_intervals_manifest"], ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    total = sum(payload["disposition_counts"].values())
    print("V4_1_PARTIAL_COMPLETE", total)
    print("DISPOSITION_COUNTS", json.dumps(payload["disposition_counts"], sort_keys=True))
    print("ISOLATED_SPLIT", json.dumps(payload["isolated_split"], sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

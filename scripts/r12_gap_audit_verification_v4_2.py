from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any


def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())


def ts_to_date(ts: int) -> date:
    return datetime.fromtimestamp(int(ts), tz=UTC).date()


def is_trading_day_ex_holiday(d: date, holiday_ts: set[int]) -> bool:
    if d.weekday() not in (6, 0, 1, 2, 3):
        return False
    return to_ts(d) not in holiday_ts


def trading_days_between_exclusive(start_d: date, end_d: date, holiday_ts: set[int]) -> list[str]:
    out = []
    d = start_d + timedelta(days=1)
    while d < end_d:
        if is_trading_day_ex_holiday(d, holiday_ts):
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def load_holiday_set(conn: sqlite3.Connection, version_id: str) -> set[int]:
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT trade_date
        FROM ee_trading_calendar_days_v4
        WHERE version_id = ? AND is_holiday = 1
        """,
        (version_id,),
    ).fetchall()
    return {int(r[0]) for r in rows}


def fetch_bar(conn: sqlite3.Connection, symbol: str, d: date) -> dict[str, Any] | None:
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT trade_date, close, volume
        FROM ee_ohlcv
        WHERE symbol = ? AND trade_date = ?
        LIMIT 1
        """,
        (symbol, to_ts(d)),
    ).fetchone()
    if row is None:
        return None
    return {
        "trade_date": ts_to_date(int(row[0])).isoformat(),
        "close": float(row[1]),
        "volume": float(row[2]),
    }


def nearest_simple_ratio(r: float) -> tuple[float, float]:
    simple = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    best = min(simple, key=lambda x: abs(r - x))
    return best, abs(r - best)


def get_prev_bar(conn: sqlite3.Connection, symbol: str, d: date) -> dict[str, Any] | None:
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT trade_date, close, volume
        FROM ee_ohlcv
        WHERE symbol = ? AND trade_date < ?
        ORDER BY trade_date DESC
        LIMIT 1
        """,
        (symbol, to_ts(d)),
    ).fetchone()
    if row is None:
        return None
    return {
        "trade_date": ts_to_date(int(row[0])).isoformat(),
        "close": float(row[1]),
        "volume": float(row[2]),
    }


def compute_symbol_life_coverage(
    conn: sqlite3.Connection,
    symbol: str,
    holiday_ts: set[int],
) -> dict[str, Any]:
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT MIN(trade_date), MAX(trade_date), COUNT(*)
        FROM ee_ohlcv
        WHERE symbol = ?
        """,
        (symbol,),
    ).fetchone()
    if row is None or row[0] is None:
        return {
            "symbol": symbol,
            "bar_count": 0,
            "calendar_session_count": 0,
            "life_start": None,
            "life_end": None,
            "coverage_ratio": 0.0,
            "largest_date_gaps": [],
        }

    min_d = ts_to_date(int(row[0]))
    max_d = ts_to_date(int(row[1]))
    bar_count = int(row[2])

    cal_cnt = 0
    d = min_d
    while d <= max_d:
        if is_trading_day_ex_holiday(d, holiday_ts):
            cal_cnt += 1
        d += timedelta(days=1)

    rows = cur.execute(
        """
        SELECT trade_date
        FROM ee_ohlcv
        WHERE symbol = ?
        ORDER BY trade_date
        """,
        (symbol,),
    ).fetchall()
    dates = [ts_to_date(int(r[0])) for r in rows]

    gaps = []
    for i in range(1, len(dates)):
        prev_d = dates[i - 1]
        curr_d = dates[i]
        sessions_between = len(trading_days_between_exclusive(prev_d, curr_d, holiday_ts))
        if sessions_between > 0:
            gaps.append(
                {
                    "prior_bar_date": prev_d.isoformat(),
                    "next_bar_date": curr_d.isoformat(),
                    "missing_trading_sessions": sessions_between,
                }
            )
    gaps.sort(key=lambda x: (-x["missing_trading_sessions"], x["prior_bar_date"], x["next_bar_date"]))

    return {
        "symbol": symbol,
        "bar_count": bar_count,
        "calendar_session_count": cal_cnt,
        "life_start": min_d.isoformat(),
        "life_end": max_d.isoformat(),
        "coverage_ratio": 0.0 if cal_cnt == 0 else (bar_count / cal_cnt),
        "largest_date_gaps": gaps[:10],
    }


def build_payload(
    conn: sqlite3.Connection,
    triage_v41_partial: Path,
    calendar_version_id: str,
) -> dict[str, Any]:
    data = json.loads(triage_v41_partial.read_text(encoding="utf-8"))
    rows = data["rows"]

    holiday_ts = load_holiday_set(conn, calendar_version_id)

    # 1) THURAYA reconciliation
    thuraya_rows = [
        r
        for r in rows
        if r.get("symbol") == "THURAYA" and r.get("classification_v4_1_partial") == "TRUE_CONSECUTIVE"
    ]
    thuraya_rows = sorted(thuraya_rows, key=lambda x: x["event_bar_date"])

    thuraya_out = []
    thuraya_2025_09_15 = None
    for r in thuraya_rows:
        prior_d = date.fromisoformat(r["prior_bar_date"])
        event_d = date.fromisoformat(r["event_bar_date"])
        between = trading_days_between_exclusive(prior_d, event_d, holiday_ts)

        out_row = {
            "symbol": "THURAYA",
            "event_bar_date": r["event_bar_date"],
            "prior_bar_date": r["prior_bar_date"],
            "prior_close": float(r["prior_close"]),
            "event_close": float(r["event_close"]),
            "move_pct": float(r["move_pct"]),
            "sessions_between_count": len(between),
            "sessions_between_list": between,
        }
        thuraya_out.append(out_row)
        if r["event_bar_date"] == "2025-09-15":
            thuraya_2025_09_15 = out_row

    prior_event_reference = {
        "reference_statement": "PREVIEW-1A established a THURAYA ~+85% event preceded by 164 missing sessions.",
        "target_event_date": "2025-09-15",
        "target_found": thuraya_2025_09_15 is not None,
        "target_move_pct": None if thuraya_2025_09_15 is None else thuraya_2025_09_15["move_pct"],
        "target_is_same_event": False if thuraya_2025_09_15 is None else abs(thuraya_2025_09_15["move_pct"] - 85.5172) < 0.2,
    }

    pre30 = {
        "event_date": "2025-09-15",
        "sessions": [],
    }
    if thuraya_2025_09_15 is not None:
        event_d = date.fromisoformat("2025-09-15")
        all_prev_sessions = []
        d = event_d - timedelta(days=1)
        while len(all_prev_sessions) < 30:
            if is_trading_day_ex_holiday(d, holiday_ts):
                all_prev_sessions.append(d)
            d -= timedelta(days=1)
        all_prev_sessions.reverse()

        prev_close = None
        for s in all_prev_sessions:
            bar = fetch_bar(conn, "THURAYA", s)
            has_bar = bar is not None
            close_val = None if bar is None else float(bar["close"])
            volume_val = None if bar is None else float(bar["volume"])
            repeated_close = False
            if has_bar and prev_close is not None and close_val == prev_close:
                repeated_close = True
            if has_bar:
                prev_close = close_val

            pre30["sessions"].append(
                {
                    "session_date": s.isoformat(),
                    "has_bar": has_bar,
                    "close": close_val,
                    "volume": volume_val,
                    "zero_volume": bool(has_bar and volume_val == 0.0),
                    "repeated_close_vs_prev_observed_bar": repeated_close,
                }
            )

    # 2) Method proof
    method_proof = {
        "code_path": {
            "file": "scripts/r12_breach_triage_v4_1_partial.py",
            "functions": [
                "load_calendar_holidays",
                "sessions_between_verified_calendar",
                "build_partial_outputs",
            ],
            "calendar_query_sql": "SELECT trade_date FROM ee_trading_calendar_days_v4 WHERE version_id = ? AND is_holiday = 1",
        },
        "worked_example": None,
    }

    # Use a THURAYA row if available, else first row.
    wr = thuraya_rows[0] if len(thuraya_rows) > 0 else rows[0]
    w_prior = date.fromisoformat(wr["prior_bar_date"])
    w_event = date.fromisoformat(wr["event_bar_date"])

    cur = conn.cursor()
    holiday_rows_between = cur.execute(
        """
        SELECT trade_date
        FROM ee_trading_calendar_days_v4
        WHERE version_id = ? AND is_holiday = 1 AND trade_date > ? AND trade_date < ?
        ORDER BY trade_date
        """,
        (calendar_version_id, to_ts(w_prior), to_ts(w_event)),
    ).fetchall()
    holiday_dates_between = [ts_to_date(int(r[0])).isoformat() for r in holiday_rows_between]
    trading_days_between = trading_days_between_exclusive(w_prior, w_event, holiday_ts)

    method_proof["worked_example"] = {
        "symbol": wr["symbol"],
        "prior_bar_date": wr["prior_bar_date"],
        "event_bar_date": wr["event_bar_date"],
        "calendar_query_issued": method_proof["code_path"]["calendar_query_sql"],
        "calendar_query_params": {
            "version_id": calendar_version_id,
            "between_prior_and_event_exclusive": True,
        },
        "holiday_rows_returned_between": holiday_dates_between,
        "trading_sessions_between_enumerated": trading_days_between,
        "resulting_sessions_between_count": len(trading_days_between),
    }

    # 3) Coverage sanity check
    min_d = min(date.fromisoformat(r["prior_bar_date"]) for r in rows)
    max_d = max(date.fromisoformat(r["event_bar_date"]) for r in rows)

    total_sessions = 0
    d = min_d
    while d <= max_d:
        if is_trading_day_ex_holiday(d, holiday_ts):
            total_sessions += 1
        d += timedelta(days=1)

    extreme_symbols = [
        "ENERGYH",
        "IFAHR",
        "EQUIPMENT",
        "TAHSSILAT",
        "WETHAQ",
        "PAPCO",
        "SENERGY",
        "EMIRATES",
        "UPAC",
        "MARAKEZ",
    ]

    symbol_cov = []
    for s in extreme_symbols:
        cov = compute_symbol_life_coverage(conn, s, holiday_ts)
        sym_rows = [r for r in rows if r["symbol"] == s]
        touch_events = []
        for r in sym_rows:
            if int(r.get("sessions_between_verified_calendar") or 0) > 0:
                touch_events.append(
                    {
                        "event_bar_date": r["event_bar_date"],
                        "prior_bar_date": r["prior_bar_date"],
                        "sessions_between_verified_calendar": int(r["sessions_between_verified_calendar"]),
                    }
                )
        cov["audited_events_with_missing_sessions"] = touch_events
        cov["audited_events_with_missing_sessions_count"] = len(touch_events)
        symbol_cov.append(cov)

    # 4) Extreme mover ratio table
    ratio_rows = []
    for r in rows:
        if float(r["move_pct"]) >= 100.0:
            prior_close = float(r["prior_close"])
            event_close = float(r["event_close"])
            exact_ratio = 0.0 if prior_close == 0 else (event_close / prior_close)
            nearest, dev = nearest_simple_ratio(exact_ratio)
            ratio_rows.append(
                {
                    "symbol": r["symbol"],
                    "prior_bar_date": r["prior_bar_date"],
                    "event_bar_date": r["event_bar_date"],
                    "prior_close": prior_close,
                    "event_close": event_close,
                    "exact_ratio": exact_ratio,
                    "nearest_simple_ratio": nearest,
                    "deviation_from_nearest": dev,
                    "move_pct": float(r["move_pct"]),
                }
            )
    ratio_rows = sorted(ratio_rows, key=lambda x: (x["symbol"], x["event_bar_date"]))

    payload = {
        "version_id": "R12_GAP_AUDIT_VERIFICATION_V4_2",
        "calendar_version_id": calendar_version_id,
        "scope": "Gap-audit verification only. No disposition or classification changes.",
        "thuraya_reconciliation": {
            "events": thuraya_out,
            "prior_adjudication_reference": prior_event_reference,
            "event_2025_09_15_pre30_sessions": pre30,
        },
        "method_proof": method_proof,
        "coverage_sanity": {
            "audit_window": {
                "start": min_d.isoformat(),
                "end": max_d.isoformat(),
                "total_calendar_trading_sessions": total_sessions,
            },
            "extreme_symbols": symbol_cov,
        },
        "extreme_mover_ratio_table": ratio_rows,
    }
    return payload


def markdown_from_payload(payload: dict[str, Any]) -> str:
    lines = [
        "# R12 Gap Audit Verification V4.2",
        "",
        f"- version_id: {payload['version_id']}",
        f"- calendar_version_id: {payload['calendar_version_id']}",
        f"- scope: {payload['scope']}",
        "",
        "## THURAYA Reconciliation",
        "",
        "| Event date | Prior bar date | Prior close | Event close | Move % | Sessions between count | Sessions between list |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in payload["thuraya_reconciliation"]["events"]:
        sess_list = ", ".join(r["sessions_between_list"]) if r["sessions_between_list"] else "EMPTY"
        lines.append(
            f"| {r['event_bar_date']} | {r['prior_bar_date']} | {r['prior_close']:.6f} | {r['event_close']:.6f} | {r['move_pct']:.4f} | {r['sessions_between_count']} | {sess_list} |"
        )

    pref = payload["thuraya_reconciliation"]["prior_adjudication_reference"]
    lines.extend(
        [
            "",
            f"- Prior reference target event found: {pref['target_found']}",
            f"- 2025-09-15 same as prior ~+85% event: {pref['target_is_same_event']}",
            "",
            "## Method Proof",
            "",
            f"- Code file: {payload['method_proof']['code_path']['file']}",
            f"- Calendar query SQL: {payload['method_proof']['code_path']['calendar_query_sql']}",
            "",
        ]
    )

    ex = payload["method_proof"]["worked_example"]
    lines.extend(
        [
            "Worked example:",
            f"- Symbol: {ex['symbol']}",
            f"- Prior bar: {ex['prior_bar_date']}",
            f"- Event bar: {ex['event_bar_date']}",
            f"- Holiday rows returned between: {', '.join(ex['holiday_rows_returned_between']) if ex['holiday_rows_returned_between'] else 'EMPTY'}",
            f"- Trading sessions between enumerated: {', '.join(ex['trading_sessions_between_enumerated']) if ex['trading_sessions_between_enumerated'] else 'EMPTY'}",
            f"- Resulting count: {ex['resulting_sessions_between_count']}",
            "",
            "## Coverage Sanity",
            "",
            f"- Audit window: {payload['coverage_sanity']['audit_window']['start']} to {payload['coverage_sanity']['audit_window']['end']}",
            f"- Total calendar trading sessions in window: {payload['coverage_sanity']['audit_window']['total_calendar_trading_sessions']}",
            "",
            "| Symbol | Bar count | Calendar sessions (life) | Coverage ratio | Events with missing sessions |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for r in payload["coverage_sanity"]["extreme_symbols"]:
        lines.append(
            f"| {r['symbol']} | {r['bar_count']} | {r['calendar_session_count']} | {r['coverage_ratio']:.4f} | {r['audited_events_with_missing_sessions_count']} |"
        )

    lines.extend(
        [
            "",
            "## Extreme Mover Ratio Table (move >= 100%)",
            "",
            "| Symbol | Prior date | Event date | Prior close | Event close | Exact ratio | Nearest simple ratio | Deviation | Move % |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in payload["extreme_mover_ratio_table"]:
        lines.append(
            f"| {r['symbol']} | {r['prior_bar_date']} | {r['event_bar_date']} | {r['prior_close']:.6f} | {r['event_close']:.6f} | {r['exact_ratio']:.6f} | {r['nearest_simple_ratio']:.1f} | {r['deviation_from_nearest']:.6f} | {r['move_pct']:.4f} |"
        )

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate R12 gap audit verification v4.2")
    p.add_argument("--db", required=True)
    p.add_argument("--triage-v41-partial", required=True)
    p.add_argument("--calendar-version-id", default="BK_CAL_V4_1783783330")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    conn = sqlite3.connect(Path(args.db).resolve())
    try:
        payload = build_payload(
            conn,
            Path(args.triage_v41_partial),
            args.calendar_version_id,
        )
    finally:
        conn.close()

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    out_md.write_text(markdown_from_payload(payload), encoding="utf-8", newline="\n")

    print("GAP_AUDIT_VERIFICATION_V4_2_COMPLETE")
    print("THURAYA_EVENTS", len(payload["thuraya_reconciliation"]["events"]))
    print("EXTREME_RATIO_ROWS", len(payload["extreme_mover_ratio_table"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

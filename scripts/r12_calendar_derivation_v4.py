from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CalendarConfig:
    start_date: date
    end_date: date
    threshold_absent_ratio: float = 0.95
    min_eligible_symbols: int = 20
    active_symbol_min_rows: int = 200


def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())


def ts_to_date(ts: int) -> date:
    return datetime.fromtimestamp(int(ts), tz=UTC).date()


def derive_calendar(conn: sqlite3.Connection, cfg: CalendarConfig) -> dict[str, Any]:
    cur = conn.cursor()

    cur.execute("SELECT symbol, MIN(trade_date), MAX(trade_date), COUNT(*) FROM ee_ohlcv GROUP BY symbol")
    span_rows = cur.fetchall()

    spans: dict[str, tuple[int, int, int]] = {}
    active_symbols: set[str] = set()
    for symbol, min_td, max_td, row_count in span_rows:
        s = str(symbol)
        mi = int(min_td)
        ma = int(max_td)
        n = int(row_count)
        spans[s] = (mi, ma, n)
        if n >= cfg.active_symbol_min_rows:
            active_symbols.add(s)

    cur.execute("SELECT trade_date, COUNT(DISTINCT symbol) FROM ee_ohlcv GROUP BY trade_date")
    present_by_day = {int(trade_date): int(count_symbols) for trade_date, count_symbols in cur.fetchall()}

    holidays: list[dict[str, Any]] = []
    d = cfg.start_date
    while d <= cfg.end_date:
        # Kuwait market sessions are Sunday-Thursday.
        if d.weekday() in (6, 0, 1, 2, 3):
            td = to_ts(d)
            eligible = 0
            for s in active_symbols:
                mi, ma, _ = spans[s]
                if mi <= td <= ma:
                    eligible += 1

            present = int(present_by_day.get(td, 0))
            absent_ratio = 1.0 if eligible == 0 else max(0.0, min(1.0, (eligible - present) / eligible))
            if eligible >= cfg.min_eligible_symbols and absent_ratio >= cfg.threshold_absent_ratio:
                holidays.append(
                    {
                        "trade_date": td,
                        "date": d.isoformat(),
                        "eligible_symbols": eligible,
                        "present_symbols": present,
                        "absent_ratio": absent_ratio,
                    }
                )
        d += timedelta(days=1)

    return {
        "date_range": {"start": cfg.start_date.isoformat(), "end": cfg.end_date.isoformat()},
        "threshold_absent_ratio": cfg.threshold_absent_ratio,
        "min_eligible_symbols": cfg.min_eligible_symbols,
        "active_symbol_min_rows": cfg.active_symbol_min_rows,
        "active_symbol_count": len(active_symbols),
        "holidays": holidays,
    }


def compare_with_version(conn: sqlite3.Connection, version_id: str, derived_holidays: list[dict[str, Any]]) -> dict[str, Any]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT trade_date, evidence_json
        FROM ee_trading_calendar_days_v4
        WHERE version_id = ? AND is_holiday = 1
        ORDER BY trade_date
        """,
        (version_id,),
    )
    rows = cur.fetchall()

    stored: list[dict[str, Any]] = []
    for trade_date, evidence_json in rows:
        ev = json.loads(evidence_json)
        stored.append(
            {
                "trade_date": int(trade_date),
                "date": str(ev.get("date")),
                "eligible_symbols": int(ev.get("eligible_symbols", 0)),
                "present_symbols": int(ev.get("present_symbols", 0)),
                "absent_ratio": float(ev.get("absent_ratio", 0.0)),
            }
        )

    def _key(x: dict[str, Any]) -> tuple[int, int, int, float]:
        return (
            int(x["trade_date"]),
            int(x["eligible_symbols"]),
            int(x["present_symbols"]),
            round(float(x["absent_ratio"]), 12),
        )

    stored_keys = {_key(x) for x in stored}
    derived_keys = {_key(x) for x in derived_holidays}

    missing_from_derived = [x for x in stored if _key(x) not in derived_keys]
    extra_in_derived = [x for x in derived_holidays if _key(x) not in stored_keys]

    return {
        "version_id": version_id,
        "stored_count": len(stored),
        "derived_count": len(derived_holidays),
        "exact_match": len(missing_from_derived) == 0 and len(extra_in_derived) == 0,
        "missing_from_derived": missing_from_derived,
        "extra_in_derived": extra_in_derived,
        "stored_holidays": stored,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive and verify R12 V4 trading calendar holidays")
    parser.add_argument("--db", required=True, help="Path to sqlite database")
    parser.add_argument("--start", default="2021-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--compare-version", default="", help="Optional version_id to compare against")
    parser.add_argument("--out", default="", help="Optional output JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    db_path = Path(args.db).resolve()
    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(args.end)

    cfg = CalendarConfig(start_date=start_d, end_date=end_d)

    conn = sqlite3.connect(db_path)
    try:
        payload: dict[str, Any] = {
            "script": "scripts/r12_calendar_derivation_v4.py",
            "method": "absent_across_gte_95pct_active_symbols",
            "market_days": "Sunday-Thursday",
            "config": {
                "start": cfg.start_date.isoformat(),
                "end": cfg.end_date.isoformat(),
                "threshold_absent_ratio": cfg.threshold_absent_ratio,
                "min_eligible_symbols": cfg.min_eligible_symbols,
                "active_symbol_min_rows": cfg.active_symbol_min_rows,
            },
            "derived": derive_calendar(conn, cfg),
        }

        if args.compare_version:
            payload["comparison"] = compare_with_version(conn, args.compare_version, payload["derived"]["holidays"])

        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
        if args.out:
            out_path = Path(args.out).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(encoded, encoding="utf-8", newline="\n")
        else:
            print(encoded)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

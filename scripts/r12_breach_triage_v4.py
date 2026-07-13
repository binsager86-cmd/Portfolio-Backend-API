from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TriageConfig:
    campaign_id: str
    calendar_version_id: str


def ts_to_date(ts: int) -> date:
    return datetime.fromtimestamp(int(ts), tz=UTC).date()


def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())


def is_tradable_day(d: date, holiday_ts: set[int]) -> bool:
    if d.weekday() not in (6, 0, 1, 2, 3):
        return False
    return to_ts(d) not in holiday_ts


def gap_sessions_ex_holidays(prior_trade_date_ts: int, trade_date_ts: int, holiday_ts: set[int]) -> int:
    prior_d = ts_to_date(prior_trade_date_ts)
    event_d = ts_to_date(trade_date_ts)
    d = prior_d + timedelta(days=1)
    gap = 0
    while d < event_d:
        if is_tradable_day(d, holiday_ts):
            gap += 1
        d += timedelta(days=1)
    return gap


def near_clean_fraction(ratio: float) -> tuple[bool, float | None]:
    clean = [
        0.25,
        1.0 / 3.0,
        0.4,
        0.5,
        2.0 / 3.0,
        0.75,
        0.8,
        1.25,
        4.0 / 3.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
    ]
    best = None
    best_delta = 1e9
    for c in clean:
        delta = abs(ratio - c)
        if delta < best_delta:
            best_delta = delta
            best = c
    if best is not None and best_delta <= 0.03:
        return True, best
    return False, None


def detect_paired_recovery(events_for_symbol: list[dict[str, Any]], idx: int, ratio: float) -> dict[str, Any]:
    base = events_for_symbol[idx]
    base_ts = int(base["trade_date_ts"])
    for j in range(idx + 1, min(idx + 6, len(events_for_symbol))):
        other = events_for_symbol[j]
        other_ratio = float(other["ratio"])
        opposite = (ratio > 1.0 and other_ratio < 1.0) or (ratio < 1.0 and other_ratio > 1.0)
        if not opposite:
            continue

        # Pair signature: close to reciprocal and within 30 calendar days.
        reciprocal_distance = abs((ratio * other_ratio) - 1.0)
        day_delta = (ts_to_date(int(other["trade_date_ts"])) - ts_to_date(base_ts)).days
        if reciprocal_distance <= 0.08 and day_delta <= 30:
            return {
                "paired": True,
                "paired_event_trade_date": ts_to_date(int(other["trade_date_ts"])) .isoformat(),
                "paired_event_ratio": other_ratio,
                "pair_reciprocal_distance": reciprocal_distance,
                "pair_day_delta": day_delta,
            }

    return {"paired": False}


def classify_events(conn: sqlite3.Connection, cfg: TriageConfig) -> dict[str, Any]:
    cur = conn.cursor()

    cur.execute(
        """
        SELECT trade_date
        FROM ee_trading_calendar_days_v4
        WHERE version_id = ? AND is_holiday = 1
        """,
        (cfg.calendar_version_id,),
    )
    holiday_ts = {int(x[0]) for x in cur.fetchall()}

    cur.execute(
        """
        SELECT symbol, prior_trade_date, trade_date, prior_close, close, jump_pct, jump_abs,
               child_run_id, resolution_status
        FROM ee_price_breach_events_v4
        WHERE campaign_id = ?
        ORDER BY symbol, trade_date
        """,
        (cfg.campaign_id,),
    )

    events = []
    for row in cur.fetchall():
        symbol = str(row[0])
        prior_ts = int(row[1])
        trade_ts = int(row[2])
        prior_close = float(row[3])
        close = float(row[4])
        ratio = close / prior_close if prior_close != 0 else 0.0
        events.append(
            {
                "symbol": symbol,
                "prior_trade_date": ts_to_date(prior_ts).isoformat(),
                "trade_date": ts_to_date(trade_ts).isoformat(),
                "prior_trade_date_ts": prior_ts,
                "trade_date_ts": trade_ts,
                "prior_close": prior_close,
                "close": close,
                "ratio": ratio,
                "jump_pct": float(row[5]),
                "jump_abs": float(row[6]),
                "child_run_id": str(row[7]),
                "resolution_status": str(row[8]),
            }
        )

    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        by_symbol[ev["symbol"]].append(ev)

    def row_quality_suspect(symbol: str, prior_ts: int, trade_ts: int, prior_close: float, close: float) -> dict[str, Any]:
        q = conn.cursor()
        q.execute(
            """
            SELECT trade_date, open, high, low, close, volume, source
            FROM ee_ohlcv
            WHERE symbol = ? AND trade_date IN (?, ?)
            ORDER BY trade_date
            """,
            (symbol, prior_ts, trade_ts),
        )
        rows = q.fetchall()
        if len(rows) != 2:
            return {"suspect": True, "reasons": ["missing_ohlcv_row_around_event"], "ohlcv_rows": []}

        reasons = []
        ohlcv_payload = []
        for r in rows:
            td = int(r[0])
            o = float(r[1])
            h = float(r[2])
            l = float(r[3])
            c = float(r[4])
            v = float(r[5])
            src = str(r[6])
            ohlcv_payload.append(
                {
                    "trade_date": ts_to_date(td).isoformat(),
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                    "source": src,
                }
            )

            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                reasons.append("non_positive_ohlc")
            if h < l:
                reasons.append("high_lt_low")
            if l > min(o, c):
                reasons.append("low_gt_min_open_close")
            if h < max(o, c):
                reasons.append("high_lt_max_open_close")
            if v < 0:
                reasons.append("negative_volume")
            if v == 0:
                reasons.append("zero_volume_around_jump")

        # Check row close alignment with breach event values.
        prior_row = ohlcv_payload[0]
        trade_row = ohlcv_payload[1]
        if abs(prior_row["close"] - prior_close) > 1e-6:
            reasons.append("prior_close_mismatch_vs_event")
        if abs(trade_row["close"] - close) > 1e-6:
            reasons.append("close_mismatch_vs_event")

        return {
            "suspect": len(reasons) > 0,
            "reasons": sorted(set(reasons)),
            "ohlcv_rows": ohlcv_payload,
        }

    classified: list[dict[str, Any]] = []

    for symbol, rows in by_symbol.items():
        for idx, ev in enumerate(rows):
            gap = gap_sessions_ex_holidays(ev["prior_trade_date_ts"], ev["trade_date_ts"], holiday_ts)
            near_fraction, nearest_fraction = near_clean_fraction(float(ev["ratio"]))
            pair = detect_paired_recovery(rows, idx, float(ev["ratio"]))
            quality = row_quality_suspect(
                symbol,
                int(ev["prior_trade_date_ts"]),
                int(ev["trade_date_ts"]),
                float(ev["prior_close"]),
                float(ev["close"]),
            )

            if quality["suspect"]:
                cls = "DATA_SUSPECT"
                reason = "OHLC inconsistency or vendor artifact around event"
            elif gap > 0:
                cls = "POST_SUSPENSION_REPRICING"
                reason = "Tradable session gap immediately before jump (holidays excluded)"
            elif near_fraction or pair.get("paired", False):
                cls = "SUSPECTED_CORPORATE_ACTION"
                reason = "Clean-fraction ratio or paired drop/recovery signature"
            else:
                cls = "ISOLATED_LARGE_MOVE"
                reason = "No gap and no corporate-action signature"

            classified.append(
                {
                    "symbol": symbol,
                    "class": cls,
                    "reason": reason,
                    "prior_trade_date": ev["prior_trade_date"],
                    "trade_date": ev["trade_date"],
                    "prior_close": ev["prior_close"],
                    "close": ev["close"],
                    "ratio": ev["ratio"],
                    "jump_pct": ev["jump_pct"],
                    "gap_sessions_ex_holidays": gap,
                    "ca_signature": {
                        "near_clean_fraction": near_fraction,
                        "nearest_fraction": nearest_fraction,
                        "paired_drop_recovery": bool(pair.get("paired", False)),
                        "paired_evidence": pair,
                    },
                    "data_suspect": {
                        "flag": quality["suspect"],
                        "reasons": quality["reasons"],
                        "ohlcv_rows": quality["ohlcv_rows"],
                    },
                    "child_run_id": ev["child_run_id"],
                    "resolution_status": ev["resolution_status"],
                }
            )

    class_order = [
        "POST_SUSPENSION_REPRICING",
        "SUSPECTED_CORPORATE_ACTION",
        "ISOLATED_LARGE_MOVE",
        "DATA_SUSPECT",
    ]

    by_class: dict[str, list[dict[str, Any]]] = {k: [] for k in class_order}
    for row in classified:
        by_class[row["class"]].append(row)

    class_summary = {}
    for cls in class_order:
        rows = sorted(by_class[cls], key=lambda x: (x["symbol"], x["trade_date"]))
        class_summary[cls] = {
            "count": len(rows),
            "symbols": sorted({r["symbol"] for r in rows}),
            "sample_rows": rows[:5],
        }

    symbol_event_counts = Counter(r["symbol"] for r in classified)
    per_symbol_distribution = [
        {"symbol": s, "event_count": c}
        for s, c in sorted(symbol_event_counts.items(), key=lambda x: (-x[1], x[0]))
    ]

    return {
        "campaign_id": cfg.campaign_id,
        "calendar_version_id": cfg.calendar_version_id,
        "total_events": len(classified),
        "class_summary": class_summary,
        "events_per_symbol_distribution": per_symbol_distribution,
        "dense_event_symbols": [x for x in per_symbol_distribution if x["event_count"] >= 8],
        "classification_notes": [
            "POST_SUSPENSION_REPRICING: gap_sessions_ex_holidays > 0",
            "SUSPECTED_CORPORATE_ACTION: near clean fraction (+/-0.03) or paired reciprocal move",
            "ISOLATED_LARGE_MOVE: no gap and no CA signature",
            "DATA_SUSPECT: OHLC inconsistency or row artifact around event",
        ],
        "rows": classified,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# R12 Breach Triage V4",
        "",
        f"- campaign_id: {payload['campaign_id']}",
        f"- calendar_version_id: {payload['calendar_version_id']}",
        f"- total_events: {payload['total_events']}",
        "",
        "## Class counts",
        "",
        "| Class | Count | Symbol count |",
        "|---|---:|---:|",
    ]

    for cls, block in payload["class_summary"].items():
        lines.append(f"| {cls} | {block['count']} | {len(block['symbols'])} |")

    lines.extend(
        [
            "",
            "## Dense event symbols (suspension-heavy masking candidates)",
            "",
            "| Symbol | Events |",
            "|---|---:|",
        ]
    )
    for row in payload["dense_event_symbols"]:
        lines.append(f"| {row['symbol']} | {row['event_count']} |")

    lines.extend(
        [
            "",
            "## Sample evidence (5 rows per class)",
            "",
        ]
    )

    for cls, block in payload["class_summary"].items():
        lines.append(f"### {cls}")
        lines.append("")
        lines.append("| Symbol | Prior date | Date | Prior close | Close | Ratio | Jump % | Gap sessions | CA signature | Data suspect reasons |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---|---|")
        for r in block["sample_rows"]:
            ca = []
            if r["ca_signature"]["near_clean_fraction"]:
                ca.append(f"near={r['ca_signature']['nearest_fraction']}")
            if r["ca_signature"]["paired_drop_recovery"]:
                ca.append("paired_recovery")
            ca_txt = ",".join(ca) if ca else "-"
            ds_txt = ",".join(r["data_suspect"]["reasons"]) if r["data_suspect"]["reasons"] else "-"
            lines.append(
                f"| {r['symbol']} | {r['prior_trade_date']} | {r['trade_date']} | {r['prior_close']:.6f} | {r['close']:.6f} | {r['ratio']:.6f} | {r['jump_pct']:.4f} | {r['gap_sessions_ex_holidays']} | {ca_txt} | {ds_txt} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate R12 V4 breach triage report")
    parser.add_argument("--db", required=True, help="Path to sqlite database")
    parser.add_argument("--campaign-id", required=True, help="Campaign id")
    parser.add_argument("--calendar-version-id", required=True, help="Calendar version id")
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    parser.add_argument("--out-md", required=True, help="Output markdown path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    conn = sqlite3.connect(Path(args.db).resolve())
    try:
        payload = classify_events(
            conn,
            TriageConfig(campaign_id=args.campaign_id, calendar_version_id=args.calendar_version_id),
        )
    finally:
        conn.close()

    out_json = Path(args.out_json).resolve()
    out_md = Path(args.out_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    out_md.write_text(markdown_report(payload), encoding="utf-8", newline="\n")

    print("TRIAGE_COMPLETE", payload["total_events"])
    for cls, block in payload["class_summary"].items():
        print("CLASS", cls, block["count"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

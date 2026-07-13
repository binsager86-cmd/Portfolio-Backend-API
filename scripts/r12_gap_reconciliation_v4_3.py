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


def trading_sessions_between_exclusive(prior_d: date, event_d: date, holiday_ts: set[int]) -> list[str]:
    out = []
    d = prior_d + timedelta(days=1)
    while d < event_d:
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


def previous_bar(conn: sqlite3.Connection, symbol: str, event_d: date) -> dict[str, Any] | None:
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT trade_date, close, volume
        FROM ee_ohlcv
        WHERE symbol = ? AND trade_date < ?
        ORDER BY trade_date DESC
        LIMIT 1
        """,
        (symbol, to_ts(event_d)),
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


def collect_verified_sessions_before(event_d: date, holiday_ts: set[int], count: int) -> list[date]:
    out: list[date] = []
    d = event_d - timedelta(days=1)
    while len(out) < count:
        if is_trading_day_ex_holiday(d, holiday_ts):
            out.append(d)
        d -= timedelta(days=1)
    out.reverse()
    return out


def thuraya_window_compare(
    clean_conn: sqlite3.Connection,
    preview_conn: sqlite3.Connection,
    holiday_ts: set[int],
) -> dict[str, Any]:
    symbol = "THURAYA"
    target_event = date(2025, 7, 20)

    sessions = collect_verified_sessions_before(target_event, holiday_ts, 200)

    clean_rows = []
    preview_rows = []

    clean_dates_with_bar = set()
    preview_dates_with_bar = set()

    for s in sessions:
        b_clean = fetch_bar(clean_conn, symbol, s)
        b_prev = fetch_bar(preview_conn, symbol, s)

        clean_rows.append(
            {
                "session_date": s.isoformat(),
                "has_bar": b_clean is not None,
                "close": None if b_clean is None else b_clean["close"],
                "volume": None if b_clean is None else b_clean["volume"],
            }
        )
        preview_rows.append(
            {
                "session_date": s.isoformat(),
                "has_bar": b_prev is not None,
                "close": None if b_prev is None else b_prev["close"],
                "volume": None if b_prev is None else b_prev["volume"],
            }
        )

        if b_clean is not None:
            clean_dates_with_bar.add(s.isoformat())
        if b_prev is not None:
            preview_dates_with_bar.add(s.isoformat())

    clean_only = sorted(clean_dates_with_bar - preview_dates_with_bar)
    preview_only = sorted(preview_dates_with_bar - clean_dates_with_bar)
    both = sorted(clean_dates_with_bar & preview_dates_with_bar)

    # Gap context for target event in each DB.
    prev_clean = previous_bar(clean_conn, symbol, target_event)
    prev_preview = previous_bar(preview_conn, symbol, target_event)

    def gap_for(prev_row: dict[str, Any] | None) -> dict[str, Any] | None:
        if prev_row is None:
            return None
        prior = date.fromisoformat(prev_row["trade_date"])
        between = trading_sessions_between_exclusive(prior, target_event, holiday_ts)
        return {
            "prior_bar_date": prior.isoformat(),
            "sessions_between_count": len(between),
            "sessions_between_list": between,
        }

    return {
        "symbol": symbol,
        "target_event_date": target_event.isoformat(),
        "verified_window_session_count": len(sessions),
        "window_start": sessions[0].isoformat(),
        "window_end": sessions[-1].isoformat(),
        "clean_db_rows": clean_rows,
        "preview_db_rows": preview_rows,
        "date_set_compare": {
            "clean_dates_with_bar_count": len(clean_dates_with_bar),
            "preview_dates_with_bar_count": len(preview_dates_with_bar),
            "both_count": len(both),
            "clean_only_count": len(clean_only),
            "preview_only_count": len(preview_only),
            "clean_only_dates": clean_only,
            "preview_only_dates": preview_only,
        },
        "target_event_gap_context": {
            "clean_db": gap_for(prev_clean),
            "preview_db": gap_for(prev_preview),
        },
    }


def reconcile_hidden_gap(
    clean_conn: sqlite3.Connection,
    holiday_ts: set[int],
    v41_partial: dict[str, Any],
) -> dict[str, Any]:
    rows = v41_partial["rows"]
    isolated = [r for r in rows if r.get("original_class_v4") == "ISOLATED_LARGE_MOVE"]

    changed = []
    hidden = 0
    true_consecutive = 0

    for r in sorted(isolated, key=lambda x: (x["symbol"], x["event_bar_date"])):
        symbol = str(r["symbol"])
        prior_d = date.fromisoformat(r["prior_bar_date"])
        event_d = date.fromisoformat(r["event_bar_date"])

        between = trading_sessions_between_exclusive(prior_d, event_d, holiday_ts)
        corrected_class = "HIDDEN_GAP" if len(between) >= 1 else "TRUE_CONSECUTIVE"

        if corrected_class == "HIDDEN_GAP":
            hidden += 1
        else:
            true_consecutive += 1

        prior_class = str(r.get("classification_v4_1_partial"))
        if prior_class != corrected_class:
            changed.append(
                {
                    "symbol": symbol,
                    "event_bar_date": event_d.isoformat(),
                    "prior_bar_date": prior_d.isoformat(),
                    "prior_classification_v4_1_partial": prior_class,
                    "corrected_classification": corrected_class,
                    "sessions_between_count": len(between),
                    "sessions_between_list": between,
                }
            )

    # Explain contradiction source by code scope.
    # v4.1 split computed only over isolated rows; v4.2 coverage count looked at all rows for selected symbols.
    scope_mismatch_evidence = []
    for sym in ["PAPCO", "TAHSSILAT", "MARAKEZ", "EMIRATES"]:
        sym_rows = [x for x in rows if x["symbol"] == sym]
        sym_isolated = [x for x in sym_rows if x.get("original_class_v4") == "ISOLATED_LARGE_MOVE"]
        sym_nonisol_with_gap = [
            x
            for x in sym_rows
            if x.get("original_class_v4") != "ISOLATED_LARGE_MOVE"
            and int(x.get("sessions_between_verified_calendar") or 0) >= 1
        ]
        scope_mismatch_evidence.append(
            {
                "symbol": sym,
                "all_rows_count": len(sym_rows),
                "isolated_rows_count": len(sym_isolated),
                "non_isolated_rows_with_gap_count": len(sym_nonisol_with_gap),
                "non_isolated_gap_rows": [
                    {
                        "event_bar_date": y["event_bar_date"],
                        "prior_bar_date": y["prior_bar_date"],
                        "original_class_v4": y["original_class_v4"],
                        "sessions_between_verified_calendar": int(y.get("sessions_between_verified_calendar") or 0),
                    }
                    for y in sym_nonisol_with_gap
                ],
            }
        )

    return {
        "recount_isolated_event_count": len(isolated),
        "corrected_split": {
            "hidden_gap": hidden,
            "true_consecutive": true_consecutive,
        },
        "changed_events": changed,
        "changed_event_count": len(changed),
        "code_path_divergence": {
            "v4_1_path": {
                "file": "scripts/r12_breach_triage_v4_1_partial.py",
                "scope": "ISOLATED_LARGE_MOVE rows only for hidden_gap/true_consecutive split",
                "session_function": "sessions_between_verified_calendar",
            },
            "v4_2_path": {
                "file": "scripts/r12_gap_audit_verification_v4_2.py",
                "scope": "all rows for each selected symbol when counting audited_events_with_missing_sessions_count",
                "session_function": "trading_days_between_exclusive / existing sessions_between_verified_calendar field",
            },
            "exact_divergence": "Scope mismatch: v4.1 split is isolated-only; v4.2 audited_events_with_missing_sessions_count included non-isolated rows for the same symbols.",
            "scope_mismatch_evidence": scope_mismatch_evidence,
        },
    }


def annotate_ca_ledger_v0_1(ca_ledger_v0_1: dict[str, Any]) -> dict[str, Any]:
    annotations = []
    for r in ca_ledger_v0_1.get("extreme_mover_ratio_annotations", []):
        exact_ratio = float(r["exact_ratio"])
        nearest, deviation = nearest_simple_ratio(exact_ratio)
        suspected = "CAPITAL_DECREASE" if deviation <= 0.05 else "UNSPECIFIED"
        annotations.append(
            {
                "symbol": r["symbol"],
                "prior_bar_date": r["prior_bar_date"],
                "event_bar_date": r["event_bar_date"],
                "prior_close": float(r["prior_close"]),
                "event_close": float(r["event_close"]),
                "exact_ratio": exact_ratio,
                "nearest_simple_ratio": nearest,
                "deviation_from_nearest": deviation,
                "suspected_action": suspected,
                "official_terms_source": None,
                "official_terms_effective_date": None,
                "official_terms_ratio": None,
            }
        )

    return {
        "version": "r12_ca_ledger_v0.1_reconciliation_v4_3",
        "annotation_count": len(annotations),
        "annotations": sorted(annotations, key=lambda x: (x["symbol"], x["event_bar_date"])),
    }


def markdown_from_payload(payload: dict[str, Any]) -> str:
    lines = [
        "# R12 Gap Reconciliation V4.3",
        "",
        f"- version_id: {payload['version_id']}",
        f"- scope: {payload['scope']}",
        "",
        "## THURAYA Divergence",
        "",
        f"- target_event_date: {payload['thuraya_reconciliation']['target_event_date']}",
        f"- clean_only_count: {payload['thuraya_reconciliation']['date_set_compare']['clean_only_count']}",
        f"- preview_only_count: {payload['thuraya_reconciliation']['date_set_compare']['preview_only_count']}",
        "",
        "## Hidden-Gap Recount (170 isolated events)",
        "",
        f"- HIDDEN_GAP: {payload['hidden_gap_reconciliation']['corrected_split']['hidden_gap']}",
        f"- TRUE_CONSECUTIVE: {payload['hidden_gap_reconciliation']['corrected_split']['true_consecutive']}",
        f"- changed_event_count: {payload['hidden_gap_reconciliation']['changed_event_count']}",
        "",
        "## CA Ledger Annotations (>=100% movers)",
        "",
        f"- annotation_count: {payload['ca_ledger_v0_1_annotations']['annotation_count']}",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R12 gap reconciliation v4.3 (verification only)")
    p.add_argument("--clean-db", required=True)
    p.add_argument("--preview-db", required=True)
    p.add_argument("--v41-partial-json", required=True)
    p.add_argument("--v42-final-json", required=True)
    p.add_argument("--calendar-version-id", default="BK_CAL_V4_1783783330")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    clean_conn = sqlite3.connect(Path(args.clean_db).resolve())
    preview_conn = sqlite3.connect(Path(args.preview_db).resolve())

    try:
        holiday_ts = load_holiday_set(clean_conn, args.calendar_version_id)
        v41 = json.loads(Path(args.v41_partial_json).read_text(encoding="utf-8"))
        v42f = json.loads(Path(args.v42_final_json).read_text(encoding="utf-8"))

        th = thuraya_window_compare(clean_conn, preview_conn, holiday_ts)
        hg = reconcile_hidden_gap(clean_conn, holiday_ts, v41)
        ca_annot = annotate_ca_ledger_v0_1(v42f["final"]["ca_ledger_v0_1"])

        payload = {
            "version_id": "R12_GAP_RECONCILIATION_V4_3",
            "scope": "Verification reconciliation only. No disposition, masking, or triage re-issuance changes.",
            "calendar_version_id": args.calendar_version_id,
            "thuraya_reconciliation": th,
            "hidden_gap_reconciliation": hg,
            "ca_ledger_v0_1_annotations": ca_annot,
            "findings_only_note": "No rulings applied. Findings reported for owner adjudication.",
        }
    finally:
        clean_conn.close()
        preview_conn.close()

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    out_md.write_text(markdown_from_payload(payload), encoding="utf-8", newline="\n")

    print("GAP_RECON_V4_3_COMPLETE")
    print("THURAYA_CLEAN_ONLY", payload["thuraya_reconciliation"]["date_set_compare"]["clean_only_count"])
    print("THURAYA_PREVIEW_ONLY", payload["thuraya_reconciliation"]["date_set_compare"]["preview_only_count"])
    print("ISOLATED_RECOUNT", json.dumps(payload["hidden_gap_reconciliation"]["corrected_split"], sort_keys=True))
    print("CHANGED_EVENTS", payload["hidden_gap_reconciliation"]["changed_event_count"])
    print("CA_ANNOTATIONS", payload["ca_ledger_v0_1_annotations"]["annotation_count"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any


CALENDAR_VERSION_ID = "BK_CAL_V4_1783783330"
EXPECTED_PROFILE_MOVE_PCT = 85.0
EXPECTED_PROFILE_MISSING_SESSIONS = 164


@dataclass
class GapReferent:
    prior_date: str
    prior_close: float
    event_date: str
    event_close: float
    missing_sessions: int
    resumption_move_pct: float


def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())


def ts_to_date(ts: int) -> date:
    return datetime.fromtimestamp(int(ts), tz=UTC).date()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_trading_day_ex_holiday(d: date, holiday_ts: set[int]) -> bool:
    if d.weekday() not in (6, 0, 1, 2, 3):
        return False
    return to_ts(d) not in holiday_ts


def trading_sessions_between_exclusive(prior_d: date, event_d: date, holiday_ts: set[int]) -> list[str]:
    out: list[str] = []
    d = prior_d + timedelta(days=1)
    while d < event_d:
        if is_trading_day_ex_holiday(d, holiday_ts):
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def load_holiday_set(clean_conn: sqlite3.Connection, version_id: str) -> set[int]:
    cur = clean_conn.cursor()
    rows = cur.execute(
        """
        SELECT trade_date
        FROM ee_trading_calendar_days_v4
        WHERE version_id = ? AND is_holiday = 1
        """,
        (version_id,),
    ).fetchall()
    return {int(r[0]) for r in rows}


def load_symbol_bars(conn: sqlite3.Connection, symbol: str) -> list[tuple[int, float]]:
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT trade_date, close
        FROM ee_ohlcv
        WHERE symbol = ?
        ORDER BY trade_date ASC
        """,
        (symbol,),
    ).fetchall()
    return [(int(r[0]), float(r[1])) for r in rows]


def find_largest_gap_referent(conn: sqlite3.Connection, symbol: str, holiday_ts: set[int]) -> GapReferent:
    bars = load_symbol_bars(conn, symbol)
    if len(bars) < 2:
        raise ValueError(f"Not enough bars for {symbol}")

    best: GapReferent | None = None

    for idx in range(1, len(bars)):
        prior_ts, prior_close = bars[idx - 1]
        event_ts, event_close = bars[idx]
        prior_d = ts_to_date(prior_ts)
        event_d = ts_to_date(event_ts)
        missing = trading_sessions_between_exclusive(prior_d, event_d, holiday_ts)
        move_pct = ((event_close / prior_close) - 1.0) * 100.0

        candidate = GapReferent(
            prior_date=prior_d.isoformat(),
            prior_close=prior_close,
            event_date=event_d.isoformat(),
            event_close=event_close,
            missing_sessions=len(missing),
            resumption_move_pct=move_pct,
        )

        if best is None:
            best = candidate
            continue

        if candidate.missing_sessions > best.missing_sessions:
            best = candidate

    if best is None:
        raise ValueError(f"Unable to compute largest gap for {symbol}")
    return best


def nearest_simple_ratio(exact_ratio: float) -> tuple[float, float]:
    simple = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    nearest = min(simple, key=lambda x: abs(exact_ratio - x))
    return nearest, abs(exact_ratio - nearest)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_ca_v0_2(
    v42_verify: dict[str, Any],
    v43_recon: dict[str, Any],
    v42_final: dict[str, Any],
) -> dict[str, Any]:
    ratio_rows: list[dict[str, Any]] = v42_verify["extreme_mover_ratio_table"]
    legacy_rows: list[dict[str, Any]] = v43_recon["ca_ledger_v0_1_annotations"]["annotations"]

    legacy_keys = {(str(r["symbol"]), str(r["event_bar_date"])) for r in legacy_rows}
    ratio_keys = {(str(r["symbol"]), str(r["event_bar_date"])) for r in ratio_rows}

    # Use the 15-row v4.2 verification table as the authoritative extreme-mover surface.
    corrected_annotations: list[dict[str, Any]] = []
    for row in sorted(ratio_rows, key=lambda x: (x["symbol"], x["event_bar_date"])):
        exact_ratio = float(row["exact_ratio"])
        nearest_ratio = float(row["nearest_simple_ratio"])
        deviation = float(row["deviation_from_nearest"])
        # Preserve v4.3 threshold rule and fix source-population defect only.
        suspected_action = "CAPITAL_DECREASE" if deviation <= 0.05 else "UNSPECIFIED"

        corrected_annotations.append(
            {
                "symbol": str(row["symbol"]),
                "prior_bar_date": str(row["prior_bar_date"]),
                "event_bar_date": str(row["event_bar_date"]),
                "prior_close": float(row["prior_close"]),
                "event_close": float(row["event_close"]),
                "exact_ratio": exact_ratio,
                "nearest_simple_ratio": nearest_ratio,
                "deviation_from_nearest": deviation,
                "suspected_action": suspected_action,
                "official_terms_source": None,
                "official_terms_effective_date": None,
                "official_terms_ratio": None,
            }
        )

    missing_from_legacy = sorted(ratio_keys - legacy_keys)

    # Report disposition class for the 4 events present in 15-row table but absent from 11-row set.
    final_row_map = {
        (str(r["symbol"]), str(r["event_bar_date"])): r
        for r in v42_final["final"]["rows"]
    }
    missing_class_rows = []
    for key in missing_from_legacy:
        src = final_row_map.get(key)
        missing_class_rows.append(
            {
                "symbol": key[0],
                "event_bar_date": key[1],
                "disposition_class": None if src is None else src.get("original_class_v4"),
            }
        )

    capital_decrease_count = sum(1 for r in corrected_annotations if r["suspected_action"] == "CAPITAL_DECREASE")

    return {
        "version_id": "R12_CA_LEDGER_V0_2",
        "scope": "Annotation-only correction. No disposition class changes.",
        "r12_execution_status": "NOT_AUTHORIZED",
        "source_artifacts": {
            "verification_ratio_table": "artifacts/preview1a_prestart/review_final/r12_gap_audit_verification_v4_2.json",
            "legacy_annotations": "artifacts/preview1a_prestart/review_final/r12_gap_reconciliation_v4_3.json",
            "triage_final": "artifacts/preview1a_prestart/review_final/r12_breach_triage_v4_2_FINAL.json",
        },
        "defect_diagnosis": {
            "v4_3_observation": "Threshold rule was applied on an 11-row source and therefore never evaluated SANAM 2023-09-05 nor TAHSSILAT 2024-02-13.",
            "root_cause": "Source-population mismatch: v4.3 annotation input came from final.ca_ledger_v0_1 (11 rows) instead of extreme_mover_ratio_table verification surface (15 rows).",
            "fix": "Recompute annotations on the full 15-row verification table using the same <= 0.05 deviation threshold.",
        },
        "legacy_annotation_count": len(legacy_rows),
        "verification_table_count": len(ratio_rows),
        "corrected_annotation_count": len(corrected_annotations),
        "capital_decrease_count": capital_decrease_count,
        "unspecified_count": len(corrected_annotations) - capital_decrease_count,
        "missing_from_legacy_count": len(missing_from_legacy),
        "missing_from_legacy_events_with_disposition_class": missing_class_rows,
        "legacy_11_row_annotation_table": sorted(
            [
                {
                    "symbol": str(r["symbol"]),
                    "prior_bar_date": str(r["prior_bar_date"]),
                    "event_bar_date": str(r["event_bar_date"]),
                    "prior_close": float(r["prior_close"]),
                    "event_close": float(r["event_close"]),
                    "exact_ratio": float(r["exact_ratio"]),
                    "nearest_simple_ratio": float(r["nearest_simple_ratio"]),
                    "deviation_from_nearest": float(r["deviation_from_nearest"]),
                    "suspected_action": str(r["suspected_action"]),
                }
                for r in legacy_rows
            ],
            key=lambda x: (x["symbol"], x["event_bar_date"]),
        ),
        "corrected_annotations": corrected_annotations,
    }


def build_reconciliation_addendum(
    clean_ref: GapReferent,
    preview_ref: GapReferent,
) -> dict[str, Any]:
    def matches_profile(ref: GapReferent) -> dict[str, Any]:
        return {
            "expected_move_pct": EXPECTED_PROFILE_MOVE_PCT,
            "expected_missing_sessions": EXPECTED_PROFILE_MISSING_SESSIONS,
            "actual_move_pct": ref.resumption_move_pct,
            "actual_missing_sessions": ref.missing_sessions,
            "move_pct_delta": ref.resumption_move_pct - EXPECTED_PROFILE_MOVE_PCT,
            "missing_sessions_delta": ref.missing_sessions - EXPECTED_PROFILE_MISSING_SESSIONS,
            "matches_profile": (
                abs(ref.resumption_move_pct - EXPECTED_PROFILE_MOVE_PCT) <= 5.0
                and abs(ref.missing_sessions - EXPECTED_PROFILE_MISSING_SESSIONS) <= 5
            ),
        }

    return {
        "version_id": "R12_GAP_RECONCILIATION_V4_3_ADDENDUM_8",
        "scope": "Finding-only referent identification. No disposition or status changes.",
        "r12_execution_status": "NOT_AUTHORIZED",
        "adjudicated_referent_identification": {
            "symbol": "THURAYA",
            "method": "Largest bar-date gap by missing verified sessions over full available history.",
            "clean_db": {
                "prior_bar": {
                    "trade_date": clean_ref.prior_date,
                    "close": clean_ref.prior_close,
                },
                "event_bar": {
                    "trade_date": clean_ref.event_date,
                    "close": clean_ref.event_close,
                },
                "missing_verified_sessions": clean_ref.missing_sessions,
                "resumption_move_pct": clean_ref.resumption_move_pct,
                "profile_match_check": matches_profile(clean_ref),
            },
            "preview_db": {
                "prior_bar": {
                    "trade_date": preview_ref.prior_date,
                    "close": preview_ref.prior_close,
                },
                "event_bar": {
                    "trade_date": preview_ref.event_date,
                    "close": preview_ref.event_close,
                },
                "missing_verified_sessions": preview_ref.missing_sessions,
                "resumption_move_pct": preview_ref.resumption_move_pct,
                "profile_match_check": matches_profile(preview_ref),
            },
            "status_note": "Finding only. 2025-07-20 TRUE_CONSECUTIVE status is unchanged.",
        },
    }


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Symbol | Prior Date | Prior Close | Event Date | Event Close | Exact Ratio | Nearest | Deviation | Action |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {symbol} | {prior_bar_date} | {prior_close:.6f} | {event_bar_date} | {event_close:.6f} | {exact_ratio:.12f} | {nearest_simple_ratio:.6f} | {deviation_from_nearest:.12f} | {suspected_action} |".format(
                **r
            )
        )
    return lines


def md_for_reconciliation_addendum(payload: dict[str, Any]) -> str:
    clean = payload["adjudicated_referent_identification"]["clean_db"]
    preview = payload["adjudicated_referent_identification"]["preview_db"]
    lines = [
        "# R12 Gap Reconciliation V4.3 Addendum 8",
        "",
        f"- version_id: {payload['version_id']}",
        f"- scope: {payload['scope']}",
        "",
        "## THURAYA Referent (Largest Full-History Gap)",
        "",
        "### Clean DB",
        f"- prior_bar_date: {clean['prior_bar']['trade_date']}",
        f"- prior_close: {clean['prior_bar']['close']}",
        f"- event_bar_date: {clean['event_bar']['trade_date']}",
        f"- event_close: {clean['event_bar']['close']}",
        f"- missing_verified_sessions: {clean['missing_verified_sessions']}",
        f"- resumption_move_pct: {clean['resumption_move_pct']}",
        f"- matches_PREVIEW_1A_profile: {clean['profile_match_check']['matches_profile']}",
        "",
        "### Preview DB",
        f"- prior_bar_date: {preview['prior_bar']['trade_date']}",
        f"- prior_close: {preview['prior_bar']['close']}",
        f"- event_bar_date: {preview['event_bar']['trade_date']}",
        f"- event_close: {preview['event_bar']['close']}",
        f"- missing_verified_sessions: {preview['missing_verified_sessions']}",
        f"- resumption_move_pct: {preview['resumption_move_pct']}",
        f"- matches_PREVIEW_1A_profile: {preview['profile_match_check']['matches_profile']}",
        "",
        f"- note: {payload['adjudicated_referent_identification']['status_note']}",
        "",
    ]
    return "\n".join(lines)


def md_for_ca_v0_2(payload: dict[str, Any]) -> str:
    lines = [
        "# R12 CA Ledger V0.2",
        "",
        f"- version_id: {payload['version_id']}",
        f"- scope: {payload['scope']}",
        f"- corrected_annotation_count: {payload['corrected_annotation_count']}",
        f"- capital_decrease_count: {payload['capital_decrease_count']}",
        f"- unspecified_count: {payload['unspecified_count']}",
        "",
        "## Defect",
        f"- observation: {payload['defect_diagnosis']['v4_3_observation']}",
        f"- root_cause: {payload['defect_diagnosis']['root_cause']}",
        f"- fix: {payload['defect_diagnosis']['fix']}",
        "",
        "## Legacy 11-Row Annotation Table",
        "",
    ]
    lines.extend(markdown_table(payload["legacy_11_row_annotation_table"]))
    lines.extend([
        "",
        "## Corrected 15-Row Annotation Table",
        "",
    ])
    lines.extend(markdown_table(payload["corrected_annotations"]))
    lines.extend([
        "",
        "## 4 Events Missing From Legacy 11-Row Set",
        "",
        "| Symbol | Event Date | Disposition Class |",
        "|---|---|---|",
    ])
    for r in payload["missing_from_legacy_events_with_disposition_class"]:
        lines.append(f"| {r['symbol']} | {r['event_bar_date']} | {r['disposition_class']} |")
    lines.append("")
    return "\n".join(lines)


def build_seal(
    review_dir: Path,
    references: list[dict[str, str]],
    disposition_counts: dict[str, int],
    masked_interval_count: int,
) -> dict[str, Any]:
    ref_items = []
    for ref in references:
        path = review_dir / ref["path"]
        ref_items.append(
            {
                "label": ref["label"],
                "path": f"artifacts/preview1a_prestart/review_final/{ref['path']}",
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    return {
        "version_id": "R12_PRE_EXAM_SURFACE_SEAL_V4_3",
        "scope": "Closure statement only. No new classifications.",
        "r12_execution_status": "NOT_AUTHORIZED",
        "ratified_artifact_references": ref_items,
        "final_disposition_counts": disposition_counts,
        "masked_interval_count": masked_interval_count,
        "closure_note": "Seal records ratified surfaces only and does not alter existing dispositions.",
    }


def md_for_seal(payload: dict[str, Any]) -> str:
    lines = [
        "# R12 Pre-Exam Surface Seal V4.3",
        "",
        f"- version_id: {payload['version_id']}",
        f"- scope: {payload['scope']}",
        f"- r12_execution_status: {payload['r12_execution_status']}",
        "",
        "## Ratified Artifact References",
        "",
        "| Label | Path | SHA256 | Size (bytes) |",
        "|---|---|---|---:|",
    ]
    for r in payload["ratified_artifact_references"]:
        lines.append(f"| {r['label']} | {r['path']} | {r['sha256']} | {r['size_bytes']} |")

    lines.extend(
        [
            "",
            "## Final Counts",
            f"- final_disposition_counts: {json.dumps(payload['final_disposition_counts'], sort_keys=True)}",
            f"- masked_interval_count: {payload['masked_interval_count']}",
            "",
            f"- note: {payload['closure_note']}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R12 v4.3 closure items")
    p.add_argument("--clean-db", required=True)
    p.add_argument("--preview-db", required=True)
    p.add_argument("--review-dir", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    clean_db = Path(args.clean_db).resolve()
    preview_db = Path(args.preview_db).resolve()
    review_dir = Path(args.review_dir).resolve()

    v42_verify_path = review_dir / "r12_gap_audit_verification_v4_2.json"
    v42_final_path = review_dir / "r12_breach_triage_v4_2_FINAL.json"
    v42_mask_path = review_dir / "r12_masked_intervals_manifest_v4_2_final.json"
    v43_recon_path = review_dir / "r12_gap_reconciliation_v4_3.json"
    cal_adj_path = review_dir / "r12_calendar_owner_adjudication_v4.json"

    for p in [
        v42_verify_path,
        v42_final_path,
        v42_mask_path,
        v43_recon_path,
        cal_adj_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    with sqlite3.connect(clean_db) as clean_conn:
        holiday_ts = load_holiday_set(clean_conn, CALENDAR_VERSION_ID)
        clean_ref = find_largest_gap_referent(clean_conn, "THURAYA", holiday_ts)

    with sqlite3.connect(preview_db) as preview_conn:
        # Use the same owner-verified calendar holiday set for cross-DB comparability.
        preview_ref = find_largest_gap_referent(preview_conn, "THURAYA", holiday_ts)

    v42_verify = load_json(v42_verify_path)
    v42_final = load_json(v42_final_path)
    v43_recon = load_json(v43_recon_path)
    mask_manifest = load_json(v42_mask_path)

    recon_addendum = build_reconciliation_addendum(clean_ref, preview_ref)
    ca_v0_2 = build_ca_v0_2(v42_verify, v43_recon, v42_final)

    recon_addendum_json = review_dir / "r12_gap_reconciliation_v4_3_addendum_8.json"
    recon_addendum_md = review_dir / "r12_gap_reconciliation_v4_3_addendum_8.md"
    ca_v0_2_json = review_dir / "r12_ca_ledger_v0_2.json"
    ca_v0_2_md = review_dir / "r12_ca_ledger_v0_2.md"

    recon_addendum_json.write_text(
        json.dumps(recon_addendum, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    recon_addendum_md.write_text(md_for_reconciliation_addendum(recon_addendum), encoding="utf-8", newline="\n")

    ca_v0_2_json.write_text(
        json.dumps(ca_v0_2, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    ca_v0_2_md.write_text(md_for_ca_v0_2(ca_v0_2), encoding="utf-8", newline="\n")

    seal = build_seal(
        review_dir=review_dir,
        references=[
            {"label": "triage_v4_2_final", "path": "r12_breach_triage_v4_2_FINAL.json"},
            {"label": "mask_manifest_v4_2_final", "path": "r12_masked_intervals_manifest_v4_2_final.json"},
            {"label": "ca_ledger_v0_2", "path": "r12_ca_ledger_v0_2.json"},
            {"label": "calendar_owner_adjudication_v4", "path": "r12_calendar_owner_adjudication_v4.json"},
            {"label": "reconciliation_v4_3", "path": "r12_gap_reconciliation_v4_3.json"},
        ],
        disposition_counts={k: int(v) for k, v in v42_final["final"]["disposition_counts"].items()},
        masked_interval_count=len(mask_manifest),
    )

    seal_json = review_dir / "r12_pre_exam_surface_seal_v4_3.json"
    seal_md = review_dir / "r12_pre_exam_surface_seal_v4_3.md"

    seal_json.write_text(
        json.dumps(seal, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    seal_md.write_text(md_for_seal(seal), encoding="utf-8", newline="\n")

    # Print requested 11-row table and closure checkpoints to terminal.
    print("THURAYA_REFERENT_CLEAN", json.dumps(recon_addendum["adjudicated_referent_identification"]["clean_db"], sort_keys=True))
    print("THURAYA_REFERENT_PREVIEW", json.dumps(recon_addendum["adjudicated_referent_identification"]["preview_db"], sort_keys=True))
    print("PROFILE_MATCH_CLEAN", recon_addendum["adjudicated_referent_identification"]["clean_db"]["profile_match_check"]["matches_profile"])
    print("PROFILE_MATCH_PREVIEW", recon_addendum["adjudicated_referent_identification"]["preview_db"]["profile_match_check"]["matches_profile"])
    print("LEGACY_11_ROW_TABLE")
    for r in ca_v0_2["legacy_11_row_annotation_table"]:
        print(
            r["symbol"],
            r["prior_bar_date"],
            r["prior_close"],
            r["event_bar_date"],
            r["event_close"],
            r["exact_ratio"],
            r["nearest_simple_ratio"],
            r["deviation_from_nearest"],
            r["suspected_action"],
        )
    print("DEFECT_ROOT_CAUSE", ca_v0_2["defect_diagnosis"]["root_cause"])
    print("CAPITAL_DECREASE_COUNT_V0_2", ca_v0_2["capital_decrease_count"])
    print("MISSING_4_EVENTS_WITH_CLASS")
    for r in ca_v0_2["missing_from_legacy_events_with_disposition_class"]:
        print(r["symbol"], r["event_bar_date"], r["disposition_class"])
    print("SEAL_FINAL_COUNTS", json.dumps(seal["final_disposition_counts"], sort_keys=True))
    print("SEAL_MASKED_INTERVAL_COUNT", seal["masked_interval_count"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

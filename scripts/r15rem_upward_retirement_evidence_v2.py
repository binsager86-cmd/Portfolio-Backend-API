from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from statistics import median
from typing import Any

from r14c_invalidation_rule_candidates_v1 import load_symbol_bars

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
SEALED_EXAM_DB = REVIEW / "r15_exam_v1_harness.db"
OUT_JSON = REVIEW / "r15rem_upward_retirement_evidence_v2.json"
OUT_MD = REVIEW / "r15rem_upward_retirement_evidence_v2.md"
OUT_SHA = REVIEW / "r15rem_upward_retirement_evidence_v2.sha256"

THRESHOLDS = [0.15, 0.20, 0.25]
R15_WINDOWS = {
    "SANAM": {"start": "2021-01-06", "end": "2026-07-09"},
    "TIJARA": {"start": "2021-07-11", "end": "2026-07-09"},
    "BPCC": {"start": "2021-07-11", "end": "2026-07-09"},
    "ZAIN": {"start": "2021-07-11", "end": "2026-07-09"},
    "MABANEE": {"start": "2021-07-11", "end": "2026-07-09"},
}
WIDTH_CAP = 0.24
RANGE_SESSIONS = 20
MIN_DWELL = 10
RETIREMENT_HORIZON = 120
REFORMATION_HORIZON = 120


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "median": median(values) if values else None,
        "p75": quantile(values, 0.75),
        "max": max(values) if values else None,
    }


def filtered_bars(symbol: str) -> list[dict[str, Any]]:
    bars = load_symbol_bars(symbol)
    window = R15_WINDOWS.get(symbol)
    if window is None:
        return bars
    return [row for row in bars if window["start"] <= str(row["trade_date"]) <= window["end"]]


def freeze_candidates(symbol: str, bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, day in enumerate(bars):
        if idx + 1 < MIN_DWELL:
            continue
        window = bars[max(0, idx + 1 - RANGE_SESSIONS) : idx + 1]
        high_ref = max(float(r["high"] or 0.0) for r in window)
        low_ref = min(float(r["low"] or 0.0) for r in window)
        close_px = float(day["close"] or 0.0)
        width = 0.0 if low_ref <= 0.0 else (high_ref - low_ref) / low_ref
        if width <= WIDTH_CAP and low_ref <= close_px <= high_ref:
            out.append(
                {
                    "symbol": symbol,
                    "freeze_index": idx,
                    "freeze_date": str(day["trade_date"]),
                    "base_high_ref": high_ref,
                    "base_low_ref": low_ref,
                    "width_pct": width,
                }
            )
    return out


def load_early_execution_rows() -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{SEALED_EXAM_DB.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT prediction_id, symbol, prediction_date, execution_state, reference_price,
                   base_reference, predicate_snapshot_json
            FROM ee_v2_forward_predictions
            WHERE execution_state = 'EXECUTE_EARLY_PILOT'
            ORDER BY symbol ASC, prediction_date ASC, prediction_id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    for row in rows:
        base_reference = json.loads(str(row["base_reference"]))
        snapshot = json.loads(str(row["predicate_snapshot_json"]))
        candidate = dict(snapshot.get("candidate_intent") or {})
        out.append(
            {
                "prediction_id": str(row["prediction_id"]),
                "symbol": str(row["symbol"]),
                "prediction_date": str(row["prediction_date"]),
                "execution_state": str(row["execution_state"]),
                "reference_price": float(row["reference_price"] or 0.0),
                "base_reference_id": str(base_reference.get("base_reference_id") or ""),
                "base_origin_date": str(base_reference.get("base_origin_date") or ""),
                "base_high_ref": float(base_reference.get("base_high_ref") or 0.0),
                "base_low_ref": float(base_reference.get("base_low_ref") or 0.0),
                "extension_pct_vs_current_valid_reference": float(candidate.get("extension_pct_vs_current_valid_reference") or 0.0),
            }
        )
    return out


def materialization_for_base(base: dict[str, Any], bars: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    origin_date = str(base["base_origin_date"])
    base_high = float(base["base_high_ref"] or 0.0)
    origin_idx = next((idx for idx, row in enumerate(bars) if str(row["trade_date"]) == origin_date), None)
    if origin_idx is None or base_high <= 0.0:
        return {"materialized": False, "materialization_date": None, "sessions_to_materialization": None, "max_mfe_120": None, "origin_index": origin_idx}
    max_mfe = 0.0
    for offset, row in enumerate(bars[origin_idx + 1 : origin_idx + 1 + RETIREMENT_HORIZON], start=1):
        mfe = (float(row["high"] or 0.0) / base_high) - 1.0
        max_mfe = max(max_mfe, mfe)
        if mfe >= threshold:
            return {
                "materialized": True,
                "materialization_date": str(row["trade_date"]),
                "sessions_to_materialization": offset,
                "max_mfe_120": max_mfe,
                "origin_index": origin_idx,
            }
    return {"materialized": False, "materialization_date": None, "sessions_to_materialization": None, "max_mfe_120": max_mfe, "origin_index": origin_idx}


def reformation_for_base(base: dict[str, Any], materialization: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    materialization_date = materialization.get("materialization_date")
    sessions_to_materialization = materialization.get("sessions_to_materialization")
    if not materialization_date or sessions_to_materialization is None:
        return {"reformed_strict": False, "reformation_date": None, "sessions_to_reformation": None}
    retired_high = float(base["base_high_ref"] or 0.0)
    dated_candidates = [row for row in candidates if str(row["freeze_date"]) > str(materialization_date)]
    origin_index = materialization.get("origin_index")
    materialized_index = int(origin_index) + int(sessions_to_materialization) if origin_index is not None else None
    for candidate in dated_candidates:
        if float(candidate["base_low_ref"] or 0.0) <= retired_high:
            continue
        if materialized_index is not None and int(candidate["freeze_index"]) - materialized_index > REFORMATION_HORIZON:
            continue
        return {
            "reformed_strict": True,
            "reformation_date": str(candidate["freeze_date"]),
            "sessions_to_reformation": None if materialized_index is None else int(candidate["freeze_index"]) - materialized_index,
            "new_base_high_ref": float(candidate["base_high_ref"]),
            "new_base_low_ref": float(candidate["base_low_ref"]),
        }
    return {"reformed_strict": False, "reformation_date": None, "sessions_to_reformation": None}


def criterion_iii_options() -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{SEALED_EXAM_DB.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        counts = {
            f"{row['event_type']}::{row['execution_state']}": int(row["c"])
            for row in conn.execute(
                """
                SELECT event_type, execution_state, COUNT(*) AS c
                FROM ee_v2_forward_predictions
                WHERE symbol = 'TIJARA'
                GROUP BY event_type, execution_state
                """
            )
        }
    finally:
        conn.close()
    total_days = len(filtered_bars("TIJARA"))
    execution_rows = counts.get("EXECUTION::EXECUTE_EARLY_PILOT", 0)
    suppression_rows = counts.get("SUPPRESSION_RESTRAINT::NONE", 0)
    veto_rows = counts.get("VETO_RESTRAINT::NONE", 0)
    intent_formed_days = execution_rows + suppression_rows
    prediction_rows = execution_rows + suppression_rows + veto_rows
    option_a_no_candidate = total_days - intent_formed_days
    option_b_no_record = total_days - prediction_rows
    return {
        "total_days": total_days,
        "execution_rows": execution_rows,
        "suppression_rows": suppression_rows,
        "veto_rows": veto_rows,
        "prediction_rows": prediction_rows,
        "option_a_candidate_state_strict": {
            "wording": "TIJARA no-candidate means candidate_intent.intent_state != INTENT_FORMED; veto days remain no-candidate days.",
            "no_candidate_count": option_a_no_candidate,
            "candidate_intent_days": intent_formed_days,
            "no_candidate_rate": option_a_no_candidate / total_days if total_days else None,
            "verdict_under_25pct_rule": "PASS" if total_days and option_a_no_candidate / total_days < 0.25 else "FAIL",
        },
        "option_b_explicit_disposition_coverage": {
            "wording": "TIJARA no-candidate means no sealed forward-prediction disposition row; veto rows count as explicit non-entry dispositions.",
            "no_record_count": option_b_no_record,
            "explicit_disposition_rows": prediction_rows,
            "no_record_rate": option_b_no_record / total_days if total_days else None,
            "verdict_under_25pct_rule": "PASS" if total_days and option_b_no_record / total_days < 0.25 else "FAIL",
        },
    }


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    early_rows = load_early_execution_rows()
    bars_by_symbol = {symbol: filtered_bars(symbol) for symbol in R15_WINDOWS}
    candidates_by_symbol = {symbol: freeze_candidates(symbol, bars) for symbol, bars in bars_by_symbol.items()}
    unique_bases = {
        (row["symbol"], row["base_reference_id"]): {
            "symbol": row["symbol"],
            "base_reference_id": row["base_reference_id"],
            "base_origin_date": row["base_origin_date"],
            "base_high_ref": row["base_high_ref"],
            "base_low_ref": row["base_low_ref"],
        }
        for row in early_rows
        if row["base_reference_id"]
    }

    threshold_payload: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        materialization_by_base = {
            key: materialization_for_base(base, bars_by_symbol.get(base["symbol"], []), threshold)
            for key, base in unique_bases.items()
        }
        suppressed_rows = []
        unsuppressed_rows = []
        for row in early_rows:
            key = (row["symbol"], row["base_reference_id"])
            materialization = materialization_by_base.get(key, {})
            suppressed = bool(materialization.get("materialized")) and str(materialization.get("materialization_date")) <= row["prediction_date"]
            target = suppressed_rows if suppressed else unsuppressed_rows
            target.append({**row, "upward_materialization": materialization})

        upward_bases = [
            {**unique_bases[key], **materialization_by_base[key]}
            for key in unique_bases
            if materialization_by_base[key].get("materialized")
        ]
        reformations = []
        for base in upward_bases:
            key = (base["symbol"], base["base_reference_id"])
            reformations.append({**base, **reformation_for_base(base, materialization_by_base[key], candidates_by_symbol.get(base["symbol"], []))})
        reformed = [row for row in reformations if row.get("reformed_strict")]
        sessions_to_reformation = [int(row["sessions_to_reformation"]) for row in reformed if row.get("sessions_to_reformation") is not None]

        before_extensions = [float(row["extension_pct_vs_current_valid_reference"]) for row in early_rows]
        after_extensions = [float(row["extension_pct_vs_current_valid_reference"]) for row in unsuppressed_rows]
        threshold_payload[str(threshold)] = {
            "early_execution_rows": len(early_rows),
            "would_suppress_count": len(suppressed_rows),
            "would_suppress_share": len(suppressed_rows) / len(early_rows) if early_rows else None,
            "remaining_after_suppression_count": len(unsuppressed_rows),
            "extension_distribution_before": distribution(before_extensions),
            "extension_distribution_after": distribution(after_extensions),
            "unique_early_bases": len(unique_bases),
            "upward_retired_base_count": len(upward_bases),
            "reformation_strict_definition": "new valid base candidate with base_low_ref > retired base_high_ref within 120 sessions after upward materialization",
            "reformed_above_retired_high_count": len(reformed),
            "reformed_above_retired_high_share": len(reformed) / len(upward_bases) if upward_bases else None,
            "median_sessions_to_reformation": median(sessions_to_reformation) if sessions_to_reformation else None,
            "suppressed_by_symbol": dict(sorted({symbol: sum(1 for row in suppressed_rows if row["symbol"] == symbol) for symbol in R15_WINDOWS}.items())),
            "suppressed_sample_rows": suppressed_rows[:20],
            "reformation_sample_rows": reformations[:20],
        }

    payload = {
        "version_id": "R15REM_UPWARD_RETIREMENT_EVIDENCE_V2",
        "mode": "READ_ONLY_SEALED_R15_EXAM_DB_PLUS_RUNTIME_OHLCV_LOOKUP",
        "sealed_exam_db": str(SEALED_EXAM_DB),
        "exam_artifact_mutation": "NONE",
        "thresholds": THRESHOLDS,
        "early_execution_row_definition": "ee_v2_forward_predictions.execution_state == EXECUTE_EARLY_PILOT",
        "retirement_rule": "base retires when high/base_high_ref - 1 >= threshold within 120 sessions after base_origin_date; row suppressed when materialization_date <= intent date",
        "r15_windows": R15_WINDOWS,
        "summary_by_threshold": threshold_payload,
        "criterion_iii_options": criterion_iii_options(),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = ["# R15-REM Upward Retirement Evidence v2", "", "Mode: READ_ONLY_SEALED_R15_EXAM_DB_PLUS_RUNTIME_OHLCV_LOOKUP", "", "Exam artifact mutation: NONE", ""]
    for threshold in THRESHOLDS:
        row = threshold_payload[str(threshold)]
        lines.extend(
            [
                f"## Threshold {threshold:.2f}",
                "",
                f"Early execution rows: {row['early_execution_rows']}",
                f"Would suppress: {row['would_suppress_count']} ({row['would_suppress_share']})",
                f"Extension before: {json.dumps(row['extension_distribution_before'], sort_keys=True)}",
                f"Extension after: {json.dumps(row['extension_distribution_after'], sort_keys=True)}",
                f"Upward-retired unique early bases: {row['upward_retired_base_count']} / {row['unique_early_bases']}",
                f"Reformed above retired high: {row['reformed_above_retired_high_count']} ({row['reformed_above_retired_high_share']}); median sessions: {row['median_sessions_to_reformation']}",
                f"Suppressed by symbol: {json.dumps(row['suppressed_by_symbol'], sort_keys=True)}",
                "",
            ]
        )
    lines.extend(["## Criterion iii Options", "", json.dumps(payload["criterion_iii_options"], indent=2, sort_keys=True), ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_SHA.write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in (OUT_JSON, OUT_MD)), encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON), "md": str(OUT_MD), "summary_by_threshold": threshold_payload, "criterion_iii_options": payload["criterion_iii_options"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
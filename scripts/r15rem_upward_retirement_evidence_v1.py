from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from r14c_invalidation_rule_candidates_v1 import load_ex_set_b_symbols, load_symbol_bars

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
OUT_JSON = REVIEW / "r15rem_upward_retirement_evidence_v1.json"
OUT_MD = REVIEW / "r15rem_upward_retirement_evidence_v1.md"
OUT_SHA = REVIEW / "r15rem_upward_retirement_evidence_v1.sha256"

THRESHOLDS = [0.15, 0.20, 0.25]
WIDTH_CAP = 0.24
RANGE_SESSIONS = 20
MIN_DWELL = 10
HORIZON = 120


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


def freeze_candidates(symbol: str, bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    last_freeze_index = -HORIZON
    for idx, day in enumerate(bars):
        if idx + 1 < MIN_DWELL or idx - last_freeze_index < HORIZON:
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
                    "freeze_date": day["trade_date"],
                    "base_high_ref": high_ref,
                    "base_low_ref": low_ref,
                    "width_pct": width,
                }
            )
            last_freeze_index = idx
    return out


def annotate_candidate(candidate: dict[str, Any], bars: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    start = int(candidate["freeze_index"]) + 1
    horizon_bars = bars[start : start + HORIZON]
    base_high = float(candidate["base_high_ref"] or 0.0)
    best_mfe = 0.0
    materialized_index: int | None = None
    materialized_date: str | None = None
    for offset, day in enumerate(horizon_bars, start=1):
        high_px = float(day["high"] or 0.0)
        mfe = 0.0 if base_high <= 0.0 else (high_px / base_high) - 1.0
        if mfe > best_mfe:
            best_mfe = mfe
        if materialized_index is None and mfe >= threshold:
            materialized_index = offset
            materialized_date = str(day["trade_date"])
    return {
        **candidate,
        "threshold": threshold,
        "horizon_sessions_available": len(horizon_bars),
        "max_mfe_120": best_mfe,
        "materialized": materialized_index is not None,
        "sessions_to_materialization": materialized_index,
        "materialization_date": materialized_date,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    materialized = [r for r in rows if r["materialized"]]
    mfe_values = [float(r["max_mfe_120"] or 0.0) for r in rows]
    session_values = [int(r["sessions_to_materialization"]) for r in materialized if r["sessions_to_materialization"] is not None]
    return {
        "candidate_count": len(rows),
        "materialized_count": len(materialized),
        "materialized_rate": 0.0 if not rows else len(materialized) / len(rows),
        "max_mfe_120_mean": mean(mfe_values) if mfe_values else None,
        "max_mfe_120_median": median(mfe_values) if mfe_values else None,
        "max_mfe_120_p75": quantile(mfe_values, 0.75),
        "sessions_to_materialization_median": median(session_values) if session_values else None,
    }


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    ex_set_b, set_a, set_b = load_ex_set_b_symbols()
    rows_by_threshold: dict[str, list[dict[str, Any]]] = {str(t): [] for t in THRESHOLDS}
    symbol_count = 0
    for symbol in ex_set_b:
        bars = load_symbol_bars(symbol)
        if len(bars) < MIN_DWELL + 1:
            continue
        symbol_count += 1
        candidates = freeze_candidates(symbol, bars)
        for threshold in THRESHOLDS:
            rows_by_threshold[str(threshold)].extend(annotate_candidate(c, bars, threshold) for c in candidates)

    payload = {
        "version_id": "R15REM_UPWARD_RETIREMENT_EVIDENCE_V1",
        "mode": "READ_ONLY_EX_SET_B_DESCRIPTIVE",
        "thresholds": THRESHOLDS,
        "cohort": {
            "ex_set_b_symbol_count": len(ex_set_b),
            "symbols_with_bars": symbol_count,
            "set_a_symbols": set_a,
            "set_b_excluded_count": len(set_b),
        },
        "candidate_definition": {
            "range_sessions": RANGE_SESSIONS,
            "min_dwell": MIN_DWELL,
            "width_cap": WIDTH_CAP,
            "horizon_sessions": HORIZON,
            "note": "Descriptive threshold evidence only; not R15 attempt 2 and not owner ratification.",
        },
        "summary_by_threshold": {k: summarize(v) for k, v in rows_by_threshold.items()},
        "sample_rows_by_threshold": {k: v[:25] for k, v in rows_by_threshold.items()},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# R15-REM Upward Retirement Evidence v1", "", "Mode: READ_ONLY_EX_SET_B_DESCRIPTIVE", ""]
    for threshold in THRESHOLDS:
        summary = payload["summary_by_threshold"][str(threshold)]
        lines.append(f"## Threshold {threshold:.2f}")
        lines.append("")
        lines.append(json.dumps(summary, sort_keys=True))
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    hashes = []
    for path in (OUT_JSON, OUT_MD):
        hashes.append(f"{sha256_file(path)}  {path.name}")
    OUT_SHA.write_text("\n".join(hashes), encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON), "md": str(OUT_MD), "summary_by_threshold": payload["summary_by_threshold"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
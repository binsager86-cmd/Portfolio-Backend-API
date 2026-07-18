from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
RUNTIME_DB = REVIEW / "r12_exam_surface_v4_5_runtime.db"
SET_MEMBERSHIP_FILE = REVIEW / "r13_gate_conflict_analysis_v1_2.json"

PROVISIONAL_TAG = "PROVISIONAL_PENDING_PARAMETER_GATE"
OWNER_WINDOW_START = "2025-05-01"
OWNER_WINDOW_END = "2025-05-31"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def base_symbol(symbol: str) -> str:
    return symbol.split("__SEG", 1)[0].upper()


def load_ex_set_b_symbols() -> tuple[list[str], list[str], list[str]]:
    conf = read_json(SET_MEMBERSHIP_FILE)
    set_a = [str(s).upper() for s in conf.get("set_membership", {}).get("set_a", [])]
    set_b = {str(s).upper() for s in conf.get("set_membership", {}).get("set_b", [])}

    conn = sqlite3.connect(str(RUNTIME_DB))
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT
              CASE WHEN instr(symbol, '__SEG') > 0 THEN substr(symbol, 1, instr(symbol, '__SEG') - 1) ELSE symbol END AS s
            FROM ee_ohlcv
            ORDER BY s
            """
        ).fetchall()
        all_symbols = [str(r[0]).upper() for r in rows]
    finally:
        conn.close()

    ex_set_b = [s for s in all_symbols if s not in set_b]
    return sorted(ex_set_b), sorted(set_a), sorted(set_b)


def load_symbol_bars(symbol: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(RUNTIME_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
              date(trade_date, 'unixepoch') AS trade_date,
              open,
              high,
              low,
              close,
              volume,
              value_kwd
            FROM ee_ohlcv
            WHERE symbol LIKE ?
            ORDER BY trade_date ASC
            """,
            (f"{symbol}%",),
        ).fetchall()
        return [
            {
                "trade_date": str(r["trade_date"]),
                "open": float(r["open"] or 0.0),
                "high": float(r["high"] or 0.0),
                "low": float(r["low"] or 0.0),
                "close": float(r["close"] or 0.0),
                "volume": float(r["volume"] or 0.0),
                "value_kwd": float(r["value_kwd"] or 0.0),
            }
            for r in rows
        ]
    finally:
        conn.close()


def pct_rank(values: list[float], x: float) -> float:
    if not values:
        return 0.0
    c = sum(1 for v in values if v <= x)
    return float(c) / float(len(values))


def evaluate_invalidation(
    candidate_id: str,
    params: dict[str, Any],
    rule_state: dict[str, Any],
    *,
    close_px: float,
    base_low: float,
    base_high: float,
    atr_value: float,
    vol_pctile: float,
    flow_progress: bool,
) -> tuple[bool, str, dict[str, Any]]:
    state = dict(rule_state)

    if candidate_id == "CANDIDATE_A_CLOSE_BELOW_LOW_N":
        n = max(1, int(params.get("n_sessions") or 1))
        streak = int(state.get("streak") or 0)
        streak = streak + 1 if close_px < base_low else 0
        state["streak"] = streak
        return streak >= n, f"close_below_low_n(n={n})", state

    if candidate_id == "CANDIDATE_B_CLOSE_BELOW_LOW_BY_ATR_X_N":
        atr_mult = float(params.get("atr_mult") or 1.0)
        n = max(1, int(params.get("n_sessions") or 1))
        threshold = base_low - atr_mult * max(0.0, atr_value)
        streak = int(state.get("streak") or 0)
        streak = streak + 1 if close_px < threshold else 0
        state["streak"] = streak
        state["threshold"] = threshold
        return streak >= n, f"close_below_low_by_atr_x_n(atr_mult={atr_mult},n={n})", state

    if candidate_id == "CANDIDATE_C_VOL_BREAK_AND_STALE_FLOW_DECAY":
        vol_break_pctile = float(params.get("vol_break_pctile") or 0.9)
        min_age_sessions = max(1, int(params.get("min_age_sessions") or 40))
        flow_decay_n = max(1, int(params.get("flow_decay_n") or 8))
        age = int(state.get("age_sessions") or 0) + 1
        flow_streak = int(state.get("flow_decay_streak") or 0)
        flow_streak = flow_streak + 1 if not flow_progress else 0
        state["age_sessions"] = age
        state["flow_decay_streak"] = flow_streak
        retire = age >= min_age_sessions and flow_streak >= flow_decay_n and vol_pctile >= vol_break_pctile and close_px < base_low
        return retire, (
            "vol_break_and_stale_flow_decay("
            f"vol_break_pctile={vol_break_pctile},min_age_sessions={min_age_sessions},flow_decay_n={flow_decay_n})"
        ), state

    raise ValueError(f"Unknown candidate_id: {candidate_id}")


def simulate_symbol(symbol: str, rows: list[dict[str, Any]], candidate_id: str, params: dict[str, Any]) -> dict[str, Any]:
    base_min_sessions = 10
    base_max_width_pct = 0.24
    range_sessions = 20

    history: list[dict[str, Any]] = []
    tr_history: deque[float] = deque(maxlen=252)
    vol_history: deque[float] = deque(maxlen=252)

    base_ref: dict[str, Any] | None = None
    freeze_count = 0
    retire_count = 0
    ratchet_count = 0
    valid_days = 0
    retired_days = 0
    no_base_days = 0

    owner_window_valid_days = 0
    owner_window_retired_days = 0

    prev_close: float | None = None

    for row in rows:
        history.append(row)
        if len(history) > 300:
            history = history[-300:]

        close_px = float(row["close"])
        high_px = float(row["high"])
        low_px = float(row["low"])
        date_txt = str(row["trade_date"])

        if prev_close is None:
            tr = high_px - low_px
        else:
            tr = max(high_px - low_px, abs(high_px - prev_close), abs(low_px - prev_close))
        tr_history.append(max(0.0, tr))
        atr_value = sum(list(tr_history)[-14:]) / max(1, len(list(tr_history)[-14:]))

        range_pct = (high_px - low_px) / max(close_px, 1e-9)
        vol_history.append(range_pct)
        vol_pctile = pct_rank(list(vol_history), range_pct)

        window = history[-range_sessions:]
        highs = [float(x["high"]) for x in window]
        lows = [float(x["low"]) for x in window]
        hi = max(highs) if highs else high_px
        lo = min(lows) if lows else low_px
        width_pct = 0.0 if lo <= 0 else (hi - lo) / lo

        if base_ref is None:
            freeze_ok = len(history) >= base_min_sessions and width_pct <= base_max_width_pct and lo <= close_px <= hi
            if freeze_ok:
                base_ref = {
                    "base_high": hi,
                    "base_low": lo,
                    "valid": True,
                    "retired_reason": "NONE",
                    "rule_state": {},
                }
                freeze_count += 1
            else:
                no_base_days += 1
        else:
            if base_ref.get("valid"):
                flow_progress = prev_close is not None and close_px > prev_close
                if flow_progress and close_px > float(base_ref.get("base_high") or 0.0):
                    base_ref["base_high"] = close_px
                    ratchet_count += 1

                retire, reason, next_state = evaluate_invalidation(
                    candidate_id,
                    params,
                    dict(base_ref.get("rule_state") or {}),
                    close_px=close_px,
                    base_low=float(base_ref.get("base_low") or 0.0),
                    base_high=float(base_ref.get("base_high") or 0.0),
                    atr_value=atr_value,
                    vol_pctile=vol_pctile,
                    flow_progress=flow_progress,
                )
                base_ref["rule_state"] = next_state
                if retire:
                    base_ref["valid"] = False
                    base_ref["retired_reason"] = reason
                    retire_count += 1

            if base_ref.get("valid"):
                valid_days += 1
                if OWNER_WINDOW_START <= date_txt <= OWNER_WINDOW_END:
                    owner_window_valid_days += 1
            else:
                retired_days += 1
                if OWNER_WINDOW_START <= date_txt <= OWNER_WINDOW_END:
                    owner_window_retired_days += 1

        prev_close = close_px

    final_state = "NO_BASE"
    final_reason = "NONE"
    if base_ref is not None:
        final_state = "VALID" if base_ref.get("valid") else "RETIRED"
        final_reason = str(base_ref.get("retired_reason") or "NONE")

    return {
        "symbol": symbol,
        "bar_count": len(rows),
        "freeze_count": freeze_count,
        "retire_count": retire_count,
        "ratchet_count": ratchet_count,
        "valid_days": valid_days,
        "retired_days": retired_days,
        "no_base_days": no_base_days,
        "final_state": final_state,
        "final_retire_reason": final_reason,
        "owner_window": {
            "start": OWNER_WINDOW_START,
            "end": OWNER_WINDOW_END,
            "valid_days": owner_window_valid_days,
            "retired_days": owner_window_retired_days,
        },
    }


def aggregate(stats: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(stats)
    if n == 0:
        return {"symbol_count": 0}
    return {
        "symbol_count": n,
        "symbols_with_freeze": sum(1 for s in stats if s["freeze_count"] > 0),
        "symbols_with_retire": sum(1 for s in stats if s["retire_count"] > 0),
        "symbols_surviving_valid": sum(1 for s in stats if s["final_state"] == "VALID"),
        "total_freeze_events": sum(int(s["freeze_count"]) for s in stats),
        "total_retire_events": sum(int(s["retire_count"]) for s in stats),
        "total_ratchet_events": sum(int(s["ratchet_count"]) for s in stats),
    }


def main() -> None:
    out_json = REVIEW / "r14c_invalidation_rule_candidates_v1.json"
    out_md = REVIEW / "r14c_invalidation_rule_candidates_v1.md"

    ex_set_b_symbols, set_a_symbols, set_b_symbols = load_ex_set_b_symbols()
    bars_map = {s: load_symbol_bars(s) for s in ex_set_b_symbols}

    candidates = [
        {
            "candidate_id": "CANDIDATE_A_CLOSE_BELOW_LOW_N",
            "principle_rationale": "Retire only after a sustained close breach below frozen base_low; simplest, transparent, and auditable.",
            "named_parameters": ["base_invalidation_n_sessions"],
            "parameter_grid": [
                {"n_sessions": 1},
                {"n_sessions": 2},
            ],
        },
        {
            "candidate_id": "CANDIDATE_B_CLOSE_BELOW_LOW_BY_ATR_X_N",
            "principle_rationale": "Require breach distance scaled by ATR and persistence, adapting retirement to local volatility.",
            "named_parameters": ["base_invalidation_atr_mult", "base_invalidation_n_sessions"],
            "parameter_grid": [
                {"atr_mult": 0.5, "n_sessions": 1},
                {"atr_mult": 1.0, "n_sessions": 2},
            ],
        },
        {
            "candidate_id": "CANDIDATE_C_VOL_BREAK_AND_STALE_FLOW_DECAY",
            "principle_rationale": "Retire stale bases only when volatility regime breaks and flow stays weak, reducing single-day noise reactions.",
            "named_parameters": ["base_vol_break_pctile", "base_stale_min_age_sessions", "base_flow_decay_n"],
            "parameter_grid": [
                {"vol_break_pctile": 0.9, "min_age_sessions": 40, "flow_decay_n": 8},
                {"vol_break_pctile": 0.85, "min_age_sessions": 60, "flow_decay_n": 10},
            ],
        },
    ]

    candidate_results: list[dict[str, Any]] = []
    for cand in candidates:
        runs: list[dict[str, Any]] = []
        for params in cand["parameter_grid"]:
            per_symbol = [
                simulate_symbol(symbol=s, rows=bars_map[s], candidate_id=cand["candidate_id"], params=params)
                for s in ex_set_b_symbols
                if bars_map[s]
            ]
            runs.append(
                {
                    "provisional_parameter_values": params,
                    "provisional_status": PROVISIONAL_TAG,
                    "aggregate": aggregate(per_symbol),
                    "per_symbol_statistics": per_symbol,
                }
            )

        candidate_results.append(
            {
                "candidate_id": cand["candidate_id"],
                "principle_rationale": cand["principle_rationale"],
                "named_parameters": cand["named_parameters"],
                "runs": runs,
            }
        )

    payload = {
        "version_id": "R14C_INVALIDATION_RULE_CANDIDATES_V1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "EX_SET_B full evidence base (all symbols except Set B; Set A included within this scope)",
        "set_membership": {
            "set_a": set_a_symbols,
            "set_b": set_b_symbols,
            "ex_set_b_symbol_count": len(ex_set_b_symbols),
        },
        "selection_status": "NO_CANDIDATE_SELECTED_IN_THIS_ARTIFACT",
        "explicit_non_optimization_statement": (
            "No invalidation rule form is chosen by optimizing SANAM May-2025 owner window. "
            "SANAM owner-window rows are reported descriptively alongside every other symbol."
        ),
        "candidates": candidate_results,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines: list[str] = [
        "# R14-C Invalidation Rule Candidates v1",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "No candidate is selected in this artifact. Selection is deferred to parameter gate + owner ratification.",
        "",
        payload["explicit_non_optimization_statement"],
        "",
        f"EX_SET_B symbol count: {len(ex_set_b_symbols)}",
        f"Set A symbols included: {', '.join(set_a_symbols)}",
        f"Set B excluded: {', '.join(set_b_symbols)}",
        "",
    ]

    for c in candidate_results:
        lines.append(f"## {c['candidate_id']}")
        lines.append("")
        lines.append(f"Rationale: {c['principle_rationale']}")
        lines.append("")
        lines.append(f"Named parameters: {', '.join(c['named_parameters'])}")
        lines.append("")
        for run in c["runs"]:
            lines.append(f"### Provisional parameter values: {json.dumps(run['provisional_parameter_values'], sort_keys=True)}")
            lines.append("")
            agg = run["aggregate"]
            lines.append(
                "Aggregate behavior: "
                f"symbol_count={agg['symbol_count']}, symbols_with_freeze={agg['symbols_with_freeze']}, "
                f"symbols_with_retire={agg['symbols_with_retire']}, symbols_surviving_valid={agg['symbols_surviving_valid']}, "
                f"total_freeze_events={agg['total_freeze_events']}, total_retire_events={agg['total_retire_events']}, total_ratchet_events={agg['total_ratchet_events']}"
            )
            lines.append("")
            lines.append("Per-symbol descriptive statistics:")
            lines.append("")
            lines.append("| Symbol | Freeze | Retire | Ratchet | Final | OwnerWindowValidDays | OwnerWindowRetiredDays |")
            lines.append("|---|---:|---:|---:|---|---:|---:|")
            for s in run["per_symbol_statistics"]:
                ow = s["owner_window"]
                lines.append(
                    f"| {s['symbol']} | {s['freeze_count']} | {s['retire_count']} | {s['ratchet_count']} | {s['final_state']} | {ow['valid_days']} | {ow['retired_days']} |"
                )
            lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("R14C_INVALIDATION_RULE_CANDIDATES_V1_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()

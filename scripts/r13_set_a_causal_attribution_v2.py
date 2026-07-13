from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iso_to_ts(iso_date: str) -> int:
    return int(datetime.strptime(iso_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def ts_to_iso(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def sqlite_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def base_symbol_sql() -> str:
    return "(CASE WHEN instr(symbol, '__SEG') > 0 THEN substr(symbol, 1, instr(symbol, '__SEG') - 1) ELSE symbol END)"


def get_engine_cfg(cur: sqlite3.Cursor) -> dict[str, Any]:
    keys = [
        "base_min_sessions",
        "base_max_width_pct",
        "volume_breakout_mult",
        "rsi_regime",
        "adx_trigger",
        "cmf_floor",
        "atr_squeeze_pctile",
        "trend_join_window",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        row = cur.execute("SELECT value_json FROM ee_engine_config WHERE key = ?", (k,)).fetchone()
        out[k] = None if row is None else json.loads(row[0])
    return out


def indicators_on_day(cur: sqlite3.Cursor, symbol: str, day_ts: int) -> dict[str, Any] | None:
    row = cur.execute(
        f"SELECT payload_json FROM ee_indicators WHERE {base_symbol_sql()} = ? AND trade_date = ? ORDER BY trade_date DESC LIMIT 1",
        (symbol, day_ts),
    ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row[0]) if row[0] else {}
    except Exception:
        return {}


def signals_for_symbol(cur: sqlite3.Cursor, symbol: str) -> list[dict[str, Any]]:
    rows = cur.execute(
        f"""
        SELECT trade_date, signal_type, phase_from, phase_to, evidence_json
        FROM ee_signals
        WHERE {base_symbol_sql()} = ?
        ORDER BY trade_date, id
        """,
        (symbol,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for td, st, pf, pt, ev in rows:
        evj = json.loads(ev) if ev else {}
        out.append(
            {
                "trade_date": int(td),
                "signal_type": str(st),
                "phase_from": None if pf is None else str(pf),
                "phase_to": None if pt is None else str(pt),
                "evidence": evj,
            }
        )
    return out


def trade_rows_for_symbol(cur: sqlite3.Cursor, symbol: str) -> list[dict[str, Any]]:
    rows = cur.execute(
        f"""
        SELECT opened_at, closed_at, net_return, exit_reason
        FROM ee_backtest_trades
        WHERE {base_symbol_sql()} = ?
        ORDER BY opened_at
        """,
        (symbol,),
    ).fetchall()
    return [
        {
            "opened_at": int(r[0]),
            "closed_at": int(r[1]),
            "net_return": float(r[2]),
            "exit_reason": str(r[3]),
        }
        for r in rows
    ]


def phase_at_or_before_day(signals: list[dict[str, Any]], day_ts: int) -> str | None:
    upto = [s for s in signals if int(s["trade_date"]) <= day_ts]
    if not upto:
        return None
    return upto[-1].get("phase_to")


def classify_day(
    symbol: str,
    day_ts: int,
    signals: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    masked_intervals: list[dict[str, Any]],
    indicator: dict[str, Any] | None,
) -> dict[str, Any]:
    masked_here = []
    for m in masked_intervals:
        start_ts = iso_to_ts(str(m["start_date"]))
        end_ts = iso_to_ts(str(m["end_date"]))
        if start_ts <= day_ts <= end_ts:
            masked_here.append(m)

    signal_on_day = [s for s in signals if int(s["trade_date"]) == day_ts]
    last_sig = None
    for s in signals:
        if int(s["trade_date"]) <= day_ts:
            last_sig = s
        else:
            break

    candidate_veto = None
    for s in signal_on_day:
        attempted = s.get("evidence", {}).get("attempted_signal_type")
        if s["signal_type"] == "SIGNAL_SUPPRESSED_RISK":
            candidate_veto = "RISK_SUPPRESSION"
            break
        if s["signal_type"] == "AVOID_SET" or s.get("phase_to") == "AVOID" or attempted == "AVOID_SET":
            candidate_veto = "AVOID_GATE"
            break

    candidate_passed = any(s["signal_type"] in {"BREAKOUT_CONFIRMED", "ACCUMULATION_ALERT"} for s in signal_on_day)
    opened_same_day = any(int(t["opened_at"]) == day_ts for t in trades)

    warmup_inferred = False
    warmup_basis = None
    if indicator is not None:
        if indicator.get("sma200") is None or indicator.get("range_high_60") is None or indicator.get("range_low_60") is None:
            warmup_inferred = True
            warmup_basis = "indicator_missing_warmup_fields"
    if not warmup_inferred and last_sig is not None and str(last_sig.get("signal_type")) == "PHASE_ONLY":
        if str(last_sig.get("evidence", {}).get("reason") or "") == "warmup_pending":
            warmup_inferred = True
            warmup_basis = "latest_runtime_signal_reason_warmup_pending"

    phase_pre = phase_at_or_before_day(signals, day_ts)

    if masked_here or warmup_inferred:
        cls = "MASKED_OR_WARMUP_EXCLUDED"
    elif candidate_veto is not None:
        cls = f"CANDIDATE_VETOED({candidate_veto})"
    elif candidate_passed and not opened_same_day:
        cls = "CANDIDATE_PASSED_NO_FILL"
    elif phase_pre is not None and phase_pre != "NEUTRAL":
        cls = "PHASE_PROGRESSED_NO_CANDIDATE"
    else:
        cls = "PHASE_MACHINE_NEVER_LEFT_NEUTRAL"

    return {
        "symbol": symbol,
        "trade_date": day_ts,
        "trade_date_iso": ts_to_iso(day_ts),
        "classification": cls,
        "phase_at_or_before_day": phase_pre,
        "signal_on_day_count": len(signal_on_day),
        "last_signal_at_or_before_day": None
        if last_sig is None
        else {
            "trade_date": last_sig["trade_date"],
            "trade_date_iso": ts_to_iso(int(last_sig["trade_date"])),
            "signal_type": last_sig["signal_type"],
            "phase_from": last_sig["phase_from"],
            "phase_to": last_sig["phase_to"],
            "attempted_signal_type": last_sig.get("evidence", {}).get("attempted_signal_type"),
            "suppressed_reason": last_sig.get("evidence", {}).get("suppressed_reason"),
            "reason": last_sig.get("evidence", {}).get("reason"),
        },
        "masked_interval_hits": masked_here,
        "warmup_inferred": warmup_inferred,
        "warmup_basis": warmup_basis,
    }


def scanner_line_map(scanner_path: Path) -> dict[str, dict[str, Any]]:
    lines = scanner_path.read_text(encoding="utf-8").splitlines()

    def find(snippet: str) -> int:
        for i, line in enumerate(lines, start=1):
            if snippet in line:
                return i
        return -1

    out = {
        "warmup_guard": {
            "line": find("if warmup_ready_date is None or trade_date < warmup_ready_date:"),
            "quote": "if warmup_ready_date is None or trade_date < warmup_ready_date:",
        },
        "base_forming_entry": {
            "line": find("width <= float(get_cfg(config, \"base_max_width_pct\"))"),
            "quote": "sma200 > 0 and ema30 > 0 and width <= base_max_width_pct and sessions_in_range >= base_min_sessions and base_low_60 <= close <= base_high_60",
        },
        "accumulation_gate": {
            "line": find("if ("),
            "quote": "accumulation_gate and cmf_hits >= 5 and squeeze_ok and (close >= ema30 or close >= 0.97 * sma200) and liquidity_ok and score >= 60",
        },
        "breakout_watch_trigger": {
            "line": find("near_base_with_build ="),
            "quote": "near_base_with_build = base_high_ref > 0 and close >= (0.97 * base_high_ref) and rv_hits >= 2",
        },
    }

    # Fix accumulation gate line by exact anchor.
    acc_line = find("accumulation_gate")
    if acc_line > 0:
        out["accumulation_gate"]["line"] = acc_line
    return out


def predicate_terms_for_day(
    symbol: str,
    day_ts: int,
    cls_row: dict[str, Any],
    indicator: dict[str, Any] | None,
    cfg: dict[str, Any],
    line_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    close = None if indicator is None else indicator.get("close")
    ema10 = None if indicator is None else indicator.get("ema10")
    ema30 = None if indicator is None else indicator.get("ema30")
    sma200 = None if indicator is None else indicator.get("sma200")
    width = None if indicator is None else indicator.get("range_width_pct")
    range_low_60 = None if indicator is None else indicator.get("range_low_60")
    range_high_60 = None if indicator is None else indicator.get("range_high_60")
    rel_volume = None if indicator is None else indicator.get("rel_volume")
    cmf_10 = None if indicator is None else indicator.get("cmf_10")
    bb_width = None if indicator is None else indicator.get("bb_width")
    atr_pct_pctile = None if indicator is None else indicator.get("atr_pct_percentile_252")

    warmup_short_circuit = bool(cls_row.get("warmup_inferred"))

    base_terms = [
        {
            "term": "sma200 > 0",
            "value": sma200,
            "threshold": "> 0",
            "recoverable": sma200 is not None,
            "passes": None if sma200 is None else bool(float(sma200) > 0.0),
        },
        {
            "term": "ema30 > 0",
            "value": ema30,
            "threshold": "> 0",
            "recoverable": ema30 is not None,
            "passes": None if ema30 is None else bool(float(ema30) > 0.0),
        },
        {
            "term": "width <= base_max_width_pct",
            "value": width,
            "threshold": cfg.get("base_max_width_pct"),
            "recoverable": width is not None and cfg.get("base_max_width_pct") is not None,
            "passes": None
            if width is None or cfg.get("base_max_width_pct") is None
            else bool(float(width) <= float(cfg["base_max_width_pct"])),
        },
        {
            "term": "sessions_in_range >= base_min_sessions",
            "value": None,
            "threshold": cfg.get("base_min_sessions"),
            "recoverable": False,
            "passes": None,
            "recoverability_note": "sessions_in_range not persisted as a runtime field for the target day",
        },
        {
            "term": "base_low_60 <= close <= base_high_60",
            "value": {"base_low_60": range_low_60, "close": close, "base_high_60": range_high_60},
            "threshold": "range inclusion",
            "recoverable": close is not None and range_low_60 is not None and range_high_60 is not None,
            "passes": None
            if close is None or range_low_60 is None or range_high_60 is None
            else bool(float(range_low_60) <= float(close) <= float(range_high_60)),
        },
    ]

    acc_terms = [
        {
            "term": "accumulation_gate",
            "value": {
                "accumulation_divergence": None if indicator is None else indicator.get("accumulation_divergence"),
                "price_slope_40": None if indicator is None else indicator.get("price_slope_40"),
                "obv_slope_40": None if indicator is None else indicator.get("obv_slope_40"),
                "anv_slope_40": None if indicator is None else indicator.get("anv_slope_40"),
            },
            "threshold": "coded OR expression",
            "recoverable": False,
            "passes": None,
            "recoverability_note": "composite gate requires slope fields that are null on the target-day runtime payload",
        },
        {
            "term": "cmf_hits >= 5 over last 10",
            "value": cmf_10,
            "threshold": cfg.get("cmf_floor"),
            "recoverable": False,
            "passes": None,
            "recoverability_note": "windowed cmf_hits count is not persisted for replay",
        },
        {
            "term": "squeeze_ok",
            "value": {"bb_width": bb_width, "atr_pct_percentile_252": atr_pct_pctile},
            "threshold": {"bb_width_max": 0.12, "atr_pct_percentile_252_max": cfg.get("atr_squeeze_pctile")},
            "recoverable": bb_width is not None or atr_pct_pctile is not None,
            "passes": None
            if bb_width is None and atr_pct_pctile is None
            else bool(
                (bb_width is not None and float(bb_width) <= 0.12)
                or (
                    atr_pct_pctile is not None
                    and cfg.get("atr_squeeze_pctile") is not None
                    and float(atr_pct_pctile) <= float(cfg["atr_squeeze_pctile"])
                )
            ),
        },
        {
            "term": "close >= ema30 or close >= 0.97*sma200",
            "value": {"close": close, "ema30": ema30, "sma200": sma200},
            "threshold": "coded OR expression",
            "recoverable": close is not None and ema30 is not None,
            "passes": None
            if close is None or ema30 is None
            else bool(
                float(close) >= float(ema30)
                or (sma200 is not None and float(close) >= (0.97 * float(sma200)))
            ),
        },
    ]

    watch_terms = [
        {
            "term": "base_high_ref > 0",
            "value": None,
            "threshold": "> 0",
            "recoverable": False,
            "passes": None,
            "recoverability_note": "state.base_high_ref is not persisted as a day-level series",
        },
        {
            "term": "close >= 0.97 * base_high_ref",
            "value": {"close": close, "base_high_ref": None},
            "threshold": "close >= 0.97*base_high_ref",
            "recoverable": False,
            "passes": None,
            "recoverability_note": "base_high_ref unavailable for target day",
        },
        {
            "term": "rv_hits >= 2 over last 5 (rel_volume >= 1.5)",
            "value": rel_volume,
            "threshold": {"rel_volume_min": 1.5, "count_min": 2},
            "recoverable": False,
            "passes": None,
            "recoverability_note": "rv_hits window count not persisted; only point-in-time rel_volume is available",
        },
    ]

    unmet_base = [t["term"] for t in base_terms if t.get("passes") is False or t.get("recoverable") is False]

    return {
        "symbol": symbol,
        "trade_date": day_ts,
        "trade_date_iso": ts_to_iso(day_ts),
        "runtime_phase_context": {
            "phase_at_or_before_day": cls_row.get("phase_at_or_before_day"),
            "warmup_short_circuit_inferred": warmup_short_circuit,
            "warmup_basis": cls_row.get("warmup_basis"),
        },
        "predicate_blocks": [
            {
                "name": "warmup_guard",
                "file": "app/services/eagle_eye/scanner_service.py",
                "line": line_map["warmup_guard"]["line"],
                "quote": line_map["warmup_guard"]["quote"],
                "in_scope": True,
                "evaluation": {
                    "warmup_short_circuit": warmup_short_circuit,
                    "effective_result": "SHORT_CIRCUIT_TRUE" if warmup_short_circuit else "SHORT_CIRCUIT_FALSE",
                    "unmet_term_when_true": "warmup_ready_date is None or trade_date < warmup_ready_date",
                },
            },
            {
                "name": "base_forming_entry",
                "file": "app/services/eagle_eye/scanner_service.py",
                "line": line_map["base_forming_entry"]["line"],
                "quote": line_map["base_forming_entry"]["quote"],
                "in_scope": not warmup_short_circuit,
                "evaluation": {
                    "terms": base_terms,
                    "unmet_terms": unmet_base,
                },
            },
            {
                "name": "accumulation_gate",
                "file": "app/services/eagle_eye/scanner_service.py",
                "line": line_map["accumulation_gate"]["line"],
                "quote": line_map["accumulation_gate"]["quote"],
                "in_scope": False if warmup_short_circuit else cls_row.get("phase_at_or_before_day") == "BASE_FORMING",
                "evaluation": {
                    "terms": acc_terms,
                },
            },
            {
                "name": "breakout_watch_trigger",
                "file": "app/services/eagle_eye/scanner_service.py",
                "line": line_map["breakout_watch_trigger"]["line"],
                "quote": line_map["breakout_watch_trigger"]["quote"],
                "in_scope": False
                if warmup_short_circuit
                else cls_row.get("phase_at_or_before_day") in {"BASE_FORMING", "ACCUMULATION"},
                "evaluation": {
                    "terms": watch_terms,
                },
            },
        ],
    }


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# R13 Set A Causal Attribution v2",
        "",
        "Supersession note: this artifact supersedes the v2.1 primary_blocker labels for Set A no-trade diagnosis.",
        "",
        "Classification vocabulary (exactly one per benchmark-active day):",
        "- PHASE_MACHINE_NEVER_LEFT_NEUTRAL",
        "- PHASE_PROGRESSED_NO_CANDIDATE",
        "- CANDIDATE_VETOED(gate)",
        "- CANDIDATE_PASSED_NO_FILL",
        "- MASKED_OR_WARMUP_EXCLUDED",
        "",
        "## Per-Symbol Counts",
    ]

    for symbol in sorted(payload.get("per_symbol_counts", {}).keys()):
        lines.append(f"- {symbol}: {payload['per_symbol_counts'][symbol]}")

    lines += ["", "## Trigger-Day Predicate Surface", ""]
    for item in payload.get("named_trigger_day_predicate_audit", []):
        lines.append(f"### {item['symbol']} {item['trade_date_iso']}")
        for p in item.get("predicate_blocks", []):
            lines.append(
                f"- {p['name']} @ {p['file']}:{p['line']} in_scope={p['in_scope']} eval={p.get('evaluation')}"
            )
        lines.append("")

    lines += ["R14 remains NOT AUTHORIZED.", ""]
    return "\n".join(lines)


def main() -> None:
    v21 = read_json(REVIEW / "r12_exam_results_v2_1.json")
    spec = read_json(REVIEW / "r12_benchmark_spec_v2.json")
    mask = read_json(REVIEW / "r12_masked_intervals_manifest_v4_3_final.json")
    runtime_db = REVIEW / "r12_exam_surface_v4_5_runtime.db"

    set_a = sorted(
        next(
            (r.get("symbols", []) for r in v21.get("benchmark_parity_suite_completion", {}).get("random_top_k_portfolio_rows", []) if r.get("set") == "set_a"),
            [],
        )
    )

    set_a_rows = v21.get("benchmark_parity_suite_completion", {}).get("set_a_symbol_rows", [])
    active = []
    for r in set_a_rows:
        if r.get("symbol") not in set_a:
            continue
        if r.get("benchmark") not in {"SIMPLE_PRICE_BREAKOUT_BENCHMARK", "PRICE_PLUS_RELATIVE_VOLUME_BENCHMARK"}:
            continue
        td = r.get("benchmark_trigger_date")
        if not td:
            continue
        active.append(
            {
                "symbol": r["symbol"],
                "benchmark": r["benchmark"],
                "trade_date_iso": td,
                "trade_date": iso_to_ts(td),
                "benchmark_entry_date": r.get("benchmark_entry_date"),
                "benchmark_exit_date": r.get("benchmark_exit_date"),
                "status": r.get("status"),
            }
        )

    by_symbol_day: dict[tuple[str, int], dict[str, Any]] = {}
    for a in active:
        key = (a["symbol"], a["trade_date"])
        row = by_symbol_day.setdefault(
            key,
            {
                "symbol": a["symbol"],
                "trade_date": a["trade_date"],
                "trade_date_iso": a["trade_date_iso"],
                "benchmarks": [],
            },
        )
        row["benchmarks"].append(
            {
                "benchmark": a["benchmark"],
                "status": a["status"],
                "benchmark_entry_date": a["benchmark_entry_date"],
                "benchmark_exit_date": a["benchmark_exit_date"],
            }
        )

    mask_by_symbol = defaultdict(list)
    for m in mask.get("intervals", []):
        s = str(m.get("symbol") or "")
        if s:
            mask_by_symbol[s].append(m)

    scanner_path = ROOT / "app" / "services" / "eagle_eye" / "scanner_service.py"
    line_map = scanner_line_map(scanner_path)

    out_rows = []
    named_audit = []

    with sqlite_ro(runtime_db) as con:
        cur = con.cursor()
        cfg = get_engine_cfg(cur)
        for row in sorted(by_symbol_day.values(), key=lambda r: (r["symbol"], r["trade_date"])):
            symbol = str(row["symbol"])
            day_ts = int(row["trade_date"])
            signals = signals_for_symbol(cur, symbol)
            trades = trade_rows_for_symbol(cur, symbol)
            indicator = indicators_on_day(cur, symbol, day_ts)

            cls = classify_day(symbol, day_ts, signals, trades, mask_by_symbol.get(symbol, []), indicator)
            rec = {
                **row,
                "classification": cls["classification"],
                "classification_evidence": cls,
                "indicator_payload_on_day": indicator,
            }
            out_rows.append(rec)

            if (symbol, day_ts) in {
                ("TIJARA", iso_to_ts("2021-09-09")),
                ("BPCC", iso_to_ts("2021-08-26")),
                ("SANAM", iso_to_ts("2021-03-23")),
            }:
                named_audit.append(predicate_terms_for_day(symbol, day_ts, cls, indicator, cfg, line_map))

    per_symbol_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in out_rows:
        per_symbol_counts[r["symbol"]][r["classification"]] += 1

    payload = {
        "version_id": "R13_SET_A_CAUSAL_ATTRIBUTION_V2",
        "supersession": "Supersedes v2.1 primary_blocker labels with day-level runtime-derived classifications.",
        "constraints": {
            "read_only_db_uri": True,
            "no_engine_contact": True,
            "no_reruns": True,
            "source_of_truth": "run-2 runtime records (ee_indicators, ee_signals, ee_backtest_trades, mask manifest)",
        },
        "input_artifacts": {
            "benchmark_spec_v2": "artifacts/preview1a_prestart/review_final/r12_benchmark_spec_v2.json",
            "exam_results_v2_1": "artifacts/preview1a_prestart/review_final/r12_exam_results_v2_1.json",
            "mask_manifest": "artifacts/preview1a_prestart/review_final/r12_masked_intervals_manifest_v4_3_final.json",
            "runtime_db": "artifacts/preview1a_prestart/review_final/r12_exam_surface_v4_5_runtime.db",
            "scanner_service": "app/services/eagle_eye/scanner_service.py",
        },
        "set_a_symbols": set_a,
        "benchmark_spec_v2_execution_note": spec.get("execution"),
        "benchmark_active_day_rows": out_rows,
        "per_symbol_counts": {k: dict(sorted(v.items())) for k, v in sorted(per_symbol_counts.items())},
        "named_trigger_day_predicate_audit": named_audit,
        "scanner_predicate_source_lines": line_map,
        "tier_rule_status": "AGENT_PROPOSED_UNRATIFIED",
        "authorization_status": "R14_NOT_AUTHORIZED",
    }

    out_json = REVIEW / "r13_set_a_causal_attribution_v2.json"
    out_md = REVIEW / "r13_set_a_causal_attribution_v2.md"
    write_json(out_json, payload)
    out_md.write_text(build_markdown(payload), encoding="utf-8")

    print("R13_SET_A_CAUSAL_ATTRIBUTION_V2_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()

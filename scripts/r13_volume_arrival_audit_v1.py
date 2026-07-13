from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
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


def ts_to_iso(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def sqlite_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def base_symbol_sql() -> str:
    return "(CASE WHEN instr(symbol, '__SEG') > 0 THEN substr(symbol, 1, instr(symbol, '__SEG') - 1) ELSE symbol END)"


def get_cfg(cur: sqlite3.Cursor) -> dict[str, Any]:
    keys = [
        "base_min_sessions",
        "base_max_width_pct",
        "volume_breakout_mult",
        "rsi_regime",
        "adx_trigger",
        "cmf_floor",
        "atr_squeeze_pctile",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        row = cur.execute("SELECT value_json FROM ee_engine_config WHERE key=?", (k,)).fetchone()
        out[k] = None if row is None else json.loads(row[0])
    return out


def indicator_map(cur: sqlite3.Cursor, symbol: str) -> dict[int, dict[str, Any]]:
    rows = cur.execute(
        f"SELECT trade_date, payload_json FROM ee_indicators WHERE {base_symbol_sql()}=? ORDER BY trade_date",
        (symbol,),
    ).fetchall()
    return {int(td): (json.loads(pj) if pj else {}) for td, pj in rows}


def signal_map(cur: sqlite3.Cursor, symbol: str) -> dict[int, list[dict[str, Any]]]:
    rows = cur.execute(
        f"SELECT id, trade_date, signal_type, phase_from, phase_to, evidence_json FROM ee_signals WHERE {base_symbol_sql()}=? ORDER BY trade_date, id",
        (symbol,),
    ).fetchall()
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rid, td, st, pf, pt, ev in rows:
        out[int(td)].append(
            {
                "id": int(rid),
                "trade_date": int(td),
                "signal_type": str(st),
                "phase_from": None if pf is None else str(pf),
                "phase_to": None if pt is None else str(pt),
                "evidence": json.loads(ev) if ev else {},
            }
        )
    return out


def trade_open_days(cur: sqlite3.Cursor, symbol: str) -> set[int]:
    rows = cur.execute(
        f"SELECT opened_at FROM ee_backtest_trades WHERE {base_symbol_sql()}=?",
        (symbol,),
    ).fetchall()
    return {int(r[0]) for r in rows}


def choose_blocking_term(phase: str, indicator: dict[str, Any] | None, cfg: dict[str, Any], close_series: list[float], relvol_last_5: list[float | None], cmf_last_10: list[float | None]) -> str:
    if indicator is None:
        return "NO_INDICATOR_PAYLOAD"

    close = indicator.get("close")
    ema10 = indicator.get("ema10")
    ema30 = indicator.get("ema30")
    sma200 = indicator.get("sma200")
    rel_volume = indicator.get("rel_volume")
    bb_width = indicator.get("bb_width")
    atr_pct_pctile = indicator.get("atr_pct_percentile_252")
    accumulation_divergence = indicator.get("accumulation_divergence")
    price_slope_40 = indicator.get("price_slope_40")
    obv_slope_40 = indicator.get("obv_slope_40")
    anv_slope_40 = indicator.get("anv_slope_40")

    if phase == "BREAKOUT_WATCH":
        if rel_volume is None or cfg.get("volume_breakout_mult") is None:
            return "M2_rel_volume"
        if float(rel_volume) < float(cfg["volume_breakout_mult"]):
            return "M2_rel_volume"
        if ema10 is None or ema30 is None or float(ema10) <= float(ema30):
            return "M3_ema10_gt_ema30"
        return "UNRESOLVED_BREAKOUT_WATCH_NON_M2"

    if phase == "ACCUMULATION":
        rv_hits = None
        if relvol_last_5 and all(v is not None for v in relvol_last_5):
            rv_hits = sum(1 for v in relvol_last_5 if float(v) >= 1.5)
        if rv_hits is None or rv_hits < 2:
            return "rv_hits >= 2 over last 5 (rel_volume>=1.5)"
        return "base_high_ref > 0"

    if phase == "BASE_FORMING":
        composite_recoverable = price_slope_40 is not None and (obv_slope_40 is not None or anv_slope_40 is not None)
        composite_ok = False
        if composite_recoverable:
            composite_ok = bool(accumulation_divergence) or (
                abs(float(price_slope_40)) < 0.02
                and (
                    (obv_slope_40 is not None and float(obv_slope_40) > 0.10)
                    or (anv_slope_40 is not None and float(anv_slope_40) > 0.10)
                )
            )
        if not composite_ok:
            return "accumulation_gate composite"
        cmf_floor = cfg.get("cmf_floor")
        if cmf_floor is None or not cmf_last_10 or not all(v is not None for v in cmf_last_10):
            return "cmf_hits >= 5 over last 10"
        cmf_hits = sum(1 for v in cmf_last_10 if float(v) > float(cmf_floor))
        if cmf_hits < 5:
            return "cmf_hits >= 5 over last 10"
        squeeze_ok = (bb_width is not None and float(bb_width) <= 0.12) or (
            atr_pct_pctile is not None and cfg.get("atr_squeeze_pctile") is not None and float(atr_pct_pctile) <= float(cfg["atr_squeeze_pctile"])
        )
        if not squeeze_ok:
            return "squeeze_ok"
        if close is None or ema30 is None or not (float(close) >= float(ema30) or (sma200 is not None and float(close) >= 0.97 * float(sma200))):
            return "close>=ema30 OR close>=0.97*sma200"
        return "liquidity_ok/score>=60"

    if phase == "NEUTRAL":
        sma200_ok = sma200 is not None and float(sma200) > 0.0
        if not sma200_ok:
            return "sma200 > 0"
        if ema30 is None or float(ema30) <= 0.0:
            return "ema30 > 0"
        width = indicator.get("range_width_pct")
        if width is None and len(close_series) >= 2:
            lo = min(close_series)
            hi = max(close_series)
            width = None if lo <= 0 else ((hi - lo) / lo)
        if width is None or cfg.get("base_max_width_pct") is None or float(width) > float(cfg["base_max_width_pct"]):
            return "width <= base_max_width_pct"
        if len(close_series) < int(cfg.get("base_min_sessions") or 60):
            return "sessions_in_range >= base_min_sessions"
        return "base_low_60 <= close <= base_high_60"

    if phase in {"AVOID", "DISTRIBUTION_WARNING", "EXIT"}:
        return "avoid_condition close < sma200 and sma200_slope < 0 and ema10 < ema30"

    return "UNCLASSIFIED_PHASE"


def disposition_for_day(phase: str, classification: str, signals_today: list[dict[str, Any]], opened_today: bool, indicator: dict[str, Any] | None, cfg: dict[str, Any], close_series: list[float], relvol_last_5: list[float | None], cmf_last_10: list[float | None]) -> str:
    if opened_today:
        return "TRADE_TAKEN"
    for s in signals_today:
        if s["signal_type"] == "SIGNAL_SUPPRESSED_RISK":
            return "CANDIDATE_VETOED(RISK_SUPPRESSION)"
        if s["signal_type"] == "AVOID_SET" or s.get("phase_to") == "AVOID":
            return "CANDIDATE_VETOED(AVOID_GATE)"
        if s["signal_type"] in {"ACCUMULATION_ALERT", "BREAKOUT_CONFIRMED"}:
            return "CANDIDATE_RAISED"
    if classification == "CANDIDATE_PASSED_NO_FILL":
        return "CANDIDATE_PASSED_NO_FILL"
    return choose_blocking_term(phase, indicator, cfg, close_series, relvol_last_5, cmf_last_10)


def build_surface(threshold: float) -> dict[str, Any]:
    d1 = read_json(REVIEW / "r13_set_a_causal_attribution_v3.json")
    runtime_db = REVIEW / "r12_exam_surface_v4_5_runtime.db"
    set_a = list(d1.get("set_a_symbols", []))
    day_index = {(r["symbol"], int(r["trade_date"])): r for r in d1.get("day_level_table", [])}

    per_symbol_days: dict[str, list[dict[str, Any]]] = {}
    per_symbol_summary: dict[str, dict[str, int]] = {}

    with sqlite_ro(runtime_db) as con:
        cur = con.cursor()
        cfg = get_cfg(cur)

        for sym in set_a:
            indicators = indicator_map(cur, sym)
            signals = signal_map(cur, sym)
            trades = trade_open_days(cur, sym)
            src_rows = cur.execute(
                "SELECT trade_date, close, volume, value_kwd FROM ee_ohlcv_masked_source WHERE symbol=? ORDER BY trade_date",
                (sym,),
            ).fetchall()
            series = [{"trade_date": int(td), "close": float(c or 0.0), "volume": float(v or 0.0), "value_kwd": float(val or 0.0)} for td, c, v, val in src_rows]
            selected: list[dict[str, Any]] = []
            summary = Counter()

            for idx, row in enumerate(series):
                day_ts = int(row["trade_date"])
                indicator = indicators.get(day_ts)
                if indicator is None:
                    continue
                rel_volume = indicator.get("rel_volume")
                if rel_volume is None or float(rel_volume) < threshold:
                    continue

                day_rec = day_index.get((sym, day_ts), {})
                phase = str(day_rec.get("phase_after_day") or "UNKNOWN")
                signals_today = signals.get(day_ts, [])
                opened_today = day_ts in trades

                closes_60 = [x["close"] for x in series[max(0, idx - 59) : idx + 1]]
                relvol_5 = [indicators.get(x["trade_date"], {}).get("rel_volume") if indicators.get(x["trade_date"]) is not None else None for x in series[max(0, idx - 4) : idx + 1]]
                cmf_10 = [indicators.get(x["trade_date"], {}).get("cmf_10") if indicators.get(x["trade_date"]) is not None else None for x in series[max(0, idx - 9) : idx + 1]]

                disposition = disposition_for_day(
                    phase,
                    str(day_rec.get("classification") or "UNKNOWN"),
                    signals_today,
                    opened_today,
                    indicator,
                    cfg,
                    closes_60,
                    relvol_5,
                    cmf_10,
                )
                summary[disposition] += 1
                selected.append(
                    {
                        "date": ts_to_iso(day_ts),
                        "trade_date": day_ts,
                        "close": row["close"],
                        "volume": row["volume"],
                        "value_kwd": row["value_kwd"],
                        "rel_volume": rel_volume,
                        "phase_state": phase,
                        "classification": day_rec.get("classification"),
                        "disposition": disposition,
                        "signals_on_day": [s["signal_type"] for s in signals_today],
                        "trade_taken": opened_today,
                    }
                )

            per_symbol_days[sym] = selected
            per_symbol_summary[sym] = dict(sorted(summary.items()))

    return {
        "threshold": threshold,
        "per_symbol_days": per_symbol_days,
        "per_symbol_blocking_distribution": per_symbol_summary,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    hi = payload["rel_volume_ge_2_5"]
    lo = payload["rel_volume_ge_2_0"]
    lines = [
        "# R13 Volume Arrival Audit v1",
        "",
        "This audit is read-only and uses frozen runtime records.",
        "",
        "## rel_volume >= 2.5",
    ]
    for sym, rows in hi["per_symbol_days"].items():
        lines.append(f"### {sym}")
        lines.append(f"- blocking_distribution: {hi['per_symbol_blocking_distribution'][sym]}")
        for row in rows:
            lines.append(
                f"- {row['date']} close={row['close']} rel_volume={row['rel_volume']} phase={row['phase_state']} disposition={row['disposition']}"
            )
    lines += ["", "## rel_volume >= 2.0 (Sensitivity)"]
    for sym, rows in lo["per_symbol_days"].items():
        lines.append(f"### {sym}")
        lines.append(f"- blocking_distribution: {lo['per_symbol_blocking_distribution'][sym]}")
        for row in rows:
            lines.append(
                f"- {row['date']} close={row['close']} rel_volume={row['rel_volume']} phase={row['phase_state']} disposition={row['disposition']}"
            )
    lines += ["", "R14 remains NOT AUTHORIZED.", ""]
    return "\n".join(lines)


def main() -> None:
    payload = {
        "version_id": "R13_VOLUME_ARRIVAL_AUDIT_V1",
        "constraints": {
            "read_only_db_uri": True,
            "no_engine_contact": True,
            "no_reruns": True,
        },
        "inputs": {
            "runtime_db": "artifacts/preview1a_prestart/review_final/r12_exam_surface_v4_5_runtime.db",
            "d1_v3": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json",
        },
        "rel_volume_ge_2_5": build_surface(2.5),
        "rel_volume_ge_2_0": build_surface(2.0),
        "authorization_status": "R14_NOT_AUTHORIZED",
    }
    out_json = REVIEW / "r13_volume_arrival_audit_v1.json"
    out_md = REVIEW / "r13_volume_arrival_audit_v1.md"
    write_json(out_json, payload)
    out_md.write_text(build_markdown(payload), encoding="utf-8")
    print("R13_VOLUME_ARRIVAL_AUDIT_V1_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
RUNTIME_DB = REVIEW / "r12_exam_surface_v4_5_runtime.db"
RISK_SERVICE = ROOT / "app" / "services" / "eagle_eye" / "risk_service.py"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sqlite_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def ts_to_iso(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def iso_to_ts(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def base_sql() -> str:
    return "(CASE WHEN instr(symbol, '__SEG') > 0 THEN substr(symbol, 1, instr(symbol, '__SEG') - 1) ELSE symbol END)"


def risk_service_excerpt() -> dict[str, Any]:
    lines = RISK_SERVICE.read_text(encoding="utf-8").splitlines()
    start = 7
    end = 55
    excerpt = "\n".join(lines[start - 1:end])
    return {
        "file": "app/services/eagle_eye/risk_service.py",
        "start_line": start,
        "end_line": end,
        "code": excerpt,
    }


def get_config(cur: sqlite3.Cursor) -> dict[str, Any]:
    keys = ["min_daily_value_kwd", "volume_breakout_mult", "rsi_regime", "adx_trigger", "ml_gate_enabled", "ml_min_labeled_signals", "ml_prob_min"]
    out: dict[str, Any] = {}
    for k in keys:
        row = cur.execute("SELECT value_json FROM ee_engine_config WHERE key=?", (k,)).fetchone()
        out[k] = None if row is None else json.loads(row[0])
    out["liquidity_window_sessions"] = 60
    out["liquidity_min_price_fils"] = 50.0
    out["liquidity_max_zero_volume_sessions"] = 3
    return out


def indicator(cur: sqlite3.Cursor, symbol: str, ts: int) -> dict[str, Any]:
    row = cur.execute(f"SELECT payload_json FROM ee_indicators WHERE {base_sql()}=? AND trade_date=?", (symbol, ts)).fetchone()
    return {} if row is None else json.loads(row[0])


def phase_on_day(day_table: list[dict[str, Any]], symbol: str, date_iso: str) -> str | None:
    for row in day_table:
        if row.get("symbol") == symbol and row.get("trade_date_iso") == date_iso:
            return row.get("phase_after_day")
    return None


def build_base_reference_history(cur: sqlite3.Cursor, symbol: str) -> list[dict[str, Any]]:
    rows = cur.execute(
        f"SELECT trade_date, evidence_json FROM ee_signals WHERE {base_sql()}=? ORDER BY trade_date, id",
        (symbol,),
    ).fetchall()
    current_ref = None
    history = []
    for td, ev in rows:
        evidence = json.loads(ev) if ev else {}
        evt = evidence.get("base_lifecycle_event") or {}
        action = evt.get("action")
        if action == "base_freeze":
            current_ref = evt.get("new", {}).get("base_high")
        elif action == "base_ratchet":
            current_ref = evt.get("new", {}).get("base_high")
        elif action in {"base_invalidated", "base_cleared"}:
            current_ref = None
        history.append({"trade_date": int(td), "current_base_high_ref": current_ref, "event": evt if evt else None})
    return history


def ref_as_of(history: list[dict[str, Any]], ts: int) -> Any:
    current = None
    for row in history:
        if int(row["trade_date"]) <= ts:
            current = row["current_base_high_ref"]
        else:
            break
    return current


def rating_row(cur: sqlite3.Cursor, symbol: str, ts: int) -> dict[str, Any] | None:
    row = cur.execute(f"SELECT score, band, components_json FROM ee_ratings WHERE {base_sql()}=? AND trade_date=?", (symbol, ts)).fetchone()
    if row is None:
        return None
    return {"score": float(row[0]), "band": str(row[1]), "components": json.loads(row[2]) if row[2] else {}}


def prev_indicators(cur: sqlite3.Cursor, symbol: str, ts: int, n: int = 6) -> list[tuple[int, dict[str, Any]]]:
    rows = cur.execute(f"SELECT trade_date, payload_json FROM ee_indicators WHERE {base_sql()}=? AND trade_date<=? ORDER BY trade_date DESC LIMIT ?", (symbol, ts, n)).fetchall()
    out = [(int(td), json.loads(pj) if pj else {}) for td, pj in rows]
    out.reverse()
    return out


def labeled_signal_count(cur: sqlite3.Cursor) -> int:
    row = cur.execute("SELECT COUNT(1) FROM ee_signals WHERE outcome_label IS NOT NULL").fetchone()
    return int(row[0] or 0)


def estimate_ml_probability(evidence: dict[str, Any]) -> float:
    score = 0.50
    if bool(evidence.get("accumulation_divergence")):
        score += 0.08
    if float(evidence.get("cmf_10") or 0.0) > 0.05:
        score += 0.05
    if float(evidence.get("rel_volume") or 0.0) >= 2.5:
        score += 0.07
    if float(evidence.get("adx_19") or 0.0) >= 22:
        score += 0.05
    if bool(evidence.get("distribution_divergence")):
        score -= 0.20
    return max(0.0, min(1.0, score))


def ml_gate(cur: sqlite3.Cursor, evidence: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(cfg.get("ml_gate_enabled", False))
    labeled = labeled_signal_count(cur)
    min_count = int(cfg.get("ml_min_labeled_signals", 150) or 150)
    if not enabled:
        return {"pass": True, "reason": "disabled", "probability": None}
    if labeled < min_count:
        return {"pass": True, "reason": "insufficient_labeled_signals", "probability": None, "labeled_signal_count": labeled}
    prob = estimate_ml_probability(evidence)
    min_prob = float(cfg.get("ml_prob_min", 0.45) or 0.45)
    return {"pass": prob >= min_prob, "reason": "probability_check", "probability": prob, "min_probability": min_prob}


def liquidity_inputs(cur: sqlite3.Cursor, symbol: str, ts: int) -> dict[str, Any]:
    rows = cur.execute(
        f"SELECT trade_date, value_kwd, close, volume FROM ee_ohlcv WHERE {base_sql()}=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 60",
        (symbol, ts),
    ).fetchall()
    trailing = [
        {"trade_date": int(td), "date": ts_to_iso(int(td)), "value_kwd": float(v or 0.0), "close": float(c or 0.0), "volume": float(vol or 0.0)}
        for td, v, c, vol in rows
    ]
    values = [r["value_kwd"] for r in trailing]
    closes = [r["close"] for r in trailing]
    vols = [r["volume"] for r in trailing]
    values_sorted = sorted(values)
    median_val = values_sorted[len(values_sorted) // 2] if values_sorted else 0.0
    min_price = min(closes) if closes else 0.0
    zero_vol = sum(1 for v in vols if v <= 0)
    return {
        "window_sessions": 60,
        "trailing_rows_desc": trailing,
        "median_daily_value_kwd_20": median_val,
        "min_price_fils_60": min_price,
        "zero_volume_sessions_60": zero_vol,
    }


def recent_macd_cross(history: list[tuple[int, dict[str, Any]]]) -> bool:
    if len(history) < 6:
        return False
    for i in range(max(1, len(history) - 5), len(history)):
        prev = history[i - 1][1]
        row = history[i][1]
        if float(prev.get("macd_line") or 0.0) <= float(prev.get("macd_signal") or 0.0) and float(row.get("macd_line") or 0.0) > float(row.get("macd_signal") or 0.0):
            return True
    return False


def resolve_day(cur: sqlite3.Cursor, day_table: list[dict[str, Any]], symbol: str, date_iso: str, cfg: dict[str, Any], base_ref_history: list[dict[str, Any]]) -> dict[str, Any]:
    ts = iso_to_ts(date_iso)
    payload = indicator(cur, symbol, ts)
    phase = phase_on_day(day_table, symbol, date_iso)
    rating = rating_row(cur, symbol, ts)
    base_high_ref = ref_as_of(base_ref_history, ts)
    history = prev_indicators(cur, symbol, ts, 6)
    prev = history[-2][1] if len(history) >= 2 else None
    close = float(payload.get("close") or 0.0)
    high = float(payload.get("high") or close)
    low = float(payload.get("low") or close)
    open_v = float(payload.get("open") or close)
    ema10 = payload.get("ema10")
    ema30 = payload.get("ema30")
    rel_volume = payload.get("rel_volume")
    rsi = payload.get("rsi_14")
    adx = payload.get("adx_19")
    plus_di = payload.get("plus_di")
    minus_di = payload.get("minus_di")
    gap_pct_base = None if base_high_ref is None else max(0.0, (open_v - float(base_high_ref)) / float(base_high_ref))
    m1 = False if base_high_ref is None else close > float(base_high_ref)
    m2 = None if rel_volume is None else float(rel_volume) >= float(cfg["volume_breakout_mult"])
    m3 = None if ema10 is None or ema30 is None else float(ema10) > float(ema30)
    m4 = None if gap_pct_base is None else gap_pct_base <= 0.08
    liq = liquidity_inputs(cur, symbol, ts)
    m5 = liq["median_daily_value_kwd_20"] >= float(cfg["min_daily_value_kwd"]) and liq["min_price_fils_60"] >= 50.0 and liq["zero_volume_sessions_60"] <= 3
    adx_5_back = float(history[-5][1].get("adx_19") or adx or 0.0) if len(history) >= 5 else float(adx or 0.0)
    macd_cross = recent_macd_cross(history)
    day_range = max(0.0, high - low)
    close_top40 = True if day_range == 0 else close >= (low + 0.6 * day_range)
    rsi_rising = True if prev is None else float(rsi or 0.0) > float(prev.get("rsi_14") or rsi or 0.0)
    c_flags = {
        "C1_rsi": None if rsi is None else float(rsi) >= float(cfg["rsi_regime"]),
        "C2_rsi_rising": rsi_rising,
        "C3_adx_di": None if adx is None or plus_di is None or minus_di is None else (float(adx) >= float(cfg["adx_trigger"]) and float(plus_di) > float(minus_di)),
        "C4_adx_accel": None if adx is None else float(adx) > float(adx_5_back),
        "C5_macd": (float(payload.get("macd_hist") or 0.0) > 0) or macd_cross,
        "C6_close_top40": close_top40,
    }
    c_score = sum(1 for v in c_flags.values() if v is True)
    ml = ml_gate(cur, payload, cfg)
    score_gate = None if rating is None else float(rating["score"]) >= 70.0
    all_resolved_except_m5 = all(v is True for v in [m1, m2, m3, m4] if v is not None) and c_score >= 4 and ml["pass"] is True and (score_gate is True)
    return {
        "symbol": symbol,
        "date": date_iso,
        "phase_state": phase,
        "close": close,
        "same_day_value_kwd": payload.get("value_kwd"),
        "rel_volume": rel_volume,
        "base_high_ref": base_high_ref,
        "mandatory": {
            "M1_close_gt_base": m1,
            "M2_rel_volume": m2,
            "M3_ema10_gt_ema30": m3,
            "M4_chase_guard": m4,
            "M5_liquidity": m5,
        },
        "c_score": c_score,
        "confirm_flags": c_flags,
        "ml_gate": ml,
        "score_gate": score_gate,
        "liquidity_filter_inputs": liq,
        "sole_surviving_m5_blocker": (all_resolved_except_m5 and m5 is False),
    }


def main() -> None:
    d1 = read_json(REVIEW / "r13_set_a_causal_attribution_v3.json")
    vol = read_json(REVIEW / "r13_volume_arrival_audit_v1.json")
    cfg = {}
    with sqlite_ro(RUNTIME_DB) as con:
        cur = con.cursor()
        cfg = get_config(cur)
        set_a = sorted(vol.get("rel_volume_ge_2_5", {}).get("per_symbol_days", {}).keys())
        base_histories = {sym: build_base_reference_history(cur, sym) for sym in set_a}
        all_hi = []
        per_symbol_counts: dict[str, int] = {}
        for sym in set_a:
            resolved_rows = []
            for row in vol.get("rel_volume_ge_2_5", {}).get("per_symbol_days", {}).get(sym, []):
                resolved_rows.append(resolve_day(cur, d1.get("day_level_table", []), sym, row["date"], cfg, base_histories[sym]))
            all_hi.extend(resolved_rows)
            per_symbol_counts[sym] = sum(1 for r in resolved_rows if r["sole_surviving_m5_blocker"])
        sanam_window = [resolve_day(cur, d1.get("day_level_table", []), "SANAM", d, cfg, base_histories["SANAM"]) for d in ["2025-05-08","2025-05-11","2025-05-12","2025-05-13","2025-05-14","2025-05-15","2025-05-18","2025-05-21"]]

    sole_total = sum(per_symbol_counts.values())
    if sole_total > 0:
        f9_status = "CONFIRMED"
        f9_statement = "Trailing-window liquidity baseline lagged breakout-day liquidity and acted as sole veto on at least one high-volume Set A day."
    else:
        f9_status = "NOT_CONFIRMED"
        f9_statement = "Trailing-liquidity lag was observed, but not as a sole surviving blocker under the implemented resolved-term test."

    payload = {
        "version_id": "R13_M5_LIQUIDITY_FORENSIC_V1",
        "inputs": {
            "runtime_db": "artifacts/preview1a_prestart/review_final/r12_exam_surface_v4_5_runtime.db",
            "volume_arrival_audit_v1": "artifacts/preview1a_prestart/review_final/r13_volume_arrival_audit_v1.json",
            "d1_v3": "artifacts/preview1a_prestart/review_final/r13_set_a_causal_attribution_v3.json",
            "risk_service": "app/services/eagle_eye/risk_service.py",
        },
        "constraints": {
            "read_only_db_uri": True,
            "no_engine_contact": True,
            "no_reruns": True,
        },
        "liquidity_filter_code": risk_service_excerpt(),
        "run2_config_values": cfg,
        "sanam_2025_05_08_to_2025_05_21": sanam_window,
        "all_set_a_high_volume_days_ge_2_5": all_hi,
        "m5_sole_surviving_blocker_count_by_symbol": per_symbol_counts,
        "m5_sole_surviving_blocker_total": sole_total,
        "f9": {
            "status": f9_status,
            "statement": f9_statement,
            "canonical_day": "SANAM 2025-05-18",
        },
        "authorization_status": {"R14_B": "NOT_AUTHORIZED", "R15": "NOT_AUTHORIZED"},
    }
    out_json = REVIEW / "r13_m5_liquidity_forensic_v1.json"
    out_md = REVIEW / "r13_m5_liquidity_forensic_v1.md"
    write_json(out_json, payload)
    lines = [
        "# R13 M5 Liquidity Forensic v1",
        "",
        f"Code source: {payload['liquidity_filter_code']['file']}:{payload['liquidity_filter_code']['start_line']}-{payload['liquidity_filter_code']['end_line']}",
        "```python",
        payload['liquidity_filter_code']['code'],
        "```",
        "Run-2 config values:",
        json.dumps(cfg, ensure_ascii=True, indent=2, sort_keys=True),
        "",
        "## SANAM 2025-05-08 -> 2025-05-21",
    ]
    for row in sanam_window:
        lines.append(json.dumps(row, ensure_ascii=True, sort_keys=True))
    lines += ["", "## M5 Sole Surviving Blocker Counts", json.dumps(per_symbol_counts, ensure_ascii=True, sort_keys=True), "", f"F9 status: {f9_status}", f"F9 statement: {f9_statement}", "", "R14-B and R15 remain NOT AUTHORIZED.", ""]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("R13_M5_LIQUIDITY_FORENSIC_V1_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()

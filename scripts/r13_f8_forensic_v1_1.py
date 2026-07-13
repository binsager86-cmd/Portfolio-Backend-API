from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
RUNTIME_DB = REVIEW / "r12_exam_surface_v4_5_runtime.db"


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


def iso_to_ts(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def ts_to_iso(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def base_sql() -> str:
    return "(CASE WHEN instr(symbol, '__SEG') > 0 THEN substr(symbol, 1, instr(symbol, '__SEG') - 1) ELSE symbol END)"


def get_config(cur: sqlite3.Cursor) -> dict[str, Any]:
    keys = [
        "volume_breakout_mult",
        "rsi_regime",
        "adx_trigger",
        "ml_gate_enabled",
        "ml_min_labeled_signals",
        "ml_prob_min",
        "min_daily_value_kwd",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        row = cur.execute("SELECT value_json FROM ee_engine_config WHERE key=?", (key,)).fetchone()
        out[key] = None if row is None else json.loads(row[0])
    return out


def signals_with_base_events(cur: sqlite3.Cursor, symbol: str, start_ts: int, end_ts: int) -> list[dict[str, Any]]:
    rows = cur.execute(
        f"""
        SELECT trade_date, signal_type, phase_from, phase_to, evidence_json
        FROM ee_signals
        WHERE {base_sql()}=? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date, id
        """,
        (symbol, start_ts, end_ts),
    ).fetchall()
    out = []
    for td, st, pf, pt, ev in rows:
        evidence = json.loads(ev) if ev else {}
        base_evt = evidence.get("base_lifecycle_event") or {}
        if pt == "BASE_FORMING" or base_evt.get("action") in {"base_freeze", "base_invalidated", "base_cleared", "base_ratchet"} or evidence.get("last_phase_reason") == "base_detected":
            out.append(
                {
                    "date": ts_to_iso(int(td)),
                    "trade_date": int(td),
                    "signal_type": str(st),
                    "phase_from": None if pf is None else str(pf),
                    "phase_to": None if pt is None else str(pt),
                    "evidence_json_verbatim": evidence,
                    "base_lifecycle_event": base_evt if base_evt else None,
                    "last_phase_reason": evidence.get("last_phase_reason"),
                }
            )
    return out


def indicator(cur: sqlite3.Cursor, symbol: str, ts: int) -> dict[str, Any]:
    row = cur.execute(
        f"SELECT payload_json FROM ee_indicators WHERE {base_sql()}=? AND trade_date=?",
        (symbol, ts),
    ).fetchone()
    return {} if row is None else json.loads(row[0])


def prev_indicators(cur: sqlite3.Cursor, symbol: str, ts: int, n: int = 6) -> list[tuple[int, dict[str, Any]]]:
    rows = cur.execute(
        f"SELECT trade_date, payload_json FROM ee_indicators WHERE {base_sql()}=? AND trade_date<=? ORDER BY trade_date DESC LIMIT ?",
        (symbol, ts, n),
    ).fetchall()
    out = [(int(td), json.loads(pj) if pj else {}) for td, pj in rows]
    out.reverse()
    return out


def phase_on_day(d1: dict[str, Any], symbol: str, date_iso: str) -> tuple[str | None, str | None]:
    for row in d1.get("day_level_table", []):
        if row.get("symbol") == symbol and row.get("trade_date_iso") == date_iso:
            return row.get("phase_before_day"), row.get("phase_after_day")
    return None, None


def liquidity_ok(cur: sqlite3.Cursor, symbol: str, ts: int, min_daily_value_kwd: float) -> tuple[bool, dict[str, Any]]:
    rows = cur.execute(
        f"SELECT value_kwd, close, volume FROM ee_ohlcv WHERE {base_sql()}=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 60",
        (symbol, ts),
    ).fetchall()
    values = [float(r[0] or 0.0) for r in rows]
    closes = [float(r[1] or 0.0) for r in rows]
    vols = [float(r[2] or 0.0) for r in rows]
    values_sorted = sorted(values)
    median_val = values_sorted[len(values_sorted) // 2] if values_sorted else 0.0
    min_price = min(closes) if closes else 0.0
    zero_vol = sum(1 for v in vols if v <= 0)
    ok = median_val >= float(min_daily_value_kwd) and min_price >= 50.0 and zero_vol <= 3
    return ok, {
        "median_daily_value_kwd_20": median_val,
        "min_price_fils_60": min_price,
        "zero_volume_sessions_60": zero_vol,
    }


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


def apply_ml_gate_from_persisted(cur: sqlite3.Cursor, evidence: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(config.get("ml_gate_enabled", False))
    labeled = labeled_signal_count(cur)
    min_count = int(config.get("ml_min_labeled_signals", 150) or 150)
    if not enabled:
        return {"enabled": False, "pass": True, "probability": None, "reason": "disabled_by_config_default_or_value", "labeled_signal_count": labeled}
    if labeled < min_count:
        return {"enabled": True, "pass": True, "probability": None, "reason": "insufficient_labeled_signals", "labeled_signal_count": labeled, "min_count": min_count}
    prob = estimate_ml_probability(evidence)
    min_prob = float(config.get("ml_prob_min", 0.45) or 0.45)
    return {"enabled": True, "pass": prob >= min_prob, "probability": prob, "min_probability": min_prob, "labeled_signal_count": labeled}


def score_row(cur: sqlite3.Cursor, symbol: str, ts: int) -> dict[str, Any] | None:
    row = cur.execute(
        f"SELECT score, band, components_json FROM ee_ratings WHERE {base_sql()}=? AND trade_date=?",
        (symbol, ts),
    ).fetchone()
    if row is None:
        return None
    return {"score": float(row[0]), "band": str(row[1]), "components_json": json.loads(row[2]) if row[2] else {}}


def recent_macd_cross(history: list[tuple[int, dict[str, Any]]]) -> bool:
    if len(history) < 6:
        return False
    for i in range(max(1, len(history) - 5), len(history)):
        prev = history[i - 1][1]
        row = history[i][1]
        if float(prev.get("macd_line") or 0.0) <= float(prev.get("macd_signal") or 0.0) and float(row.get("macd_line") or 0.0) > float(row.get("macd_signal") or 0.0):
            return True
    return False


def resolve_confirm_layer(cur: sqlite3.Cursor, d1: dict[str, Any], symbol: str, date_iso: str, base_high_ref: float | None) -> dict[str, Any]:
    config = get_config(cur)
    ts = iso_to_ts(date_iso)
    payload = indicator(cur, symbol, ts)
    history = prev_indicators(cur, symbol, ts, 6)
    prev_payload = history[-2][1] if len(history) >= 2 else None
    _, phase_after = phase_on_day(d1, symbol, date_iso)
    rating = score_row(cur, symbol, ts)

    close = float(payload.get("close") or 0.0)
    high = float(payload.get("high") or close)
    low = float(payload.get("low") or close)
    open_v = float(payload.get("open") or close)
    rel_volume = payload.get("rel_volume")
    ema10 = payload.get("ema10")
    ema30 = payload.get("ema30")
    rsi = payload.get("rsi_14")
    adx = payload.get("adx_19")
    plus_di = payload.get("plus_di")
    minus_di = payload.get("minus_di")

    gap_pct_base = None if base_high_ref is None else max(0.0, (open_v - float(base_high_ref)) / float(base_high_ref))
    m1 = False if base_high_ref is None else close > float(base_high_ref)
    m2 = None if rel_volume is None or config.get("volume_breakout_mult") is None else float(rel_volume) >= float(config["volume_breakout_mult"])
    m3 = None if ema10 is None or ema30 is None else float(ema10) > float(ema30)
    m4 = None if gap_pct_base is None else gap_pct_base <= 0.08
    m5_ok, m5_meta = liquidity_ok(cur, symbol, ts, float(config.get("min_daily_value_kwd") or 100000.0))

    adx_5_back = float(history[-5][1].get("adx_19") or adx or 0.0) if len(history) >= 5 else float(adx or 0.0)
    macd_cross = recent_macd_cross(history)
    day_range = max(0.0, high - low)
    close_top40 = True if day_range == 0 else close >= (low + 0.6 * day_range)
    rsi_rising = True if prev_payload is None else float(rsi or 0.0) > float(prev_payload.get("rsi_14") or rsi or 0.0)

    c_flags = {
        "C1_rsi": {"pass": None if rsi is None or config.get("rsi_regime") is None else float(rsi) >= float(config["rsi_regime"]), "value": rsi, "threshold": config.get("rsi_regime")},
        "C2_rsi_rising": {"pass": rsi_rising, "value": {"today_rsi": rsi, "prev_rsi": None if prev_payload is None else prev_payload.get("rsi_14")}, "threshold": "today_rsi > prev_rsi"},
        "C3_adx_di": {"pass": None if adx is None or plus_di is None or minus_di is None or config.get("adx_trigger") is None else (float(adx) >= float(config["adx_trigger"]) and float(plus_di) > float(minus_di)), "value": {"adx_19": adx, "plus_di": plus_di, "minus_di": minus_di}, "threshold": {"adx_trigger": config.get("adx_trigger"), "plus_di_gt_minus_di": True}},
        "C4_adx_accel": {"pass": None if adx is None else float(adx) > float(adx_5_back), "value": {"adx_19": adx, "adx_5_back": adx_5_back}, "threshold": "adx_19 > adx_5_back"},
        "C5_macd": {"pass": (float(payload.get("macd_hist") or 0.0) > 0) or macd_cross, "value": {"macd_hist": payload.get("macd_hist"), "recent_cross": macd_cross}, "threshold": "macd_hist > 0 OR recent_cross"},
        "C6_close_top40": {"pass": close_top40, "value": {"close": close, "high": high, "low": low}, "threshold": "close >= low + 0.6*(high-low)"},
    }
    c_score = sum(1 for v in c_flags.values() if v.get("pass") is True)
    ml = apply_ml_gate_from_persisted(cur, payload, config)
    score_gate = {
        "persisted": rating is not None,
        "score": None if rating is None else rating["score"],
        "pass": None if rating is None else float(rating["score"]) >= 70.0,
        "threshold": 70.0,
        "band": None if rating is None else rating["band"],
    }

    resolved = {
        "phase_state": phase_after,
        "mandatory": {
            "M1_close_gt_base": {"pass": m1, "value": {"close": close, "base_high_ref": base_high_ref}, "persisted_input": base_high_ref is not None},
            "M2_rel_volume": {"pass": m2, "value": rel_volume, "threshold": config.get("volume_breakout_mult"), "persisted_input": rel_volume is not None},
            "M3_ema10_gt_ema30": {"pass": m3, "value": {"ema10": ema10, "ema30": ema30}, "persisted_input": ema10 is not None and ema30 is not None},
            "M4_chase_guard": {"pass": m4, "value": {"open": open_v, "base_high_ref": base_high_ref, "gap_pct_base": gap_pct_base}, "threshold": 0.08, "persisted_input": base_high_ref is not None},
            "M5_liquidity": {"pass": m5_ok, "value": m5_meta, "threshold": {"min_daily_value_kwd": config.get("min_daily_value_kwd"), "min_price_fils_60": 50.0, "max_zero_volume_sessions_60": 3}, "persisted_input": True, "reconstructed_from": "ee_ohlcv via risk_service logic"},
        },
        "confirm_flags": c_flags,
        "c_score": {"value": c_score, "threshold": 4, "pass": c_score >= 4},
        "ml_gate": ml,
        "score_gate": score_gate,
        "post_mandatory_code_path": {
            "confirm_flags_line": "app/services/eagle_eye/scanner_service.py#L721",
            "c_score_line": "app/services/eagle_eye/scanner_service.py#L729",
            "ml_gate_line": "app/services/eagle_eye/scanner_service.py#L750",
            "breakout_confirm_line": "app/services/eagle_eye/scanner_service.py#L751",
            "ml_service_line": "app/services/eagle_eye/ml_service.py#L33",
            "liquidity_service_line": "app/services/eagle_eye/risk_service.py#L12",
        },
    }

    all_mandatory = all(resolved["mandatory"][k]["pass"] is True for k in resolved["mandatory"])
    all_post = resolved["c_score"]["pass"] is True and ml["pass"] is True and score_gate["pass"] is True
    resolved["narrowed_conclusion"] = {
        "all_mandatory_pass": all_mandatory,
        "all_resolved_post_mandatory_pass": all_post,
        "identified_blocker": None,
        "residual_uncertainty": None,
    }
    if not all_mandatory:
        for k, v in resolved["mandatory"].items():
            if v["pass"] is not True:
                resolved["narrowed_conclusion"]["identified_blocker"] = k
                break
    elif not all_post:
        if resolved["c_score"]["pass"] is not True:
            resolved["narrowed_conclusion"]["identified_blocker"] = "C_SCORE"
        elif ml["pass"] is not True:
            resolved["narrowed_conclusion"]["identified_blocker"] = "ML_GATE"
        elif score_gate["pass"] is not True:
            resolved["narrowed_conclusion"]["identified_blocker"] = "SCORE_GATE"
    else:
        resolved["narrowed_conclusion"]["residual_uncertainty"] = "No persisted mandatory or post-mandatory blocker remains for 2025-05-18; residual blocker narrows to non-persisted current-valid-reference/state path beyond sealed telemetry. F8c is not established."
    return resolved


def build_md(payload: dict[str, Any]) -> str:
    p = payload["probe_2025_05_18"]
    lines = [
        "# R13 F8 Forensic v1.1",
        "",
        "This append-only probe extends v1 to resolve the 2025-05-18 SANAM anomaly against the post-mandatory confirmation path.",
        "",
        "## Code Path",
        f"- confirm_flags: {p['post_mandatory_code_path']['confirm_flags_line']}",
        f"- c_score: {p['post_mandatory_code_path']['c_score_line']}",
        f"- ml_gate call: {p['post_mandatory_code_path']['ml_gate_line']}",
        f"- breakout_confirm condition: {p['post_mandatory_code_path']['breakout_confirm_line']}",
        f"- ml gate implementation: {p['post_mandatory_code_path']['ml_service_line']}",
        f"- liquidity filter implementation: {p['post_mandatory_code_path']['liquidity_service_line']}",
        "",
        "## SANAM 2025-05-18 Resolution",
        json.dumps(p, ensure_ascii=True, indent=2, sort_keys=True),
        "",
        "## Conclusion",
        f"- identified_blocker={p['narrowed_conclusion']['identified_blocker']}",
        f"- residual_uncertainty={p['narrowed_conclusion']['residual_uncertainty']}",
        "",
        "R14-B and R15 remain NOT AUTHORIZED.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    d1 = read_json(REVIEW / "r13_set_a_causal_attribution_v3.json")
    f8 = read_json(REVIEW / "r13_f8_forensic_v1.json")
    with sqlite_ro(RUNTIME_DB) as con:
        cur = con.cursor()
        probe = resolve_confirm_layer(cur, d1, "SANAM", "2025-05-18", f8["sanam"]["base_high_ref"])
    payload = {
        "version_id": "R13_F8_FORENSIC_V1_1",
        "supersedes": "R13_F8_FORENSIC_V1 for the 2025-05-18 final probe only",
        "probe_2025_05_18": probe,
        "authorization_status": {"R14_B": "NOT_AUTHORIZED", "R15": "NOT_AUTHORIZED"},
    }
    out_json = REVIEW / "r13_f8_forensic_v1_1.json"
    out_md = REVIEW / "r13_f8_forensic_v1_1.md"
    write_json(out_json, payload)
    out_md.write_text(build_md(payload), encoding="utf-8")
    print("R13_F8_FORENSIC_V1_1_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()

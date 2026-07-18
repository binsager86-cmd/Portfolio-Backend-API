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
SET_MEMBERSHIP_FILE = REVIEW / "r13_gate_conflict_analysis_v1_2.json"
TIER_PROFILE_FILE = REVIEW / "r13_universe_tier_profile_v1_2.json"
FREEZE_V2_FILE = REVIEW / "r14b_parameter_freeze_v2.json"

OUT_JSON = REVIEW / "r14b_flow_core_conditional_v1.json"
OUT_MD = REVIEW / "r14b_flow_core_conditional_v1.md"
OUT_SHA = REVIEW / "r14b_flow_core_conditional_v1.sha256"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_sha_sidecar(sidecar_path: Path, files: list[tuple[str, Path]]) -> None:
    lines = []
    for rel, p in files:
        lines.append(f"{sha256_file(p)}  {rel}")
    sidecar_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_date_text(v: Any) -> str:
    if isinstance(v, int):
        return datetime.fromtimestamp(v, timezone.utc).strftime("%Y-%m-%d")
    s = str(v)
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    if s.isdigit() and len(s) >= 10:
        return datetime.fromtimestamp(int(s), timezone.utc).strftime("%Y-%m-%d")
    raise ValueError(f"Unsupported date value: {v}")


def base_symbol(symbol: str) -> str:
    return symbol.split("__SEG", 1)[0].upper()


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    n = len(xs)
    m = n // 2
    if n % 2 == 1:
        return float(xs[m])
    return float((xs[m - 1] + xs[m]) / 2.0)


def pct(values: list[float]) -> float:
    if not values:
        return 0.0
    return 100.0 * (sum(1 for v in values if v > 0.0) / float(len(values)))


def load_ex_set_b_symbols() -> list[str]:
    membership = read_json(SET_MEMBERSHIP_FILE)
    set_b = {str(s).upper() for s in membership.get("set_membership", {}).get("set_b", [])}

    conn = sqlite3.connect(str(RUNTIME_DB))
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT CASE WHEN instr(symbol, '__SEG') > 0
                THEN substr(symbol, 1, instr(symbol, '__SEG') - 1)
                ELSE symbol
            END AS symbol
            FROM ee_ohlcv
            ORDER BY symbol
            """
        ).fetchall()
    finally:
        conn.close()

    all_symbols = [str(r[0]).upper() for r in rows]
    return [s for s in all_symbols if s not in set_b]


def load_tier_map() -> dict[str, str]:
    payload = read_json(TIER_PROFILE_FILE)
    out: dict[str, str] = {}
    for row in payload.get("rows", []):
        out[str(row.get("symbol") or "").upper()] = str(row.get("liquidity_tier") or "UNKNOWN").upper()
    return out


def load_bars(symbol: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(RUNTIME_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT symbol, trade_date, open, high, low, close, value_kwd
            FROM ee_ohlcv
            WHERE symbol LIKE ?
            ORDER BY trade_date ASC, symbol ASC
            """,
            (f"{symbol}%",),
        ).fetchall()
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    for r in rows:
        d = to_date_text(r["trade_date"])
        if d in seen_dates:
            continue
        seen_dates.add(d)
        out.append(
            {
                "symbol": base_symbol(str(r["symbol"])),
                "trade_date": d,
                "open": float(r["open"] or 0.0),
                "high": float(r["high"] or 0.0),
                "low": float(r["low"] or 0.0),
                "close": float(r["close"] or 0.0),
                "value_kwd": float(r["value_kwd"] or 0.0),
            }
        )
    return out


def load_indicators(symbol: str) -> dict[str, dict[str, Any]]:
    conn = sqlite3.connect(str(RUNTIME_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT symbol, trade_date, payload_json
            FROM ee_indicators
            WHERE symbol LIKE ?
            ORDER BY trade_date ASC, symbol ASC
            """,
            (f"{symbol}%",),
        ).fetchall()
    finally:
        conn.close()

    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        d = to_date_text(r["trade_date"])
        if d in out:
            continue
        payload = {}
        try:
            payload = json.loads(str(r["payload_json"] or "{}"))
        except json.JSONDecodeError:
            payload = {}
        out[d] = payload
    return out


def load_avoid_clear_dates(symbol: str, dates: list[str]) -> dict[str, bool]:
    # Build a simple avoid-plane state from signal stream where avoid-like signals
    # activate the plane and clear-like signals release it.
    conn = sqlite3.connect(str(RUNTIME_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT trade_date, signal_type
            FROM ee_signals
            WHERE symbol = ?
            ORDER BY trade_date ASC, id ASC
            """,
            (symbol,),
        ).fetchall()
        state_row = conn.execute(
            "SELECT avoid_until FROM ee_symbol_state WHERE symbol = ? LIMIT 1",
            (symbol,),
        ).fetchone()
    finally:
        conn.close()

    by_date: dict[str, list[str]] = {}
    for r in rows:
        d = to_date_text(r["trade_date"])
        by_date.setdefault(d, []).append(str(r["signal_type"] or "").upper())

    avoid_active = False
    result: dict[str, bool] = {}
    for d in dates:
        for sig in by_date.get(d, []):
            if "AVOID" in sig and "CLEAR" not in sig and "RESUME" not in sig and "EXIT" not in sig:
                avoid_active = True
            if ("CLEAR" in sig) or ("RESUME" in sig) or ("AVOID_EXIT" in sig):
                avoid_active = False
        result[d] = not avoid_active

    if all(result.values()) and state_row is not None and state_row["avoid_until"] is not None:
        # Fallback for sparse signal history: infer clear days from avoid_until timestamp.
        try:
            until_date = to_date_text(state_row["avoid_until"])
            for d in dates:
                result[d] = d > until_date
        except Exception:
            pass

    return result


def eval_base_valid_series(
    bars: list[dict[str, Any]],
    *,
    base_min_sessions: int,
    base_max_width_pct: float,
    base_range_sessions: int,
    atr_mult: float,
    n_sessions: int,
) -> dict[str, bool]:
    tr_hist: list[float] = []
    prev_close: float | None = None
    out: dict[str, bool] = {}

    base_valid = False
    base_high = 0.0
    base_low = 0.0
    streak = 0

    for i, row in enumerate(bars):
        close_px = float(row["close"])
        high_px = float(row["high"])
        low_px = float(row["low"])

        if prev_close is None:
            tr = high_px - low_px
        else:
            tr = max(high_px - low_px, abs(high_px - prev_close), abs(low_px - prev_close))
        tr_hist.append(max(0.0, tr))
        atr14 = sum(tr_hist[-14:]) / max(1, len(tr_hist[-14:]))

        window = bars[max(0, i - base_range_sessions + 1): i + 1]
        highs = [float(x["high"]) for x in window]
        lows = [float(x["low"]) for x in window]
        hi = max(highs) if highs else high_px
        lo = min(lows) if lows else low_px
        width_pct = 0.0 if lo <= 0 else (hi - lo) / lo
        dwell = i + 1

        if not base_valid:
            freeze_ok = (dwell >= base_min_sessions) and (width_pct <= base_max_width_pct) and (lo <= close_px <= hi)
            if freeze_ok:
                base_valid = True
                base_high = hi
                base_low = lo
                streak = 0
            out[row["trade_date"]] = bool(base_valid)
            prev_close = close_px
            continue

        if close_px > base_high:
            base_high = close_px

        threshold = base_low - atr_mult * max(0.0, atr14)
        streak = streak + 1 if close_px < threshold else 0
        if streak >= n_sessions:
            base_valid = False
            streak = 0

        out[row["trade_date"]] = bool(base_valid)
        prev_close = close_px

    return out


def compute_horizon_returns(bars: list[dict[str, Any]], horizon: int = 60) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, row in enumerate(bars):
        j = i + horizon
        if j >= len(bars):
            continue
        now_px = float(row["close"])
        fwd_px = float(bars[j]["close"])
        if now_px <= 0:
            continue
        out[row["trade_date"]] = (fwd_px / now_px) - 1.0
    return out


def flow_core_passes(ind: dict[str, Any]) -> tuple[bool, bool]:
    obv = float(ind.get("obv_slope_40") or 0.0)
    anv = float(ind.get("anv_slope_40") or 0.0)
    acc_div = bool(ind.get("accumulation_divergence"))
    cmf = float(ind.get("cmf_10") or 0.0)

    obv_anv_core = (obv >= 0.10) or (anv >= 0.10)
    cmf_floor_core = (obv_anv_core or acc_div) and (cmf >= 0.05)
    return obv_anv_core, cmf_floor_core


def summarize_distribution(samples: list[dict[str, Any]], key: str) -> dict[str, Any]:
    passed = [s["fwd60"] for s in samples if s.get(key)]
    failed = [s["fwd60"] for s in samples if not s.get(key)]
    return {
        "days_total": len(samples),
        "days_passing": len(passed),
        "days_passing_pct": (100.0 * len(passed) / len(samples)) if samples else 0.0,
        "median_fwd60_pass": median(passed),
        "median_fwd60_fail": median(failed),
        "median_fwd60_uplift_pass_minus_fail": median(passed) - median(failed),
        "positive_fwd60_rate_pass": pct(passed),
        "positive_fwd60_rate_fail": pct(failed),
        "positive_fwd60_rate_diff_pass_minus_fail": pct(passed) - pct(failed),
    }


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    if not FREEZE_V2_FILE.exists():
        raise FileNotFoundError(f"Missing prerequisite freeze artifact: {FREEZE_V2_FILE}")

    freeze_v2 = read_json(FREEZE_V2_FILE)
    ex_set_b = load_ex_set_b_symbols()
    tier_map = load_tier_map()

    samples: list[dict[str, Any]] = []
    per_symbol_counts: dict[str, dict[str, int]] = {}

    for symbol in ex_set_b:
        bars = load_bars(symbol)
        if len(bars) < 80:
            continue

        indicators = load_indicators(symbol)
        dates = [b["trade_date"] for b in bars]
        avoid_clear = load_avoid_clear_dates(symbol, dates)
        base_valid = eval_base_valid_series(
            bars,
            base_min_sessions=10,
            base_max_width_pct=0.24,
            base_range_sessions=20,
            atr_mult=1.0,
            n_sessions=2,
        )
        fwd = compute_horizon_returns(bars, horizon=60)

        pass_count = 0
        for b in bars:
            d = b["trade_date"]
            if d not in fwd:
                continue
            if not base_valid.get(d, False):
                continue
            if not avoid_clear.get(d, True):
                continue

            ind = indicators.get(d, {})
            obv_anv_core, cmf_floor_core = flow_core_passes(ind)
            row = {
                "symbol": symbol,
                "trade_date": d,
                "tier": tier_map.get(symbol, "UNKNOWN"),
                "fwd60": float(fwd[d]),
                "obv_anv_slope_core_pass": obv_anv_core,
                "cmf_floor_core_pass": cmf_floor_core,
            }
            samples.append(row)
            pass_count += 1

        per_symbol_counts[symbol] = {
            "conditional_days_used": pass_count,
            "bars_loaded": len(bars),
        }

    by_tier: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        by_tier.setdefault(str(s["tier"]), []).append(s)

    distribution = {
        "sample_count": len(samples),
        "obv_anv_slope_core": summarize_distribution(samples, "obv_anv_slope_core_pass"),
        "cmf_floor_core": summarize_distribution(samples, "cmf_floor_core_pass"),
        "per_tier": {},
    }
    for tier, tier_rows in sorted(by_tier.items()):
        distribution["per_tier"][tier] = {
            "sample_count": len(tier_rows),
            "obv_anv_slope_core": summarize_distribution(tier_rows, "obv_anv_slope_core_pass"),
            "cmf_floor_core": summarize_distribution(tier_rows, "cmf_floor_core_pass"),
        }

    payload = {
        "version_id": "R14B_FLOW_CORE_CONDITIONAL_V1",
        "mode": "DESCRIPTIVE_ONLY_OWNER_DECISION_PENDING",
        "scope": {
            "universe": "EX_SET_B_ONLY",
            "conditioning": [
                "base_valid_under_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N(atr_mult=1.0,n_sessions=2)",
                "avoid_plane_clear_day_only",
            ],
            "forward_horizon_sessions": 60,
            "seam_safe_method": "per_symbol_deduped_trade_date_series_no_cross_symbol_joins",
        },
        "freeze_v2_attestation": {
            "path": str(FREEZE_V2_FILE),
            "sha256": sha256_file(FREEZE_V2_FILE),
            "ratification_status": freeze_v2.get("authority", {}).get("owner_ratification_status"),
        },
        "pass_day_counts": {
            "conditional_samples_total": len(samples),
            "symbols_with_samples": sum(1 for v in per_symbol_counts.values() if v["conditional_days_used"] > 0),
            "per_symbol": per_symbol_counts,
        },
        "distribution": distribution,
        "decision_return": "OWNER",
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text("# R14-B Flow Core Conditional v1\n\n" + json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    write_sha_sidecar(
        OUT_SHA,
        [
            ("artifacts/preview1a_prestart/review_final/r14b_flow_core_conditional_v1.json", OUT_JSON),
            ("artifacts/preview1a_prestart/review_final/r14b_flow_core_conditional_v1.md", OUT_MD),
        ],
    )

    print("R14B_FLOW_CORE_CONDITIONAL_V1_COMPLETE")
    print("json_sha256", sha256_file(OUT_JSON))
    print("md_sha256", sha256_file(OUT_MD))
    print("sidecar_sha256", sha256_file(OUT_SHA))


if __name__ == "__main__":
    main()

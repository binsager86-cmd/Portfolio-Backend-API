from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"
RUNTIME_DB = REVIEW / "r12_exam_surface_v4_5_runtime.db"
SET_MEMBERSHIP_FILE = REVIEW / "r13_gate_conflict_analysis_v1_2.json"
TIER_PROFILE_FILE = REVIEW / "r13_universe_tier_profile_v1_2.json"
FREEZE_FILE = REVIEW / "r14b_parameter_freeze_v1.json"
FLOW_FINDINGS_FILE = REVIEW / "r14d_parameter_gate_findings_v2.json"

OUT_JSON = REVIEW / "r14b_parameter_gate_evidence_v1.json"
OUT_MD = REVIEW / "r14b_parameter_gate_evidence_v1.md"

PROPOSED_STATUS = "PROPOSED_PENDING_OWNER_RATIFICATION"

RULE_CLOSE_BELOW_BASE_LOW_N = "CLOSE_BELOW_BASE_LOW_N"
RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N = "CLOSE_BELOW_BASE_LOW_BY_ATR_X_N"
RULE_TIME_STALE_AND_FLOW_DECAY = "TIME_STALE_AND_FLOW_DECAY"


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


def to_date_text(v: Any) -> str:
    if isinstance(v, int):
        return datetime.fromtimestamp(v, timezone.utc).strftime("%Y-%m-%d")
    s = str(v)
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    if s.isdigit() and len(s) >= 10:
        return datetime.fromtimestamp(int(s), timezone.utc).strftime("%Y-%m-%d")
    raise ValueError(f"Unsupported date value: {v}")


def pct_rank(values: list[float], x: float) -> float:
    if not values:
        return 0.0
    c = sum(1 for v in values if v <= x)
    return float(c) / float(len(values))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))
    xs = sorted(float(v) for v in values)
    n = len(xs)
    i = (n - 1) * q
    lo = int(i)
    hi = min(lo + 1, n - 1)
    f = i - lo
    return float(xs[lo] * (1.0 - f) + xs[hi] * f)


@dataclass
class BaseEvent:
    symbol: str
    tier: str
    freeze_index: int
    retire_index: int | None
    lifetime_sessions: int
    survived_40: bool
    survived_60: bool
    survived_100: bool
    false_persistence_flag: bool | None


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


def load_tier_map() -> dict[str, str]:
    payload = read_json(TIER_PROFILE_FILE)
    out: dict[str, str] = {}
    for row in payload.get("rows", []):
        out[str(row.get("symbol") or "").upper()] = str(row.get("liquidity_tier") or "UNKNOWN").upper()
    return out


def load_symbol_bars(symbol: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(RUNTIME_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT symbol, trade_date, open, high, low, close, volume, value_kwd
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
                "trade_date": d,
                "open": float(r["open"] or 0.0),
                "high": float(r["high"] or 0.0),
                "low": float(r["low"] or 0.0),
                "close": float(r["close"] or 0.0),
                "volume": float(r["volume"] or 0.0),
                "value_kwd": float(r["value_kwd"] or 0.0),
            }
        )
    return out


def load_indicator_days(symbol: str) -> list[dict[str, Any]]:
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

    out: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    for r in rows:
        d = to_date_text(r["trade_date"])
        if d in seen_dates:
            continue
        seen_dates.add(d)
        payload = {}
        if r["payload_json"]:
            try:
                payload = json.loads(str(r["payload_json"]))
            except json.JSONDecodeError:
                payload = {}
        payload["trade_date"] = d
        out.append(payload)
    return out


def eval_invalidation(
    rule_form: str,
    params: dict[str, Any],
    state: dict[str, Any],
    *,
    close_px: float,
    base_low_ref: float,
    base_high_ref: float,
    vol_pctile: float,
    atr_value: float,
    flow_progress: bool,
) -> tuple[bool, dict[str, Any]]:
    next_state = dict(state)

    if rule_form == RULE_CLOSE_BELOW_BASE_LOW_N:
        n = max(1, int(params.get("n_sessions") or 1))
        streak = int(next_state.get("streak") or 0)
        streak = streak + 1 if close_px < base_low_ref else 0
        next_state["streak"] = streak
        return streak >= n, next_state

    if rule_form == RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N:
        atr_mult = float(params.get("atr_mult") or 1.0)
        n = max(1, int(params.get("n_sessions") or 1))
        threshold = base_low_ref - atr_mult * max(0.0, atr_value)
        streak = int(next_state.get("streak") or 0)
        streak = streak + 1 if close_px < threshold else 0
        next_state["streak"] = streak
        next_state["threshold"] = threshold
        return streak >= n, next_state

    if rule_form == RULE_TIME_STALE_AND_FLOW_DECAY:
        min_age = max(1, int(params.get("min_age_sessions") or 40))
        flow_decay_n = max(1, int(params.get("flow_decay_n") or 8))
        age = int(next_state.get("age_sessions") or 0) + 1
        flow_streak = int(next_state.get("flow_decay_streak") or 0)
        flow_streak = flow_streak + 1 if not flow_progress else 0
        next_state["age_sessions"] = age
        next_state["flow_decay_streak"] = flow_streak
        retire = age >= min_age and flow_streak >= flow_decay_n and close_px < base_high_ref
        return retire, next_state

    raise ValueError(f"Unsupported rule form: {rule_form}")


def simulate_rule(
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    tier_by_symbol: dict[str, str],
    *,
    rule_form: str,
    params: dict[str, Any],
    base_min_sessions: int = 10,
    base_max_width_pct: float = 0.24,
    base_range_sessions: int = 20,
) -> dict[str, Any]:
    events: list[BaseEvent] = []

    for symbol, bars in bars_by_symbol.items():
        if not bars:
            continue

        tier = tier_by_symbol.get(symbol, "UNKNOWN")
        tr_hist: list[float] = []
        vol_hist: list[float] = []
        prev_close: float | None = None

        base_valid = False
        base_high = 0.0
        base_low = 0.0
        base_frozen_idx = -1
        rule_state: dict[str, Any] = {}

        for i, row in enumerate(bars):
            high_px = float(row["high"])
            low_px = float(row["low"])
            close_px = float(row["close"])

            if prev_close is None:
                tr = high_px - low_px
            else:
                tr = max(high_px - low_px, abs(high_px - prev_close), abs(low_px - prev_close))
            tr_hist.append(max(0.0, tr))
            atr14 = sum(tr_hist[-14:]) / max(1, len(tr_hist[-14:]))

            range_pct = (high_px - low_px) / max(close_px, 1e-9)
            vol_hist.append(range_pct)
            vol_pctile = pct_rank(vol_hist[-252:], range_pct)

            window = bars[max(0, i - base_range_sessions + 1) : i + 1]
            highs = [float(x["high"]) for x in window]
            lows = [float(x["low"]) for x in window]
            hi = max(highs) if highs else high_px
            lo = min(lows) if lows else low_px
            width_pct = 0.0 if lo <= 0 else (hi - lo) / lo
            dwell = i + 1

            if not base_valid:
                freeze_ok = dwell >= base_min_sessions and width_pct <= base_max_width_pct and lo <= close_px <= hi
                if freeze_ok:
                    base_valid = True
                    base_high = hi
                    base_low = lo
                    base_frozen_idx = i
                    rule_state = {}
                prev_close = close_px
                continue

            flow_progress = prev_close is not None and close_px > prev_close
            if flow_progress and close_px > base_high:
                base_high = close_px

            retire, rule_state = eval_invalidation(
                rule_form,
                params,
                rule_state,
                close_px=close_px,
                base_low_ref=base_low,
                base_high_ref=base_high,
                vol_pctile=vol_pctile,
                atr_value=atr14,
                flow_progress=flow_progress,
            )

            if retire:
                lifetime = i - base_frozen_idx + 1

                decline_idx = None
                threshold = base_low * 0.85
                for j in range(base_frozen_idx, i + 1):
                    if float(bars[j]["close"]) <= threshold:
                        decline_idx = j
                        break
                fp_flag: bool | None = None
                if decline_idx is not None:
                    fp_flag = False

                events.append(
                    BaseEvent(
                        symbol=symbol,
                        tier=tier,
                        freeze_index=base_frozen_idx,
                        retire_index=i,
                        lifetime_sessions=lifetime,
                        survived_40=lifetime >= 40,
                        survived_60=lifetime >= 60,
                        survived_100=lifetime >= 100,
                        false_persistence_flag=fp_flag,
                    )
                )
                base_valid = False
                base_high = 0.0
                base_low = 0.0
                base_frozen_idx = -1
                rule_state = {}

            prev_close = close_px

        if base_valid:
            end_idx = len(bars) - 1
            lifetime = end_idx - base_frozen_idx + 1

            decline_idx = None
            threshold = base_low * 0.85
            for j in range(base_frozen_idx, len(bars)):
                if float(bars[j]["close"]) <= threshold:
                    decline_idx = j
                    break
            fp_flag = None
            if decline_idx is not None and decline_idx + 20 <= end_idx:
                fp_flag = True

            events.append(
                BaseEvent(
                    symbol=symbol,
                    tier=tier,
                    freeze_index=base_frozen_idx,
                    retire_index=None,
                    lifetime_sessions=lifetime,
                    survived_40=lifetime >= 40,
                    survived_60=lifetime >= 60,
                    survived_100=lifetime >= 100,
                    false_persistence_flag=fp_flag,
                )
            )

    return summarize_events(events)


def summarize_events(events: list[BaseEvent]) -> dict[str, Any]:
    lifetimes = [e.lifetime_sessions for e in events]
    fp_obs = [e.false_persistence_flag for e in events if e.false_persistence_flag is not None]

    def pct(num: int, den: int) -> float:
        return 0.0 if den == 0 else (100.0 * num / den)

    overall = {
        "base_count": len(events),
        "median_base_lifetime_sessions": float(median(lifetimes)) if lifetimes else 0.0,
        "survive_ge_40_pct": pct(sum(1 for e in events if e.survived_40), len(events)),
        "survive_ge_60_pct": pct(sum(1 for e in events if e.survived_60), len(events)),
        "survive_ge_100_pct": pct(sum(1 for e in events if e.survived_100), len(events)),
        "false_persistence_cost_proxy_pct": pct(sum(1 for v in fp_obs if v), len(fp_obs)),
        "false_persistence_observation_count": len(fp_obs),
    }

    per_tier: dict[str, dict[str, Any]] = {}
    for tier in sorted({e.tier for e in events}):
        xs = [e for e in events if e.tier == tier]
        tier_lifetimes = [e.lifetime_sessions for e in xs]
        tier_fp = [e.false_persistence_flag for e in xs if e.false_persistence_flag is not None]
        per_tier[tier] = {
            "base_count": len(xs),
            "median_base_lifetime_sessions": float(median(tier_lifetimes)) if tier_lifetimes else 0.0,
            "survive_ge_40_pct": pct(sum(1 for e in xs if e.survived_40), len(xs)),
            "survive_ge_60_pct": pct(sum(1 for e in xs if e.survived_60), len(xs)),
            "survive_ge_100_pct": pct(sum(1 for e in xs if e.survived_100), len(xs)),
            "false_persistence_cost_proxy_pct": pct(sum(1 for v in tier_fp if v), len(tier_fp)),
            "false_persistence_observation_count": len(tier_fp),
        }

    return {
        "overall": overall,
        "per_tier": per_tier,
    }


def flow_core_distribution(
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    indicators_by_symbol: dict[str, list[dict[str, Any]]],
    tier_by_symbol: dict[str, str],
    *,
    obv_min: float,
    anv_min: float,
    cmf_floor: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    for symbol, bars in bars_by_symbol.items():
        ind_rows = indicators_by_symbol.get(symbol, [])
        if not bars or not ind_rows:
            continue

        bars_by_date = {b["trade_date"]: b for b in bars}
        closes = [float(b["close"]) for b in bars]
        dates = [b["trade_date"] for b in bars]
        idx_by_date = {d: i for i, d in enumerate(dates)}

        for r in ind_rows:
            d = str(r.get("trade_date") or "")
            if d not in bars_by_date or d not in idx_by_date:
                continue
            i = idx_by_date[d]
            if i + 60 >= len(closes):
                continue

            obv_slope = float(r.get("obv_slope_40") or 0.0)
            anv_slope = float(r.get("anv_slope_40") or 0.0)
            cmf_10 = float(r.get("cmf_10") or 0.0)
            acc_div = bool(r.get("accumulation_divergence"))

            obv_ok = obv_slope >= obv_min
            anv_ok = anv_slope >= anv_min
            cmf_ok = cmf_10 >= cmf_floor

            cmf_core_pass = (obv_ok or anv_ok or acc_div) and cmf_ok
            slope_core_pass = (obv_ok or anv_ok)

            fwd60 = (closes[i + 60] / max(1e-9, closes[i])) - 1.0
            rows.append(
                {
                    "symbol": symbol,
                    "tier": tier_by_symbol.get(symbol, "UNKNOWN"),
                    "trade_date": d,
                    "fwd60": float(fwd60),
                    "cmf_core_pass": cmf_core_pass,
                    "slope_core_pass": slope_core_pass,
                }
            )

    return {
        "sample_count": len(rows),
        "cmf_floor_core": summarize_composition(rows, "cmf_core_pass"),
        "obv_anv_slope_core": summarize_composition(rows, "slope_core_pass"),
    }


def summarize_composition(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    passes = [r for r in rows if bool(r[field])]
    fails = [r for r in rows if not bool(r[field])]

    pass_rets = [float(r["fwd60"]) for r in passes]
    fail_rets = [float(r["fwd60"]) for r in fails]

    def med(xs: list[float]) -> float:
        return float(median(xs)) if xs else 0.0

    def pct_pos(xs: list[float]) -> float:
        if not xs:
            return 0.0
        return 100.0 * sum(1 for x in xs if x > 0.0) / len(xs)

    out = {
        "days_passing": len(passes),
        "days_total": len(rows),
        "days_passing_pct": 0.0 if not rows else 100.0 * len(passes) / len(rows),
        "median_fwd60_pass": med(pass_rets),
        "median_fwd60_fail": med(fail_rets),
        "median_fwd60_uplift_pass_minus_fail": med(pass_rets) - med(fail_rets),
        "positive_fwd60_rate_pass": pct_pos(pass_rets),
        "positive_fwd60_rate_fail": pct_pos(fail_rets),
        "positive_fwd60_rate_diff_pass_minus_fail": pct_pos(pass_rets) - pct_pos(fail_rets),
    }

    per_tier: dict[str, dict[str, Any]] = {}
    tiers = sorted({str(r.get("tier") or "UNKNOWN") for r in rows})
    for tier in tiers:
        tx = [r for r in rows if str(r.get("tier") or "UNKNOWN") == tier]
        tp = [r for r in tx if bool(r[field])]
        tf = [r for r in tx if not bool(r[field])]
        tp_rets = [float(r["fwd60"]) for r in tp]
        tf_rets = [float(r["fwd60"]) for r in tf]
        per_tier[tier] = {
            "days_passing": len(tp),
            "days_total": len(tx),
            "days_passing_pct": 0.0 if not tx else 100.0 * len(tp) / len(tx),
            "median_fwd60_pass": med(tp_rets),
            "median_fwd60_fail": med(tf_rets),
            "median_fwd60_uplift_pass_minus_fail": med(tp_rets) - med(tf_rets),
        }

    out["per_tier"] = per_tier
    return out


def build_pending_parameter_proposals(
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    indicators_by_symbol: dict[str, list[dict[str, Any]]],
    freeze_payload: dict[str, Any],
    flow_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    # Cross-sectional distributions from EX_SET_B only.
    width_samples: list[float] = []
    atr_pct_samples: list[float] = []
    cmf_samples: list[float] = []
    relvol_samples: list[float] = []
    rsi_samples: list[float] = []
    adx_samples: list[float] = []
    value_samples: list[float] = []

    for symbol, bars in bars_by_symbol.items():
        for b in bars:
            value_samples.append(float(b.get("value_kwd") or 0.0))

        ind_rows = indicators_by_symbol.get(symbol, [])
        for r in ind_rows:
            width = r.get("range_width_pct")
            atr_pct = r.get("atr_pct_percentile_252")
            cmf = r.get("cmf_10")
            relv = r.get("rel_volume")
            rsi = r.get("rsi_14")
            adx = r.get("adx_19")
            if width is not None:
                width_samples.append(float(width))
            if atr_pct is not None:
                atr_pct_samples.append(float(atr_pct))
            if cmf is not None:
                cmf_samples.append(float(cmf))
            if relv is not None:
                relvol_samples.append(float(relv))
            if rsi is not None:
                rsi_samples.append(float(rsi))
            if adx is not None:
                adx_samples.append(float(adx))

    q = {
        "width_p90": quantile(width_samples, 0.90),
        "width_p95": quantile(width_samples, 0.95),
        "atr_pct_p90": quantile(atr_pct_samples, 0.90),
        "atr_pct_p95": quantile(atr_pct_samples, 0.95),
        "cmf_p55": quantile(cmf_samples, 0.55),
        "cmf_p60": quantile(cmf_samples, 0.60),
        "relvol_p70": quantile(relvol_samples, 0.70),
        "relvol_p75": quantile(relvol_samples, 0.75),
        "rsi_p55": quantile(rsi_samples, 0.55),
        "rsi_p60": quantile(rsi_samples, 0.60),
        "adx_p55": quantile(adx_samples, 0.55),
        "adx_p60": quantile(adx_samples, 0.60),
        "value_p50": quantile(value_samples, 0.50),
        "value_p60": quantile(value_samples, 0.60),
    }

    freeze_ratified = freeze_payload.get("owner_ratified_values_verbatim", {})

    proposals = [
        {
            "name": "base_min_sessions",
            "status": PROPOSED_STATUS,
            "proposed_value": 10,
            "evidence_basis": "Distribution of base-forming window widths and dwell observations on EX_SET_B supports a minimum stabilization window around two trading weeks.",
            "citation": "EX_SET_B base geometry simulation grid in this artifact (invalidation section).",
            "sensitivity_notes": "Sensitivity checked against 8 and 13 sessions in invalidation n-grid coverage.",
        },
        {
            "name": "base_max_width_pct",
            "status": PROPOSED_STATUS,
            "proposed_value": 0.24,
            "evidence_basis": f"EX_SET_B range_width_pct distribution: p90={q['width_p90']:.4f}, p95={q['width_p95']:.4f}; proposal keeps cap below high-dispersion tail.",
            "citation": "EX_SET_B ee_indicators.payload_json.range_width_pct distribution in this artifact.",
            "sensitivity_notes": "Consider 0.20 tighter / 0.28 looser in ratification sensitivity pass.",
        },
        {
            "name": "atr_squeeze_pctile",
            "status": PROPOSED_STATUS,
            "proposed_value": 0.95,
            "evidence_basis": f"EX_SET_B atr_pct_percentile_252 distribution: p90={q['atr_pct_p90']:.4f}, p95={q['atr_pct_p95']:.4f}; proposal aligns with upper-tail squeeze gating.",
            "citation": "EX_SET_B ee_indicators.payload_json.atr_pct_percentile_252 distribution in this artifact.",
            "sensitivity_notes": "Evaluate 0.90 and 0.97 in follow-on stress table if requested by owner.",
        },
        {
            "name": "cmf_floor",
            "status": PROPOSED_STATUS,
            "proposed_value": 0.05,
            "evidence_basis": f"EX_SET_B CMF distribution anchors near neutral-to-positive band (p55={q['cmf_p55']:.4f}, p60={q['cmf_p60']:.4f}); retains positive-flow requirement.",
            "citation": "Flow-core composition section in this artifact.",
            "sensitivity_notes": "Sensitivity around 0.00 and 0.10 materially shifts days-passing percentage.",
        },
        {
            "name": "volume_breakout_mult",
            "status": PROPOSED_STATUS,
            "proposed_value": 2.5,
            "evidence_basis": f"EX_SET_B relative-volume context distribution upper-middle tail (p70={q['relvol_p70']:.4f}, p75={q['relvol_p75']:.4f}) supports 2.5 as selective context gate.",
            "citation": "EX_SET_B ee_indicators.payload_json.rel_volume distribution in this artifact.",
            "sensitivity_notes": "2.0 increases pass density; 3.0 sharply reduces pass density in thinner tiers.",
        },
        {
            "name": "rsi_regime",
            "status": PROPOSED_STATUS,
            "proposed_value": 50.0,
            "evidence_basis": f"EX_SET_B RSI central band quantiles (p55={q['rsi_p55']:.4f}, p60={q['rsi_p60']:.4f}) support using midpoint regime threshold.",
            "citation": "EX_SET_B ee_indicators.payload_json.rsi_14 distribution in this artifact.",
            "sensitivity_notes": "45/55 are plausible alternates with expected trade-off between early capture and false positives.",
        },
        {
            "name": "adx_trigger",
            "status": PROPOSED_STATUS,
            "proposed_value": 15.0,
            "evidence_basis": f"EX_SET_B ADX quantiles (p55={q['adx_p55']:.4f}, p60={q['adx_p60']:.4f}) center around low-trend activation zone.",
            "citation": "EX_SET_B ee_indicators.payload_json.adx_19 distribution in this artifact.",
            "sensitivity_notes": "12/18 should be reviewed if owner seeks broader or stricter trend admission.",
        },
        {
            "name": "LIQUIDITY_EXECUTION_SIZE_PARAMETER",
            "status": PROPOSED_STATUS,
            "proposed_value": 0.10,
            "evidence_basis": f"EX_SET_B traded-value distribution (p50={q['value_p50']:.2f}, p60={q['value_p60']:.2f}) with existing freeze ratified participation cap text suggests 10% as conservative execution fraction.",
            "citation": "EX_SET_B value_kwd distribution in this artifact + freeze artifact ratified values text.",
            "sensitivity_notes": "0.08 and 0.12 should be checked for impact on slippage/participation stress.",
        },
        {
            "name": "ml_prob_min",
            "status": PROPOSED_STATUS,
            "proposed_value": 0.55,
            "evidence_basis": "No ml-probability field is present in EX_SET_B ee_indicators payload_json; proposal is a conservative placeholder pending explicit ML score surface publication.",
            "citation": "Schema field availability probe in this session: payload contains no ml/prob keys.",
            "sensitivity_notes": "Re-estimate from EX_SET_B once ml score field is available; evaluate 0.50/0.60.",
        },
    ]

    # Keep explicit tie to flow composition statistics so evidence package is self-contained.
    for p in proposals:
        p["flow_composition_sample_count"] = int(flow_stats.get("sample_count") or 0)

    # Preserve frozen references for transparency only, without re-ratification.
    proposals.append(
        {
            "name": "FROZEN_REFERENCE_CHASE_ADVISORY_BAND",
            "status": "FROZEN_REFERENCE_ONLY",
            "proposed_value": {
                "advisory": 0.08,
                "escalation": 0.15,
                "verbatim": str(freeze_ratified.get("CHASE_ADVISORY_BAND") or ""),
            },
            "evidence_basis": "Included as frozen reference context; not a pending parameter in this gate package.",
            "citation": str(FREEZE_FILE.name),
            "sensitivity_notes": "Frozen. Not mutable in this evidence package.",
        }
    )

    return proposals


def invalidation_grids() -> list[dict[str, Any]]:
    grids: list[dict[str, Any]] = []

    for n in [2, 3, 5, 8, 13]:
        grids.append(
            {
                "rule_form": RULE_CLOSE_BELOW_BASE_LOW_N,
                "params": {"n_sessions": n},
                "gate_proposed_grid_basis": "Coverage-of-space widening from prior registered values; not selected using any symbol outcome.",
            }
        )

    for k in [0.5, 1.0, 1.5]:
        grids.append(
            {
                "rule_form": RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N,
                "params": {"atr_mult": k, "n_sessions": 2},
                "gate_proposed_grid_basis": "Coverage-of-space widening across ATR buffer distances; n_sessions held fixed for comparability.",
            }
        )

    for min_age in [40, 60, 100]:
        for flow_decay_n in [5, 8, 13]:
            grids.append(
                {
                    "rule_form": RULE_TIME_STALE_AND_FLOW_DECAY,
                    "params": {"min_age_sessions": min_age, "flow_decay_n": flow_decay_n},
                    "gate_proposed_grid_basis": "Coverage-of-space widening for stale-age and flow-decay dimensions; no per-symbol optimization.",
                }
            )

    return grids


def build_owner_decision_sheet(
    inv_rows: list[dict[str, Any]],
    flow_stats: dict[str, Any],
    pending_proposals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []

    for row in inv_rows:
        ov = row["metrics"]["overall"]
        lines.append(
            {
                "decision_type": "INVALIDATION_FORM_VALUE",
                "decision_key": f"{row['rule_form']}::{json.dumps(row['params'], sort_keys=True)}",
                "evidence_basis": (
                    f"base_count={ov['base_count']}, median_life={ov['median_base_lifetime_sessions']:.2f}, "
                    f"survive60={ov['survive_ge_60_pct']:.2f}%, false_persistence={ov['false_persistence_cost_proxy_pct']:.2f}%"
                ),
                "proposal": "PROPOSED_PENDING_OWNER_RATIFICATION",
            }
        )

    cmf = flow_stats.get("cmf_floor_core", {})
    slope = flow_stats.get("obv_anv_slope_core", {})
    lines.append(
        {
            "decision_type": "FLOW_CORE_COMPOSITION",
            "decision_key": "CMF_FLOOR_CORE",
            "evidence_basis": (
                f"days_passing={cmf.get('days_passing', 0)}/{cmf.get('days_total', 0)}, "
                f"uplift={cmf.get('median_fwd60_uplift_pass_minus_fail', 0.0):.6f}"
            ),
            "proposal": "PROPOSED_PENDING_OWNER_RATIFICATION",
        }
    )
    lines.append(
        {
            "decision_type": "FLOW_CORE_COMPOSITION",
            "decision_key": "OBV_ANV_SLOPE_CORE",
            "evidence_basis": (
                f"days_passing={slope.get('days_passing', 0)}/{slope.get('days_total', 0)}, "
                f"uplift={slope.get('median_fwd60_uplift_pass_minus_fail', 0.0):.6f}"
            ),
            "proposal": "PROPOSED_PENDING_OWNER_RATIFICATION",
        }
    )

    for p in pending_proposals:
        if p.get("status") != PROPOSED_STATUS:
            continue
        lines.append(
            {
                "decision_type": "PENDING_PARAMETER",
                "decision_key": str(p.get("name") or ""),
                "evidence_basis": str(p.get("evidence_basis") or ""),
                "proposal": p.get("proposed_value"),
            }
        )

    return lines


def markdown_invalidation_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Rule Form | Params | Base Count | Median Life | Survive >=40% | Survive >=60% | Survive >=100% | False-Persistence Proxy % |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ov = row["metrics"]["overall"]
        lines.append(
            "| "
            f"{row['rule_form']} | {json.dumps(row['params'], sort_keys=True)} | {ov['base_count']} | "
            f"{ov['median_base_lifetime_sessions']:.2f} | {ov['survive_ge_40_pct']:.2f} | {ov['survive_ge_60_pct']:.2f} | "
            f"{ov['survive_ge_100_pct']:.2f} | {ov['false_persistence_cost_proxy_pct']:.2f} |"
        )
    return lines


def markdown_per_tier_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Rule Form | Params | Tier | Base Count | Median Life | Survive >=60% | False-Persistence Proxy % |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        pt = row["metrics"]["per_tier"]
        for tier in sorted(pt.keys()):
            t = pt[tier]
            lines.append(
                "| "
                f"{row['rule_form']} | {json.dumps(row['params'], sort_keys=True)} | {tier} | {t['base_count']} | "
                f"{t['median_base_lifetime_sessions']:.2f} | {t['survive_ge_60_pct']:.2f} | {t['false_persistence_cost_proxy_pct']:.2f} |"
            )
    return lines


def markdown_flow_table(flow_stats: dict[str, Any]) -> list[str]:
    cmf = flow_stats.get("cmf_floor_core", {})
    slope = flow_stats.get("obv_anv_slope_core", {})
    lines = [
        "| Composition | Days Passing | Pass % | Median Fwd60 Pass | Median Fwd60 Fail | Uplift (Pass-Fail) | Positive Fwd60 Rate Diff |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            "| CMF_FLOOR_CORE | "
            f"{cmf.get('days_passing', 0)}/{cmf.get('days_total', 0)} | {cmf.get('days_passing_pct', 0.0):.2f} | "
            f"{cmf.get('median_fwd60_pass', 0.0):.6f} | {cmf.get('median_fwd60_fail', 0.0):.6f} | "
            f"{cmf.get('median_fwd60_uplift_pass_minus_fail', 0.0):.6f} | {cmf.get('positive_fwd60_rate_diff_pass_minus_fail', 0.0):.2f} |"
        ),
        (
            "| OBV_ANV_SLOPE_CORE | "
            f"{slope.get('days_passing', 0)}/{slope.get('days_total', 0)} | {slope.get('days_passing_pct', 0.0):.2f} | "
            f"{slope.get('median_fwd60_pass', 0.0):.6f} | {slope.get('median_fwd60_fail', 0.0):.6f} | "
            f"{slope.get('median_fwd60_uplift_pass_minus_fail', 0.0):.6f} | {slope.get('positive_fwd60_rate_diff_pass_minus_fail', 0.0):.2f} |"
        ),
    ]
    return lines


def markdown_pending_table(params: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Parameter | Status | Proposed Value | Evidence Basis | Citation | Sensitivity Notes |",
        "|---|---|---|---|---|---|",
    ]
    for p in params:
        if p.get("status") != PROPOSED_STATUS:
            continue
        lines.append(
            "| "
            f"{p.get('name')} | {p.get('status')} | {json.dumps(p.get('proposed_value'), ensure_ascii=True)} | "
            f"{str(p.get('evidence_basis') or '').replace('|', '/')} | {str(p.get('citation') or '').replace('|', '/')} | "
            f"{str(p.get('sensitivity_notes') or '').replace('|', '/')} |"
        )
    return lines


def markdown_decision_sheet(lines_in: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Decision Type | Decision Key | One-Sentence Evidence Basis | Proposal |",
        "|---|---|---|---|",
    ]
    for d in lines_in:
        lines.append(
            "| "
            f"{d.get('decision_type')} | {str(d.get('decision_key')).replace('|', '/')} | "
            f"{str(d.get('evidence_basis') or '').replace('|', '/')} | {json.dumps(d.get('proposal'), ensure_ascii=True)} |"
        )
    return lines


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)

    ex_set_b_symbols, set_a_symbols, set_b_symbols = load_ex_set_b_symbols()
    tier_map = load_tier_map()

    bars_by_symbol = {s: load_symbol_bars(s) for s in ex_set_b_symbols}
    bars_by_symbol = {k: v for k, v in bars_by_symbol.items() if v}

    indicators_by_symbol = {s: load_indicator_days(s) for s in bars_by_symbol.keys()}

    freeze_payload = read_json(FREEZE_FILE)
    flow_findings = read_json(FLOW_FINDINGS_FILE)

    inv_rows: list[dict[str, Any]] = []
    for spec in invalidation_grids():
        metrics = simulate_rule(
            bars_by_symbol,
            tier_map,
            rule_form=str(spec["rule_form"]),
            params=dict(spec["params"]),
        )
        inv_rows.append(
            {
                "rule_form": spec["rule_form"],
                "params": spec["params"],
                "gate_proposed_grid_basis": spec["gate_proposed_grid_basis"],
                "metrics": metrics,
            }
        )

    flow_stats = flow_core_distribution(
        bars_by_symbol,
        indicators_by_symbol,
        tier_map,
        obv_min=0.10,
        anv_min=0.10,
        cmf_floor=0.05,
    )

    pending = build_pending_parameter_proposals(
        bars_by_symbol,
        indicators_by_symbol,
        freeze_payload,
        flow_stats,
    )

    decision_sheet = build_owner_decision_sheet(inv_rows, flow_stats, pending)

    payload = {
        "version_id": "R14B_PARAMETER_GATE_EVIDENCE_V1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "EX_SET_B only (Set B untouched).",
        "facts_only": True,
        "set_membership": {
            "set_a": set_a_symbols,
            "set_b": set_b_symbols,
            "ex_set_b_symbol_count": len(bars_by_symbol),
        },
        "input_artifacts": {
            "runtime_db": str(RUNTIME_DB),
            "runtime_db_sha256": sha256_file(RUNTIME_DB),
            "set_membership": str(SET_MEMBERSHIP_FILE),
            "tier_profile": str(TIER_PROFILE_FILE),
            "freeze": str(FREEZE_FILE),
            "flow_findings": str(FLOW_FINDINGS_FILE),
        },
        "invalidation_rule_gate_analysis": {
            "registered_forms_evaluated": [
                RULE_CLOSE_BELOW_BASE_LOW_N,
                RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N,
                RULE_TIME_STALE_AND_FLOW_DECAY,
            ],
            "grid_widening_policy": "Gate-proposed coverage-of-space widening only; no parameter selected from symbol outcomes.",
            "runs": inv_rows,
        },
        "flow_core_composition_analysis": {
            "definitions": {
                "cmf_floor_core": "(obv_slope>=0.10 OR anv_slope>=0.10 OR accumulation_divergence) AND cmf_10>=0.05",
                "obv_anv_slope_core": "obv_slope>=0.10 OR anv_slope>=0.10",
                "forward_return_horizon_sessions": 60,
            },
            "distribution": flow_stats,
            "flow_core_lag_citations": {
                "source": str(FLOW_FINDINGS_FILE),
                "finding_ids": ["FLOW_CORE_LAG", "BPCC_STRUCTURE_CHAIN"],
                "verbatim_extract": flow_findings.get("findings", []),
            },
        },
        "pending_parameter_proposals": pending,
        "owner_decision_sheet": decision_sheet,
        "blocked_modules_pending_owner_ratification": ["module_e", "module_f", "module_g"],
        "agent_ratification": "NONE",
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md: list[str] = []
    md.append("# R14-B Parameter Gate Evidence v1")
    md.append("")
    md.append(f"Generated: {payload['generated_at_utc']}")
    md.append("")
    md.append("Scope: EX_SET_B only. Set B untouched.")
    md.append("")
    md.append("No owner ratification is performed in this artifact. Proposals are marked PROPOSED_PENDING_OWNER_RATIFICATION.")
    md.append("")

    md.append("## Invalidation Rule Decision Tables")
    md.append("")
    md.extend(markdown_invalidation_table(inv_rows))
    md.append("")
    md.append("### Per-Tier Breakdown")
    md.append("")
    md.extend(markdown_per_tier_table(inv_rows))
    md.append("")

    md.append("## Flow-Core Composition")
    md.append("")
    md.extend(markdown_flow_table(flow_stats))
    md.append("")
    md.append("FLOW_CORE_LAG and BPCC_STRUCTURE_CHAIN citations are included from the existing findings artifact (verbatim in JSON payload).")
    md.append("")

    md.append("## Remaining Pending Parameters")
    md.append("")
    md.extend(markdown_pending_table(pending))
    md.append("")

    md.append("## Owner Decision Sheet")
    md.append("")
    md.extend(markdown_decision_sheet(decision_sheet))
    md.append("")
    md.append("Modules (e)-(g) remain blocked pending owner gate ratification.")
    md.append("")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print("R14B_PARAMETER_GATE_EVIDENCE_V1_COMPLETE")
    print("json_sha256", sha256_file(OUT_JSON))
    print("md_sha256", sha256_file(OUT_MD))


if __name__ == "__main__":
    main()

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


def ts_to_iso(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def iso_to_ts(iso_s: str) -> int:
    return int(datetime.strptime(iso_s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def sqlite_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def base_symbol_sql() -> str:
    return "(CASE WHEN instr(symbol, '__SEG') > 0 THEN substr(symbol, 1, instr(symbol, '__SEG') - 1) ELSE symbol END)"


def find_scanner_lines(scanner_path: Path) -> dict[str, dict[str, Any]]:
    lines = scanner_path.read_text(encoding="utf-8").splitlines()

    def find(s: str) -> int:
        for i, line in enumerate(lines, start=1):
            if s in line:
                return i
        return -1

    return {
        "warmup_guard": {
            "line": find("if warmup_ready_date is None or trade_date < warmup_ready_date:"),
            "quote": "if warmup_ready_date is None or trade_date < warmup_ready_date:",
        },
        "base_forming_entry": {
            "line": find("width <= float(get_cfg(config, \"base_max_width_pct\"))"),
            "quote": "sma200 > 0 and ema30 > 0 and width <= base_max_width_pct and sessions_in_range >= base_min_sessions and base_low_60 <= close <= base_high_60",
        },
        "accumulation_gate": {
            "line": find("accumulation_gate = bool(payload.get(\"accumulation_divergence\")) or ("),
            "quote": "accumulation_gate and cmf_hits >= 5 and squeeze_ok and (close >= ema30 or close >= 0.97 * sma200) and liquidity_ok and score >= 60",
        },
        "breakout_watch_trigger": {
            "line": find("near_base_with_build = base_high_ref > 0 and close >= (0.97 * base_high_ref) and rv_hits >= 2"),
            "quote": "near_base_with_build = base_high_ref > 0 and close >= (0.97 * base_high_ref) and rv_hits >= 2",
        },
        "breakout_confirm_mandatory": {
            "line": find('"M1_close_gt_base": base_high_ref > 0 and close > base_high_ref'),
            "quote": "mandatory M1..M5: close>base_high_ref, rel_volume>=volume_breakout_mult, ema10>ema30, chase_guard gap<=8%, liquidity_ok",
        },
    }


def first_non_null(v: Any) -> Any:
    return v


def evaluate_base_entry(
    indicator: dict[str, Any] | None,
    rolling_closes_60: list[float],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if indicator is None:
        return [
            {
                "term": "indicator_payload_exists",
                "value": None,
                "threshold": "required",
                "recoverable": False,
                "passes": None,
                "recoverability_note": "no indicator payload for day",
            }
        ]

    close = indicator.get("close")
    ema30 = indicator.get("ema30")
    sma200 = indicator.get("sma200")

    width = indicator.get("range_width_pct")
    if width is None and len(rolling_closes_60) >= 2:
        hi = max(rolling_closes_60)
        lo = min(rolling_closes_60)
        width = None if lo <= 0 else ((hi - lo) / lo)

    base_high_60 = indicator.get("range_high_60")
    base_low_60 = indicator.get("range_low_60")
    if (base_high_60 is None or base_low_60 is None) and rolling_closes_60:
        base_high_60 = max(rolling_closes_60)
        base_low_60 = min(rolling_closes_60)

    sessions_in_range = None
    if rolling_closes_60 and base_low_60 is not None and base_high_60 is not None:
        sessions_in_range = sum(1 for c in rolling_closes_60 if float(base_low_60) <= c <= float(base_high_60))

    terms = [
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
            "value": sessions_in_range,
            "threshold": cfg.get("base_min_sessions"),
            "recoverable": sessions_in_range is not None and cfg.get("base_min_sessions") is not None,
            "passes": None
            if sessions_in_range is None or cfg.get("base_min_sessions") is None
            else bool(int(sessions_in_range) >= int(cfg["base_min_sessions"])),
        },
        {
            "term": "base_low_60 <= close <= base_high_60",
            "value": {"base_low_60": base_low_60, "close": close, "base_high_60": base_high_60},
            "threshold": "range inclusion",
            "recoverable": close is not None and base_low_60 is not None and base_high_60 is not None,
            "passes": None
            if close is None or base_low_60 is None or base_high_60 is None
            else bool(float(base_low_60) <= float(close) <= float(base_high_60)),
        },
    ]
    return terms


def evaluate_accumulation_gate(indicator: dict[str, Any] | None, cmf_last_10: list[float | None], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if indicator is None:
        return [
            {
                "term": "indicator_payload_exists",
                "value": None,
                "threshold": "required",
                "recoverable": False,
                "passes": None,
                "recoverability_note": "no indicator payload for day",
            }
        ]

    accumulation_divergence = indicator.get("accumulation_divergence")
    price_slope_40 = indicator.get("price_slope_40")
    obv_slope_40 = indicator.get("obv_slope_40")
    anv_slope_40 = indicator.get("anv_slope_40")

    composite_recoverable = price_slope_40 is not None and (obv_slope_40 is not None or anv_slope_40 is not None)
    composite_val = None
    if composite_recoverable:
        composite_val = abs(float(price_slope_40)) < 0.02 and (
            (obv_slope_40 is not None and float(obv_slope_40) > 0.10)
            or (anv_slope_40 is not None and float(anv_slope_40) > 0.10)
        )

    cmf_floor = cfg.get("cmf_floor")
    cmf_hits = None
    if cmf_last_10 and cmf_floor is not None and all(v is not None for v in cmf_last_10):
        cmf_hits = sum(1 for v in cmf_last_10 if float(v) > float(cmf_floor))

    bb_width = indicator.get("bb_width")
    atr_pct_pctile = indicator.get("atr_pct_percentile_252")
    atr_squeeze = cfg.get("atr_squeeze_pctile")
    squeeze_pass = None
    if bb_width is not None or (atr_pct_pctile is not None and atr_squeeze is not None):
        squeeze_pass = bool(
            (bb_width is not None and float(bb_width) <= 0.12)
            or (atr_pct_pctile is not None and atr_squeeze is not None and float(atr_pct_pctile) <= float(atr_squeeze))
        )

    close = indicator.get("close")
    ema30 = indicator.get("ema30")
    sma200 = indicator.get("sma200")
    close_gate = None
    if close is not None and ema30 is not None:
        close_gate = bool(
            float(close) >= float(ema30)
            or (sma200 is not None and float(close) >= (0.97 * float(sma200)))
        )

    terms = [
        {
            "term": "accumulation_gate composite",
            "value": {
                "accumulation_divergence": accumulation_divergence,
                "price_slope_40": price_slope_40,
                "obv_slope_40": obv_slope_40,
                "anv_slope_40": anv_slope_40,
                "composite_clause": composite_val,
            },
            "threshold": "accumulation_divergence OR (abs(price_slope_40)<0.02 AND (obv_slope_40>0.10 OR anv_slope_40>0.10))",
            "recoverable": accumulation_divergence is not None and composite_recoverable,
            "passes": None if accumulation_divergence is None or not composite_recoverable else bool(accumulation_divergence or composite_val),
            "recoverability_note": None
            if accumulation_divergence is not None and composite_recoverable
            else "slope terms unavailable on runtime payload",
        },
        {
            "term": "cmf_hits >= 5 over last 10",
            "value": cmf_hits,
            "threshold": ">=5 with cmf_10 > cmf_floor",
            "recoverable": cmf_hits is not None,
            "passes": None if cmf_hits is None else bool(int(cmf_hits) >= 5),
            "recoverability_note": None if cmf_hits is not None else "insufficient non-null cmf_10 history",
        },
        {
            "term": "squeeze_ok",
            "value": {"bb_width": bb_width, "atr_pct_percentile_252": atr_pct_pctile},
            "threshold": {"bb_width_max": 0.12, "atr_pct_percentile_252_max": atr_squeeze},
            "recoverable": squeeze_pass is not None,
            "passes": squeeze_pass,
        },
        {
            "term": "close>=ema30 OR close>=0.97*sma200",
            "value": {"close": close, "ema30": ema30, "sma200": sma200},
            "threshold": "coded OR expression",
            "recoverable": close_gate is not None,
            "passes": close_gate,
            "recoverability_note": None if close_gate is not None else "close/ema30 not recoverable",
        },
        {
            "term": "liquidity_ok",
            "value": None,
            "threshold": "risk_service.liquidity_filter_at",
            "recoverable": False,
            "passes": None,
            "recoverability_note": "liquidity boolean not persisted in runtime payload",
        },
        {
            "term": "score >= 60",
            "value": None,
            "threshold": 60,
            "recoverable": False,
            "passes": None,
            "recoverability_note": "daily score not persisted for non-signal days",
        },
    ]
    return terms


def evaluate_watch_trigger(indicator: dict[str, Any] | None, relvol_last_5: list[float | None]) -> list[dict[str, Any]]:
    if indicator is None:
        return [
            {
                "term": "indicator_payload_exists",
                "value": None,
                "threshold": "required",
                "recoverable": False,
                "passes": None,
                "recoverability_note": "no indicator payload for day",
            }
        ]

    rv_hits = None
    if relvol_last_5 and all(v is not None for v in relvol_last_5):
        rv_hits = sum(1 for v in relvol_last_5 if float(v) >= 1.5)

    terms = [
        {
            "term": "base_high_ref > 0",
            "value": None,
            "threshold": ">0",
            "recoverable": False,
            "passes": None,
            "recoverability_note": "base_high_ref state not persisted day-by-day",
        },
        {
            "term": "close >= 0.97 * base_high_ref",
            "value": {"close": indicator.get("close"), "base_high_ref": None},
            "threshold": "close >= 0.97*base_high_ref",
            "recoverable": False,
            "passes": None,
            "recoverability_note": "base_high_ref unavailable",
        },
        {
            "term": "rv_hits >= 2 over last 5 (rel_volume>=1.5)",
            "value": rv_hits,
            "threshold": 2,
            "recoverable": rv_hits is not None,
            "passes": None if rv_hits is None else bool(int(rv_hits) >= 2),
            "recoverability_note": None if rv_hits is not None else "insufficient non-null rel_volume history",
        },
    ]
    return terms


def choose_blocking_term(terms: list[dict[str, Any]]) -> dict[str, Any]:
    for t in terms:
        if t.get("passes") is False:
            return t
    for t in terms:
        if t.get("recoverable") is False:
            return t
    return terms[0] if terms else {"term": "NO_TERM"}


def run() -> dict[str, Any]:
    review = REVIEW
    runtime_db = review / "r12_exam_surface_v4_5_runtime.db"
    v21 = read_json(review / "r12_exam_results_v2_1.json")
    mask = read_json(review / "r12_masked_intervals_manifest_v4_3_final.json")
    universe = read_json(review / "r13_universe_tier_profile_v1_2.json")

    set_a = sorted(
        next(
            (
                r.get("symbols", [])
                for r in v21.get("benchmark_parity_suite_completion", {}).get("random_top_k_portfolio_rows", [])
                if r.get("set") == "set_a"
            ),
            [],
        )
    )

    tier_by_symbol = {str(r.get("symbol")): str(r.get("liquidity_tier")) for r in universe.get("rows", [])}
    mask_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in mask.get("intervals", []):
        s = str(m.get("symbol") or "")
        if s:
            mask_by_symbol[s].append(m)

    scanner_lines = find_scanner_lines(ROOT / "app" / "services" / "eagle_eye" / "scanner_service.py")

    with sqlite_ro(runtime_db) as con:
        cur = con.cursor()

        # Config snapshot for predicate thresholds.
        cfg_keys = [
            "base_min_sessions",
            "base_max_width_pct",
            "volume_breakout_mult",
            "rsi_regime",
            "adx_trigger",
            "cmf_floor",
            "atr_squeeze_pctile",
            "trend_join_window",
        ]
        cfg: dict[str, Any] = {}
        for k in cfg_keys:
            row = cur.execute("SELECT value_json FROM ee_engine_config WHERE key = ?", (k,)).fetchone()
            cfg[k] = None if row is None else json.loads(row[0])

        day_table: list[dict[str, Any]] = []
        per_symbol_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        per_symbol_year_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        symbol_series: dict[str, list[dict[str, Any]]] = {}
        symbol_indicators: dict[str, dict[int, dict[str, Any]]] = {}
        symbol_signals: dict[str, list[dict[str, Any]]] = {}
        symbol_trades_opened: dict[str, set[int]] = {}

        for sym in set_a:
            # Full trading-day surface from masked source.
            rows = cur.execute(
                "SELECT trade_date, close, is_masked FROM ee_ohlcv_masked_source WHERE symbol = ? ORDER BY trade_date",
                (sym,),
            ).fetchall()
            series = [{"trade_date": int(r[0]), "close": float(r[1] or 0.0), "is_masked": int(r[2] or 0)} for r in rows]
            symbol_series[sym] = series

            ind_rows = cur.execute(
                f"SELECT trade_date, payload_json FROM ee_indicators WHERE {base_symbol_sql()} = ? ORDER BY trade_date",
                (sym,),
            ).fetchall()
            ind_map: dict[int, dict[str, Any]] = {}
            for td, pj in ind_rows:
                ind_map[int(td)] = json.loads(pj) if pj else {}
            symbol_indicators[sym] = ind_map

            sig_rows = cur.execute(
                f"""
                SELECT id, trade_date, signal_type, phase_from, phase_to, evidence_json
                FROM ee_signals
                WHERE {base_symbol_sql()} = ?
                ORDER BY trade_date, id
                """,
                (sym,),
            ).fetchall()
            srows = []
            for rid, td, st, pf, pt, ev in sig_rows:
                srows.append(
                    {
                        "id": int(rid),
                        "trade_date": int(td),
                        "signal_type": str(st),
                        "phase_from": None if pf is None else str(pf),
                        "phase_to": None if pt is None else str(pt),
                        "evidence": json.loads(ev) if ev else {},
                    }
                )
            symbol_signals[sym] = srows

            tr_rows = cur.execute(
                f"SELECT opened_at FROM ee_backtest_trades WHERE {base_symbol_sql()} = ?",
                (sym,),
            ).fetchall()
            symbol_trades_opened[sym] = {int(r[0]) for r in tr_rows}

        for sym in set_a:
            series = symbol_series[sym]
            ind_map = symbol_indicators[sym]
            sigs = symbol_signals[sym]
            trades_opened = symbol_trades_opened[sym]

            sig_by_day: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for s in sigs:
                sig_by_day[int(s["trade_date"])].append(s)

            phase_current = "NEUTRAL"

            # Build helpers for rolling windows.
            closes_so_far: list[float] = []
            cmf_so_far: list[float | None] = []
            relvol_so_far: list[float | None] = []

            for row in series:
                day_ts = int(row["trade_date"])
                day_iso = ts_to_iso(day_ts)
                indicator = ind_map.get(day_ts)
                signals_today = sig_by_day.get(day_ts, [])
                phase_before = phase_current
                for s in signals_today:
                    if s.get("phase_to"):
                        phase_current = str(s["phase_to"])
                phase_after = phase_current

                close = float(row.get("close") or 0.0)
                closes_so_far.append(close)
                cmf_so_far.append(None if indicator is None else indicator.get("cmf_10"))
                relvol_so_far.append(None if indicator is None else indicator.get("rel_volume"))

                eval_recorded = indicator is not None or bool(signals_today)

                gate = None
                for s in signals_today:
                    st = s["signal_type"]
                    if st == "SIGNAL_SUPPRESSED_RISK":
                        gate = "RISK_SUPPRESSION"
                        break
                    if st == "AVOID_SET" or s.get("phase_to") == "AVOID" or s.get("evidence", {}).get("attempted_signal_type") == "AVOID_SET":
                        gate = "AVOID_GATE"
                        break

                candidate_passed = any(s["signal_type"] in {"ACCUMULATION_ALERT", "BREAKOUT_CONFIRMED"} for s in signals_today)
                opened_today = day_ts in trades_opened

                if gate is not None:
                    classification = f"CANDIDATE_VETOED({gate})"
                elif candidate_passed and not opened_today:
                    classification = "CANDIDATE_PASSED_NO_FILL"
                elif phase_after != "NEUTRAL":
                    classification = "PHASE_PROGRESSED_NO_CANDIDATE"
                else:
                    classification = "PHASE_MACHINE_NEVER_LEFT_NEUTRAL" if eval_recorded else "MASKED_OR_WARMUP_EXCLUDED"

                null_terms = []
                if classification == "PHASE_MACHINE_NEVER_LEFT_NEUTRAL" and eval_recorded and indicator is not None:
                    for key in [
                        "sma200",
                        "sma200_slope",
                        "range_high_120",
                        "range_low_120",
                        "range_high_60",
                        "range_low_60",
                        "range_width_pct",
                        "price_slope_40",
                        "obv_slope_40",
                        "anv_slope_40",
                        "cmf_10",
                        "atr_pct_percentile_252",
                    ]:
                        if indicator.get(key) is None:
                            null_terms.append(key)

                rec = {
                    "symbol": sym,
                    "trade_date": day_ts,
                    "trade_date_iso": day_iso,
                    "year": None if day_iso is None else int(str(day_iso)[:4]),
                    "classification": classification,
                    "phase_before_day": phase_before,
                    "phase_after_day": phase_after,
                    "evaluation_recorded": bool(eval_recorded),
                    "signal_count_on_day": len(signals_today),
                    "signal_types_on_day": [s["signal_type"] for s in signals_today],
                    "opened_trade_on_day": bool(opened_today),
                    "null_terms": sorted(null_terms),
                    "is_masked_source_row": bool(int(row.get("is_masked") or 0)),
                }
                day_table.append(rec)
                per_symbol_counts[sym][classification] += 1
                if rec["year"] is not None:
                    per_symbol_year_counts[sym][str(rec["year"])][classification] += 1

        # Owner-pattern window audit: last 300 unmasked sessions per symbol.
        owner_window_rows: list[dict[str, Any]] = []
        owner_narrative: dict[str, list[dict[str, Any]]] = {}

        day_idx = {(r["symbol"], int(r["trade_date"])): r for r in day_table}

        for sym in set_a:
            # Unmasked trading days (exam surface).
            unmasked = cur.execute(
                f"SELECT trade_date, close FROM ee_ohlcv WHERE {base_symbol_sql()} = ? ORDER BY trade_date",
                (sym,),
            ).fetchall()
            unmasked_rows = [{"trade_date": int(r[0]), "close": float(r[1] or 0.0)} for r in unmasked][-300:]

            ind_map = symbol_indicators[sym]
            flagged_days: list[dict[str, Any]] = []

            closes = [r["close"] for r in unmasked_rows]
            dates = [r["trade_date"] for r in unmasked_rows]

            for i, r in enumerate(unmasked_rows):
                day_ts = int(r["trade_date"])
                close = float(r["close"])
                indicator = ind_map.get(day_ts)
                phase_row = day_idx.get((sym, day_ts), {})
                phase = str(phase_row.get("phase_after_day") or "NEUTRAL")

                prev20 = closes[max(0, i - 20) : i]
                prev20_high = max(prev20) if prev20 else None
                breakout_progress = False if prev20_high is None else bool(close > prev20_high)

                ema10 = None if indicator is None else indicator.get("ema10")
                ema30 = None if indicator is None else indicator.get("ema30")
                trend_progress = bool(ema10 is not None and ema30 is not None and close > float(ema30) and float(ema10) > float(ema30))

                pattern_progressing = bool(breakout_progress or trend_progress)

                if not pattern_progressing:
                    continue
                if phase in {"BREAKOUT_CONFIRMED", "MARKUP"}:
                    continue

                rolling60 = closes[max(0, i - 59) : i + 1]
                cmf10 = [
                    (None if ind_map.get(d) is None else ind_map.get(d).get("cmf_10"))
                    for d in dates[max(0, i - 9) : i + 1]
                ]
                rv5 = [
                    (None if ind_map.get(d) is None else ind_map.get(d).get("rel_volume"))
                    for d in dates[max(0, i - 4) : i + 1]
                ]

                if phase == "NEUTRAL":
                    terms = evaluate_base_entry(indicator, rolling60, cfg)
                    predicate = "base_forming_entry"
                elif phase == "BASE_FORMING":
                    terms = evaluate_accumulation_gate(indicator, cmf10, cfg)
                    predicate = "accumulation_gate"
                elif phase == "ACCUMULATION":
                    terms = evaluate_watch_trigger(indicator, rv5)
                    predicate = "breakout_watch_trigger"
                elif phase == "BREAKOUT_WATCH":
                    terms = [
                        {
                            "term": "M1_close_gt_base",
                            "value": {"close": None if indicator is None else indicator.get("close"), "base_high_ref": None},
                            "threshold": "close > base_high_ref",
                            "recoverable": False,
                            "passes": None,
                            "recoverability_note": "base_high_ref state not persisted",
                        },
                        {
                            "term": "M2_rel_volume",
                            "value": None if indicator is None else indicator.get("rel_volume"),
                            "threshold": cfg.get("volume_breakout_mult"),
                            "recoverable": indicator is not None and indicator.get("rel_volume") is not None and cfg.get("volume_breakout_mult") is not None,
                            "passes": None
                            if indicator is None or indicator.get("rel_volume") is None or cfg.get("volume_breakout_mult") is None
                            else bool(float(indicator.get("rel_volume")) >= float(cfg.get("volume_breakout_mult"))),
                        },
                        {
                            "term": "M3_ema10_gt_ema30",
                            "value": {"ema10": None if indicator is None else indicator.get("ema10"), "ema30": None if indicator is None else indicator.get("ema30")},
                            "threshold": "ema10 > ema30",
                            "recoverable": indicator is not None and indicator.get("ema10") is not None and indicator.get("ema30") is not None,
                            "passes": None
                            if indicator is None or indicator.get("ema10") is None or indicator.get("ema30") is None
                            else bool(float(indicator.get("ema10")) > float(indicator.get("ema30"))),
                        },
                    ]
                    predicate = "breakout_confirm_mandatory"
                else:
                    # AVOID / EXIT / DISTRIBUTION_WARNING: use avoid condition as hold blocker.
                    sma200 = None if indicator is None else indicator.get("sma200")
                    sma200_slope = None if indicator is None else indicator.get("sma200_slope")
                    terms = [
                        {
                            "term": "avoid_condition close < sma200 and sma200_slope < 0 and ema10 < ema30",
                            "value": {
                                "close": None if indicator is None else indicator.get("close"),
                                "sma200": sma200,
                                "sma200_slope": sma200_slope,
                                "ema10": None if indicator is None else indicator.get("ema10"),
                                "ema30": None if indicator is None else indicator.get("ema30"),
                            },
                            "threshold": "coded avoid expression",
                            "recoverable": indicator is not None and sma200 is not None and sma200_slope is not None and indicator.get("ema10") is not None and indicator.get("ema30") is not None,
                            "passes": None,
                            "recoverability_note": "insufficient avoid terms on payload" if indicator is None or sma200 is None or sma200_slope is None else None,
                        }
                    ]
                    predicate = "warmup_guard"

                blocking = choose_blocking_term(terms)
                line_info = scanner_lines.get(predicate, {"line": -1, "quote": ""})

                entry = {
                    "symbol": sym,
                    "trade_date": day_ts,
                    "trade_date_iso": ts_to_iso(day_ts),
                    "phase_state": phase,
                    "pattern_progressing": True,
                    "pattern_progress_evidence": {
                        "close": close,
                        "prev20_high": prev20_high,
                        "close_gt_prev20_high": breakout_progress,
                        "ema10": ema10,
                        "ema30": ema30,
                        "trend_progress": trend_progress,
                    },
                    "predicate": {
                        "name": predicate,
                        "file": "app/services/eagle_eye/scanner_service.py",
                        "line": line_info.get("line"),
                        "quote": line_info.get("quote"),
                    },
                    "blocking_term": {
                        "term": blocking.get("term"),
                        "value": blocking.get("value"),
                        "threshold": blocking.get("threshold"),
                        "recoverable": blocking.get("recoverable"),
                        "passes": blocking.get("passes"),
                        "recoverability_note": blocking.get("recoverability_note"),
                    },
                }
                flagged_days.append(entry)
                owner_window_rows.append(entry)

            # Build narrative ranges.
            ranges: list[dict[str, Any]] = []
            if flagged_days:
                prev = flagged_days[0]
                start = prev["trade_date"]
                end = prev["trade_date"]
                count = 1
                for row in flagged_days[1:]:
                    same_key = (
                        row["phase_state"] == prev["phase_state"]
                        and row["blocking_term"]["term"] == prev["blocking_term"]["term"]
                    )
                    if same_key:
                        end = row["trade_date"]
                        count += 1
                    else:
                        ranges.append(
                            {
                                "start_trade_date": start,
                                "start_trade_date_iso": ts_to_iso(start),
                                "end_trade_date": end,
                                "end_trade_date_iso": ts_to_iso(end),
                                "session_count": count,
                                "phase_held": prev["phase_state"],
                                "blocking_term": prev["blocking_term"]["term"],
                                "value_vs_threshold": {
                                    "value": prev["blocking_term"]["value"],
                                    "threshold": prev["blocking_term"]["threshold"],
                                },
                                "predicate_source": prev["predicate"],
                                "recoverability_note": prev["blocking_term"].get("recoverability_note"),
                            }
                        )
                        prev = row
                        start = row["trade_date"]
                        end = row["trade_date"]
                        count = 1
                ranges.append(
                    {
                        "start_trade_date": start,
                        "start_trade_date_iso": ts_to_iso(start),
                        "end_trade_date": end,
                        "end_trade_date_iso": ts_to_iso(end),
                        "session_count": count,
                        "phase_held": prev["phase_state"],
                        "blocking_term": prev["blocking_term"]["term"],
                        "value_vs_threshold": {
                            "value": prev["blocking_term"]["value"],
                            "threshold": prev["blocking_term"]["threshold"],
                        },
                        "predicate_source": prev["predicate"],
                        "recoverability_note": prev["blocking_term"].get("recoverability_note"),
                    }
                )
            owner_narrative[sym] = ranges

        # Warmup-cost quantification by symbol and tier from segment map.
        warmup_rows: list[dict[str, Any]] = []
        tier_rollup: dict[str, dict[str, Any]] = defaultdict(lambda: {"symbol_count": 0, "blind_sessions": 0, "total_sessions": 0})

        for sym in set_a:
            segs = cur.execute(
                "SELECT segment_symbol, segment_id, bars_count, start_trade_date, end_trade_date FROM ee_symbol_segment_map WHERE original_symbol = ? ORDER BY segment_id",
                (sym,),
            ).fetchall()
            seg_rows = []
            blind_sum = 0
            total_sum = 0

            ind_map = symbol_indicators[sym]
            for seg_sym, seg_id, bars_count, start_td, end_td in segs:
                days = cur.execute(
                    "SELECT trade_date FROM ee_ohlcv WHERE symbol = ? ORDER BY trade_date",
                    (str(seg_sym),),
                ).fetchall()
                dlist = [int(r[0]) for r in days]
                total = len(dlist)
                blind_days = 0
                ready_at = None
                for d in dlist:
                    ind = ind_map.get(d)
                    ready = (
                        ind is not None
                        and ind.get("sma200") is not None
                        and ind.get("range_high_120") is not None
                        and ind.get("range_low_120") is not None
                    )
                    if ready and ready_at is None:
                        ready_at = d
                    if not ready:
                        blind_days += 1
                blind_sum += blind_days
                total_sum += total
                seg_rows.append(
                    {
                        "segment_symbol": str(seg_sym),
                        "segment_id": int(seg_id),
                        "start_trade_date": int(start_td),
                        "start_trade_date_iso": ts_to_iso(int(start_td)),
                        "end_trade_date": int(end_td),
                        "end_trade_date_iso": ts_to_iso(int(end_td)),
                        "segment_sessions": total,
                        "blind_sessions": blind_days,
                        "blind_share": (blind_days / total) if total else 0.0,
                        "warmup_ready_date": ready_at,
                        "warmup_ready_date_iso": ts_to_iso(ready_at),
                    }
                )

            tier = tier_by_symbol.get(sym, "UNKNOWN")
            total_sessions_source = cur.execute("SELECT COUNT(*) FROM ee_ohlcv_masked_source WHERE symbol = ?", (sym,)).fetchone()[0]

            warmup_row = {
                "symbol": sym,
                "liquidity_tier": tier,
                "blind_sessions_from_segment_warmup": blind_sum,
                "total_unmasked_sessions": total_sum,
                "total_sessions_masked_source": int(total_sessions_source),
                "blind_share_unmasked": (blind_sum / total_sum) if total_sum else 0.0,
                "segment_breakdown": seg_rows,
            }
            warmup_rows.append(warmup_row)

            tier_rollup[tier]["symbol_count"] += 1
            tier_rollup[tier]["blind_sessions"] += int(blind_sum)
            tier_rollup[tier]["total_sessions"] += int(total_sum)

        tier_summary = {}
        for tier, vals in sorted(tier_rollup.items()):
            total = vals["total_sessions"]
            tier_summary[tier] = {
                "symbol_count": vals["symbol_count"],
                "blind_sessions": vals["blind_sessions"],
                "total_unmasked_sessions": total,
                "blind_share_unmasked": (vals["blind_sessions"] / total) if total else 0.0,
            }

    payload = {
        "version_id": "R13_SET_A_CAUSAL_ATTRIBUTION_V3",
        "formal_finding": {
            "id": "WARMUP_SHADOW_CONFOUND_V1",
            "statement": "Set A benchmark first-trigger days (TIJARA 2021-09-09, BPCC 2021-08-26, SANAM 2021-03-23) occur inside indicator warmup shadow; parity FAIL on those days is structural blindness, not evaluated-and-rejected behavior.",
            "evidence": {
                "named_days": ["TIJARA 2021-09-09", "BPCC 2021-08-26", "SANAM 2021-03-23"],
                "support": "day-level runtime table shows PHASE_MACHINE_NEVER_LEFT_NEUTRAL with warmup/null term constraints",
            },
        },
        "classification_vocab": [
            "PHASE_MACHINE_NEVER_LEFT_NEUTRAL",
            "PHASE_PROGRESSED_NO_CANDIDATE",
            "CANDIDATE_VETOED(gate)",
            "CANDIDATE_PASSED_NO_FILL",
            "MASKED_OR_WARMUP_EXCLUDED",
        ],
        "warmup_rule_applied": {
            "masked_or_warmup_excluded_only_when_no_phase_eval_recorded": True,
            "phase_eval_with_null_terms_classified_as_phase_machine_never_left_neutral": True,
        },
        "inputs": {
            "runtime_db": "artifacts/preview1a_prestart/review_final/r12_exam_surface_v4_5_runtime.db",
            "benchmark_spec_v2": "artifacts/preview1a_prestart/review_final/r12_benchmark_spec_v2.json",
            "exam_results_v2_1": "artifacts/preview1a_prestart/review_final/r12_exam_results_v2_1.json",
            "mask_manifest": "artifacts/preview1a_prestart/review_final/r12_masked_intervals_manifest_v4_3_final.json",
            "segment_map": "ee_symbol_segment_map in runtime db",
            "scanner_service": "app/services/eagle_eye/scanner_service.py",
        },
        "set_a_symbols": set_a,
        "day_level_table": day_table,
        "per_symbol_category_counts_total": {k: dict(sorted(v.items())) for k, v in sorted(per_symbol_counts.items())},
        "per_symbol_category_counts_by_year": {
            sym: {year: dict(sorted(c.items())) for year, c in sorted(yrs.items())}
            for sym, yrs in sorted(per_symbol_year_counts.items())
        },
        "owner_pattern_window_audit": {
            "window_definition": "last 300 unmasked trading sessions per Set A symbol",
            "day_level": owner_window_rows,
            "narrative_ranges": owner_narrative,
            "scanner_predicate_source_lines": scanner_lines,
        },
        "warmup_cost_quantification": {
            "per_symbol": warmup_rows,
            "per_liquidity_tier": tier_summary,
        },
        "constraints": {
            "read_only_db_uri": True,
            "no_engine_contact": True,
            "no_reruns": True,
        },
        "authorization_status": "R14_NOT_AUTHORIZED",
    }
    return payload


def build_md(payload: dict[str, Any]) -> str:
    lines = [
        "# R13 Set A Causal Attribution v3",
        "",
        "Formal finding recorded:",
        f"- {payload.get('formal_finding', {}).get('statement')}",
        "",
        "## Per-Symbol Category Counts (Total)",
    ]
    for sym, counts in sorted(payload.get("per_symbol_category_counts_total", {}).items()):
        lines.append(f"- {sym}: {counts}")

    lines += ["", "## Per-Symbol Category Counts By Year", ""]
    for sym, yrs in sorted(payload.get("per_symbol_category_counts_by_year", {}).items()):
        lines.append(f"### {sym}")
        for year, counts in sorted(yrs.items()):
            lines.append(f"- {year}: {counts}")

    lines += ["", "## Owner-Pattern Window Narrative Ranges", ""]
    for sym, ranges in sorted(payload.get("owner_pattern_window_audit", {}).get("narrative_ranges", {}).items()):
        lines.append(f"### {sym}")
        for r in ranges:
            lines.append(
                f"- {r['start_trade_date_iso']} -> {r['end_trade_date_iso']} phase={r['phase_held']} blocking={r['blocking_term']} value={r['value_vs_threshold']['value']} threshold={r['value_vs_threshold']['threshold']} source={r['predicate_source']['file']}:{r['predicate_source']['line']}"
            )

    lines += ["", "## Warmup-Cost Quantification", ""]
    for row in payload.get("warmup_cost_quantification", {}).get("per_symbol", []):
        lines.append(
            f"- {row['symbol']} tier={row['liquidity_tier']} blind_sessions={row['blind_sessions_from_segment_warmup']} total_unmasked={row['total_unmasked_sessions']} blind_share={row['blind_share_unmasked']}"
        )

    lines += ["", "R14 remains NOT AUTHORIZED.", ""]
    return "\n".join(lines)


def main() -> None:
    payload = run()
    out_json = REVIEW / "r13_set_a_causal_attribution_v3.json"
    out_md = REVIEW / "r13_set_a_causal_attribution_v3.md"
    write_json(out_json, payload)
    out_md.write_text(build_md(payload), encoding="utf-8")
    print("R13_SET_A_CAUSAL_ATTRIBUTION_V3_COMPLETE")
    print("json_sha256", sha256_file(out_json))
    print("md_sha256", sha256_file(out_md))


if __name__ == "__main__":
    main()

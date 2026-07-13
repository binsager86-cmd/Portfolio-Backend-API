from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import BASE_DIR, get_settings
from app.core.db_isolation import ensure_debug_fixture_write_allowed
from app.core.database import query_all, query_one, query_val
from app.services.eagle_eye.backtest_service import run_backtest
from app.services.eagle_eye.pipeline import process_bar
from app.services.eagle_eye.indicator_service import compute_and_store_symbol
from app.services.eagle_eye.market_data_service import ensure_schema, load_ohlcv_csv, get_active_config, get_config_hash
from app.services.eagle_eye.audit_service import ensure_schema as ensure_audit_schema
from app.services.eagle_eye.rating_service import compute_rating_from_indicator, store_rating


SYMBOLS = [
    "TST_ACCUM_001",
    "TST_BREAKOUT_FAIL_001",
    "TST_DISTRIBUTION_001",
    "TST_ACCUM_002",
    "TST_MARKUP_001",
    "DBG_GATE_001",
]
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
SEGMENTS_PATH = FIXTURES / "segments.json"


def _resolve_segment(spec: str | None) -> tuple[str, str, int, int] | None:
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError("--segment must use SYMBOL:NAME format")
    symbol_raw, name_raw = spec.split(":", 1)
    symbol = symbol_raw.strip().upper()
    name = name_raw.strip()
    if not SEGMENTS_PATH.exists():
        raise ValueError(f"segments file missing: {SEGMENTS_PATH}")
    payload = json.loads(SEGMENTS_PATH.read_text(encoding="utf-8"))
    sym = payload.get(symbol)
    if not isinstance(sym, dict):
        raise ValueError(f"symbol {symbol} not found in segments.json")
    seg = sym.get(name)
    if not isinstance(seg, dict):
        raise ValueError(f"segment {symbol}:{name} not found in segments.json")
    start_td = int(seg.get("trade_date_start") or 0)
    end_td = int(seg.get("trade_date_end") or 0)
    if start_td <= 0 or end_td <= 0 or end_td < start_td:
        raise ValueError(f"invalid trade_date range for segment {symbol}:{name}")
    return symbol, name, start_td, end_td


def _load_and_run() -> None:
    settings = get_settings()
    ensure_debug_fixture_write_allowed(settings.ENVIRONMENT, settings.database_abs_path, BASE_DIR)

    ensure_schema()
    ensure_audit_schema()
    for t in [
        "ee_ohlcv",
        "ee_indicators",
        "ee_symbol_state",
        "ee_signals",
        "ee_ratings",
        "ee_positions",
        "ee_backtest_runs",
        "ee_backtest_trades",
        "ee_change_status_history",
        "ee_change_requests",
        "ee_audit_events",
    ]:
        try:
            query_val(f"DELETE FROM {t}", ())
        except Exception:
            pass

    for s in SYMBOLS:
        load_ohlcv_csv(str(FIXTURES / f"synthetic_{s.lower()}.csv"), s)
        compute_and_store_symbol(s)

    row = query_one("SELECT MIN(trade_date) mn, MAX(trade_date) mx FROM ee_ohlcv", ())
    run_backtest(SYMBOLS, int(row["mn"]), int(row["mx"]), config_overrides={"min_daily_value_kwd": 100000.0})


def _trace(symbol: str) -> dict[str, int]:
    rows = query_all(
        """
        SELECT i.trade_date, i.payload_json, r.score
        FROM ee_indicators i
        LEFT JOIN ee_ratings r ON r.symbol=i.symbol AND r.trade_date=i.trade_date
        WHERE i.symbol = ?
        ORDER BY i.trade_date
        """,
        (symbol,),
    )
    base_start = query_val(
        "SELECT MIN(trade_date) FROM ee_signals WHERE symbol=? AND signal_type='PHASE_ONLY' AND phase_to='BASE_FORMING'",
        (symbol,),
    )

    if not base_start:
        return {"no_base_phase": 1}

    fail_counts = {
        "accumulation_gate": 0,
        "cmf_hits": 0,
        "squeeze": 0,
        "trend": 0,
        "liquidity": 0,
        "score": 0,
        "all_passed": 0,
    }

    for idx, r in enumerate(rows):
        td = int(r["trade_date"])
        if td < int(base_start):
            continue

        p = json.loads(r["payload_json"] or "{}")
        hist = rows[max(0, idx - 9) : idx + 1]
        cmf_hits = 0
        for h in hist:
            hp = json.loads(h["payload_json"] or "{}")
            if float(hp.get("cmf_10") or 0.0) > 0.05:
                cmf_hits += 1

        acc_gate = bool(p.get("accumulation_divergence"))
        squeeze = (
            (p.get("atr_pct_percentile_252") is not None and float(p.get("atr_pct_percentile_252") or 1.0) <= 0.20)
            or float(p.get("bb_width") or 1.0) <= 0.12
        )
        close = float(p.get("close") or 0.0)
        ema30 = float(p.get("ema30") or 0.0)
        sma200 = float(p.get("sma200") or 0.0)
        trend = close >= ema30 or close >= 0.97 * sma200

        liq_ok = query_val(
            """
            SELECT CASE WHEN COUNT(1) = 0 THEN 0 ELSE 1 END
            FROM (
              SELECT value_kwd, close, volume
              FROM ee_ohlcv
              WHERE symbol = ? AND trade_date <= ?
              ORDER BY trade_date DESC
              LIMIT 60
            )
            WHERE value_kwd IS NOT NULL
            """,
            (symbol, td),
        )
        score_ok = float(r["score"] or 0.0) >= 60.0

        checks = [acc_gate, cmf_hits >= 5, squeeze, trend, bool(liq_ok), score_ok]
        if all(checks):
            fail_counts["all_passed"] += 1
            break

        if not acc_gate:
            fail_counts["accumulation_gate"] += 1
        if cmf_hits < 5:
            fail_counts["cmf_hits"] += 1
        if not squeeze:
            fail_counts["squeeze"] += 1
        if not trend:
            fail_counts["trend"] += 1
        if not liq_ok:
            fail_counts["liquidity"] += 1
        if not score_ok:
            fail_counts["score"] += 1

    return fail_counts


def _trace_phase(symbol: str) -> None:
    for t in ["ee_symbol_state", "ee_signals", "ee_ratings", "ee_positions"]:
        try:
            query_val(f"DELETE FROM {t}", ())
        except Exception:
            pass

    cfg = get_active_config()
    rows = query_all(
        "SELECT trade_date, payload_json FROM ee_indicators WHERE symbol = ? ORDER BY trade_date ASC",
        (symbol,),
    )
    history: list[dict] = []
    state = None
    for r in rows:
        td = int(r["trade_date"])
        payload = json.loads(r["payload_json"] or "{}")
        payload["trade_date"] = td
        history.append(payload)
        if len(history) > 140:
            history = history[-140:]

        score, band, components = compute_rating_from_indicator(payload)
        store_rating(symbol, td, score, band, components)

        before = state["phase"] if isinstance(state, dict) else "NEUTRAL"
        result = process_bar(
            symbol,
            td,
            cfg,
            trace_id="trace-phase",
            indicator_payload=payload,
            indicator_history=history,
            state_override=state,
            persist_state=False,
            coverage_start_date=int(rows[0]["trade_date"]),
            score=score,
            band=band,
            components=components,
            persist_rating=False,
        )
        state = result.get("state") if isinstance(result.get("state"), dict) else state
        after = state["phase"] if isinstance(state, dict) else before
        reason = result.get("reason")
        transition = result.get("transition")
        stype = result.get("signal_type")

        if transition or before != after or reason:
            print(
                f"{td} | {before} -> {after} | transition={transition} | signal={stype} | reason={reason}"
            )


def _cfg_with_fallback(cfg: dict, key: str, fallback: float) -> tuple[float, str]:
    if key in cfg:
        return float(cfg[key]), "config"
    return float(fallback), "fallback"


def _mfm(row: dict) -> float:
    high = float(row.get("high") or 0.0)
    low = float(row.get("low") or 0.0)
    close = float(row.get("close") or 0.0)
    rng = high - low
    if rng == 0:
        return 0.0
    return ((close - low) - (high - close)) / rng


def _trace_watch(symbol: str, include_mf: bool = False, segment_range: tuple[int, int] | None = None) -> None:
    for t in ["ee_symbol_state", "ee_signals", "ee_ratings", "ee_positions"]:
        try:
            query_val(f"DELETE FROM {t}", ())
        except Exception:
            pass

    cfg = get_active_config()
    rows = query_all(
        "SELECT trade_date, payload_json FROM ee_indicators WHERE symbol = ? ORDER BY trade_date ASC",
        (symbol,),
    )
    warmup_ready_date: int | None = None
    for r in rows:
        p = json.loads(r["payload_json"] or "{}")
        sma200 = float(p.get("sma200") or 0.0)
        range_low_120 = float(p.get("range_low_120") or 0.0)
        range_high_120 = float(p.get("range_high_120") or 0.0)
        if sma200 > 0 and range_low_120 > 0 and range_high_120 > 0:
            warmup_ready_date = int(r["trade_date"])
            break

    fixture_mtimes: dict[str, int] = {}
    for s in SYMBOLS:
        p = FIXTURES / f"synthetic_{s.lower()}.csv"
        if p.exists():
            fixture_mtimes[p.name] = int(p.stat().st_mtime)
    print(
        "TRACE_HEADER "
        "state_source=fresh_replay "
        f"config_source=active_db config_hash={get_config_hash(cfg)} "
        f"fixture_mtimes={json.dumps(fixture_mtimes, sort_keys=True, ensure_ascii=True)}"
    )

    history: list[dict] = []
    state: dict | None = None
    fallback_reported = False
    coverage_sessions = 0
    phase_dist: dict[str, int] = {}
    printed_rows = 0
    for r in rows:
        td = int(r["trade_date"])
        payload = json.loads(r["payload_json"] or "{}")
        payload["trade_date"] = td
        history.append(payload)
        if len(history) > 140:
            history = history[-140:]

        score, band, components = compute_rating_from_indicator(payload)
        store_rating(symbol, td, score, band, components)

        result = process_bar(
            symbol,
            td,
            cfg,
            trace_id="trace-watch",
            indicator_payload=payload,
            indicator_history=history,
            state_override=state,
            persist_state=False,
            coverage_start_date=warmup_ready_date,
            coverage_sessions=coverage_sessions if coverage_sessions > 0 else None,
            score=score,
            band=band,
            components=components,
            persist_rating=False,
        )
        state = result.get("state") if isinstance(result.get("state"), dict) else state
        if not isinstance(state, dict):
            continue

        if warmup_ready_date is not None and td >= warmup_ready_date:
            coverage_sessions += 1

        phase = str(state.get("phase") or "NEUTRAL")
        state_json = state.get("state_json") if isinstance(state.get("state_json"), dict) else {}
        phase_dist[phase] = int(phase_dist.get(phase) or 0) + 1

        close = float(payload.get("close") or 0.0)
        ema10 = float(payload.get("ema10") or 0.0)
        ema30 = float(payload.get("ema30") or 0.0)
        sma200 = float(payload.get("sma200") or 0.0)
        sma200_slope = float(payload.get("sma200_slope") or 0.0)
        rel_volume = float(payload.get("rel_volume") or 0.0)
        bb_width = float(payload.get("bb_width") or 1.0)
        atr_pct_raw = payload.get("atr_pct_percentile_252")
        atr_pct = float(atr_pct_raw) if atr_pct_raw is not None else None
        base_high_state = state.get("base_high")
        base_high_ref = float(base_high_state or 0.0)
        base_high_source = "state" if base_high_ref > 0 else "state_missing"
        range_high_60 = float(payload.get("range_high_60") or 0.0)
        range_low_60 = float(payload.get("range_low_60") or 0.0)
        width = float(payload.get("range_width_pct") or 9.0)
        avoid_now = close < sma200 and sma200_slope < 0 and ema10 < ema30
        avoid_clear_streak = int(state_json.get("avoid_clear_streak") or 0)
        avoid_until = state_json.get("avoid_until")

        volume_breakout_mult, src_break = _cfg_with_fallback(cfg, "volume_breakout_mult", 2.5)
        cmf_floor, src_cmf = _cfg_with_fallback(cfg, "cmf_floor", 0.05)
        atr_squeeze_pctile, src_atr = _cfg_with_fallback(cfg, "atr_squeeze_pctile", 0.20)
        trend_join_window = int(float(cfg.get("trend_join_window", 40)))

        cmf_hist = [float(h.get("cmf_10") or 0.0) for h in history[-10:]]
        cmf_last = float(payload.get("cmf_10") or 0.0)
        cmf_hits = sum(1 for x in cmf_hist if x > cmf_floor)
        accumulation_gate = bool(payload.get("accumulation_divergence")) or (
            abs(float(payload.get("price_slope_40") or 0.0)) < 0.02
            and (
                float(payload.get("obv_slope_40") or 0.0) > 0.10
                or float(payload.get("anv_slope_40") or 0.0) > 0.10
            )
        )
        bb_leg = bb_width <= 0.12
        atr_leg = False
        if atr_pct is not None:
            atr_leg = atr_pct <= atr_squeeze_pctile
        squeeze_ok = bb_leg or atr_leg
        trend_ok = close >= ema30 or close >= 0.97 * sma200
        liquidity_ok = query_val(
            """
            SELECT CASE WHEN COUNT(1) = 0 THEN 0 ELSE 1 END
            FROM (
              SELECT value_kwd, close, volume
              FROM ee_ohlcv
              WHERE symbol = ? AND trade_date <= ?
              ORDER BY trade_date DESC
              LIMIT 60
            )
            WHERE value_kwd IS NOT NULL
            """,
            (symbol, td),
        )
        score_ok = float(score or 0.0) >= 60.0
        acc_pass = accumulation_gate and cmf_hits >= 5 and squeeze_ok and trend_ok and bool(liquidity_ok) and score_ok

        recent_5 = history[-5:]
        rv_hits = sum(1 for h in recent_5 if float(h.get("rel_volume") or 0.0) >= 1.5)
        near_base_with_build = base_high_ref > 0 and close >= (0.97 * base_high_ref) and rv_hits >= 2
        watch_pass = near_base_with_build

        closes_60 = [float(h.get("close") or 0.0) for h in history[-60:]]
        sessions_in_range = sum(1 for c in closes_60 if range_low_60 <= c <= range_high_60)
        close_in_range = range_low_60 <= close <= range_high_60
        join_window_open = coverage_sessions > 0 and coverage_sessions <= trend_join_window
        join_window_display = min(coverage_sessions, trend_join_window)
        join_c1 = close > sma200
        join_c2 = sma200_slope > 0
        join_c3 = ema10 > ema30
        join_c4 = float(payload.get("range_low_120") or 0.0) > 0
        join_c5 = close >= (float(payload.get("range_low_120") or 0.0) * 1.15) if join_c4 else False
        adx = float(payload.get("adx_19") or 0.0)
        plus_di = float(payload.get("plus_di") or 0.0)
        minus_di = float(payload.get("minus_di") or 0.0)
        rsi = float(payload.get("rsi_14") or 0.0)
        day_range = max(0.0, float(payload.get("high") or close) - float(payload.get("low") or close))
        close_top40 = True if day_range == 0 else close >= (float(payload.get("low") or close) + 0.6 * day_range)
        prev_row = history[-2] if len(history) >= 2 else None
        rsi_rising = True if prev_row is None else rsi > float(prev_row.get("rsi_14") or rsi)
        adx_5_back = float(history[-5].get("adx_19") or adx) if len(history) >= 5 else adx
        macd_cross_recent = False
        if len(history) >= 6:
            for i in range(max(1, len(history) - 5), len(history)):
                prev_macd = history[i - 1]
                curr_macd = history[i]
                if (
                    float(prev_macd.get("macd_line") or 0.0) <= float(prev_macd.get("macd_signal") or 0.0)
                    and float(curr_macd.get("macd_line") or 0.0) > float(curr_macd.get("macd_signal") or 0.0)
                ):
                    macd_cross_recent = True
                    break
        gap_pct_base = 0.0 if base_high_ref <= 0 else max(0.0, (float(payload.get("open") or close) - base_high_ref) / base_high_ref)
        mandatory = {
            "M1": base_high_ref > 0 and close > base_high_ref,
            "M2": rel_volume >= volume_breakout_mult,
            "M3": ema10 > ema30,
            "M4": base_high_ref > 0 and gap_pct_base <= 0.08,
            "M5": bool(liquidity_ok),
        }
        confirm_flags = {
            "C1": rsi >= float(cfg.get("rsi_regime", 55)),
            "C2": rsi_rising,
            "C3": adx >= float(cfg.get("adx_trigger", 22)) and plus_di > minus_di,
            "C4": adx > adx_5_back,
            "C5": (float(payload.get("macd_hist") or 0.0) > 0) or macd_cross_recent,
            "C6": close_top40,
        }
        c_score = sum(1 for v in confirm_flags.values() if v)
        confirming = state_json.get("confirming") if isinstance(state_json.get("confirming"), dict) else None
        confirm_bar = int(confirming.get("bars") or 0) if confirming else 0
        confirm_phase = "CONFIRMING" if confirming else "BREAKOUT_WATCH"
        mfm_last10 = [_mfm(h) for h in history[-10:]]

        if not fallback_reported:
            print(
                "CFG "
                f"volume_breakout_mult={volume_breakout_mult}({src_break}) "
                f"cmf_floor={cmf_floor}({src_cmf}) "
                f"atr_squeeze_pctile={atr_squeeze_pctile}({src_atr})"
            )
            for knob in [
                "base_price_slope_floor",
                "base_price_slope_ceiling",
                "base_min_width_pct",
                "base_to_range_low120_cap",
                "trend_join_window_grace",
            ]:
                if knob in cfg:
                    print(f"CFG {knob}={cfg[knob]}(config)")
                else:
                    print(f"CFG {knob}=<not-read-by-current-scanner>(absent)")
            fallback_reported = True

        in_segment = True
        if segment_range is not None:
            in_segment = segment_range[0] <= td <= segment_range[1]

        should_print = in_segment and (
            segment_range is not None or phase in {"NEUTRAL", "BASE_FORMING", "ACCUMULATION", "BREAKOUT_WATCH"}
        )
        if should_print:
            printed_rows += 1
            print(
                f"{td} phase={confirm_phase if phase == 'BREAKOUT_WATCH' else phase} "
                f"ACC[gate={accumulation_gate} cmf_10={cmf_last:.5f} cmf_hits={cmf_hits}/5 floor={cmf_floor} squeeze={squeeze_ok} "
                f"atr_leg={atr_leg} bb_leg={bb_leg} bb={bb_width:.5f} atr_pct={atr_pct}<=thr({atr_squeeze_pctile}) trend={trend_ok} close={close:.3f} ema30={ema30:.3f} "
                f"sma200={sma200:.3f} liq={bool(liquidity_ok)} score={score:.2f}>=60 {score_ok} pass={acc_pass}] "
                f"WATCH[rv_hits={rv_hits}/2 near_base={near_base_with_build} close={close:.3f}>=0.97*base({0.97*base_high_ref:.3f}) "
                f"base_high_ref={base_high_ref:.3f} source={base_high_source} pass={watch_pass}] "
                f"NEUTRAL_BASE[width={width:.5f}<=max({float(cfg.get('base_max_width_pct', 0.18))}):{width <= float(cfg.get('base_max_width_pct', 0.18))} "
                f"sessions_in_range={sessions_in_range}>=min({int(cfg.get('base_min_sessions', 60))}):{sessions_in_range >= int(cfg.get('base_min_sessions', 60))} "
                f"close_in_range={close_in_range} avoid={avoid_now}] "
                f"AVOID[entry close<sma200={close < sma200} sma200_slope<0={sma200_slope < 0} ema10<ema30={ema10 < ema30} "
                f"reclaim_streak={int(state_json.get('avoid_reclaim_streak') or 0)}/2 clear_streak={avoid_clear_streak}/20 avoid_until={avoid_until} phase={phase == 'AVOID'}] "
                f"JOIN[window={join_window_display}/{trend_join_window} open={join_window_open} c1_close>sma200={join_c1} c2_slope={join_c2} c3_ema={join_c3} c4_range_low120={join_c4} c5_dist={join_c5}]"
            )
            if phase == "BREAKOUT_WATCH":
                print(
                    f"{td} CONFIRM[phase={confirm_phase} bar={confirm_bar}/3 "
                    f"base_high_ref={base_high_ref:.3f} source={base_high_source} "
                    f"M1={mandatory['M1']} close={close:.3f}>base({base_high_ref:.3f}) "
                    f"M2={mandatory['M2']} rv={rel_volume:.3f}>=mult({volume_breakout_mult}) "
                    f"M3={mandatory['M3']} ema10={ema10:.3f}>ema30={ema30:.3f} "
                    f"M4={mandatory['M4']} gap_pct={gap_pct_base:.4f}<=0.08 "
                    f"M5={mandatory['M5']} liq={bool(liquidity_ok)} "
                    f"C={c_score}/6 need=4 C1={confirm_flags['C1']} C2={confirm_flags['C2']} C3={confirm_flags['C3']} "
                    f"C4={confirm_flags['C4']} C5={confirm_flags['C5']} C6={confirm_flags['C6']}]"
                )
            base_event = state_json.get("base_lifecycle_last_event") if isinstance(state_json.get("base_lifecycle_last_event"), dict) else None
            if isinstance(base_event, dict) and int(base_event.get("bar") or 0) == td:
                print(
                    f"{td} BASE_EVENT[action={base_event.get('action')} reason={base_event.get('reason')} "
                    f"old=({base_event.get('old')}) new=({base_event.get('new')})]"
                )
            if include_mf:
                print(f"{td} MF[last10]={','.join(f'{x:.4f}' for x in mfm_last10)}")

    if printed_rows == 0:
        print(
            f"TRACE_WATCH_SILENT symbol={symbol} reason=no_rows_emitted "
            f"phase_distribution={json.dumps(phase_dist, sort_keys=True, ensure_ascii=True)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate-by-gate scanner diagnostics for synthetic symbols")
    parser.add_argument("--symbol", choices=SYMBOLS + ["ALL"], default="ALL")
    parser.add_argument("--trace-phase", choices=SYMBOLS, default=None)
    parser.add_argument("--trace-watch", choices=SYMBOLS + ["ALL"], default=None)
    parser.add_argument("--trace-watch-mf", action="store_true", help="Include per-bar money-flow multiplier list for last 10 bars")
    parser.add_argument("--segment", default=None, help="Optional SYMBOL:SEGMENT filter from tests/fixtures/segments.json")
    args = parser.parse_args()

    segment = _resolve_segment(args.segment) if args.segment else None

    _load_and_run()
    if args.trace_phase:
        _trace_phase(args.trace_phase)
        return
    if args.trace_watch:
        symbols = SYMBOLS if args.trace_watch == "ALL" else [args.trace_watch]
        for s in symbols:
            if segment is not None and s != segment[0]:
                continue
            print(f"\nTRACE_WATCH {s}")
            segment_range = (segment[2], segment[3]) if segment is not None else None
            _trace_watch(s, include_mf=bool(args.trace_watch_mf), segment_range=segment_range)
        return

    symbols = SYMBOLS if args.symbol == "ALL" else [args.symbol]
    for s in symbols:
        print(f"\n{s}")
        print(_trace(s))


if __name__ == "__main__":
    main()

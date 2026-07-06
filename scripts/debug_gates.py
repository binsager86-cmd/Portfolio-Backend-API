from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.database import query_all, query_one, query_val
from app.services.eagle_eye.backtest_service import run_backtest
from app.services.eagle_eye.pipeline import process_bar
from app.services.eagle_eye.indicator_service import compute_and_store_symbol
from app.services.eagle_eye.market_data_service import ensure_schema, load_ohlcv_csv, get_active_config, get_config_hash
from app.services.eagle_eye.audit_service import ensure_schema as ensure_audit_schema
from app.services.eagle_eye.rating_service import compute_rating_from_indicator, store_rating


SYMBOLS = ["TIJARA", "BPCC", "ZAIN", "SANAM", "MABANEE", "JOINER"]
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def _load_and_run() -> None:
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


def _trace_watch(symbol: str, include_mf: bool = False) -> None:
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
        base_high_ref = float(state.get("base_high") or payload.get("range_high_60") or payload.get("range_high_120") or 0.0)
        range_high_60 = float(payload.get("range_high_60") or 0.0)
        range_low_60 = float(payload.get("range_low_60") or 0.0)
        width = float(payload.get("range_width_pct") or 9.0)

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
        squeeze_ok = bb_width <= 0.12
        if atr_pct is not None:
            squeeze_ok = squeeze_ok or (atr_pct <= atr_squeeze_pctile)
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
        near_base_with_build = close >= (0.97 * base_high_ref) and rv_hits >= 2
        watch_pass = near_base_with_build

        closes_60 = [float(h.get("close") or 0.0) for h in history[-60:]]
        sessions_in_range = sum(1 for c in closes_60 if range_low_60 <= c <= range_high_60)
        close_in_range = range_low_60 <= close <= range_high_60
        avoid_now = close < sma200 and sma200_slope < 0 and ema10 < ema30
        join_window_open = coverage_sessions > 0 and coverage_sessions <= trend_join_window
        join_c1 = close > sma200
        join_c2 = sma200_slope > 0
        join_c3 = ema10 > ema30
        join_c4 = float(payload.get("range_low_120") or 0.0) > 0
        join_c5 = close >= (float(payload.get("range_low_120") or 0.0) * 1.15) if join_c4 else False
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

        if phase in {"NEUTRAL", "BASE_FORMING", "ACCUMULATION"}:
            printed_rows += 1
            print(
                f"{td} phase={phase} "
                f"ACC[gate={accumulation_gate} cmf_10={cmf_last:.5f} cmf_hits={cmf_hits}/5 floor={cmf_floor} squeeze={squeeze_ok} bb={bb_width:.5f} "
                f"atr_pct={atr_pct}<=thr({atr_squeeze_pctile}) trend={trend_ok} close={close:.3f} ema30={ema30:.3f} "
                f"sma200={sma200:.3f} liq={bool(liquidity_ok)} score={score:.2f}>=60 {score_ok} pass={acc_pass}] "
                f"WATCH[rv_hits={rv_hits}/2 near_base={near_base_with_build} close={close:.3f}>=0.97*base({0.97*base_high_ref:.3f}) "
                f"base_high_ref={base_high_ref:.3f} pass={watch_pass}] "
                f"NEUTRAL_BASE[width={width:.5f}<=max({float(cfg.get('base_max_width_pct', 0.18))}):{width <= float(cfg.get('base_max_width_pct', 0.18))} "
                f"sessions_in_range={sessions_in_range}>=min({int(cfg.get('base_min_sessions', 60))}):{sessions_in_range >= int(cfg.get('base_min_sessions', 60))} "
                f"close_in_range={close_in_range} avoid={avoid_now}] "
                f"JOIN[window={coverage_sessions}/{trend_join_window} open={join_window_open} c1_close>sma200={join_c1} c2_slope={join_c2} c3_ema={join_c3} c4_range_low120={join_c4} c5_dist={join_c5}]"
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
    args = parser.parse_args()

    _load_and_run()
    if args.trace_phase:
        _trace_phase(args.trace_phase)
        return
    if args.trace_watch:
        symbols = SYMBOLS if args.trace_watch == "ALL" else [args.trace_watch]
        for s in symbols:
            print(f"\nTRACE_WATCH {s}")
            _trace_watch(s, include_mf=bool(args.trace_watch_mf))
        return

    symbols = SYMBOLS if args.symbol == "ALL" else [args.symbol]
    for s in symbols:
        print(f"\n{s}")
        print(_trace(s))


if __name__ == "__main__":
    main()

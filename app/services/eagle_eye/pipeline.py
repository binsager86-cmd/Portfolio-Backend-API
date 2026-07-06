"""
Canonical Eagle Eye bar-processing pipeline.
"""
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from app.core.database import query_all
from app.services.eagle_eye.adapter import DataAdapter
from app.services.eagle_eye.config import CONFIG
from app.services.eagle_eye.dna_extractor import dna_to_dict, extract_dna
from app.services.eagle_eye.indicators import compute_all_indicators
from app.services.eagle_eye.indicator_service import load_latest_indicator
from app.services.eagle_eye.market_data_service import get_active_config, get_cfg
from app.services.eagle_eye.move_detector import detect_fakeouts, detect_moves
from app.services.eagle_eye.recorder import record_all_events
from app.services.eagle_eye.rating_service import compute_rating_from_indicator, store_rating
from app.services.eagle_eye.risk_service import liquidity_filter_at
from app.services.eagle_eye.scanner_service import evaluate_symbol


def _load_recent_indicators(symbol: str, trade_date: int, limit: int = 140) -> list[dict[str, Any]]:
    rows = query_all(
        """
        SELECT trade_date, payload_json
        FROM ee_indicators
        WHERE symbol = ? AND trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT ?
        """,
        (symbol, trade_date, limit),
    )
    out: list[dict[str, Any]] = []
    for row in reversed(rows or []):
        try:
            payload = json.loads(str(row.get("payload_json") or "{}"))
        except Exception:
            payload = {}
        payload["trade_date"] = int(row.get("trade_date") or 0)
        out.append(payload)
    return out


def _derive_warmup_coverage(history: list[dict[str, Any]], trade_date: int) -> tuple[int | None, int | None]:
    warmup_start: int | None = None
    for row in history:
        td = int(row.get("trade_date") or 0)
        if td <= 0 or td > trade_date:
            continue
        sma200 = float(row.get("sma200") or 0.0)
        range_low_120 = float(row.get("range_low_120") or 0.0)
        range_high_120 = float(row.get("range_high_120") or 0.0)
        if sma200 > 0 and range_low_120 > 0 and range_high_120 > 0:
            warmup_start = td
            break

    if warmup_start is None:
        return None, None

    sessions = sum(1 for row in history if warmup_start <= int(row.get("trade_date") or 0) <= trade_date)
    return warmup_start, sessions


def process_bar(
    symbol: str,
    trade_date: int,
    cfg: dict[str, Any] | None = None,
    *,
    trace_id: str | None = None,
    indicator_payload: dict[str, Any] | None = None,
    indicator_history: list[dict[str, Any]] | None = None,
    state_override: dict[str, Any] | None = None,
    persist_state: bool = True,
    liquidity_snapshot: tuple[bool, dict[str, Any]] | None = None,
    coverage_start_date: int | None = None,
    coverage_sessions: int | None = None,
    score: float | None = None,
    band: str | None = None,
    components: dict[str, Any] | None = None,
    persist_rating: bool = False,
) -> dict[str, Any]:
    cfg = cfg or get_active_config()
    payload = indicator_payload if indicator_payload is not None else load_latest_indicator(symbol, trade_date)
    if not payload:
        return {"symbol": symbol, "status": "no_indicator"}

    history = indicator_history if indicator_history is not None else _load_recent_indicators(symbol, trade_date, 140)
    if not history:
        return {"symbol": symbol, "status": "no_history"}

    if coverage_start_date is None or coverage_sessions is None:
        inferred_start, inferred_sessions = _derive_warmup_coverage(history, trade_date)
        if coverage_start_date is None:
            coverage_start_date = inferred_start
        if coverage_sessions is None:
            coverage_sessions = inferred_sessions

    if liquidity_snapshot is None:
        min_daily_value_kwd = float(get_cfg(cfg, "min_daily_value_kwd"))
        liquidity_snapshot = liquidity_filter_at(symbol, trade_date, min_daily_value_kwd)

    if score is None or band is None or components is None:
        liquidity_score = 100.0 if liquidity_snapshot[0] else 20.0
        score, band, components = compute_rating_from_indicator(payload, liquidity_score=liquidity_score)

    if persist_rating:
        store_rating(symbol, trade_date, float(score), str(band), components or {})

    return evaluate_symbol(
        symbol,
        trade_date,
        float(score),
        cfg,
        trace_id=trace_id,
        indicator_payload=payload,
        indicator_history=history,
        state_override=state_override,
        persist_state=persist_state,
        liquidity_snapshot=liquidity_snapshot,
        coverage_start_date=coverage_start_date,
        coverage_sessions=coverage_sessions,
    )


def run_phase1(
    adapter: DataAdapter,
    output_dir: str = "./output",
    verbose: bool = True,
) -> Dict[str, dict]:
    """Run the full Phase 1 pipeline. Returns dict of {ticker: dna_dict}."""

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "events").mkdir(exist_ok=True)
    (out_path / "dna").mkdir(exist_ok=True)
    (out_path / "indicators").mkdir(exist_ok=True)

    stocks = adapter.list_stocks()
    if verbose:
        print(f"Phase 1 starting — {len(stocks)} stocks to study")
        print(f"History window: {CONFIG.HISTORY_YEARS} years")
        print(f"Move thresholds (%): {CONFIG.MOVE_THRESHOLDS_PCT}")
        print(f"Pre-move snapshot lookbacks (days): {CONFIG.PRE_MOVE_LOOKBACK_DAYS}")
        print("=" * 70)

    end_date = date.today()
    start_date = end_date - timedelta(days=int(CONFIG.HISTORY_YEARS * 365.25))

    all_dna = {}

    for stock in stocks:
        ticker = stock.ticker
        if verbose:
            print(f"\n[{ticker}] {stock.name_en}")

        ohlcv = adapter.get_ohlcv_daily(ticker, start_date, end_date)
        if len(ohlcv) < CONFIG.MIN_HISTORY_DAYS_REQUIRED:
            if verbose:
                print(f"  skip: only {len(ohlcv)} days available, need {CONFIG.MIN_HISTORY_DAYS_REQUIRED}")
            continue

        if verbose:
            print(f"  loaded {len(ohlcv)} bars ({ohlcv.index[0].date()} → {ohlcv.index[-1].date()})")

        try:
            indicators_df = compute_all_indicators(ohlcv)
        except Exception as e:
            print(f"  error computing indicators: {e}")
            continue

        indicators_df.to_csv(out_path / "indicators" / f"{ticker}.csv")

        events = detect_moves(ticker, ohlcv)
        if verbose:
            event_counts = {}
            for e in events:
                event_counts[e.threshold_pct] = event_counts.get(e.threshold_pct, 0) + 1
            print(f"  detected moves: {dict(sorted(event_counts.items()))}")

        fakeouts = detect_fakeouts(ticker, ohlcv)
        if verbose:
            print(f"  detected fakeouts: {len(fakeouts)}")

        all_event_snapshots = record_all_events(events + fakeouts, indicators_df)
        real_snapshots = [s for s in all_event_snapshots if not s.event.is_fakeout]
        fake_snapshots = [s for s in all_event_snapshots if s.event.is_fakeout]

        events_records = []
        for s in all_event_snapshots:
            rec = {
                **{k: v for k, v in vars(s.event).items()},
                "snapshot_lookbacks_captured": list(s.indicator_snapshots.keys()),
                "signals_fired": len(s.signal_sequence),
                "earliest_signal_days_before": (
                    s.signal_sequence[0]['days_before_acceleration']
                    if s.signal_sequence else 0
                ),
            }
            for k in ('start_date', 'acceleration_date', 'peak_date'):
                if rec.get(k) is not None:
                    rec[k] = rec[k].isoformat()
            events_records.append(rec)
        if events_records:
            pd.DataFrame(events_records).to_csv(
                out_path / "events" / f"{ticker}_events.csv", index=False
            )

        dna = extract_dna(ticker, real_snapshots, fake_snapshots, indicators_df=indicators_df)
        if dna is None:
            if verbose:
                print(f"  insufficient events to build DNA")
            continue

        dna_dict = dna_to_dict(dna)
        all_dna[ticker] = dna_dict
        with open(out_path / "dna" / f"{ticker}_dna.json", 'w') as f:
            json.dump(dna_dict, f, indent=2)

        if verbose:
            print(f"  personality: {dna.personality_tag}")
            print(f"  avg consolidation before move: {dna.avg_pre_move_consolidation_days:.1f} days")
            print(f"  avg move duration: {dna.avg_move_duration_days:.1f} days")
            print(f"  avg move magnitude: {dna.avg_move_magnitude_pct:.1f}%")
            if dna.most_reliable_signals_overall:
                top = dna.most_reliable_signals_overall[0]
                print(f"  most reliable early warning: {top.signal} "
                      f"(avg lead {top.avg_lead_days:.0f}d, "
                      f"reliability {top.reliability_pct:.0f}%, "
                      f"FPR {top.false_positive_rate:.0f}%)")

    summary = {
        "phase": 1,
        "config": {
            "history_years": CONFIG.HISTORY_YEARS,
            "move_thresholds_pct": list(CONFIG.MOVE_THRESHOLDS_PCT),
            "pre_move_lookbacks": list(CONFIG.PRE_MOVE_LOOKBACK_DAYS),
        },
        "stocks_studied": len(all_dna),
        "stocks_attempted": len(stocks),
        "tickers": list(all_dna.keys()),
    }
    with open(out_path / "phase1_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print("\n" + "=" * 70)
        print(f"Phase 1 complete — {len(all_dna)}/{len(stocks)} stocks have behavioral DNA")
        print(f"Output: {out_path.resolve()}")

    return all_dna

"""
Phase 1 Pipeline Orchestrator.

Runs the full forensic learning pipeline:
  1. Pull historical data via the DataAdapter
  2. Compute indicators for the full history
  3. Detect every significant move at every threshold
  4. Detect fakeouts (control group)
  5. Capture forensic snapshots for every event
  6. Extract behavioral DNA per stock
  7. Persist results

After this runs, every stock has a stored "behavioral DNA" that becomes the
foundation Phase 2 (ML) and Phase 3 (live rating) build on.
"""
import json
from datetime import date, timedelta
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np

from core.config import CONFIG
from data.adapters import DataAdapter
from indicators.engine import compute_all_indicators
from forensics.move_detector import detect_moves, detect_fakeouts
from forensics.recorder import record_all_events
from forensics.dna_extractor import extract_dna, dna_to_dict


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

        # 1. Pull data
        ohlcv = adapter.get_ohlcv_daily(ticker, start_date, end_date)
        if len(ohlcv) < CONFIG.MIN_HISTORY_DAYS_REQUIRED:
            if verbose:
                print(f"  skip: only {len(ohlcv)} days available, need {CONFIG.MIN_HISTORY_DAYS_REQUIRED}")
            continue

        if verbose:
            print(f"  loaded {len(ohlcv)} bars ({ohlcv.index[0].date()} → {ohlcv.index[-1].date()})")

        # 2. Compute indicators
        try:
            indicators_df = compute_all_indicators(ohlcv)
        except Exception as e:
            print(f"  error computing indicators: {e}")
            continue

        # Persist indicator history (CSV — switch to parquet in production)
        indicators_df.to_csv(out_path / "indicators" / f"{ticker}.csv")

        # 3. Detect real moves at all thresholds
        events = detect_moves(ticker, ohlcv)
        if verbose:
            event_counts = {}
            for e in events:
                event_counts[e.threshold_pct] = event_counts.get(e.threshold_pct, 0) + 1
            print(f"  detected moves: {dict(sorted(event_counts.items()))}")

        # 4. Detect fakeouts (control group)
        fakeouts = detect_fakeouts(ticker, ohlcv)
        if verbose:
            print(f"  detected fakeouts: {len(fakeouts)}")

        # 5. Record forensic snapshots
        all_event_snapshots = record_all_events(events + fakeouts, indicators_df)
        real_snapshots = [s for s in all_event_snapshots if not s.event.is_fakeout]
        fake_snapshots = [s for s in all_event_snapshots if s.event.is_fakeout]

        # Persist event data (one row per event with metadata)
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
            # Stringify dates
            for k in ('start_date', 'acceleration_date', 'peak_date'):
                if rec.get(k) is not None:
                    rec[k] = rec[k].isoformat()
            events_records.append(rec)
        if events_records:
            pd.DataFrame(events_records).to_csv(
                out_path / "events" / f"{ticker}_events.csv", index=False
            )

        # 6. Extract DNA
        dna = extract_dna(ticker, real_snapshots, fake_snapshots)
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

    # Save a master summary
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

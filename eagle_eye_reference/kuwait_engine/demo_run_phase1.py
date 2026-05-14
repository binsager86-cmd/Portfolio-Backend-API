"""
Phase 1 Demo Runner.

Runs the full forensic learning pipeline end-to-end on synthetic Kuwait-like
stocks. Validates that every component works together before we plug in your
real ticker chart API.

Usage:
    python demo_run_phase1.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.adapters import SyntheticAdapter
from core.pipeline import run_phase1


if __name__ == "__main__":
    print("=" * 70)
    print("Kuwait Stock Analysis Engine — Phase 1 Demo Run")
    print("Using SyntheticAdapter (real adapter pending API discovery)")
    print("=" * 70)

    adapter = SyntheticAdapter(seed=2024)
    all_dna = run_phase1(adapter, output_dir="./output", verbose=True)

    print("\n" + "=" * 70)
    print("SUMMARY — Per-stock behavioral DNA learned")
    print("=" * 70)
    for ticker, dna in all_dna.items():
        print(f"\n{ticker} ({dna['personality_tag']}):")
        print(f"  Events studied: {dna['total_events_studied']} real, {dna['fakeouts_studied']} fakeouts")
        print(f"  Avg pre-move consolidation: {dna['avg_pre_move_consolidation_days']} days")
        print(f"  Avg move duration: {dna['avg_move_duration_days']} days")
        print(f"  Avg move magnitude: {dna['avg_move_magnitude_pct']}%")
        if dna['most_reliable_signals_overall']:
            print(f"  Top 3 reliable early-warning signals:")
            for s in dna['most_reliable_signals_overall'][:3]:
                print(f"    • {s['signal']:40s}  lead={s['avg_lead_days']:5.1f}d "
                      f"reliability={s['reliability_pct']:5.1f}%  FPR={s['false_positive_rate']:5.1f}%  "
                      f"discriminative={s['discriminative_power']:+.1f}")

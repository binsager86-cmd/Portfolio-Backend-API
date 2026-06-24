"""
Eagle Eye - Audit Snapshot Completeness (READ-ONLY)

Goal verification: the tracker must save the FULL set of indicators that drove
each rating, every day, append-only - so weekly reviews can diagnose WHY a
rating was given and whether the indicators that justified it actually predicted
the outcome.

This script inspects what's actually stored in ee_rating_snapshots.indicators_json
to confirm completeness (all ~48 indices) vs a partial subset.

Usage (from backend-api root):
    python scripts/audit_snapshot_completeness.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())

from app.core.database import query_all, query_val


EXPECTED_INDICATORS = [
    "traded_value_ratio_20d",
    "cmf_20",
    "obv_slope_20d",
    "up_down_volume_ratio_20d",
    "high_volume_weak_close_flag",
    "stock_close_vs_200sma",
    "stock_close_vs_50sma",
    "stock_50sma_slope_20d",
    "market_close_vs_200sma",
    "rsi_14",
    "cci_20",
    "relative_strength_3m",
    "macd_histogram_slope_5d",
    "return_3m",
    "bb_width_percentile_252d",
    "donchian_breakout_50d",
    "close_location_value",
    "failed_breakout_flag",
    "distance_to_major_resistance",
    "risk_reward_ratio",
    "price_extension_from_50sma",
    "plus_di",
    "minus_di",
    "cmf_20_change_5d",
    "cmf_20_change_10d",
    "obv_slope_change_10d",
    "rsi_14_change_5d",
    "consecutive_up_closes",
    "pct_above_60d_low",
    "ema_10",
    "ema_20",
    "ema_30",
    "macd_histogram",
    "volume_ratio_20d",
    "active_trading_days_ratio_60d",
    "near_zero_volume_flag",
    "data_quality_score",
    "close",
]


def main() -> None:
    print("\nEAGLE EYE - SNAPSHOT COMPLETENESS AUDIT (read-only)")
    print("=" * 70)

    total = query_val("SELECT COUNT(*) FROM ee_rating_snapshots", ()) or 0
    if total == 0:
        print("No snapshots stored yet. Run a recompute first.")
        return

    samples = query_all(
        """
        SELECT ticker, rating, snapshot_date, indicators_json,
               liquidity_score, trend_score, momentum_score,
               geometry_score, rr_score
        FROM ee_rating_snapshots
        WHERE indicators_json IS NOT NULL
        ORDER BY snapshot_date DESC
        LIMIT 5
        """,
        (),
    )

    if not samples:
        print("Snapshots exist but indicators_json is empty/null - THIS IS THE GAP.")
        print("The 'why' behind ratings is not being saved.")
        return

    print(f"\nInspecting {len(samples)} sample snapshot rows:\n")

    for sample in samples:
        ticker = sample["ticker"]
        rating = sample["rating"]
        snapshot_date = sample["snapshot_date"]
        try:
            indicators = json.loads(sample["indicators_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            indicators = {}

        stored_keys = set(indicators.keys())
        expected = set(EXPECTED_INDICATORS)
        present = expected & stored_keys
        missing = expected - stored_keys
        extra = stored_keys - expected

        print(f"--- {ticker} ({rating}, {snapshot_date}) ---")
        print(f"  Total keys stored in indicators_json: {len(stored_keys)}")
        print(f"  Expected indicators present: {len(present)}/{len(expected)}")
        if missing:
            print(f"  MISSING ({len(missing)}): {sorted(missing)}")
        else:
            print("  MISSING: none - full indicator set captured")
        if extra:
            print(f"  Extra keys also stored: {len(extra)}")
        print(
            f"  Family scores (columns): liq={sample.get('liquidity_score')} "
            f"trend={sample.get('trend_score')} mom={sample.get('momentum_score')} "
            f"geom={sample.get('geometry_score')} rr={sample.get('rr_score')}"
        )
        print()

    print("=" * 70)
    first = samples[0]
    try:
        indicators = json.loads(first["indicators_json"] or "{}")
    except (json.JSONDecodeError, TypeError):
        indicators = {}
    present_count = len(set(EXPECTED_INDICATORS) & set(indicators.keys()))
    coverage = present_count / len(EXPECTED_INDICATORS) * 100

    print(
        f"VERDICT: indicator coverage ~= {coverage:.0f}% "
        f"({present_count}/{len(EXPECTED_INDICATORS)})"
    )
    if coverage >= 90:
        print("GOAL MET: the full 'why' behind each rating is being saved daily.")
        print("Weekly reviews can diagnose which indicators drove each call.")
    elif coverage >= 50:
        print("PARTIAL: some indicators saved, but gaps exist. Review missing list.")
    else:
        print("GAP: indicators_json is missing most indices. The 'why' is not")
        print("fully captured - weekly diagnosis will be limited. Needs a fix to")
        print("persist the complete indicator dict in the snapshot.")
    print("=" * 70)

    fam_check = query_all(
        """
        SELECT
          SUM(CASE WHEN liquidity_score IS NOT NULL THEN 1 ELSE 0 END) AS liq,
          SUM(CASE WHEN trend_score IS NOT NULL THEN 1 ELSE 0 END) AS trend,
          SUM(CASE WHEN momentum_score IS NOT NULL THEN 1 ELSE 0 END) AS mom,
          SUM(CASE WHEN geometry_score IS NOT NULL THEN 1 ELSE 0 END) AS geom,
          SUM(CASE WHEN rr_score IS NOT NULL THEN 1 ELSE 0 END) AS rr,
          COUNT(*) AS total
        FROM ee_rating_snapshots
        """,
        (),
    )
    if fam_check:
        row = fam_check[0]
        print(f"\nFamily score column population (of {row['total']} rows):")
        print(
            f"  liquidity: {row['liq']}  trend: {row['trend']}  "
            f"momentum: {row['mom']}  geometry: {row['geom']}  risk_reward: {row['rr']}"
        )
        if row["liq"] == 0:
            print("  Family score columns are empty - they're extracted from")
            print("  indicators_json which may not contain family scores.")
            print("  Family scores may need to be added to the indicator dict")
            print("  before snapshotting, or read from a different source.")


if __name__ == "__main__":
    main()
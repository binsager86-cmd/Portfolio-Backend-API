#!/usr/bin/env python3
"""
Pre-Phase-2: Dampener fire-test + composition matrix.

Task 1 — Synthetic dampener fire-test (3 cases):
  • Trigger case:  thin volume + rise   → dampener fires, confidence capped at 60
  • Variant A:     thin volume, no rise → dampener must NOT fire
  • Variant B:     rise, normal volume  → dampener must NOT fire

Task 2 — Composition matrix (6 scenarios, real code execution).

Runs the REAL code paths from indicators.py (compute_all_indicators)
and applies the VERBATIM dampener logic from ingest.py lines 426-441.
"""
from __future__ import annotations

import sys
import os

# Make sure we can import from the backend-api package
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import pandas as pd
import numpy as np
from datetime import date, timedelta

# ── Real code imports ────────────────────────────────────────────────────────
from app.services.eagle_eye.indicators import compute_all_indicators

# ============================================================================
# Helpers
# ============================================================================

def build_synthetic_df(last_close: float, last_volume: int,
                        n_baseline: int = 49) -> pd.DataFrame:
    """
    OHLCV fixture: n_baseline days at baseline, then one custom test day.

    The task spec called for 30 bars but compute_all_indicators() requires >= 50.
    We use 49 baseline days + 1 test day = 50 bars; the dampener math is identical
    because the 20-day rolling median (shifted by 1) still uses all-baseline bars.

    Baseline: close=100, volume=1_000_000 → dollar_volume=100_000_000 per day.
    """
    baseline_close = 100.0
    baseline_volume = 1_000_000

    n_total = n_baseline + 1
    dates = pd.date_range(start="2025-01-01", periods=n_total, freq="B")

    closes  = [baseline_close]  * n_baseline + [last_close]
    volumes = [baseline_volume] * n_baseline + [last_volume]
    opens   = [c * 0.995 for c in closes]
    highs   = [c * 1.01  for c in closes]
    lows    = [c * 0.99  for c in closes]
    # turnover_kwd used by indicators.py for dollar_volume (preferred over vol×close)
    turnovers = [v * c for v, c in zip(volumes, closes)]

    return pd.DataFrame({
        "open":         opens,
        "high":         highs,
        "low":          lows,
        "close":        closes,
        "volume":       volumes,
        "turnover_kwd": turnovers,
    }, index=dates)


# ── Verbatim dampener block from ingest.py lines 426-441 ────────────────────
def _apply_dampener(ind_df: pd.DataFrame, df: pd.DataFrame, confidence: float) -> dict:
    """
    Exact copy of the thin-volume-on-rise dampener from ingest.py.
    Returns a result dict; does NOT call log_compute (no DB in tests).
    """
    dampener_fired = False
    score_reasons: list = []
    rel_liq = None
    today_ret = None
    median_dv = None
    today_dv = None

    # ── VERBATIM from ingest.py lines 426-441 ────────────────────────────────
    if "dollar_volume" in ind_df.columns and len(ind_df) >= 21:
        _dv = ind_df["dollar_volume"]
        _median_dv = float(_dv.rolling(20).median().shift(1).iloc[-1])
        _today_dv = float(_dv.iloc[-1])
        _rel_liq = _today_dv / _median_dv if _median_dv > 0 else 0.0
        _today_ret = (
            float((df["close"].iloc[-1] / df["close"].iloc[-2]) - 1)
            if len(df) >= 2 else 0.0
        )
        if _rel_liq < 0.5 and _today_ret > 0.02:
            confidence = min(confidence, 60)
            dampener_fired = True
            score_reasons.append(
                f"thin_volume_on_rise: rel_liq={_rel_liq:.4f} ret={_today_ret:.4f} conf_capped=60"
            )
        rel_liq  = _rel_liq
        today_ret = _today_ret
        median_dv = _median_dv
        today_dv  = _today_dv
    # ── END VERBATIM ─────────────────────────────────────────────────────────

    return {
        "confidence_final": round(confidence, 2),
        "dampener_fired":   dampener_fired,
        "score_reasons":    score_reasons,
        "rel_liq":          round(rel_liq,  4) if rel_liq  is not None else None,
        "today_ret":        round(today_ret, 4) if today_ret is not None else None,
        "median_dv":        round(median_dv, 0) if median_dv is not None else None,
        "today_dv":         round(today_dv,  0) if today_dv  is not None else None,
    }


def run_case(label: str, last_close: float, last_volume: int,
             pre_confidence: float = 85.0) -> dict:
    df      = build_synthetic_df(last_close, last_volume)
    ind_df  = compute_all_indicators(df)
    result  = _apply_dampener(ind_df, df, pre_confidence)
    result["label"] = label
    return result


# ── Verbatim Adjustment-1 block from ingest.py lines 413-423 ────────────────
def _apply_adj1(volume_context: dict, confidence: float) -> float:
    """
    Exact copy of the volume-context confidence multiplier from ingest.py.
    Returns confidence after Adjustment 1.
    """
    tier = volume_context["liquidity_tier"]
    _LIQUIDITY_MULTIPLIERS = {"ILLIQUID": 0.50, "WATCH_ONLY": 0.70}
    if tier in _LIQUIDITY_MULTIPLIERS:
        confidence = confidence * _LIQUIDITY_MULTIPLIERS[tier]
    elif not volume_context["is_volume_confirmed"]:
        confidence = confidence * 0.85
    elif volume_context["relative_volume_percentile"] > 80:
        confidence = min(confidence * 1.10, 100)
    confidence = round(min(confidence, 100), 2)
    return confidence


# ============================================================================
# Task 1 — Dampener fire-test
# ============================================================================

def task1_fire_test() -> list[dict]:
    cases = [
        # (label, last_close, last_volume, expect_fired, expect_conf)
        ("TRIGGER: thin_vol + rise (+3%, 25% vol)",    103.0, 250_000,   True,  60.0),
        ("VARIANT_A: thin_vol, no_rise (0%, 25% vol)", 100.0, 250_000,   False, 85.0),
        ("VARIANT_B: rise, normal_vol (+3%, 100% vol)", 103.0, 1_000_000, False, 85.0),
    ]

    results = []
    all_pass = True

    print("=" * 70)
    print("TASK 1 — DAMPENER FIRE-TEST")
    print("=" * 70)

    for label, d_close, d_vol, exp_fired, exp_conf in cases:
        r = run_case(label, last_close=d_close, last_volume=d_vol, pre_confidence=85.0)

        fired_ok = r["dampener_fired"] == exp_fired
        conf_ok  = r["confidence_final"] == exp_conf

        status = "PASS" if (fired_ok and conf_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"\n  [{status}] {label}")
        print(f"         median_dv   = {r['median_dv']:,.0f}" if r['median_dv'] else "         median_dv   = N/A")
        print(f"         today_dv    = {r['today_dv']:,.0f}"  if r['today_dv']  else "         today_dv    = N/A")
        print(f"         rel_liq     = {r['rel_liq']}"  if r['rel_liq']  is not None else "         rel_liq     = N/A")
        print(f"         today_ret   = {r['today_ret']}" if r['today_ret'] is not None else "         today_ret   = N/A")
        print(f"         fired       = {r['dampener_fired']}  (expected {exp_fired})")
        print(f"         confidence  = {r['confidence_final']}  (expected {exp_conf})")
        if r["score_reasons"]:
            print(f"         reasons     = {r['score_reasons']}")

        r.update({"expected_fired": exp_fired, "expected_conf": exp_conf, "status": status})
        results.append(r)

    print()
    print("TASK 1 OVERALL:", "ALL PASS" if all_pass else "FAILURES DETECTED -- STOP")
    return results


# ============================================================================
# Task 2 — Composition matrix
# ============================================================================

def task2_composition_matrix() -> list[dict]:
    """
    Run 6 scenarios through real Adjustment-1 + verbatim Adjustment-2 code.
    volume_context dicts constructed to match each scenario specification.
    """
    PRE_CONF = 85.0

    scenarios = [
        # (label, tier, vol_confirmed, high_pct, rel_liq, today_ret)
        ("TRADEABLE | confirmed | low_pct | liq=0.8 | ret=0.01",
         "TRADEABLE", True,  False, 0.80, 0.010),
        ("TRADEABLE | confirmed | high_pct | liq=0.9 | ret=0.005",
         "TRADEABLE", True,  True,  0.90, 0.005),
        ("TRADEABLE | NOT confirmed | low_pct | liq=0.4 | ret=0.025",
         "TRADEABLE", False, False, 0.40, 0.025),
        ("WATCH_ONLY | confirmed | low_pct | liq=0.6 | ret=0.005",
         "WATCH_ONLY", True, False, 0.60, 0.005),
        ("WATCH_ONLY | NOT confirmed | low_pct | liq=0.3 | ret=0.025",
         "WATCH_ONLY", False, False, 0.30, 0.025),
        ("ILLIQUID | NOT confirmed | low_pct | liq=0.2 | ret=0.03",
         "ILLIQUID", False, False, 0.20, 0.030),
    ]

    results = []
    print()
    print("=" * 70)
    print("TASK 2 — COMPOSITION MATRIX (pre-confidence = 85)")
    print("=" * 70)

    header = (
        f"  {'Scenario':<55} | {'Adj1':>6} | {'Adj2':>6} | {'Final':>6} | "
        f"{'D-fired':<8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, tier, vol_confirmed, high_pct, rel_liq, today_ret in scenarios:
        # Build a minimal volume_context that matches the scenario
        volume_context = {
            "liquidity_tier":             tier,
            "is_volume_confirmed":        vol_confirmed,
            # rv_percentile >80 iff high_pct=True
            "relative_volume_percentile": 85.0 if high_pct else 50.0,
        }

        # Adjustment 1 — verbatim from ingest.py
        after_adj1 = _apply_adj1(volume_context, PRE_CONF)

        # Adjustment 2 — construct synthetic ind_df/df to exercise real dampener
        # Build an ind_df where dollar_volume produces exactly the specified rel_liq
        # on day 30, given a 20-day median of 100_000_000.
        MEDIAN_DV = 100_000_000.0
        today_dv  = rel_liq * MEDIAN_DV

        # 30-bar index; days 1-29 must all have MEDIAN_DV so median is exact
        dates = pd.date_range(start="2025-01-01", periods=30, freq="B")
        dv_series = pd.Series([MEDIAN_DV] * 29 + [today_dv], index=dates)

        # Minimal ind_df containing only dollar_volume (sufficient for dampener block)
        ind_df = pd.DataFrame({"dollar_volume": dv_series})

        # df with close to produce today_ret
        base_close = 100.0
        day30_close = base_close * (1 + today_ret)
        closes = pd.Series([base_close] * 29 + [day30_close], index=dates)
        df_mini = pd.DataFrame({"close": closes})

        result = _apply_dampener(ind_df, df_mini, after_adj1)
        after_adj2 = result["confidence_final"]
        fired      = result["dampener_fired"]

        row = {
            "label":      label,
            "tier":       tier,
            "vol_conf":   vol_confirmed,
            "high_pct":   high_pct,
            "rel_liq":    rel_liq,
            "today_ret":  today_ret,
            "after_adj1": after_adj1,
            "after_adj2": after_adj2,
            "final":      after_adj2,
            "dampened":   fired,
        }
        results.append(row)

        print(
            f"  {label:<55} | {after_adj1:>6.2f} | {after_adj2:>6.2f} | {after_adj2:>6.2f} | "
            f"{'Yes' if fired else 'No':<8}"
        )

    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    task1_results = task1_fire_test()
    task2_results = task2_composition_matrix()

    # Final verdict
    print()
    print("=" * 70)
    all_t1_pass = all(r["status"] == "PASS" for r in task1_results)
    if all_t1_pass:
        print("VERDICT: DAMPENER VERIFIED -- ALL 3 FIRE-TEST CASES PASS")
    else:
        failed = [r["label"] for r in task1_results if r["status"] != "PASS"]
        print("VERDICT: FIRE-TEST FAILED --", failed)
    print("=" * 70)

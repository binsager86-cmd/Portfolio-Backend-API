"""
run_rating_reality_check.py — Phase 2B: Rating Reality Check

For each ticker in [KFH, NBK, BOUBYAN, MABANEE, AGILITY]:
  For each sampled trading day T in 2024-01-01 through 2025-09-30:
    - Compute Phase 1 rule-based rating + confidence (only data <= T)
    - Record what actually happened in next 20 trading days
    - Bucket predictions by confidence band [0-49, 50-59, ..., 90-100]
    - Compute: n, tp1_hit_rate, mean_return, mean_drawdown, calibration_delta

Sampling: every 5th trading day (~440 total prediction days).

Output:
  tools/rating_reality_check.json
  tools/rating_reality_check.txt
"""
from __future__ import annotations

import json
import traceback
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.services.eagle_eye.indicators import compute_all_indicators
from app.services.eagle_eye.rating_engine import (
    classify_stage,
    compute_confidence,
    compute_entry_stop_targets,
    compute_rating,
    compute_support_resistance,
)
from app.services.eagle_eye.store import load_ohlcv

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TICKERS = ["KFH", "NBK", "BOUBYAN", "MABANEE", "AGILITY"]
EVAL_START = pd.Timestamp("2024-01-01")
EVAL_END = pd.Timestamp("2025-09-30")
HORIZON_DAYS = 20          # forward window
SAMPLE_EVERY_N = 5         # sample every 5th trading day
MIN_HISTORY_BARS = 200     # min bars before T to compute indicators

CONFIDENCE_BANDS = [
    (0,  49,  "00-49"),
    (50, 59,  "50-59"),
    (60, 69,  "60-69"),
    (70, 79,  "70-79"),
    (80, 89,  "80-89"),
    (90, 100, "90-100"),
]

ML_MODELS_ROOT = Path("ml_models")


# ---------------------------------------------------------------------------
# ML model loader (graceful — returns None if no model found)
# ---------------------------------------------------------------------------
def _try_load_ml_model(ticker: str):
    """Return ML model bundle or None if not trained yet."""
    try:
        from app.services.eagle_eye.ml.tier_resolver import resolve_model_for_ticker
        bundle = resolve_model_for_ticker(ticker, models_root=str(ML_MODELS_ROOT))
        return bundle
    except Exception:
        return None


def _ml_predict(bundle, indicators_df: pd.DataFrame) -> Optional[float]:
    """
    Run ML prediction using the feature list from bundle.
    Returns calibrated probability 0-1, or None on failure.
    """
    if bundle is None:
        return None
    try:
        feature_list = bundle.feature_list
        last_row = indicators_df.iloc[[-1]][feature_list]
        last_row = last_row.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        raw_prob = float(bundle.model.predict(last_row.values)[0])
        if bundle.calibrator is not None:
            cal_prob = float(bundle.calibrator.predict([[raw_prob]])[0])
        else:
            cal_prob = raw_prob
        return float(np.clip(cal_prob, 0.0, 1.0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core: rate one stock at one date T
# ---------------------------------------------------------------------------
def _rate_at_T(
    ticker: str,
    ohlcv_full: pd.DataFrame,
    T: pd.Timestamp,
    ml_bundle,
) -> Optional[Dict[str, Any]]:
    """
    Compute rule-based rating (and ML if available) at date T.
    Returns dict with all fields or None if insufficient history.
    """
    hist = ohlcv_full.loc[ohlcv_full.index <= T].copy()
    if len(hist) < MIN_HISTORY_BARS:
        return None

    try:
        ind_df = compute_all_indicators(hist)
    except Exception:
        return None

    if ind_df.empty or len(ind_df) == 0:
        return None

    indicators_row = ind_df.iloc[-1].to_dict()
    current_close = float(hist["close"].iloc[-1])

    # Phase 1 rule-based
    stage = classify_stage(indicators_row)
    confidence = compute_confidence(indicators_row, stage, dna=None, regime="NEUTRAL")
    rating = compute_rating(confidence)
    sr = compute_support_resistance(hist, indicators_row)
    targets = compute_entry_stop_targets(hist, indicators_row, sr, stage=stage)

    # TP1: compute_entry_stop_targets now enforces a stage-aware ATR floor,
    # so tp1 should always be a valid price above current_close.
    # The fallback here handles the unlikely case of a None or stale value.
    tp1 = targets.get("tp1")
    atr_val = float(indicators_row.get("atr", current_close * 0.02) or current_close * 0.02)
    if tp1 is None or tp1 <= current_close:
        # Use the same default multiple as "UNKNOWN" stage (1.5× ATR)
        tp1 = current_close + 1.5 * atr_val  # fallback: +1.5 ATR

    # ML confidence (if model available)
    ml_confidence: Optional[float] = None
    if ml_bundle is not None:
        ml_prob = _ml_predict(ml_bundle, ind_df)
        if ml_prob is not None:
            ml_confidence = round(ml_prob * 100, 2)

    return {
        "date": T.strftime("%Y-%m-%d"),
        "close": round(current_close, 4),
        "stage": stage,
        "confidence": round(confidence, 2),
        "rating": rating,
        "tp1": round(tp1, 4),
        "atr": round(atr_val, 4),
        "ml_confidence": ml_confidence,
    }


# ---------------------------------------------------------------------------
# Core: measure outcome for one prediction day
# ---------------------------------------------------------------------------
def _measure_outcome(
    ohlcv_full: pd.DataFrame,
    T: pd.Timestamp,
    tp1: float,
    close_at_T: float,
) -> Dict[str, Any]:
    """
    Look forward HORIZON_DAYS trading days and measure:
      - tp1_hit: did high reach tp1?
      - max_gain_pct: best gain achievable (high-based)
      - max_drawdown_pct: worst intraday dip (low-based)
      - days_to_tp1: first day high >= tp1 (or None)
    """
    future = ohlcv_full.loc[ohlcv_full.index > T].head(HORIZON_DAYS)
    if len(future) == 0:
        return {
            "tp1_hit": False,
            "max_gain_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "days_to_tp1": None,
            "n_future_bars": 0,
        }

    highs = future["high"].values
    lows = future["low"].values

    max_high = float(np.max(highs))
    min_low = float(np.min(lows))

    tp1_hit = bool(max_high >= tp1)
    max_gain_pct = round((max_high / close_at_T - 1.0) * 100, 4) if close_at_T > 0 else 0.0
    max_drawdown_pct = round((min_low / close_at_T - 1.0) * 100, 4) if close_at_T > 0 else 0.0

    days_to_tp1 = None
    if tp1_hit:
        for i, h in enumerate(highs):
            if h >= tp1:
                days_to_tp1 = i + 1
                break

    return {
        "tp1_hit": tp1_hit,
        "max_gain_pct": max_gain_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "days_to_tp1": days_to_tp1,
        "n_future_bars": len(future),
    }


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------
def _confidence_bucket_label(conf: float) -> str:
    for lo, hi, label in CONFIDENCE_BANDS:
        if lo <= conf <= hi:
            return label
    return "90-100"


def _aggregate_buckets(rows: List[Dict[str, Any]], conf_key: str = "confidence") -> Dict[str, Dict]:
    """Group prediction rows into confidence buckets and compute stats."""
    buckets: Dict[str, List] = {label: [] for _, _, label in CONFIDENCE_BANDS}

    for row in rows:
        conf = row.get(conf_key)
        if conf is None:
            continue
        label = _confidence_bucket_label(conf)
        buckets[label].append(row)

    result = {}
    for _, _, label in CONFIDENCE_BANDS:
        grp = buckets[label]
        n = len(grp)
        if n == 0:
            result[label] = {
                "n": 0,
                "tp1_hit_rate": None,
                "mean_return_pct": None,
                "mean_drawdown_pct": None,
                "calibration_delta": None,
            }
            continue

        # Midpoint of bucket
        lo_str, hi_str = label.split("-")
        midpoint = (float(lo_str) + float(hi_str)) / 2.0 / 100.0

        hit_rate = float(np.mean([r["tp1_hit"] for r in grp]))
        mean_ret = float(np.mean([r["max_gain_pct"] for r in grp]))
        mean_dd = float(np.mean([r["max_drawdown_pct"] for r in grp]))
        cal_delta = abs(midpoint - hit_rate)

        result[label] = {
            "n": n,
            "tp1_hit_rate": round(hit_rate, 4),
            "mean_return_pct": round(mean_ret, 4),
            "mean_drawdown_pct": round(mean_dd, 4),
            "calibration_delta": round(cal_delta, 4),
        }

    return result


# ---------------------------------------------------------------------------
# Main per-ticker evaluation
# ---------------------------------------------------------------------------
def evaluate_ticker(ticker: str) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  {ticker}")
    print(f"{'='*60}")

    ohlcv = load_ohlcv(ticker)
    if ohlcv.empty:
        print(f"  [SKIP] No OHLCV data for {ticker}")
        return {"ticker": ticker, "error": "no_ohlcv_data", "rows": []}

    ohlcv = ohlcv.sort_index()
    print(f"  Loaded {len(ohlcv)} bars: {ohlcv.index.min().date()} → {ohlcv.index.max().date()}")

    # All trading days in eval window
    eval_days = ohlcv.index[
        (ohlcv.index >= EVAL_START) & (ohlcv.index <= EVAL_END)
    ]
    sampled_days = eval_days[::SAMPLE_EVERY_N]
    print(f"  Eval window: {len(eval_days)} trading days → sampling every {SAMPLE_EVERY_N}th → {len(sampled_days)} days")

    # Try to load ML model
    ml_bundle = _try_load_ml_model(ticker)
    ml_status = "available" if ml_bundle is not None else "not_trained"
    print(f"  ML model: {ml_status}")

    prediction_rows: List[Dict[str, Any]] = []
    skipped = 0

    for i, T in enumerate(sampled_days):
        rating_result = _rate_at_T(ticker, ohlcv, T, ml_bundle)
        if rating_result is None:
            skipped += 1
            continue

        outcome = _measure_outcome(ohlcv, T, rating_result["tp1"], rating_result["close"])
        if outcome["n_future_bars"] < 5:
            # Too close to end of data — skip
            skipped += 1
            continue

        row = {**rating_result, **outcome}
        prediction_rows.append(row)

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(sampled_days)} done, {len(prediction_rows)} valid rows so far")

    print(f"  Done: {len(prediction_rows)} valid predictions, {skipped} skipped")

    # Build buckets for rule-based
    rb_buckets = _aggregate_buckets(prediction_rows, conf_key="confidence")

    # Build buckets for ML (only if model existed)
    ml_rows = [r for r in prediction_rows if r.get("ml_confidence") is not None]
    ml_buckets = _aggregate_buckets(ml_rows, conf_key="ml_confidence") if ml_rows else None

    return {
        "ticker": ticker,
        "ml_status": ml_status,
        "n_predictions": len(prediction_rows),
        "n_skipped": skipped,
        "date_range": {
            "start": sampled_days[0].strftime("%Y-%m-%d") if len(sampled_days) > 0 else None,
            "end": sampled_days[-1].strftime("%Y-%m-%d") if len(sampled_days) > 0 else None,
        },
        "rule_based_buckets": rb_buckets,
        "ml_buckets": ml_buckets,
        "prediction_rows": prediction_rows,
    }


# ---------------------------------------------------------------------------
# Table formatter
# ---------------------------------------------------------------------------
def _format_bucket_table(ticker: str, label: str, buckets: Optional[Dict]) -> str:
    if buckets is None:
        return f"  [{ticker}] {label}: No data available\n"

    lines = []
    lines.append(f"\n  {ticker} — {label}")
    lines.append(f"  {'Band':>8}  {'N':>5}  {'TP1 Hit%':>9}  {'Mean Ret%':>10}  {'Mean DD%':>9}  {'Cal Δ':>7}")
    lines.append(f"  {'-'*8}  {'-'*5}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*7}")

    for _, _, band in CONFIDENCE_BANDS:
        b = buckets.get(band, {})
        n = b.get("n", 0)
        if n == 0:
            lines.append(f"  {band:>8}  {n:>5}  {'—':>9}  {'—':>10}  {'—':>9}  {'—':>7}")
        else:
            hit = b.get("tp1_hit_rate", 0)
            ret = b.get("mean_return_pct", 0)
            dd = b.get("mean_drawdown_pct", 0)
            cal = b.get("calibration_delta", 0)
            lines.append(
                f"  {band:>8}  {n:>5}  {hit*100:>8.1f}%  {ret:>9.2f}%  {dd:>8.2f}%  {cal*100:>6.1f}%"
            )

    return "\n".join(lines) + "\n"


def _format_stage_breakdown(prediction_rows: List[Dict]) -> str:
    if not prediction_rows:
        return ""
    from collections import Counter
    stages = Counter(r.get("stage", "UNKNOWN") for r in prediction_rows)
    lines = ["  Stage distribution:"]
    for stage, cnt in sorted(stages.items(), key=lambda x: -x[1]):
        pct = cnt / len(prediction_rows) * 100
        lines.append(f"    {stage:30s}  {cnt:>4}  ({pct:.0f}%)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aggregate across all tickers
# ---------------------------------------------------------------------------
def _aggregate_all(results: List[Dict]) -> Tuple[Dict, Optional[Dict]]:
    """Merge all prediction rows across tickers and re-bucket."""
    all_rb = []
    all_ml = []
    for r in results:
        all_rb.extend(r.get("prediction_rows", []))
        ml_rows = [row for row in r.get("prediction_rows", []) if row.get("ml_confidence") is not None]
        all_ml.extend(ml_rows)

    rb_buckets = _aggregate_buckets(all_rb, conf_key="confidence")
    ml_buckets = _aggregate_buckets(all_ml, conf_key="ml_confidence") if all_ml else None

    return rb_buckets, ml_buckets


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("Rating Reality Check — Phase 2B")
    print(f"Tickers: {TICKERS}")
    print(f"Window: {EVAL_START.date()} → {EVAL_END.date()}")
    print(f"Horizon: {HORIZON_DAYS} trading days forward")
    print(f"Sampling: every {SAMPLE_EVERY_N}th trading day")
    print()

    all_results: List[Dict] = []

    for ticker in TICKERS:
        try:
            result = evaluate_ticker(ticker)
        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")
            traceback.print_exc()
            result = {"ticker": ticker, "error": str(e), "prediction_rows": [], "rule_based_buckets": None, "ml_buckets": None}
        all_results.append(result)

    # Aggregate
    all_rb_buckets, all_ml_buckets = _aggregate_all(all_results)

    # ---------------------------------------------------------------------------
    # Build .txt output
    # ---------------------------------------------------------------------------
    txt_lines = []
    txt_lines.append("=" * 70)
    txt_lines.append("EAGLE EYE RATING REALITY CHECK")
    txt_lines.append(f"Window: {EVAL_START.date()} → {EVAL_END.date()}")
    txt_lines.append(f"Horizon: {HORIZON_DAYS} trading days | Sample every {SAMPLE_EVERY_N}th day")
    txt_lines.append("=" * 70)
    txt_lines.append("")
    txt_lines.append("COLUMN GUIDE:")
    txt_lines.append("  Band     = Confidence score bucket")
    txt_lines.append("  N        = Number of prediction days in bucket")
    txt_lines.append("  TP1 Hit% = % of days where price hit TP1 within 20 days")
    txt_lines.append("  Mean Ret = Average max gain achievable in 20-day window")
    txt_lines.append("  Mean DD  = Average worst drawdown in 20-day window")
    txt_lines.append("  Cal Δ    = |bucket_midpoint - actual_hit_rate| (calibration error)")
    txt_lines.append("             Lower is better. 0% = perfectly calibrated.")
    txt_lines.append("")

    for result in all_results:
        ticker = result["ticker"]
        if result.get("error") and not result.get("prediction_rows"):
            txt_lines.append(f"\n{ticker}: SKIPPED — {result.get('error')}")
            continue

        txt_lines.append(f"\n{'─'*70}")
        txt_lines.append(f"TICKER: {ticker}  |  ML: {result.get('ml_status','N/A')}  |  N Predictions: {result.get('n_predictions',0)}")
        txt_lines.append(f"{'─'*70}")

        txt_lines.append(_format_stage_breakdown(result.get("prediction_rows", [])))
        txt_lines.append(_format_bucket_table(ticker, "RULE-BASED", result.get("rule_based_buckets")))

        if result.get("ml_buckets"):
            txt_lines.append(_format_bucket_table(ticker, "ML", result.get("ml_buckets")))
        else:
            txt_lines.append(f"  [{ticker}] ML: Model not trained yet — N/A")

    txt_lines.append(f"\n{'='*70}")
    txt_lines.append("OVERALL AGGREGATE (all 5 tickers combined)")
    txt_lines.append(f"{'='*70}")

    total_preds = sum(r.get("n_predictions", 0) for r in all_results)
    txt_lines.append(f"Total predictions: {total_preds}")
    txt_lines.append(_format_bucket_table("ALL", "RULE-BASED AGGREGATE", all_rb_buckets))

    if all_ml_buckets:
        txt_lines.append(_format_bucket_table("ALL", "ML AGGREGATE", all_ml_buckets))
    else:
        txt_lines.append("  [ALL] ML AGGREGATE: No ML models trained yet")

    # Calibration verdict
    txt_lines.append(f"\n{'─'*70}")
    txt_lines.append("CALIBRATION VERDICT:")
    for _, _, band in CONFIDENCE_BANDS:
        b = all_rb_buckets.get(band, {})
        n = b.get("n", 0)
        if n == 0:
            continue
        cal = b.get("calibration_delta", 0)
        hit = b.get("tp1_hit_rate", 0)
        mid_pct = (float(band.split("-")[0]) + float(band.split("-")[1])) / 2
        verdict = "WELL-CAL" if cal * 100 < 15 else ("OVER-CONF" if hit * 100 < mid_pct else "UNDER-CONF")
        txt_lines.append(f"  Band {band}: hit={hit*100:.1f}% target≈{mid_pct:.0f}%  cal_err={cal*100:.1f}%  → {verdict}")

    txt_lines.append("")
    txt_lines.append("INTERPRETING THIS OUTPUT:")
    txt_lines.append("  If TP1 hit rate RISES with confidence band → signal is discriminative")
    txt_lines.append("  If Cal Δ < 15% across all bands → ratings are well-calibrated")
    txt_lines.append("  If high-conf (80-100) bands show hit_rate > 60% → system is actionable")
    txt_lines.append("  If all bands show ~same hit rate → confidence score has no predictive value")

    txt_content = "\n".join(txt_lines)

    # ---------------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------------
    out_dir = Path("tools")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strip prediction_rows from JSON to keep it manageable (save separately)
    json_output = {
        "meta": {
            "tickers": TICKERS,
            "eval_start": str(EVAL_START.date()),
            "eval_end": str(EVAL_END.date()),
            "horizon_days": HORIZON_DAYS,
            "sample_every_n": SAMPLE_EVERY_N,
            "total_predictions": total_preds,
        },
        "per_ticker": [
            {
                "ticker": r["ticker"],
                "ml_status": r.get("ml_status", "N/A"),
                "n_predictions": r.get("n_predictions", 0),
                "n_skipped": r.get("n_skipped", 0),
                "date_range": r.get("date_range"),
                "rule_based_buckets": r.get("rule_based_buckets"),
                "ml_buckets": r.get("ml_buckets"),
            }
            for r in all_results
        ],
        "aggregate": {
            "rule_based_buckets": all_rb_buckets,
            "ml_buckets": all_ml_buckets,
        },
    }

    json_path = out_dir / "rating_reality_check.json"
    txt_path = out_dir / "rating_reality_check.txt"

    json_path.write_text(json.dumps(json_output, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(txt_content, encoding="utf-8")

    print("\n" + txt_content)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()

"""
ml/shadow_runner.py — Phase 3: Daily shadow scoring for SHADOW-status models.

Runs after market close (≈14:30 Asia/Kuwait, Sun–Thu).  For each stock with a
SHADOW model the runner:

1. Loads today's OHLCV and builds a feature row identical to training.
2. Runs the calibrated model to get raw + calibrated probabilities.
3. Reads the current rule-engine rating from ee_ratings_cache.
4. Computes the band label (LOW / MEDIUM / HIGH / INSUFFICIENT_DATA / NO_VARIANCE)
   via band_display.compute_band().
5. Writes one idempotent row to ml_shadow_log (INSERT OR IGNORE on UNIQUE
   constraint model_id+log_date).
6. Writes one row to phase3_evaluation_log comparing ML and rule directions.

The run is fully idempotent — calling it twice on the same date produces a single
row thanks to the UNIQUE INDEX on (model_id, log_date).
"""
from __future__ import annotations

import hashlib
import json
import logging
import traceback
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

# ── Shadow roster (14 stocks in SHADOW phase) ────────────────────────────────
SHADOW_ROSTER: List[str] = [
    "AAYANRE", "ALTIJARIA", "ARGAN", "BOURSA", "FACIL", "IFA",
    "JAZEERA", "JTC", "KCEM", "KPPC", "MKHZN", "OOREDOO", "URC", "WARBACAP",
]
PRIMARY_LABEL = "y_10pct_20d"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_shadow_scoring(signal_date: Optional[str] = None) -> Dict:
    """
    Score all SHADOW-roster stocks for *signal_date* (ISO string, default = today).

    Returns a summary dict:
      {
        "signal_date": "YYYY-MM-DD",
        "scored": int,
        "skipped": int,
        "errors": int,
        "details": [{ticker, band, calibrated_prob, error?}, ...]
      }
    """
    from app.core.database import exec_sql, query_all, query_one
    from app.services.eagle_eye.ml.model_store import load_model_bundle
    from app.services.eagle_eye.ml.feature_builder import build_inference_row
    from app.services.eagle_eye.store import load_ohlcv
    from app.services.eagle_eye.ml import band_display

    today_str = signal_date or date.today().isoformat()
    today = date.fromisoformat(today_str)

    summary = {
        "signal_date": today_str,
        "scored": 0,
        "skipped": 0,
        "errors": 0,
        "details": [],
    }

    for ticker in SHADOW_ROSTER:
        detail: Dict = {"ticker": ticker}
        try:
            result = _score_one(
                ticker=ticker,
                today=today,
                today_str=today_str,
                load_model_bundle=load_model_bundle,
                build_inference_row=build_inference_row,
                load_ohlcv=load_ohlcv,
                band_display=band_display,
                exec_sql=exec_sql,
                query_one=query_one,
            )
            detail.update(result)
            if result.get("skipped"):
                summary["skipped"] += 1
            else:
                summary["scored"] += 1
        except Exception as exc:
            LOGGER.warning("shadow_runner: error scoring %s: %s", ticker, exc)
            LOGGER.debug(traceback.format_exc())
            detail["error"] = str(exc)
            summary["errors"] += 1

        summary["details"].append(detail)

    LOGGER.info(
        "shadow_runner: date=%s scored=%d skipped=%d errors=%d",
        today_str,
        summary["scored"],
        summary["skipped"],
        summary["errors"],
    )
    return summary


# ---------------------------------------------------------------------------
# Per-stock scoring
# ---------------------------------------------------------------------------

def _score_one(
    ticker: str,
    today: date,
    today_str: str,
    *,
    load_model_bundle,
    build_inference_row,
    load_ohlcv,
    band_display,
    exec_sql,
    query_one,
) -> Dict:
    """Score one ticker.  Returns a dict describing the outcome."""

    # ── 1. Load model bundle ──────────────────────────────────────────────
    model_id_str = f"{ticker}_{PRIMARY_LABEL}"
    bundle = load_model_bundle(tier="per_stock", identifier=model_id_str)
    if bundle is None:
        LOGGER.debug("shadow_runner: no bundle for %s — skip", ticker)
        return {"skipped": True, "reason": "no_bundle"}

    # Resolve model_id (the UUID stored in models table, not the filesystem id)
    model_row = query_one(
        "SELECT model_id FROM ml_models WHERE stock_ticker = ? AND status = 'SHADOW' "
        "ORDER BY trained_at DESC LIMIT 1",
        (ticker,),
    )
    if model_row is None:
        LOGGER.debug("shadow_runner: %s has no SHADOW row in ml_models — skip", ticker)
        return {"skipped": True, "reason": "no_shadow_model_row"}

    model_id = model_row["model_id"]

    # ── 2. Load OHLCV and build feature row ───────────────────────────────
    try:
        ohlcv = load_ohlcv(ticker)
    except Exception as exc:
        return {"skipped": True, "reason": f"ohlcv_load_failed: {exc}"}

    feature_row = build_inference_row(ticker=ticker, ohlcv=ohlcv, T=today)
    if feature_row is None:
        return {"skipped": True, "reason": "insufficient_ohlcv"}

    # ── 3. Align to feature_list ──────────────────────────────────────────
    feature_list = bundle.feature_list
    X = np.array(
        [float(feature_row.get(f, float("nan"))) for f in feature_list],
        dtype=np.float32,
    ).reshape(1, -1)
    # Replace NaN with 0.0 (same as training imputation fallback)
    X = np.nan_to_num(X, nan=0.0)

    # ── 4. Inference ──────────────────────────────────────────────────────
    import pandas as pd
    X_df = pd.DataFrame(X, columns=feature_list)
    raw_prob = float(bundle.model.predict(X_df)[0])

    calibrated_prob: float
    if bundle.calibrator is not None:
        try:
            calibrated_prob = float(bundle.calibrator.predict_proba(X_df)[:, 1][0])
        except Exception:
            calibrated_prob = raw_prob
    else:
        calibrated_prob = raw_prob

    # ── 5. Features hash for audit ────────────────────────────────────────
    features_hash = hashlib.md5(
        json.dumps({k: round(float(v), 6) for k, v in feature_row.items() if k in feature_list}, sort_keys=True).encode()
    ).hexdigest()[:16]

    # ── 6. Rule engine data ───────────────────────────────────────────────
    rule_row = query_one(
        "SELECT stage, rating, confidence FROM ee_ratings_cache WHERE ticker = ? "
        "ORDER BY computed_at DESC LIMIT 1",
        (ticker,),
    )
    rule_stage = rule_row["stage"] if rule_row else None
    rule_confidence = float(rule_row["confidence"]) if rule_row and rule_row["confidence"] is not None else None
    ml_score = calibrated_prob
    ml_bucket = _prob_to_bucket(calibrated_prob)

    # ── 7. Band computation ───────────────────────────────────────────────
    band_label, band_low, band_high = band_display.compute_band(
        ticker=ticker,
        calibrated_prob=calibrated_prob,
        model_id=model_id,
        signal_date=today_str,
    )

    # ── 8. Write ml_shadow_log (idempotent) ───────────────────────────────
    exec_sql(
        """
        INSERT OR IGNORE INTO ml_shadow_log
            (model_id, stock_ticker, log_date,
             ml_score, ml_bucket, rule_score, rule_bucket,
             raw_prob, calibrated_prob, band_label,
             rule_stage, rule_confidence, features_hash,
             outcome_filled, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, datetime('now'))
        """,
        (
            model_id, ticker, today_str,
            ml_score, ml_bucket, rule_confidence, rule_stage,
            raw_prob, calibrated_prob, band_label,
            rule_stage, rule_confidence, features_hash,
        ),
    )

    # ── 9. Write phase3_evaluation_log (idempotent) ───────────────────────
    rule_rating = rule_row["rating"] if rule_row else None
    agreement = _compute_agreement(ml_bucket, rule_rating)
    exec_sql(
        """
        INSERT OR IGNORE INTO phase3_evaluation_log
            (log_date, stock_ticker, model_id, band_label,
             rule_rating, rule_confidence, agreement, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (today_str, ticker, model_id, band_label, rule_rating, rule_confidence, agreement),
    )

    LOGGER.info(
        "shadow_runner: %s → band=%s (cal=%.3f raw=%.3f) rule=%s",
        ticker, band_label, calibrated_prob, raw_prob, rule_stage,
    )
    return {
        "band": band_label,
        "calibrated_prob": calibrated_prob,
        "raw_prob": raw_prob,
        "rule_stage": rule_stage,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prob_to_bucket(prob: float) -> str:
    if prob >= 0.65:
        return "STRONG"
    if prob >= 0.50:
        return "MODERATE"
    if prob >= 0.35:
        return "WEAK"
    return "LOW"


def _compute_agreement(ml_bucket: str, rule_rating: Optional[str]) -> Optional[int]:
    if not rule_rating:
        return None
    ml_positive = ml_bucket in ("STRONG", "MODERATE")
    rule_positive = rule_rating.upper() in ("BULL", "STAGE_2", "ACCUMULATION", "WATCH", "STRONG_BULL")
    return 1 if ml_positive == rule_positive else 0

"""
Eagle Eye — Ride Quality Evaluator  (Phase R3)
==============================================

Inference layer for the Ride Quality Model.  Called daily for each stock
with an active position to answer: HOLD / ADD / EXIT?

Primary entry point:
    evaluate_ride(ticker, entry_price, entry_date) → RideQualityResult

The evaluator:
  1. Loads the per-stock ride model (falls back to pooled)
  2. Computes live market indicators for *ticker*
  3. Builds ride context features (days held, drawdown from peak, etc.)
  4. Runs the classifier: P(HOLD), P(ADD), P(EXIT)
  5. Runs the regression: remaining_upside_pct estimate
  6. Returns a structured result the recommendation engine can consume

Position State Tracking
-----------------------
The evaluator needs to know the "running peak" for a position — the highest
price reached since entry.  This is persisted in the ``ee_ride_state`` table
(see db_tables.py) and updated on every evaluation call.

Thread Safety
-------------
The model is loaded once per process and cached in module-level _MODEL_CACHE.
Safe for single-threaded Uvicorn with async offload (asyncio.to_thread).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.services.eagle_eye.indicators import compute_all_indicators
from app.services.eagle_eye.ml.feature_builder import build_curated_feature_row
from app.services.eagle_eye.ml.model_store import ModelBundle, load_model_bundle
from app.services.eagle_eye.ml.ride_feature_builder import (
    LABEL_DECODING,
    RIDE_FEATURE_NAMES,
    build_ride_context_features,
    build_ride_feature_row,
    _load_dna_context,
)
from app.services.eagle_eye.ml.ride_labeler import LabeledRideDay
from app.services.eagle_eye.ml.ride_trainer import (
    RIDE_MODEL_TIER,
    RIDE_REGRESSION_TIER,
)
from app.services.eagle_eye.store import load_ohlcv

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model cache (avoids reloading on every request)
# ---------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, Optional[ModelBundle]] = {}


def _get_model(ticker: str, tier: str) -> Optional[ModelBundle]:
    """Load ride model for *ticker*, falling back to pooled."""
    cache_key = f"{tier}/{ticker}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Try per-stock model first
    bundle = load_model_bundle(tier=tier, identifier=ticker.upper())
    if bundle is None:
        # Fall back to pooled
        bundle = load_model_bundle(tier=tier, identifier="__pooled__")

    _MODEL_CACHE[cache_key] = bundle
    return bundle


def clear_model_cache() -> None:
    """Evict all cached ride models (call after retraining)."""
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RideQualityResult:
    """
    Output of evaluate_ride() — one result per active position per day.

    All probability values are in [0, 1].
    remaining_upside_est is in % (e.g., 6.5 means +6.5% estimated gain left).
    """
    ticker: str
    evaluation_date: date

    # Ride state (inputs)
    days_held: int
    entry_price: float
    current_price: float
    unrealized_pct: float
    peak_gain_pct: float
    drawdown_from_peak: float

    # Model output
    ride_action: str            # "HOLD" | "ADD" | "EXIT"
    ride_confidence: float      # max(p_hold, p_add, p_exit) * 100
    p_hold: float
    p_add: float
    p_exit: float
    remaining_upside_est: float # regression estimate (%)

    # Source
    model_source: str           # "per_stock" | "pooled" | "rules_fallback"
    model_available: bool

    # Human-readable summary
    summary: str = field(default="")

    def __post_init__(self) -> None:
        if not self.summary:
            self.summary = self._build_summary()

    def _build_summary(self) -> str:
        icon = {"HOLD": "🟢", "ADD": "🟡", "EXIT": "🔴"}.get(self.ride_action, "⚪")
        dd_str = f"  ↩ Pullback: -{self.drawdown_from_peak:.1f}% from peak" if self.drawdown_from_peak > 0.5 else ""
        upside_str = f"  Est. remaining upside: +{self.remaining_upside_est:.1f}%" if self.remaining_upside_est > 0 else ""
        return (
            f"▲ Riding {self.days_held}d (+{self.unrealized_pct:.1f}%)"
            f"{dd_str}"
            f"\n{icon} ML: {self.ride_action} ({self.ride_confidence:.0f}%) — {self._action_hint()}"
            f"{upside_str}"
        )

    def _action_hint(self) -> str:
        hints = {
            "HOLD": "trend intact",
            "ADD": "healthy dip — consider adding",
            "EXIT": "tighten stop / take profits",
        }
        return hints.get(self.ride_action, "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "evaluation_date": self.evaluation_date.isoformat(),
            "days_held": self.days_held,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pct": round(self.unrealized_pct, 2),
            "peak_gain_pct": round(self.peak_gain_pct, 2),
            "drawdown_from_peak": round(self.drawdown_from_peak, 2),
            "ride_action": self.ride_action,
            "ride_confidence": round(self.ride_confidence, 1),
            "p_hold": round(self.p_hold, 4),
            "p_add": round(self.p_add, 4),
            "p_exit": round(self.p_exit, 4),
            "remaining_upside_est": round(self.remaining_upside_est, 2),
            "model_source": self.model_source,
            "model_available": self.model_available,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Position state helpers (running peak)
# ---------------------------------------------------------------------------

def _get_running_peak(
    ticker: str,
    entry_date: date,
    current_price: float,
) -> float:
    """
    Return the highest price reached since entry_date for *ticker*.
    Uses DB-persisted ride state; falls back to max of OHLCV since entry.
    """
    # Try DB-persisted peak first (fastest)
    try:
        from app.core.database import query_one
        row = query_one(
            "SELECT running_peak_price FROM ee_ride_state WHERE ticker = ? AND entry_date = ?",
            (ticker.upper(), entry_date.isoformat()),
        )
        if row and row.get("running_peak_price"):
            return max(float(row["running_peak_price"]), current_price)
    except Exception:
        pass

    # Fallback: compute from OHLCV
    try:
        ohlcv = load_ohlcv(ticker, start=entry_date)
        if not ohlcv.empty:
            return max(float(ohlcv["high"].max()), current_price)
    except Exception:
        pass

    return current_price


def _update_running_peak(
    ticker: str,
    entry_date: date,
    entry_price: float,
    running_peak: float,
    current_price: float,
    ride_action: str,
) -> None:
    """Upsert the ride state row in DB."""
    try:
        from app.core.database import exec_sql
        exec_sql(
            """
            INSERT INTO ee_ride_state
                (ticker, entry_date, entry_price, running_peak_price, last_evaluated, last_action)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, entry_date) DO UPDATE SET
                running_peak_price = excluded.running_peak_price,
                last_evaluated     = excluded.last_evaluated,
                last_action        = excluded.last_action
            """,
            (
                ticker.upper(),
                entry_date.isoformat(),
                entry_price,
                running_peak,
                date.today().isoformat(),
                ride_action,
            ),
        )
    except Exception as exc:
        LOGGER.debug("Could not update ee_ride_state for %s: %s", ticker, exc)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _rules_fallback(
    ticker: str,
    days_held: int,
    unrealized_pct: float,
    drawdown_from_peak: float,
    peak_gain_pct: float,
) -> RideQualityResult:
    """
    Rules-based fallback when no ML model is available.
    Replicates the design document's threshold logic.
    """
    evaluation_date = date.today()

    # Simple heuristic: drawdown > 8% → EXIT, pullback 2-5% in uptrend → ADD, else HOLD
    if drawdown_from_peak > 8.0:
        action = "EXIT"
        p_exit, p_hold, p_add = 0.75, 0.15, 0.10
    elif drawdown_from_peak > 2.0 and peak_gain_pct > 5.0 and unrealized_pct > 0:
        action = "ADD"
        p_exit, p_hold, p_add = 0.15, 0.30, 0.55
    else:
        action = "HOLD"
        p_exit, p_hold, p_add = 0.20, 0.60, 0.20

    return RideQualityResult(
        ticker=ticker,
        evaluation_date=evaluation_date,
        days_held=days_held,
        entry_price=0.0,
        current_price=0.0,
        unrealized_pct=unrealized_pct,
        peak_gain_pct=peak_gain_pct,
        drawdown_from_peak=drawdown_from_peak,
        ride_action=action,
        ride_confidence=max(p_hold, p_add, p_exit) * 100.0,
        p_hold=p_hold,
        p_add=p_add,
        p_exit=p_exit,
        remaining_upside_est=0.0,
        model_source="rules_fallback",
        model_available=False,
    )


def evaluate_ride(
    ticker: str,
    entry_price: float,
    entry_date: date,
    *,
    current_price: Optional[float] = None,
    today: Optional[date] = None,
) -> RideQualityResult:
    """
    Evaluate the current ride quality for *ticker*.

    Parameters
    ----------
    ticker        : stock ticker (e.g. "ZAIN")
    entry_price   : price at which the position was entered
    entry_date    : date the position was opened
    current_price : override live price (uses latest OHLCV close if None)
    today         : override evaluation date (defaults to date.today())

    Returns
    -------
    RideQualityResult with HOLD/ADD/EXIT prediction, probabilities, and
    estimated remaining upside.
    """
    evaluation_date = today or date.today()
    ticker = ticker.upper()

    # ── Load OHLCV ────────────────────────────────────────────────────────
    ohlcv = load_ohlcv(ticker)
    if ohlcv.empty or len(ohlcv) < 20:
        days_held = max(0, (evaluation_date - entry_date).days)
        return _rules_fallback(ticker, days_held, 0.0, 0.0, 0.0)

    # Current price from latest bar if not provided
    if current_price is None or not math.isfinite(current_price) or current_price <= 0:
        current_price = float(ohlcv["close"].iloc[-1])

    # ── Ride state ────────────────────────────────────────────────────────
    days_held = max(0, (evaluation_date - entry_date).days)
    unrealized_pct = (current_price / entry_price - 1.0) * 100.0 if entry_price > 0 else 0.0

    running_peak = _get_running_peak(ticker, entry_date, current_price)
    peak_gain_pct = (running_peak / entry_price - 1.0) * 100.0 if entry_price > 0 else unrealized_pct
    drawdown_from_peak = (running_peak - current_price) / running_peak * 100.0 if running_peak > 0 else 0.0

    # ── Compute indicators (full history, current bar is last) ────────────
    try:
        indicators = compute_all_indicators(ohlcv)
    except Exception as exc:
        LOGGER.warning("evaluate_ride %s: indicator compute failed: %s", ticker, exc)
        return _rules_fallback(ticker, days_held, unrealized_pct, drawdown_from_peak, peak_gain_pct)

    if indicators.empty:
        return _rules_fallback(ticker, days_held, unrealized_pct, drawdown_from_peak, peak_gain_pct)

    # Current indicator snapshot (latest bar)
    ind_row = indicators.iloc[-1].to_dict()
    bar_idx = len(indicators) - 1

    # Entry bar index
    entry_candidates = np.where(ohlcv.index.date == entry_date)[0]
    entry_idx = int(entry_candidates[0]) if len(entry_candidates) > 0 else max(0, bar_idx - days_held)

    # ── Build a LabeledRideDay-like struct for context feature computation ─
    # (We reuse the same compute path, just with live values, no future labels)
    live_day = LabeledRideDay(
        ticker=ticker,
        event_id="live",
        bar_date=evaluation_date,
        bar_idx=bar_idx,
        entry_price=entry_price,
        days_held=days_held,
        unrealized_pct=unrealized_pct,
        peak_gain_pct=peak_gain_pct,
        drawdown_from_peak=drawdown_from_peak,
        gain_velocity=unrealized_pct / max(days_held, 1),
        label="HOLD",          # placeholder — not used at inference
        remaining_upside_pct=0.0,
        remaining_max_dd_pct=0.0,
        makes_new_high_20d=False,
    )

    dna_ctx = _load_dna_context(ticker)
    rc = build_ride_context_features(
        live_day,
        indicators_history=indicators,
        ohlcv=ohlcv,
        **dna_ctx,
    )
    feat_row = build_ride_feature_row(ind_row, rc)

    # ── Load model ────────────────────────────────────────────────────────
    clf_bundle = _get_model(ticker, RIDE_MODEL_TIER)
    reg_bundle = _get_model(ticker, RIDE_REGRESSION_TIER)

    if clf_bundle is None or clf_bundle.model is None:
        action = _rules_fallback(ticker, days_held, unrealized_pct, drawdown_from_peak, peak_gain_pct)
        _update_running_peak(ticker, entry_date, entry_price, running_peak, current_price, action.ride_action)
        return action

    model_source = (
        "per_stock"
        if clf_bundle.identifier != "__pooled__"
        else "pooled"
    )

    # ── Inference ─────────────────────────────────────────────────────────
    feature_names = clf_bundle.feature_list or list(RIDE_FEATURE_NAMES)
    X = pd.DataFrame([feat_row])[feature_names]

    try:
        raw_pred = clf_bundle.model.predict(X)
        arr = np.asarray(raw_pred, dtype=float)
        if arr.ndim == 1 and arr.size == 3:
            probs = arr
        elif arr.ndim == 2 and arr.shape[1] == 3:
            probs = arr[0]
        else:
            probs = np.array([0.50, 0.10, 0.40])  # conservative fallback
    except Exception as exc:
        LOGGER.warning("evaluate_ride %s: classifier predict failed: %s", ticker, exc)
        return _rules_fallback(ticker, days_held, unrealized_pct, drawdown_from_peak, peak_gain_pct)

    # Normalize probabilities
    probs = np.clip(probs, 0.0, 1.0)
    total = float(probs.sum())
    if total > 0:
        probs = probs / total

    p_hold, p_add, p_exit = float(probs[0]), float(probs[1]), float(probs[2])
    action_idx = int(np.argmax(probs))
    ride_action = LABEL_DECODING.get(action_idx, "HOLD")
    ride_confidence = float(probs[action_idx]) * 100.0

    # ── Remaining upside regression ───────────────────────────────────────
    remaining_upside_est = 0.0
    if reg_bundle is not None and reg_bundle.model is not None:
        try:
            reg_feat_names = reg_bundle.feature_list or list(RIDE_FEATURE_NAMES)
            X_reg = pd.DataFrame([feat_row])[reg_feat_names]
            reg_pred = reg_bundle.model.predict(X_reg)
            remaining_upside_est = float(np.clip(np.asarray(reg_pred).flatten()[0], 0.0, 100.0))
        except Exception:
            pass

    # ── Persist ride state ────────────────────────────────────────────────
    _update_running_peak(ticker, entry_date, entry_price, running_peak, current_price, ride_action)

    result = RideQualityResult(
        ticker=ticker,
        evaluation_date=evaluation_date,
        days_held=days_held,
        entry_price=entry_price,
        current_price=current_price,
        unrealized_pct=round(unrealized_pct, 2),
        peak_gain_pct=round(peak_gain_pct, 2),
        drawdown_from_peak=round(drawdown_from_peak, 2),
        ride_action=ride_action,
        ride_confidence=round(ride_confidence, 1),
        p_hold=round(p_hold, 4),
        p_add=round(p_add, 4),
        p_exit=round(p_exit, 4),
        remaining_upside_est=round(remaining_upside_est, 2),
        model_source=model_source,
        model_available=True,
    )

    LOGGER.debug(
        "evaluate_ride %s  days=%d  unreal=%.1f%%  dd=%.1f%%  → %s (%.0f%%)",
        ticker, days_held, unrealized_pct, drawdown_from_peak, ride_action, ride_confidence,
    )
    return result


# ---------------------------------------------------------------------------
# Batch evaluation (all active positions)
# ---------------------------------------------------------------------------

def evaluate_all_active_rides() -> List[RideQualityResult]:
    """
    Evaluate ride quality for all tickers with an active ``ee_ride_state`` row.
    Called from the daily nightly pipeline.
    """
    try:
        from app.core.database import query_all
        rows = query_all(
            "SELECT ticker, entry_date, entry_price FROM ee_ride_state WHERE last_action != 'EXIT'",
            (),
        )
    except Exception as exc:
        LOGGER.warning("evaluate_all_active_rides: DB query failed: %s", exc)
        return []

    results: List[RideQualityResult] = []
    for row in rows:
        try:
            ticker = str(row["ticker"])
            entry_date = date.fromisoformat(str(row["entry_date"]))
            entry_price = float(row["entry_price"] or 0.0)
            if entry_price <= 0:
                continue
            result = evaluate_ride(ticker, entry_price, entry_date)
            results.append(result)
        except Exception as exc:
            LOGGER.warning("evaluate_all_active_rides: failed for %s: %s", row.get("ticker"), exc)

    return results

"""
Eagle Eye — Ride Quality Feature Builder  (Phase R1)
=====================================================

Builds the combined feature vector for each labeled ride-day:

  1. Curated v14 market indicators (48 features — reused from entry model)
  2. Ride context features (15 new features — the key addition)
  3. DNA context features (3 per-stock historical norms)

The ride context features answer WHERE IN THE RIDE the model is, which
the entry model can't do.  Without them the model can't distinguish:
  • Day 3 of a fresh breakout (pullback → HOLD)
  • Day 50 of an exhausted run (same indicators → EXIT)

Public API
----------
  build_ride_feature_row(indicators_row, ride_context, dna_context) → dict
  build_ride_training_matrix(ticker) → pd.DataFrame (all rides, labeled)
  RIDE_FEATURE_NAMES → Tuple[str, ...]  (canonical feature name order)
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from app.services.eagle_eye.indicators import compute_all_indicators
from app.services.eagle_eye.ml.feature_builder import (
    CURATED_FEATURE_ORDER,
    build_curated_feature_row,
)
from app.services.eagle_eye.ml.ride_labeler import (
    LabeledRideDay,
    build_ride_labeled_days,
    detect_historical_rides,
    label_ride_days,
)
from app.services.eagle_eye.store import load_ohlcv

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ride-context feature names (canonical order)
# ---------------------------------------------------------------------------

RIDE_CONTEXT_FEATURE_NAMES: Tuple[str, ...] = (
    # Position state
    "ride_days_held",
    "ride_unrealized_pct",
    "ride_peak_gain_pct",
    "ride_drawdown_from_peak",
    "ride_gain_velocity",
    # Trend persistence (computed from indicator history)
    "ride_days_above_ema10",
    "ride_days_above_ema20",
    "ride_ema10_slope_vs_entry",
    "ride_obv_vs_entry",
    # Volume character during pullback
    "ride_pullback_volume_ratio",
    "ride_pullback_bar_count",
    # Behavioral DNA context
    "dna_typical_pullback_pct",
    "dna_typical_run_length",
    "dna_pullback_recovery_rate",
    "dna_optimal_hold_days",
)

# Full feature set = curated v14 + ride context
RIDE_FEATURE_NAMES: Tuple[str, ...] = tuple(CURATED_FEATURE_ORDER) + RIDE_CONTEXT_FEATURE_NAMES

# Label → integer encoding for LightGBM multiclass
LABEL_ENCODING: Dict[str, int] = {"HOLD": 0, "ADD": 1, "EXIT": 2}
LABEL_DECODING: Dict[int, str] = {v: k for k, v in LABEL_ENCODING.items()}

# Minimum labeled ride-day samples required to train a per-stock model
MIN_PER_STOCK_SAMPLES: int = 200
# Minimum samples required for the pooled (cross-stock) model
MIN_POOLED_SAMPLES: int = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sf(v: Any, default: float = float("nan")) -> float:
    """Safe float conversion."""
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def _nan() -> float:
    return float("nan")


# ---------------------------------------------------------------------------
# Ride context feature computation
# ---------------------------------------------------------------------------

def build_ride_context_features(
    labeled_day: LabeledRideDay,
    indicators_history: pd.DataFrame,
    ohlcv: pd.DataFrame,
    *,
    dna_typical_pullback: float = 3.5,
    dna_typical_run_length: int = 25,
    dna_pullback_recovery_rate: float = 0.65,
    dna_optimal_hold_days: int = 40,
) -> Dict[str, float]:
    """
    Compute all 15 ride-context features for one labeled ride-day.

    Parameters
    ----------
    labeled_day       : LabeledRideDay from ride_labeler
    indicators_history: Full indicator DataFrame for this ticker (from compute_all_indicators)
    ohlcv             : Full OHLCV DataFrame for this ticker
    dna_*             : Per-stock behavioral DNA values
    """
    bar_idx = labeled_day.bar_idx
    entry_idx = bar_idx - labeled_day.days_held

    # ── Basic ride-state (already in labeled_day) ─────────────────────────
    days_held = labeled_day.days_held
    unrealized_pct = labeled_day.unrealized_pct
    peak_gain_pct = labeled_day.peak_gain_pct
    drawdown_from_peak = labeled_day.drawdown_from_peak
    gain_velocity = labeled_day.gain_velocity

    # ── Trend persistence features ────────────────────────────────────────
    ride_days_above_ema10 = _nan()
    ride_days_above_ema20 = _nan()
    ride_ema10_slope_vs_entry = _nan()
    ride_obv_vs_entry = _nan()

    closes = ohlcv["close"].to_numpy(dtype=float)
    vols = ohlcv["volume"].to_numpy(dtype=float)

    if not indicators_history.empty and bar_idx < len(indicators_history):
        try:
            # Slice of indicator rows from entry to today
            ind_slice = indicators_history.iloc[max(0, entry_idx):bar_idx + 1]
            close_slice = closes[max(0, entry_idx):bar_idx + 1]

            # Count bars above EMA10 in last 10 days (or since entry)
            lookback_10 = min(10, len(ind_slice))
            lookback_20 = min(20, len(ind_slice))

            if "ema_10" in ind_slice.columns and lookback_10 > 0:
                ema10_slice = ind_slice["ema_10"].to_numpy(dtype=float)[-lookback_10:]
                close_10 = close_slice[-lookback_10:]
                ride_days_above_ema10 = float(np.sum(close_10 > ema10_slice))

            if "ema_20" in ind_slice.columns and lookback_20 > 0:
                ema20_slice = ind_slice["ema_20"].to_numpy(dtype=float)[-lookback_20:]
                close_20 = close_slice[-lookback_20:]
                ride_days_above_ema20 = float(np.sum(close_20 > ema20_slice))

            # EMA10 slope change since entry
            if "ema_10" in ind_slice.columns and len(ind_slice) >= 2:
                ema10_today = _sf(ind_slice["ema_10"].iloc[-1])
                ema10_entry = _sf(ind_slice["ema_10"].iloc[0])
                if entry_idx > 0 and ema10_entry > 0:
                    entry_price_raw = _sf(closes[entry_idx]) if closes[entry_idx] > 0 else ema10_entry
                    ride_ema10_slope_vs_entry = (ema10_today - ema10_entry) / ema10_entry * 100.0

            # OBV change since ride started
            if "obv" in ind_slice.columns and len(ind_slice) >= 2:
                obv_today = _sf(ind_slice["obv"].iloc[-1])
                obv_entry = _sf(ind_slice["obv"].iloc[0])
                if math.isfinite(obv_entry) and math.isfinite(obv_today) and obv_entry != 0:
                    ride_obv_vs_entry = (obv_today - obv_entry) / max(abs(obv_entry), 1.0) * 100.0
        except Exception:
            pass

    # ── Volume character during pullback ──────────────────────────────────
    ride_pullback_volume_ratio = _nan()
    ride_pullback_bar_count = _nan()

    if bar_idx > entry_idx and bar_idx < len(closes):
        try:
            # Find the peak of the ride (highest high between entry and today)
            highs_slice = ohlcv["high"].to_numpy(dtype=float)[entry_idx:bar_idx + 1]
            peak_offset = int(np.argmax(highs_slice))
            upleg_end = entry_idx + peak_offset

            # Upleg volume (entry to peak)
            upleg_vols = vols[entry_idx:upleg_end + 1]
            upleg_avg = float(np.nanmean(upleg_vols)) if len(upleg_vols) > 0 else 1.0

            # Pullback volume (peak to today)
            pullback_vols = vols[upleg_end:bar_idx + 1]
            pullback_avg = float(np.nanmean(pullback_vols)) if len(pullback_vols) > 0 else 1.0

            if upleg_avg > 0:
                ride_pullback_volume_ratio = pullback_avg / upleg_avg

            # Count consecutive down bars since peak
            pullback_closes = closes[upleg_end:bar_idx + 1]
            count = 0
            for j in range(len(pullback_closes) - 1, 0, -1):
                if pullback_closes[j] < pullback_closes[j - 1]:
                    count += 1
                else:
                    break
            ride_pullback_bar_count = float(count)
        except Exception:
            pass

    return {
        # Position state
        "ride_days_held": float(days_held),
        "ride_unrealized_pct": float(unrealized_pct),
        "ride_peak_gain_pct": float(peak_gain_pct),
        "ride_drawdown_from_peak": float(drawdown_from_peak),
        "ride_gain_velocity": float(gain_velocity),
        # Trend persistence
        "ride_days_above_ema10": ride_days_above_ema10,
        "ride_days_above_ema20": ride_days_above_ema20,
        "ride_ema10_slope_vs_entry": ride_ema10_slope_vs_entry,
        "ride_obv_vs_entry": ride_obv_vs_entry,
        # Volume character
        "ride_pullback_volume_ratio": ride_pullback_volume_ratio,
        "ride_pullback_bar_count": ride_pullback_bar_count,
        # DNA context
        "dna_typical_pullback_pct": float(dna_typical_pullback),
        "dna_typical_run_length": float(dna_typical_run_length),
        "dna_pullback_recovery_rate": float(dna_pullback_recovery_rate),
        "dna_optimal_hold_days": float(dna_optimal_hold_days),
    }


# ---------------------------------------------------------------------------
# Combined feature row (market indicators + ride context)
# ---------------------------------------------------------------------------

def build_ride_feature_row(
    indicators_row: Mapping[str, Any],
    ride_context: Mapping[str, float],
) -> Dict[str, float]:
    """
    Merge curated v14 market features with ride context into one flat dict.
    NaN for any missing value — LightGBM handles missing natively.
    """
    market_features = build_curated_feature_row(indicators_row)
    row: Dict[str, float] = {}

    # Market features first (canonical order)
    for feat in CURATED_FEATURE_ORDER:
        row[feat] = market_features.get(feat, _nan())

    # Ride context features
    for feat in RIDE_CONTEXT_FEATURE_NAMES:
        raw = ride_context.get(feat, _nan())
        row[feat] = _sf(raw)

    return row


# ---------------------------------------------------------------------------
# Per-ticker training matrix
# ---------------------------------------------------------------------------

def build_ride_training_matrix(ticker: str) -> pd.DataFrame:
    """
    Build the complete labeled feature matrix for *ticker*.

    Returns a DataFrame with:
      - One row per labeled ride-day
      - All RIDE_FEATURE_NAMES columns (market + ride context)
      - ``label`` (str: HOLD/ADD/EXIT)
      - ``label_encoded`` (int: 0/1/2)
      - ``remaining_upside_pct`` (regression target)
      - ``ticker``, ``event_id``, ``bar_date`` (metadata — not features)

    Returns empty DataFrame if insufficient data.
    """
    ohlcv = load_ohlcv(ticker)
    if ohlcv.empty or len(ohlcv) < 150:
        return pd.DataFrame()

    # Load DNA for this ticker (best-effort)
    dna_ctx = _load_dna_context(ticker)

    # Compute indicators for the full history once
    try:
        indicators = compute_all_indicators(ohlcv)
    except Exception as exc:
        LOGGER.warning("%s: compute_all_indicators failed: %s", ticker, exc)
        return pd.DataFrame()

    # Detect all historical rides
    rides = detect_historical_rides(ticker, ohlcv)
    if not rides:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for ride in rides:
        labeled_days = label_ride_days(ride, ohlcv)
        for lday in labeled_days:
            # Market indicator snapshot at this bar
            if lday.bar_idx < len(indicators):
                ind_row = indicators.iloc[lday.bar_idx].to_dict()
            else:
                ind_row = {}

            # Ride context features
            rc = build_ride_context_features(
                lday,
                indicators_history=indicators,
                ohlcv=ohlcv,
                **dna_ctx,
            )

            # Combine
            feat_row = build_ride_feature_row(ind_row, rc)

            # Add labels and metadata
            feat_row["label"] = lday.label
            feat_row["label_encoded"] = LABEL_ENCODING.get(lday.label, 0)
            feat_row["remaining_upside_pct"] = lday.remaining_upside_pct
            feat_row["ticker"] = ticker
            feat_row["event_id"] = lday.event_id
            feat_row["bar_date"] = lday.bar_date.isoformat()
            feat_row["days_held"] = lday.days_held
            rows.append(feat_row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    LOGGER.info(
        "%s: ride training matrix  %d rows  labels=%s",
        ticker,
        len(df),
        df["label"].value_counts().to_dict() if "label" in df.columns else {},
    )
    return df


# ---------------------------------------------------------------------------
# DNA context loader
# ---------------------------------------------------------------------------

def _load_dna_context(ticker: str) -> Dict[str, Any]:
    """
    Load behavioral DNA statistics for *ticker* from DB.
    Returns safe defaults if DNA not yet built.
    """
    defaults: Dict[str, Any] = {
        "dna_typical_pullback": 3.5,
        "dna_typical_run_length": 25,
        "dna_pullback_recovery_rate": 0.65,
        "dna_optimal_hold_days": 40,
    }

    try:
        import json
        from app.core.database import query_one

        row = query_one(
            "SELECT dna_json FROM ee_dna_profiles WHERE ticker = ?",
            (ticker.upper(),),
        )
        if row is None or not row.get("dna_json"):
            return defaults

        dna = json.loads(row["dna_json"])
        pep = dna.get("pullback_entry_profile") or {}
        defaults["dna_typical_pullback"] = float(pep.get("median_pullback_pct") or 3.5)
        defaults["dna_typical_run_length"] = int(pep.get("recovery_days") or 25)
        defaults["dna_pullback_recovery_rate"] = float(pep.get("pullback_success_rate") or 0.65)
        defaults["dna_optimal_hold_days"] = int(dna.get("optimal_hold_window_days") or 40)
    except Exception:
        pass

    return defaults


# ---------------------------------------------------------------------------
# Pooled training matrix (all tickers)
# ---------------------------------------------------------------------------

def build_pooled_ride_training_matrix(tickers: List[str]) -> pd.DataFrame:
    """
    Build a pooled training matrix from all provided tickers.
    Skips tickers with insufficient data; logs progress.
    """
    frames: List[pd.DataFrame] = []
    for i, ticker in enumerate(tickers):
        try:
            df = build_ride_training_matrix(ticker)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            LOGGER.warning("Skipping %s during ride matrix build: %s", ticker, exc)

        if (i + 1) % 20 == 0:
            LOGGER.info("Ride matrix progress: %d/%d tickers", i + 1, len(tickers))

    if not frames:
        return pd.DataFrame()

    pooled = pd.concat(frames, ignore_index=True)
    LOGGER.info(
        "Pooled ride matrix: %d rows, %d tickers, labels=%s",
        len(pooled),
        pooled["ticker"].nunique(),
        pooled["label"].value_counts().to_dict() if "label" in pooled.columns else {},
    )
    return pooled

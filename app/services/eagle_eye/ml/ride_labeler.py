"""
Eagle Eye — Ride Quality Labeler  (Phase R1)
============================================

Detects every historical "ride" from forensic move events and labels
EVERY day of the ride with the correct hold decision:

  HOLD  — trend intact, stay in position
  ADD   — healthy pullback in uptrend, opportunity to increase
  EXIT  — trend weakening, protect profits

Also produces a regression target: ``remaining_upside_pct`` — how much
additional gain was available from that day forward.

Labeling is forward-looking (uses future data) and is ONLY used during
training.  At inference time, ``ride_evaluator.py`` runs the trained
model on live indicators + ride context.

Architecture
------------
  detect_historical_rides()   →  List[RideRecord]
  label_ride_days()           →  adds label column to ride DataFrame
  build_ride_training_matrix()→  full labeled feature table for one ticker
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.services.eagle_eye.move_detector import MoveEvent, detect_moves
from app.services.eagle_eye.store import load_ohlcv

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Label-forward windows (trading days)
HOLD_DD_LIMIT_DEFAULT: float = 5.0     # max drawdown before HOLD → EXIT
EXIT_DD_LIMIT_DEFAULT: float = 8.0     # drawdown that forces EXIT label
ADD_PULLBACK_ATR_MULT: float = 2.0     # approved global default multiplier
ADD_PULLBACK_FLOOR: float = 2.5        # approved minimum pullback floor (%)
ADD_RECOVERY_DAYS: int = 20            # window to recover and make new high
REMAINING_UPSIDE_WINDOW: int = 40      # days over which remaining_upside is measured
MAX_RIDE_DAYS: int = 180               # cap ride duration for labeling
MIN_RIDE_DAYS: int = 10                # skip rides shorter than this


def _atr_percent_series(ohlcv: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Compute ATR% (ATR / close * 100) with Wilder-style smoothing."""
    if ohlcv.empty:
        return np.array([], dtype=float)

    high = pd.to_numeric(ohlcv["high"], errors="coerce")
    low = pd.to_numeric(ohlcv["low"], errors="coerce")
    close = pd.to_numeric(ohlcv["close"], errors="coerce")

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
    atr_pct = (atr / close.replace(0, np.nan)) * 100.0
    return atr_pct.to_numpy(dtype=float)


@dataclass
class RideRecord:
    """A single detected historical ride extracted from a MoveEvent."""

    ticker: str
    event_id: str
    entry_date: date
    entry_idx: int          # integer position in OHLCV DataFrame
    entry_price: float
    peak_date: date
    peak_idx: int
    peak_price: float
    peak_gain_pct: float
    duration_days: int      # calendar days from entry to peak
    # Per-stock DNA-calibrated label thresholds
    hold_dd_limit: float = HOLD_DD_LIMIT_DEFAULT
    exit_dd_limit: float = EXIT_DD_LIMIT_DEFAULT
    max_dd_during_ride: float = 0.0
    # Running peak tracking during ride
    running_peak_prices: List[float] = field(default_factory=list)


@dataclass
class LabeledRideDay:
    """One training sample: one day during one historical ride."""

    ticker: str
    event_id: str
    bar_date: date
    bar_idx: int
    entry_price: float
    # Ride-state at this day
    days_held: int
    unrealized_pct: float
    peak_gain_pct: float            # best P&L during ride so far
    drawdown_from_peak: float       # current pullback from ride's running peak
    gain_velocity: float            # unrealized_pct / max(days_held, 1)
    # Labels (forward-looking — TRAINING ONLY)
    label: str                      # HOLD | ADD | EXIT
    remaining_upside_pct: float     # regression target
    remaining_max_dd_pct: float     # forward max drawdown (informational)
    makes_new_high_20d: bool


# ---------------------------------------------------------------------------
# DNA threshold calibration helpers
# ---------------------------------------------------------------------------

def _calibrate_thresholds(
    ticker: str,
    events: List[MoveEvent],
    ohlcv: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Derive per-stock label thresholds from historical pullback statistics.

    Returns (hold_dd_limit, exit_dd_limit).
    Falls back to global defaults if insufficient history.
    """
    if len(events) < 3 or ohlcv.empty:
        return HOLD_DD_LIMIT_DEFAULT, EXIT_DD_LIMIT_DEFAULT

    closes = ohlcv["close"].to_numpy(dtype=float)
    highs = ohlcv["high"].to_numpy(dtype=float)
    lows = ohlcv["low"].to_numpy(dtype=float)
    dates_arr = ohlcv.index.date

    pullbacks: List[float] = []
    for ev in events:
        # Find entry and peak indices in OHLCV
        entry_candidates = np.where(dates_arr == ev.acceleration_date)[0]
        peak_candidates = np.where(dates_arr == ev.peak_date)[0]
        if len(entry_candidates) == 0 or len(peak_candidates) == 0:
            continue
        ei = int(entry_candidates[0])
        pi = int(peak_candidates[0])
        if pi <= ei:
            continue
        slice_lows = lows[ei:pi + 1]
        slice_highs = highs[ei:pi + 1]
        if slice_highs.max() <= 0:
            continue
        running_max = np.maximum.accumulate(slice_highs)
        dd_series = (running_max - slice_lows) / running_max * 100.0
        pullbacks.append(float(np.nanmax(dd_series)))

    if len(pullbacks) < 3:
        return HOLD_DD_LIMIT_DEFAULT, EXIT_DD_LIMIT_DEFAULT

    median_pb = float(np.median(pullbacks))
    p75_pb = float(np.percentile(pullbacks, 75))

    # HOLD limit = median pullback (clamped 2-8%)
    hold_limit = float(np.clip(median_pb, 2.0, 8.0))
    # EXIT limit = 75th-percentile pullback (clamped 4-12%, always > hold)
    exit_limit = float(np.clip(max(p75_pb, hold_limit + 2.0), 4.0, 12.0))
    return hold_limit, exit_limit


# ---------------------------------------------------------------------------
# Ride detection
# ---------------------------------------------------------------------------

def detect_historical_rides(
    ticker: str,
    ohlcv: pd.DataFrame,
    *,
    min_gain_pct: float = 8.0,
) -> List[RideRecord]:
    """
    Detect all historical rides for *ticker* from MoveEvents.

    A "ride" starts at the acceleration_date of a MoveEvent (the day a
    detectable breakout began) and ends at the event's peak_date.
    Only moves with gain >= min_gain_pct and duration >= MIN_RIDE_DAYS
    are included.
    """
    if ohlcv.empty or len(ohlcv) < 100:
        return []

    dates_arr = ohlcv.index.date
    closes = ohlcv["close"].to_numpy(dtype=float)
    highs = ohlcv["high"].to_numpy(dtype=float)

    # Detect moves at the 8% threshold (smallest interesting trend)
    events = detect_moves(ticker, ohlcv, thresholds_pct=(min_gain_pct,))
    # Calibrate per-stock thresholds from all available events
    hold_dd, exit_dd = _calibrate_thresholds(ticker, events, ohlcv)

    rides: List[RideRecord] = []
    seen_entries: set = set()

    for ev in events:
        entry_date = ev.acceleration_date
        peak_date = ev.peak_date
        if ev.is_fakeout:
            continue
        if ev.gain_pct < min_gain_pct:
            continue

        entry_candidates = np.where(dates_arr == entry_date)[0]
        peak_candidates = np.where(dates_arr == peak_date)[0]
        if len(entry_candidates) == 0 or len(peak_candidates) == 0:
            continue

        ei = int(entry_candidates[0])
        pi = int(peak_candidates[0])
        duration = (peak_date - entry_date).days

        if duration < MIN_RIDE_DAYS or pi <= ei:
            continue
        if entry_date in seen_entries:
            continue
        seen_entries.add(entry_date)

        entry_price = float(closes[ei]) if closes[ei] > 0 else ev.acceleration_price
        peak_price = float(highs[pi])

        # Pre-compute running peak prices over the ride (for drawdown_from_peak feature)
        ride_highs = highs[ei:pi + 1]
        running_peak = np.maximum.accumulate(ride_highs).tolist()

        # Max drawdown during ride
        lows_slice = ohlcv["low"].to_numpy(dtype=float)[ei:pi + 1]
        ride_dd = float(np.nanmax((np.maximum.accumulate(ride_highs) - lows_slice) / np.maximum.accumulate(ride_highs) * 100.0))

        rides.append(
            RideRecord(
                ticker=ticker,
                event_id=ev.event_id,
                entry_date=entry_date,
                entry_idx=ei,
                entry_price=entry_price,
                peak_date=peak_date,
                peak_idx=pi,
                peak_price=peak_price,
                peak_gain_pct=ev.gain_pct,
                duration_days=duration,
                hold_dd_limit=hold_dd,
                exit_dd_limit=exit_dd,
                max_dd_during_ride=ride_dd,
                running_peak_prices=running_peak,
            )
        )

    LOGGER.debug("%s: detected %d historical rides", ticker, len(rides))
    return rides


# ---------------------------------------------------------------------------
# Ride-day labeling
# ---------------------------------------------------------------------------

def label_ride_days(
    ride: RideRecord,
    ohlcv: pd.DataFrame,
) -> List[LabeledRideDay]:
    """
    Label every day of a ride with HOLD / ADD / EXIT.

    Uses FUTURE price data — call this only during training.
    For each day D in [entry_idx, min(peak_idx, entry_idx+MAX_RIDE_DAYS)]:
      - Compute forward windows from D
      - Apply label logic (DNA-calibrated thresholds)
      - Record remaining_upside_pct regression target
    """
    closes = ohlcv["close"].to_numpy(dtype=float)
    highs = ohlcv["high"].to_numpy(dtype=float)
    lows = ohlcv["low"].to_numpy(dtype=float)
    dates_arr = ohlcv.index.date
    n = len(ohlcv)
    atr_pct_arr = _atr_percent_series(ohlcv, period=14)

    end_idx = min(ride.peak_idx, ride.entry_idx + MAX_RIDE_DAYS)
    labeled: List[LabeledRideDay] = []

    for day_offset in range(end_idx - ride.entry_idx + 1):
        day_idx = ride.entry_idx + day_offset

        if day_idx >= n - ADD_RECOVERY_DAYS - 1:
            # Not enough forward data to label reliably
            break

        close_now = closes[day_idx]
        if not math.isfinite(close_now) or close_now <= 0:
            continue

        # ── Ride-state features ───────────────────────────────────────────
        days_held = day_offset
        unrealized_pct = (close_now / ride.entry_price - 1.0) * 100.0

        # Running peak up to and including this day
        if day_offset < len(ride.running_peak_prices):
            running_peak_price = ride.running_peak_prices[day_offset]
        else:
            running_peak_price = float(np.nanmax(highs[ride.entry_idx:day_idx + 1]))

        if running_peak_price > 0:
            peak_gain_pct = (running_peak_price / ride.entry_price - 1.0) * 100.0
            drawdown_from_peak = (running_peak_price - close_now) / running_peak_price * 100.0
        else:
            peak_gain_pct = unrealized_pct
            drawdown_from_peak = 0.0

        gain_velocity = unrealized_pct / max(days_held, 1)

        # ── Forward windows (FUTURE DATA — training labels only) ──────────
        fwd_20_end = min(day_idx + 21, n)
        fwd_40_end = min(day_idx + 41, n)

        future_highs_20 = highs[day_idx + 1:fwd_20_end]
        future_lows_20 = lows[day_idx + 1:fwd_20_end]
        future_highs_40 = highs[day_idx + 1:fwd_40_end]

        if len(future_highs_20) < 5:
            # Not enough forward bars — skip
            break

        # Historical high up to and including today (for "new high" test)
        historical_high = float(np.nanmax(highs[ride.entry_idx:day_idx + 1]))
        if not math.isfinite(historical_high):
            historical_high = close_now

        remaining_max_gain = float((np.nanmax(future_highs_40) / close_now - 1.0) * 100.0) if len(future_highs_40) > 0 else 0.0
        remaining_max_dd = float((1.0 - np.nanmin(future_lows_20) / close_now) * 100.0) if len(future_lows_20) > 0 else 0.0
        makes_new_high_20d = bool(np.nanmax(future_highs_20) > historical_high) if len(future_highs_20) > 0 else False

        # ── Label logic (DNA-calibrated) ──────────────────────────────────
        hold_dd = ride.hold_dd_limit
        exit_dd = ride.exit_dd_limit
        atr_pct = float("nan")
        if day_idx < len(atr_pct_arr):
            atr_pct = float(atr_pct_arr[day_idx])

        if math.isfinite(atr_pct) and atr_pct > 0:
            add_pullback_min = max(ADD_PULLBACK_FLOOR, ADD_PULLBACK_ATR_MULT * atr_pct)
        else:
            add_pullback_min = ADD_PULLBACK_FLOOR

        if remaining_max_dd > exit_dd:
            label = "EXIT"
        elif makes_new_high_20d and remaining_max_dd < hold_dd:
            if drawdown_from_peak > add_pullback_min and remaining_max_gain > 5.0:
                label = "ADD"
            else:
                label = "HOLD"
        elif remaining_max_gain > remaining_max_dd * 2.0:
            label = "HOLD"
        else:
            label = "EXIT"

        labeled.append(
            LabeledRideDay(
                ticker=ride.ticker,
                event_id=ride.event_id,
                bar_date=dates_arr[day_idx],
                bar_idx=day_idx,
                entry_price=ride.entry_price,
                days_held=days_held,
                unrealized_pct=round(unrealized_pct, 4),
                peak_gain_pct=round(peak_gain_pct, 4),
                drawdown_from_peak=round(drawdown_from_peak, 4),
                gain_velocity=round(gain_velocity, 6),
                label=label,
                remaining_upside_pct=round(remaining_max_gain, 4),
                remaining_max_dd_pct=round(remaining_max_dd, 4),
                makes_new_high_20d=makes_new_high_20d,
            )
        )

    return labeled


# ---------------------------------------------------------------------------
# Label distribution audit
# ---------------------------------------------------------------------------

def audit_label_distribution(labeled_days: List[LabeledRideDay]) -> Dict[str, Any]:
    """Summarize label counts and class balance for QA."""
    if not labeled_days:
        return {"total": 0, "hold": 0, "add": 0, "exit": 0, "hold_pct": 0, "add_pct": 0, "exit_pct": 0}

    counts = {"HOLD": 0, "ADD": 0, "EXIT": 0}
    for d in labeled_days:
        counts[d.label] = counts.get(d.label, 0) + 1

    total = len(labeled_days)
    return {
        "total": total,
        "hold": counts["HOLD"],
        "add": counts["ADD"],
        "exit": counts["EXIT"],
        "hold_pct": round(counts["HOLD"] / total * 100, 1),
        "add_pct": round(counts["ADD"] / total * 100, 1),
        "exit_pct": round(counts["EXIT"] / total * 100, 1),
    }


# ---------------------------------------------------------------------------
# Full pipeline: ticker → List[LabeledRideDay]
# ---------------------------------------------------------------------------

def build_ride_labeled_days(ticker: str) -> List[LabeledRideDay]:
    """
    End-to-end: load OHLCV, detect rides, label every day.
    Convenience function for the trainer to call per-ticker.
    """
    ohlcv = load_ohlcv(ticker)
    if ohlcv.empty:
        LOGGER.warning("%s: no OHLCV data, skipping ride labeling", ticker)
        return []

    rides = detect_historical_rides(ticker, ohlcv)
    if not rides:
        return []

    all_days: List[LabeledRideDay] = []
    for ride in rides:
        days = label_ride_days(ride, ohlcv)
        all_days.extend(days)

    LOGGER.debug(
        "%s: %d rides → %d labeled ride-days  %s",
        ticker,
        len(rides),
        len(all_days),
        audit_label_distribution(all_days),
    )
    return all_days

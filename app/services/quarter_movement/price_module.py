"""
Module 1: Quarterly Price Movement.

Computes quarterly high/low percentage changes from the baseline closing price
for each calendar quarter from 2023-Q1 to the current date.

Input:
    daily_records: list of dicts with keys:
        date (ISO str "YYYY-MM-DD"), high (float), low (float), close (float).
        Missing or non-positive fields are skipped per spec §1.2.
    today: current system date.

Output dict keys:
    years                        — list[int] of years present in the table
    price_movement_table         — {year_str: {q1..q4: QuarterPriceCell | None}}
    price_movement_means         — {q1..q4: QuarterPriceMeanCell}
    active_quarter_baseline_price — float | None (baseline close for forecast)

QuarterPriceCell:
    high_pct        — ((max_high - baseline) / baseline) × 100, 1 d.p., or None
    low_pct         — ((min_low  - baseline) / baseline) × 100, 1 d.p., or None
    in_progress     — True when the quarter end date is after today
    insufficient_data — True when data was missing; cell is excluded from means

Known limitations:
    * If a baseline date falls on a non-trading day the most recent prior
      trading day is used (spec §1.3).
    * In-progress quarter cells are computed but excluded from mean calculation
      (spec §5.2).
    * Quarters with fewer than two complete historical instances carry an
      asterisk flag in the mean cell (spec §7.3).
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Quarter boundary constants ─────────────────────────────────────────────

_QUARTER_START_MONTHS = {"q1": 1, "q2": 4, "q3": 7, "q4": 10}

_QUARTER_END_MONTH_DAY: Dict[str, Tuple[int, int]] = {
    "q1": (3, 31),
    "q2": (6, 30),
    "q3": (9, 30),
    "q4": (12, 31),
}

_QUARTERS_ORDERED = ("q1", "q2", "q3", "q4")


# ── Date helpers ──────────────────────────────────────────────────────────


def _quarter_start_date(year: int, quarter_key: str) -> date:
    return date(year, _QUARTER_START_MONTHS[quarter_key], 1)


def _quarter_end_date(year: int, quarter_key: str) -> date:
    end_month, end_day = _QUARTER_END_MONTH_DAY[quarter_key]
    return date(year, end_month, end_day)


def _baseline_date_for_quarter(year: int, quarter_key: str) -> date:
    """Nominal (calendar) baseline date per spec §1.3."""
    if quarter_key == "q1":
        return date(year - 1, 12, 31)   # Dec 31 prior year
    elif quarter_key == "q2":
        return date(year, 3, 31)
    elif quarter_key == "q3":
        return date(year, 6, 30)
    else:
        return date(year, 9, 30)


def _active_quarter_for_date(d: date) -> Tuple[int, str]:
    month = d.month
    if month <= 3:
        return d.year, "q1"
    elif month <= 6:
        return d.year, "q2"
    elif month <= 9:
        return d.year, "q3"
    return d.year, "q4"


def _enumerate_quarters_since_2023(today: date) -> List[Tuple[int, str]]:
    """All quarters from Q1-2023 through the active quarter, inclusive."""
    active_year, active_q = _active_quarter_for_date(today)
    result: List[Tuple[int, str]] = []
    for year in range(2023, active_year + 1):
        for q_key in _QUARTERS_ORDERED:
            result.append((year, q_key))
            if year == active_year and q_key == active_q:
                return result
    return result


# ── Module ────────────────────────────────────────────────────────────────


class QuarterlyPriceMovementModule:
    """
    Module 1: Quarterly Price Movement.
    Computes high/low percentage changes relative to each quarter's baseline price.
    """

    def compute(
        self,
        daily_records: List[Dict],
        today: date,
    ) -> Dict:
        """
        Build the quarterly price movement table and per-quarter-type means.

        Args:
            daily_records: OHLCV records (any order; sorted internally).
                           Each record needs: date, high, low, close.
            today:         Current system date used to detect in-progress quarters.

        Returns:
            Dict with keys: years, price_movement_table, price_movement_means,
            active_quarter_baseline_price.
        """
        # ── Build date-indexed price lookup ──────────────────────────────
        price_lookup: Dict[str, Dict[str, float]] = {}
        for record in daily_records:
            date_str = record.get("date")
            if not date_str:
                continue
            high_val = record.get("high")
            low_val = record.get("low")
            close_val = record.get("close")
            # Skip dates with any missing price field (spec §1.2)
            if high_val is None or low_val is None or close_val is None:
                continue
            try:
                price_lookup[date_str] = {
                    "high": float(high_val),
                    "low": float(low_val),
                    "close": float(close_val),
                }
            except (TypeError, ValueError):
                continue

        all_trading_dates = sorted(price_lookup.keys())

        def find_baseline_close(nominal_baseline: date) -> Optional[float]:
            """
            Return the closing price on the most recent trading day on or before
            nominal_baseline (spec §1.3 non-trading-day fallback).
            """
            iso_target = nominal_baseline.isoformat()
            for trading_date in reversed(all_trading_dates):
                if trading_date <= iso_target:
                    return price_lookup[trading_date]["close"]
            return None

        # ── Iterate all quarters from 2023 through active quarter ────────
        quarters = _enumerate_quarters_since_2023(today)
        active_year, active_quarter_key = _active_quarter_for_date(today)
        years_in_range = sorted({y for y, _ in quarters})

        price_movement_table: Dict[str, Dict[str, Optional[Dict]]] = {
            str(year): {q: None for q in _QUARTERS_ORDERED}
            for year in years_in_range
        }

        active_quarter_baseline_price: Optional[float] = None

        for year, quarter_key in quarters:
            nominal_baseline = _baseline_date_for_quarter(year, quarter_key)
            baseline_closing_price = find_baseline_close(nominal_baseline)

            if baseline_closing_price is None or baseline_closing_price == 0:
                logger.warning(
                    "No baseline price for %s %s (nominal baseline %s) — cell omitted",
                    year, quarter_key, nominal_baseline,
                )
                price_movement_table[str(year)][quarter_key] = {
                    "high_pct": None,
                    "low_pct": None,
                    "in_progress": False,
                    "insufficient_data": True,
                }
                continue

            quarter_start = _quarter_start_date(year, quarter_key)
            quarter_end = _quarter_end_date(year, quarter_key)
            is_in_progress = quarter_end > today

            quarter_start_iso = quarter_start.isoformat()
            # For in-progress quarters, only use data up to today
            effective_end_iso = min(quarter_end.isoformat(), today.isoformat())

            # ── Find max high and min low within the quarter period ───────
            quarter_highest_price: Optional[float] = None
            quarter_lowest_price: Optional[float] = None

            for trading_date, prices in price_lookup.items():
                if quarter_start_iso <= trading_date <= effective_end_iso:
                    h = prices["high"]
                    l = prices["low"]
                    if quarter_highest_price is None or h > quarter_highest_price:
                        quarter_highest_price = h
                    if quarter_lowest_price is None or l < quarter_lowest_price:
                        quarter_lowest_price = l

            if quarter_highest_price is None or quarter_lowest_price is None:
                price_movement_table[str(year)][quarter_key] = {
                    "high_pct": None,
                    "low_pct": None,
                    "in_progress": is_in_progress,
                    "insufficient_data": True,
                }
                continue

            # ── Apply formulas from spec §2.1 and §2.2 ───────────────────
            # §2.1: ((max_high - baseline) / baseline) × 100
            highest_price_percentage_increase = round(
                ((quarter_highest_price - baseline_closing_price) / baseline_closing_price) * 100.0,
                1,
            )
            # §2.2: ((min_low - baseline) / baseline) × 100  (negative sign preserved)
            lowest_price_percentage_change = round(
                ((quarter_lowest_price - baseline_closing_price) / baseline_closing_price) * 100.0,
                1,
            )

            price_movement_table[str(year)][quarter_key] = {
                "high_pct": highest_price_percentage_increase,
                "low_pct": lowest_price_percentage_change,
                "in_progress": is_in_progress,
                "insufficient_data": False,
            }

            if year == active_year and quarter_key == active_quarter_key:
                active_quarter_baseline_price = baseline_closing_price

        # ── Compute arithmetic means per quarter type (spec §2.4) ────────
        # Exclude in-progress quarters and quarters with insufficient data.
        price_movement_means: Dict[str, Dict] = {}
        for q_key in _QUARTERS_ORDERED:
            high_pct_values: List[float] = []
            low_pct_values: List[float] = []

            for year in years_in_range:
                cell = price_movement_table[str(year)].get(q_key)
                if cell is None or cell.get("insufficient_data") or cell.get("in_progress"):
                    continue
                high_v = cell.get("high_pct")
                low_v = cell.get("low_pct")
                if high_v is not None:
                    high_pct_values.append(high_v)
                if low_v is not None:
                    low_pct_values.append(low_v)

            high_pct_mean = (
                round(sum(high_pct_values) / len(high_pct_values), 1)
                if high_pct_values else None
            )
            low_pct_mean = (
                round(sum(low_pct_values) / len(low_pct_values), 1)
                if low_pct_values else None
            )
            # Spec §7.3: flag reduced sample sizes (< 2 complete years)
            reduced_sample = len(high_pct_values) < 2

            price_movement_means[q_key] = {
                "high_pct_mean": high_pct_mean,
                "low_pct_mean": low_pct_mean,
                "reduced_sample": reduced_sample,    # asterisk flag per §7.3
                "sample_count": len(high_pct_values),
            }

        return {
            "years": years_in_range,
            "price_movement_table": price_movement_table,
            "price_movement_means": price_movement_means,
            "active_quarter_baseline_price": active_quarter_baseline_price,
        }
